import copy
import gc
import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
from _kernel_lib import _exp_val
from packaging import version
from scipy.special import binom
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .utils import safe_isinstance
from .utils._legacy import (
    convert_to_data,
    convert_to_instance,
    convert_to_link,
    convert_to_model,
    match_instance_to_data,
    match_model_to_data,
)

log = logging.getLogger("shap")


class KernelExplainer():
    """Uses the Kernel SHAP method to explain the output of any function.

    Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature. The computed importance values
    are Shapley values from game theory and also coefficients from a local linear
    regression.

    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame or shap.common.DenseData or any scipy.sparse matrix
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. For small problems,
        this background dataset can be the whole training set, but for larger problems consider
        using a single reference value or using the ``kmeans`` function to summarize the dataset.
        Note: for the sparse case, we accept any sparse matrix but convert to lil format for
        performance.

    feature_names : list
        The names of the features in the background dataset. If the background dataset is
        supplied as a pandas.DataFrame, then ``feature_names`` can be set to ``None`` (default),
        and the feature names will be taken as the column names of the dataframe.

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the output with a link function where link(output) = sum(phi).
        Default is "identity" (a no-op).
        If the model output is a probability, then "logit" can be used to transform the SHAP values
        into log-odds units.

    Examples
    --------
    See :ref:`Kernel Explainer Examples <kernel_explainer_examples>`.

    """

    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):
        # if feature_names is not None:
        #     print('2')
        #     self.data_feature_names = feature_names
        # elif isinstance(data, pd.DataFrame):
        #     print('3')
        #     self.data_feature_names = list(data.columns)

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        # Model là 1 class trong file _legacy.py công dụng của nó là gọi hàm model.f tương tức là gọi hàm prediction mà chúng ta mong muốn
        self.model = convert_to_model(model, keep_index=self.keep_index)
        # self.data là vector binary 0 đầu tiên để model convert và tính toán các thuộc tính cần thiết
        self.data = convert_to_data(data, keep_index=self.keep_index)
        # Nếu không có gì thay đổi thì đây chính là expected value do kq trả về là phép tính đầu tiên của binary vector 0 
        model_null = match_model_to_data(self.model, self.data)

        # enforce our current input type limitations
        # if not isinstance(self.data, (DenseData, SparseData)):
        #     emsg = "Shap explainer only supports the DenseData and SparseData input currently."
        #     raise TypeError(emsg)
        # if self.data.transposed:
        #     emsg = "Shap explainer does not support transposed DenseData or SparseData currently."
        #     raise DimensionError(emsg)

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning(
                "Using "
                + str(len(self.data.weights))
                + " background data samples could cause "
                + "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to "
                + "summarize the background as K samples."
            )

        # init our parameters
        # Nếu là 1 ảnh thì thì N là số ảnh và P chính là số segment
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        # if isinstance(model_null, (pd.DataFrame, pd.Series)):
        #     model_null = np.squeeze(model_null.values)
        # if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
        #     model_null = model_null.numpy()
        # elif safe_isinstance(model_null, "tensorflow.python.framework.ops.SymbolicTensor"):
        #     model_null = self._convert_symbolic_tensor(model_null)


        # trong numpy . T là để chuyển vịm và weights khởi tạo là 1.0, nếu có 1 ảnh thì không cần quan tâm 
        # model_null là giá trị [số ảnh, giá trị output của số lớp]
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]
        
   

    

    def shap_values(self, X, **kwargs):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "aic", "bic", or float
            The l1 regularization to use for feature selection. The estimation
            procedure is based on a debiased lasso.

            * "num_features(int)" selects a fixed number of top features.
            * "aic" and "bic" options use the AIC and BIC rules for regularization.
            * Passing a float directly sets the "alpha" parameter of the
              ``sklearn.linear_model.Lasso`` model used for feature selection.
            * "auto" (deprecated): uses "aic" when less than
              20% of the possible sample space is enumerated, otherwise it uses
              no regularization.

            .. versionchanged:: 0.47.0
                The default value changed from ``"auto"`` to ``"num_features(10)"``.

        silent: bool
            If True, hide tqdm progress bar. Default False.

        gc_collect : bool
           Run garbage collection after each explanation round. Sometime needed for memory intensive explanations (default False).

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as the ``expected_value``
            attribute of the explainer).

            The type and shape of the return value depends on the number of model inputs and outputs:

            * one input, one output: array of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: array of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs: list of arrays of corresponding shape above.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs and one input changed from list to np.ndarray.

        """
       
        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            explanation = self.explain(data, **kwargs)
            # vector-output
            s = explanation.shape
            out = np.zeros(s)
            out[:] = explanation
            return out
        # explain the whole dataset
        elif len(X.shape) == 2:
            # Hình ảnh shape là 2 bao gồm segment và số ảnh (1,42)
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X[i : i + 1, :]
                explanations.append(self.explain(data, **kwargs))
                if kwargs.get("gc_collect", False):
                    gc.collect()

            # vector-output
            s = explanations[0].shape
            # Kiểm tra xem giá trị có thuốc (42,2) tương đương giá trị segment và số lớp không
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i][:, j]
                outs = np.stack(outs, axis=-1)
                # Trả về mảng 3 chiều (số ảnh, số segment, số lớp)
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i]
                return out

        else:
            raise "Instance must have 1 or 2 dimensions!"

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        # input incoming là 1 vector (1)  đại diện cho 1 hình ảnh chuẩn
        instance = convert_to_instance(incoming_instance)
        # instance sẽ là 1 class gồm 2 thuộc tính . class là vector1 và group_display_values = None
        match_instance_to_data(instance, self.data)
        # Trả về instance.group_display_values là 1 mảng gồm n_segments với mỗi segment là 1 np.floast(1.0)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        # instance.x là  ma trận 2 chiều bao gồm số ảnh và số segment với giá trị là binary vector full 1 -> ảnh gốc
        # Hàm này giá trị trả về mục đích nó tìm hiểu là coi có feature nào không quan trọng, có thể bỏ qua bước tính toán không
        # Đối với hình ảnh thì cái nào cũng quan trọng
        self.varyingInds = self.varying_groups(instance.x)

        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()
        # Self.M tức là số features cần tính toán là 42 -> y như cũ
        # Self.varyingFeatureGroups -> sẽ là 1 mảng numpy chứa  index của các segment (features) 
        # Lưu ý mảng này mảng 1 chiều và phần tử đầu tiên được gán label là 0 -> có thể đây chính là thứ gây ra nguyên nhân start_label = 1

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            # Tính toán giá trị logic gốc.
            model_out = self.model.f(instance.x)
        # Hình ảnh thì không có vô đoạn code phía dưới
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        elif safe_isinstance(model_out, "tensorflow.python.framework.ops.SymbolicTensor"):
            model_out = self._convert_symbolic_tensor(model_out)
        # Lấy giá trị predict thứ nhất cho ảnh 1
        self.fx = model_out[0]
        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            # Bài toán hình ảnh thì vô đây.
            # self.l1_reg là ràng buộc L1 chỉ chọn 10 features đặc biệt
            self.l1_reg = kwargs.get("l1_reg", "num_features(10)")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2**30
            if self.M <= 30:
                self.max_samples = 2**self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            # Khởi tạo các giá trị cần thiết
            self.allocate()

            # weight the different subset sizes
            # Vì tính đối xứng, 1 liên minh có K features có trọng số tương ứng, nên chỉ cần lặp qua num_subset_sizes là được
            # np.ceil để làm tròn
            # Vì tính chất đối xứng lên chỉ cần chạy đến segment/2 là đủ
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            # Lấy các cặp đối sứng ví dụ 1 và self.M -1  và trong pool self.M chỉ có num_paired cặp đối xứng
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            # 3 hàm dưới để gán trọng số, lí do mất cái số cách chọn vì lúc này
            # Ta không quan tâm đến 1 giá trị feature nào đó, ta chỉ quan tâm trọng số khi nếu S có 1 có 2 hoặc có i lần.
            # Ví dụ A,B,C nếu chọn phần tử đầu A là thì trọng số cũng tương tự như B như C => nên là coi như chúng là 1
            # Công thức kernelshap weight cũng khử đi xác suất tổ hợp 0 và tổ hợp full 1
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            # Vì tính chất đối xứng nên nổ lực để lấy cái tổ hợp gốc và phần bù trở nên x2 
            weight_vector[:num_paired_subset_sizes] *= 2
            # Sum lại để ép xác suất về 1 xác suất thể hiện nên lấy mẫu như thế nào
            weight_vector /= np.sum(weight_vector)
            log.debug(f"{weight_vector = }")
            log.debug(f"{num_subset_sizes = }")
            log.debug(f"{num_paired_subset_sizes = }")
            log.debug(f"{self.M = }")

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            # Tạo ra 1 mảng chưa các segments -> segment 1 được chuyển thành 0 nhé
            group_inds = np.arange(self.M, dtype="int64")
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):
                # determine how many subsets (and their complements) are of the current size
                # Thuạt toán này là có bao nhiêu cách chọn n phần tử trong N tập hợp
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    # Nhân đôi để tính phần bù
                    nsubsets *= 2
                log.debug(f"{subset_size = }")
                log.debug(f"{nsubsets = }")
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1] = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1]}"
                )
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1]/nsubsets = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets}"
                )

                # see if we have enough samples to enumerate all subsets of this size
                # Hàm này là nó sẽ lấy giá trị còn lại của n_samples chưa sử dụng * với xác suất của index  S(liên minh) cặp tương ứng dung *2
                # Sau đó nó chia với số cách chọn (nsubsets) của index tiếp theo của Liên minh
                
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    # Ví dụ với cách chọn 1 ta sẽ có prop
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    # Chuẩn  hóa về 1 để các prop tiếp theo không bị sai lệch
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

                    # add all the samples of the current subset size
                    # Phân bổ cái prop từ ban đầu chia đều cho các thành viên
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        # Tại vì lấy phần bù nên phải chia 2 đó cu
                        w /= 2.0
                    # group_inds = số segment nhớ là phần tử đầu là 0
                    # duyệt qua tất cả các segment á mà
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype="int64")] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info(f"{num_full_subsets = }")

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug(f"{samples_left = }")
            if num_full_subsets != num_subset_sizes:
                # weight vector vẫn là phân phối ban đầu.
                remaining_weight_vector = copy.copy(weight_vector)
                # Chia 2 để không cần tính phần bù
                remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
                # num_full_subsets là coi đã xử lí tổ hợp S nào rồi á
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                # Chia lại phân phối để ép vào 1, bởi vì khi ta ép lại xác suất và để tính khi đã loại bỏ 1 tập hợp S ra 
                # Thì weight_vector gốc chưa được thay đổi.
                # Nên thành ra xuống đây khi lấy xác suất của phần tử nào thỏa ra thì phải ép xác suất lại
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info(f"{remaining_weight_vector = }")
                log.info(f"{num_paired_subset_sizes = }")
                # Chọn ra các tổ hợp liên minh
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                consecutive_failures = 0
                MAX_FAILURES = 500
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    if consecutive_failures > MAX_FAILURES:
                        log.warning(f"Stopping random sampling early due to {consecutive_failures} consecutive collisions.")
                        break
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                    # ind có thể = 0 và num_full_sets cũng có thể = 0 cho nên +1 thêm để ra tổ hợp liên minh hợp lí nhất
                    subset_size = ind + num_full_subsets + 1
                    # Tạo ra 1 vector nhị phân mà chỉ hiển thị subset_size (Kích thước liên minh) segmentation
                    # Lưu ý là việc chọn segment nào là ngãu nhiên, miễn sao đủ số lượng là được
                    # Lưu ý là ngẫu nhiên có trọng số, nhưng do vì không đủ samples nên chỉ chọn đúng 1 tổ hợp S là [0] số lượng là subsersize
                    # print(f"Debug Kernel: M={self.M}, subset_size={subset_size}, type={type(subset_size)}")
                    # print(f"Debug Mask Shape: {mask.shape}")
                    mask[np.random.permutation(self.M)[:int(subset_size)]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    # nếu mẫu này chưa có thì add
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0)
                        consecutive_failures = 0
                    else:
                        # Nếu mẫu này có rồi thì +1 vô, nhớ là so sánh = tức là số phần tử, trọng số... mới gọi là = nha
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0
                        consecutive_failures += 1

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0
                    ind_set_pos += 1

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info(f"{weight_left = }")
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()
            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var
                #vphi_var có thể sử dụng làm phương sai

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    @staticmethod
    def not_equal(i, j):
        number_types = (int, float, np.number)
        if isinstance(i, number_types) and isinstance(j, number_types):
            return 0 if np.allclose(i, j, equal_nan=True) else 1
        elif hasattr(i, "dtype") and hasattr(j, "dtype"):
            if np.issubdtype(i.dtype, np.number) and np.issubdtype(j.dtype, np.number):
                return 0 if np.allclose(i, j, equal_nan=True) else 1
            if np.issubdtype(i.dtype, np.bool_) and np.issubdtype(j.dtype, np.bool_):
                return 0 if np.allclose(i, j, equal_nan=True) else 1
            return 0 if all(i == j) else 1
        else:
            return 0 if i == j else 1

    def varying_groups(self, x):
        # Groupsize = segmentation size -> tạo ra dãy numpy [0] với n_segments
        varying = np.zeros(self.data.groups_size)
        for i in range(self.data.groups_size):
            # lấy idx = label của segment
            inds = self.data.groups[i]
            # lấy giá trị của segment -> luôn là 1.0
            x_group = x[0, inds]
            if scipy.sparse.issparse(x_group):
                if all(j not in x.nonzero()[1] for j in inds):
                    varying[i] = False
                    continue
                x_group = x_group.todense()
            varying[i] = self.not_equal(x_group, self.data.data[:, inds])
        varying_indices = np.nonzero(varying)[0]
        return varying_indices
       

    def allocate(self):
        # if scipy.sparse.issparse(self.data.data):
        #     # We tile the sparse matrix in csr format but convert it to lil
        #     # for performance when adding samples
        #     print('allocate1')

        #     shape = self.data.data.shape
        #     nnz = self.data.data.nnz
        #     data_rows, data_cols = shape
        #     rows = data_rows * self.nsamples
        #     shape = rows, data_cols
        #     if nnz == 0:
        #         self.synth_data = scipy.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
        #     else:
        #         data = self.data.data.data
        #         indices = self.data.data.indices
        #         indptr = self.data.data.indptr
        #         last_indptr_idx = indptr[len(indptr) - 1]
        #         indptr_wo_last = indptr[:-1]
        #         new_indptrs = []
        #         for i in range(self.nsamples - 1):
        #             new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
        #         new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
        #         new_indptr = np.concatenate(new_indptrs)
        #         new_data = np.tile(data, self.nsamples)
        #         new_indices = np.tile(indices, self.nsamples)
        #         self.synth_data = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        # else:
        #     self.synth_data = np.tile(self.data.data, (self.nsamples, 1))
        
        # dùng để chứa hình ảnh bị nhiễu 
        self.synth_data = np.tile(self.data.data, (self.nsamples, 1))
        # Tạo mask => Binary nhị phân với n_samples tương ứng
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        # Tạo kernel_weights cho từng tổ hợp Coaliation
        self.kernelWeights = np.zeros(self.nsamples)
        # self.D là số chiều output, self.y là kết quả dự đoán của N samples, và cho N ảnh mong muốn
        self.y = np.zeros((self.nsamples * self.N, self.D))
        # Tương tự như trên nhưng chỉ cho 1 ảnh 
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

       
    def addsample(self, x, m, w):
        # Self.N là số ảnh muốn giải đó
        # nsamplesAdded là số ảnh được thêm vào
        offset = self.nsamplesAdded * self.N

        if isinstance(self.varyingFeatureGroups, (list,)):
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset : offset + self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            # mặt nạ boolean mới ( chố nào = 1 thì giá trị True)
            mask = m == 1.0
            # Phần tử nào true mới được giữ lại
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    self.synth_data[offset : offset + self.N, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if scipy.sparse.issparse(x) and not scipy.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset : offset + self.N, groups] = evaluation_data
        # Ma trận binary vector nè
        self.maskMatrix[self.nsamplesAdded, :] = m
        # Thêm weight dựa vào giá trị phân bổ đã được chia ra
        # Lưu ý KernelWeight này khác cái xác suất để chọn nha
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        # Self.N là số ảnh muốn giải thích, không biết nsamplesRun là gì nhưng nó = 0
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :]
        # Bước này để predict trên các tập mẫu nè
        # Data ở đây là 360 binary vector dựa theo các chiến lược đó ku

        modelOut = self.model.f(data)
        # Self.y bây giờ kết quả dự đoán của N_samples với  D output
        self.y[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))
        # find the expected value of each output
        # self.samplesRun là các mẫu đã được tạo rồi, còn self.y về lý thuyết là tính kỳ vọng
        # nhưng shape vẫn là (n_samples, class)
        self.ey, self.nsamplesRun = _exp_val(
            self.nsamplesRun, self.nsamplesAdded, self.D, self.N, self.data.weights, self.y, self.ey
        )
        # ey_array = np.asarray(self.ey)
        # print(ey_array[0], self.y[0]) 
        # Tính trung bình có trọng số nhưng chắc tại N = 1 nên trung bình như nhau
        
       

    def solve(self, fraction_evaluated, dim):
        # dim là  số chiều out_put á nha
        # self.ey trong đây là giá trị y chang self.y
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])

      
        #  self.maskMatrix là 1 mặt nạ nhị phân đại diện cho S ngẫu nhiên
        # s là tính tổng số 1 -> tại vì chỉ có 0,1 nên sum = số feature được xuất hiên
        # Lưu ý là self.maskMatrix nếu theo chiến lược 2 thì nó chỉ chọn đúng 1 tổ hợp S
        # Và phần bù 1 - S lý do là phải chọn đủ mẫu cho 1 tổ hợp á.       
        s = np.sum(self.maskMatrix, 1)
              

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        # Chọn các phần tử không bị loại khỏi danh sách
        log.debug(f"{fraction_evaluated = }")
        if self.l1_reg == "auto":
            warnings.warn("l1_reg='auto' is deprecated and will be removed in a future version.", DeprecationWarning)
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            # self.M - S sẽ ra 1 kích thước của 1 liên minh bù, ví dụ  s là (1,42,1,42,29,12) -> thì self.M - s sẽ là (42,1,42,1,12,29)
            # self.KernelWeihts sẽ nhân trọng số với phần bù, tại vì theo tính chất kernelWeight thì trọng số S và 1 - S = nhau
            # Cuối cùng là w_aug sẽ tạo ra 1 mảng sếp chồng theo chiều ngang
            # Nối chúng lại sẽ có 1 ngàn mẫu nhiễu với trọng số tương ứng được tính từ kernelWeights
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            
            log.info(f"{np.sum(w_aug) = }")
            log.info(f"{np.sum(self.kernelWeights) = }")
            # Căn bậc 2, lý do là chúng ta muốn giải quyết bài toán giảm thiểu sai số của [weight * sai số ^2]
            # mà sckit-learn không phù hợp nên chia căn 2 (sqrt(w) * Y - sqrt(w) * X * β)² = w * (Y - X * β)²
            # Nói chung là để đưa vào bình phương á mà, hệ hệ.
            w_sqrt_aug = np.sqrt(w_aug)

            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            # eyAjd_aug chính là giá trị Dự đoán - baseline * với trọng số
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            # vector nhị phân hoàn chỉnh của bộ dữ liệu tăng cường
            # var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                # Thằng r là ràng buộc cho L1 regulization, tức là chỉ chọn ra 10 features quan trọng nhất
                # có thể thay đổi ở l1_reg
                r = int(self.l1_reg[len("num_features(") : -1])
                # Least Angle Regression thứ này sẽ tính các segment quan trọng -> giảm kích thước
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg in ("auto", "bic", "aic"):

                c = "aic" if self.l1_reg == "auto" else self.l1_reg

                # "Normalize" parameter of LassoLarsIC was deprecated in sklearn version 1.2
                if version.parse(sklearn.__version__) < version.parse("1.2.0"):

                    kwg = dict(normalize=False)
                else:
                    kwg = {}
                model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion=c, **kwg))
                nonzero_inds = np.nonzero(model.fit(mask_aug, eyAdj_aug)[1].coef_)[0]

            # use a fixed regularization coefficient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        # Cập nhật lại danh sách chưa các segment quan trọng nhất chính là nonzero_inds
        # Nếu không thì lấy mặc định
        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        # Lý do sử dụng M-1 là kiểu như phương pháp thế y = 3x vào phuognw trình x + y/3x(stand_for) = 10 
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])
        )
        # chuyền về ma trận có M-1 segment (số lượng ảnh, segment đã trích lọc - nonzero_inds)
        # Etmp là 1 vector nhị phân được tính dựa vào tham chiếu
        # self.maskMatrix[:, , nonzero_inds[:-1]] nó sẽ lấy trạng tái bật/tắt của các feature quan trọng mà Lasso đã tìm ra
        # Sau đó trừ đi trạng thái của feature cuối cùng là nonzero_inds[-1] để làm tham chiếu.
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])

        log.debug(f"{etmp[:4, :] = }")

        # solve a weighted least squares equation to estimate phi
        # least squares:
        #     phi = min_w ||W^(1/2) (y - X w)||^2
        # the corresponding normal equation:
        #     (X' W X) phi = X' W y
        # with
        #     X = etmp
        #     W = np.diag(self.kernelWeights)
        #     y = eyAdj2
        #
        # We could just rely on sciki-learn
        #     from sklearn.linear_model import LinearRegression
        #     lm = LinearRegression(fit_intercept=False).fit(etmp, eyAdj2, sample_weight=self.kernelWeights)
        # Under the hood, as of scikit-learn version 1.3, LinearRegression still uses np.linalg.lstsq and
        # there are more performant options. See https://github.com/scikit-learn/scikit-learn/issues/22855.
        # Y = kết quả
        y = np.asarray(eyAdj2)
        # X = ma trận binayry với các segment đã thay đổi trạng thái so với tham chiếu. 
        # Bao gồm 3 giá trị 1 là cùng trái chiều dương, 0 là cùng chiều, -1 là trái chiều âm
        X = etmp
        # Trọng số nhân với ma trận binary 
        WX = self.kernelWeights[:, None] * X
        try:
            w = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            # XWX = np.linalg.pinv(X.T @ WX)
            # w = np.dot(XWX, np.dot(np.transpose(WX), y))
            sqrt_W = np.sqrt(self.kernelWeights)
            w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]
        log.debug(f"{np.sum(w) = }")
        log.debug(
            f"self.link(self.fx) - self.link(self.fnull) = {self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])}"
        )
        log.debug(f"self.fx = {self.fx[dim]}")
        log.debug(f"self.link(self.fx) = {self.link.f(self.fx[dim])}")
        log.debug(f"self.fnull = {self.fnull[dim]}")
        log.debug(f"self.link(self.fnull) = {self.link.f(self.fnull[dim])}")
        phi = np.zeros(self.M)
        # Các giá trị từ 0 đến len - 1 = w
        phi[nonzero_inds[:-1]] = w
        # F(x) - f(expect) - sum(w)
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info(f"{phi = }")

        # clean up any rounding errors
        for i in range(self.M):
            # Giá trị shapely bé hơn 1*10^-10 = 0
            # Cập nhật phi về lại các segment gốc, tức là segment nào không được đề cập thì = 0,
            # Ví dụ ở trên là 500,10 -> 500,42
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
