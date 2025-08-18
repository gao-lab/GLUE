import time

import joblib
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm


def tf_idf(data, scale_factor=100000, idf=None):
    sparse_input = issparse(data)

    if idf is None:
        # add small value in case down sample creates empty feature
        _col_sum = data.sum(axis=0)
        if sparse_input:
            col_sum = _col_sum.A1.astype(np.float32) + 0.00001
        else:
            col_sum = _col_sum.ravel().astype(np.float32) + 0.00001
        idf = np.log(1 + data.shape[0] / col_sum).astype(np.float32)
    else:
        idf = idf.astype(np.float32)

    _row_sum = data.sum(axis=1)
    if sparse_input:
        row_sum = _row_sum.A1.astype(np.float32) + 0.00001
    else:
        row_sum = _row_sum.ravel().astype(np.float32) + 0.00001

    tf = data.astype(np.float32)

    if sparse_input:
        tf.data = tf.data / np.repeat(row_sum, tf.getnnz(axis=1))
        tf.data = np.log1p(np.multiply(tf.data, scale_factor, dtype="float32"))
        tf = tf.multiply(idf).tocsr()
    else:
        tf = tf / row_sum[:, np.newaxis]
        tf = np.log1p(np.multiply(tf, scale_factor, dtype="float32"))
        tf = tf * idf
    return tf, idf


class LSI:
    def __init__(
        self,
        scale_factor=100000,
        n_components=100,
        algorithm="arpack",
        random_state=0,
        n_iter=5,
        idf=None,
        model=None,
    ):
        self.scale_factor = scale_factor
        if idf is not None:
            self.idf = idf.copy()
        else:
            self.idf = None
        if model is not None:
            self.model = model
        else:
            self.model = TruncatedSVD(
                n_components=n_components,
                n_iter=n_iter,
                algorithm=algorithm,
                random_state=random_state,
            )
        self.random_state = random_state
        self.fitted = False

    def _downsample_data(self, data, downsample):
        np.random.seed(self.random_state)
        if downsample is not None and downsample < data.shape[0]:
            use_row_idx = np.sort(
                np.random.choice(np.arange(0, data.shape[0]), downsample, replace=False)
            )
            data = csr_matrix(data[use_row_idx, :])
        return data

    @staticmethod
    def _get_data(data):
        if isinstance(data, AnnData):
            data = data.X
        return data

    def fit(self, data, downsample=None, verbose=False):
        data = self._get_data(data)
        data = self._downsample_data(data, downsample)
        if verbose:
            print("start tf idf")
            start = time.time()
        tf, idf = tf_idf(data, self.scale_factor)
        if self.idf is not None:
            idf = self.idf.copy()
        if verbose:
            print("finish tf idf", time.time() - start)
            start = time.time()
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows - 1, n_cols - 1, self.model.n_components)
        self.model.fit(tf)
        if verbose:
            print("finish svd", time.time() - start)
        self.fitted = True
        return self

    def fit_transform(
        self, data, downsample=None, obsm_name="X_lsi", verbose=False, normalize=True
    ):
        _data = self._get_data(data)
        _data = self._downsample_data(_data, downsample)
        start = time.time()
        if verbose:
            print("start tf idf")
        tf, idf = tf_idf(_data, self.scale_factor)
        if self.idf is not None:
            idf = self.idf.copy()
        if verbose:
            print("finish tf idf", time.time() - start)
            start = time.time()
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows - 1, n_cols - 1, self.model.n_components)
        tf_reduce = self.model.fit_transform(tf)
        self.fitted = True
        if normalize:
            tf_reduce = tf_reduce / self.model.singular_values_
        if verbose:
            print("finish svd", time.time() - start)
        if isinstance(data, AnnData):
            data.obsm[obsm_name] = tf_reduce
        else:
            return tf_reduce

    def transform(
        self, data, chunk_size=50000, obsm_name="X_lsi", verbose=False, normalize=True
    ):
        _data = self._get_data(data)

        check_is_fitted(self.model)
        tf_reduce = []
        if verbose:
            chunks = tqdm(np.arange(0, _data.shape[0], chunk_size))
        else:
            chunks = np.arange(0, _data.shape[0], chunk_size)
        for chunk_start in chunks:
            tf, _ = tf_idf(
                _data[chunk_start : (chunk_start + chunk_size)],
                self.scale_factor,
                self.idf,
            )
            tf_reduce.append(self.model.transform(tf))

        tf_reduce = np.concatenate(tf_reduce, axis=0)

        if normalize:
            tf_reduce = tf_reduce / self.model.singular_values_

        if isinstance(data, AnnData):
            data.obsm[obsm_name] = tf_reduce
        else:
            return tf_reduce

    def save(self, path):
        joblib.dump(self, path)
