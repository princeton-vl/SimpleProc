
# Evaluate MVSAnywhere Model on RobustMVD Benchmark
Clone RobustMVD: https://github.com/lmb-freiburg/robustmvd.

Follow its instruction to download the datasets. And for ETH3D please use the undistorted images and undistort the depth maps (e.g., according to https://github.com/sadrasafa/StereoGS/issues/4). We provide a list of all used images and depth maps in `SimpleProc/data_lists/data_lists.txt`. You can delete unneeded data not listed in this file, but make sure you have all the listed data.

We have the sample_list of both our validation split and test split in `SimpleProc/data_lists`. You need replace the `robustmvd-dev/rmvd/data/sample_lists` with either `sample_lists_val` or `sample_lists_test`.  Note that for ScanNet our selection is different from official RMVD benchmark.

Please see the RobustMVD repository for evaluation instruction. Make sure you point the dataset path to your downloaded dataset.


Note that several edits are needed:

- Edit `rmvd/data/eth3d.py` to recognize `.npy`.

```
class ETH3DDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        if self.path.endswith('.npy'):
            depth = np.load(osp.join(root, self.path))
        else:
            height, width = 4032, 6048
            depth = np.fromfile(osp.join(root, self.path), dtype=np.float32).reshape(height, width)
            depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
            depth = np.expand_dims(depth, 0)  # 1HW
        return depth
```

- add `drop_quantile` to `valid_mean` to make the metric more robust to outliers in the ground truth of ETH3D.
```
def valid_mean(arr, mask, axis=None, keepdims=np._NoValue, drop_quantile=None):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    if drop_quantile is not None:
        if not (0 <= drop_quantile < 1):
            raise ValueError("drop_quantile must be in the interval [0, 1).")
        valid_mask = mask > 0
        if np.any(valid_mask):
            arr_valid = np.where(valid_mask, arr, np.nan)
            q = np.nanquantile(arr_valid, 1 - drop_quantile, axis=axis)

            if axis is None:
                q_broadcast = q
            else:
                axis_tuple = (axis,) if not isinstance(axis, tuple) else axis
                axis_tuple = tuple([a if a >= 0 else arr.ndim + a for a in axis_tuple])
                q_broadcast = q
                for ax in sorted(axis_tuple):
                    q_broadcast = np.expand_dims(q_broadcast, ax)

            mask = mask * (arr <= q_broadcast)
        else:
            mask = mask * 0
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, copy=False, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid
```

In `def m_rel_ae`, call it with drop_quantile for ETH3D:
```
    if rel_ae.shape[0] > 3000:
        m_rel_ae, valid = valid_mean(rel_ae, mask, drop_quantile=0.01)
    else:
        m_rel_ae, valid = valid_mean(rel_ae, mask)
```