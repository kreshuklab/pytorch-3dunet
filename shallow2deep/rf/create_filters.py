import argparse
import os

import h5py
import numpy as np
from elf.parallel.filters import apply_filter
from skimage.segmentation.boundaries import find_boundaries

from shallow2deep.rf.rf_utils import MULTI_FILTERS, FILTER_NAMES

SIGMAS = [0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
SIGMAS_GAUSSIAN_SMOOTHING = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]


def compute_boundaries(x):
    dims = x.ndim
    assert dims in [2, 3]
    if dims == 2:
        boundaries = find_boundaries(x, connectivity=2, mode="thick")
    else:
        boundaries = np.array(
            [
                find_boundaries(x[i], connectivity=2, mode="thick")
                for i in range(x.shape[0])
            ]
        )

    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Compute filters")
    parser.add_argument("--source_path", type=str, required=True, help="Data path for volume")
    parser.add_argument("--save_path", type=str, required=True, help="Save path for volume with filters")
    parser.add_argument("--raw_internal_path", type=str, default="raw", help="Internal path for raw data")
    parser.add_argument("--label_internal_path", type=str, default="label", help="Internal path for label data")
    parser.add_argument(
        "--compute_boundaries",
        action="store_true",
        help="Set flag to compute boundaries for gt",
    )

    args = parser.parse_args()

    source_path = args.source_path
    save_path = args.save_path
    raw_int_path = args.raw_internal_path
    label_int_path = args.label_internal_path

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with h5py.File(source_path, "r") as file:
        raw = (
            file[raw_int_path][:]
            if file[raw_int_path].ndim > 2
            else file[raw_int_path][:][None, :]
        ).astype("float32")
        label = (
            file[label_int_path][:]
            if file[label_int_path].ndim > 2
            else file[label_int_path][:][None, :]
        )
    if args.compute_boundaries:
        label = compute_boundaries(label)
    else:
        label = label

    print("Saving to: ", save_path)
    with h5py.File(save_path, "a") as file:
        file.create_dataset(
            "raw",
            data=raw,
            compression="gzip",
        )

        file.create_dataset(
            "label",
            data=label,
            compression="gzip",
        )

    slices = [[i, i + 1] for i in range(raw.shape[0])]

    for filter_name in FILTER_NAMES:

        if filter_name == "gaussianSmoothing":
            filter_sigmas = SIGMAS_GAUSSIAN_SMOOTHING
        else:
            filter_sigmas = SIGMAS

        for sigma in filter_sigmas:
            out_file = compute_filter_for_volume(
                raw, filter_name, slices, sigma
            )

            with h5py.File(save_path, "a") as file:
                file.create_dataset(
                    f"{filter_name}_sigma_{sigma}", data=out_file, compression="gzip",
                )

            print(f"{filter_name} with sigma {sigma} computed")


def compute_filter_for_volume(raw_data, filtr, slices, sigma):
    if filtr in MULTI_FILTERS:
        out_file = np.empty((2,) + raw_data.shape)
    else:
        out_file = np.empty(raw_data.shape)

    for (slice_start, slice_end) in slices:

        curr_slice = raw_data[slice_start:slice_end]
        if not filtr in MULTI_FILTERS:
            out_d = np.empty(out_file.shape[1:])
        else:
            out_d = np.zeros((out_file.shape[0],) + out_file.shape[2:])

        filtered_raw = apply_filter(
            curr_slice[0],
            filtr,
            sigma,
            out=out_d,
            block_shape=(curr_slice[0].shape[0], curr_slice[0].shape[1]),
            outer_scale=2 * sigma if filtr == "structureTensorEigenvalues" else None,
        )
        if filtered_raw.ndim == 2:
            out_file[slice_start:slice_end] = filtered_raw[None, :].copy()
        elif filtered_raw.ndim == 3:
            out_file[:, slice_start:slice_end, :] = filtered_raw[:, None, :].copy()
        else:
            out_file[:] = filtered_raw[:].copy()

    return out_file


if __name__ == "__main__":
    main()
