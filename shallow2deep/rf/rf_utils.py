import h5py
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MULTI_FILTERS = ["hessianOfGaussianEigenvalues", "structureTensorEigenvalues"]

FILTER_NAMES = [
    "gaussianSmoothing",
    "gaussianGradientMagnitude",
    "hessianOfGaussianEigenvalues",
    "structureTensorEigenvalues",
    "laplacianOfGaussian",
]


def create_patch_generator(
        path, label_internal_path, min_patch_len, max_patch_len, patch_num
):
    with h5py.File(path, "r") as f:
        volume_shape = np.array(
            f[label_internal_path].shape
            if f[label_internal_path].ndim > 2
            else (1,) + f[label_internal_path].shape
        )

    with h5py.File(path, "r") as f:
        dataset_keys = list(f.keys())
        dataset_keys.sort()

    i = 0
    while True:
        curr_patch_len_x = np.random.randint(
            min_patch_len, min(max_patch_len, volume_shape[1])
        )
        curr_patch_len_z = np.random.randint(
            min_patch_len, min(max_patch_len, volume_shape[2])
        )

        x_coord = np.random.randint(0, volume_shape[0])
        y_coord = np.random.randint(0, volume_shape[1] - curr_patch_len_x)
        z_coord = np.random.randint(0, volume_shape[2] - curr_patch_len_z)

        patch_coordinates = [
            [x_coord, x_coord + 1],
            [y_coord, y_coord + curr_patch_len_x],
            [z_coord, z_coord + curr_patch_len_z],
        ]

        train_data, label_data, coordinate_data, filter_names = load_patch_data(
            path, patch_coordinates, dataset_keys, label_internal_path
        )

        if len(np.unique(label_data)) == 1:
            continue

        yield train_data, label_data, coordinate_data, filter_names
        i += 1

        if i == patch_num:
            break


def load_filter(h5_path, filter_name, coordinates):
    with h5py.File(h5_path, "r") as f:

        dims = f[filter_name].ndim

        [x1, x2], [y1, y2], [z1, z2] = coordinates

        assert dims in [2, 3, 4], f"Dimensions of filter are {dims} but should be between 2 and 4"

        if dims == 2:
            filter_vol = f[filter_name][y1:y2, z1:z2][None, None, :]

        elif dims == 3:

            if filter_name in MULTI_FILTERS:
                filter_vol = f[filter_name][:, y1:y2, z1:z2][:, None, :]

            else:
                filter_vol = f[filter_name][x1:x2, y1:y2, z1:z2][None, :]

        else:
            filter_vol = f[filter_name][:, x1:x2, y1:y2, z1:z2]

    return filter_vol


def load_patch_data(path, patch_coordinates, dataset_keys, label_internal_path):
    rf_training_data = []
    filter_names = []
    for filter_name in dataset_keys:
        if filter_name == label_internal_path:
            continue

        filter_volume = load_filter(h5_path=path, filter_name=filter_name, coordinates=patch_coordinates)
        for filter_volume_channel in filter_volume:
            rf_training_data.append(filter_volume_channel.flatten())

        filter_names.append(filter_name)

    labels = np.array(
        load_filter(h5_path=path, filter_name=label_internal_path, coordinates=patch_coordinates)).flatten()

    return np.array(rf_training_data).T, labels, patch_coordinates, filter_names


def train_and_save_rf(
        train_data,
        label_data,
        coordinate_data,
        filter_names,
        save_path

):
    rf_model = RandomForestClassifier(n_jobs=-1)
    rf_model.fit(train_data, label_data)
    pickle.dump({"rf_model": rf_model, "coordinate_data": coordinate_data, "filter_names": filter_names},
                open(save_path, "wb"))


def load_whole_volume(
        path, raw_internal_path, filter_names
):
    volume_slices = []

    with h5py.File(path, "r") as f:
        volume_shape = np.array(
            f[raw_internal_path].shape
            if f[raw_internal_path].ndim > 2
            else (1,) + f[raw_internal_path].shape
        )

    slices = (
        [[idx, idx + 1] for idx in range(volume_shape[0])]

    )

    print(f"Loading filters for all slices of file: {path}")
    for i, (slice_start, slice_end) in enumerate(slices):
        slice_data = []
        print(f"loading filters for slice {i} of {volume_shape[0] - 1}")

        slice_coordinatess = [
            [slice_start, slice_end],
            [0, volume_shape[1]],
            [0, volume_shape[2]],
        ]

        for filter_name in filter_names:

            filter_volume = load_filter(path, filter_name, slice_coordinatess)

            for filter_volume_channel in filter_volume:
                slice_data.append(filter_volume_channel.flatten())

        volume_slices.append(np.array(slice_data).T)

    return volume_slices


def predict_with_rf(rf_model, volume_slices):
    pred_data = []

    for volume_slice in volume_slices:
        pred_data.append(rf_model.predict_proba(volume_slice)[:, 1])

    return pred_data
