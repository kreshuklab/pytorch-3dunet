import argparse
import os
import pickle

import h5py
import numpy as np

from shallow2deep.rf import rf_utils


def main():
    parser = argparse.ArgumentParser(description="RF prediction")
    parser.add_argument("--volume_path", required=True, type=str, help="Path to volume with filters")
    parser.add_argument("--rf_path", type=str, required=True, help="Path to rf folder or to a single rf")
    parser.add_argument("--save_folder", type=str, required=True, help="Path prediction folder")
    parser.add_argument("--raw_internal_path", type=str, default="raw", help="Raw internal path")
    parser.add_argument("--label_internal_path", type=str, default="label", help="Ground truth internal path")
    parser.add_argument("--unequal_filter_order", action="store_true",
                        help="Set this flag if not all rfs were trained with same filter names and order")
    args = parser.parse_args()

    volume_path = args.volume_path
    rf_path = args.rf_path
    save_folder = args.save_folder
    raw_internal_path = args.raw_internal_path
    label_internal_path = args.label_internal_path
    unequal_filter_order = args.unequal_filter_order

    print(f"Generating predictions for file {volume_path}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if os.path.isdir(rf_path):
        rf_filepaths = [os.path.join(os.path.dirname(rf_path), file) for file in os.listdir(rf_path)]
    else:
        rf_filepaths = [rf_path]

    volume_name = volume_path.split("/")[-1].split(".")[0]

    with h5py.File(volume_path, "r") as f:
        assert f[raw_internal_path].shape == f[label_internal_path].shape
        raw = f[raw_internal_path][:]
        label = f[label_internal_path][:]

    if not unequal_filter_order:
        filter_names = pickle.load(open(rf_filepaths[0], "rb"))["filter_names"]
        volume_slices = rf_utils.load_whole_volume(
            path=volume_path, raw_internal_path=raw_internal_path, filter_names=filter_names
        )

    for idx, rf_filepath in enumerate(rf_filepaths):

        rf_model_name = os.path.basename(rf_filepath).split(".")[0]
        save_path = os.path.join(save_folder, f"{volume_name}_{rf_model_name}.h5")

        print(f"Predicting with RF from {rf_filepath} and saving to {save_path} ({idx+1} out of {len(rf_filepaths)})")

        rf_data = pickle.load(open(rf_filepath, "rb"))

        if unequal_filter_order:
            volume_slices = rf_utils.load_whole_volume(
                path=volume_path, raw_internal_path=raw_internal_path, filter_names=rf_data["filter_names"]
            )

        pred_slices = rf_utils.predict_with_rf(rf_data["rf_model"], volume_slices)

        assert len(pred_slices) == label.shape[0], (len(pred_slices), label.shape[0])

        pred_volume = np.array([pred_slice.reshape(label[idx].shape) for idx, pred_slice in enumerate(pred_slices)])

        with h5py.File(save_path, "a") as f:
            f.create_dataset("rf_pred", data=pred_volume, compression="gzip")
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("orig_label", data=label, compression="gzip")


if __name__ == "__main__":
    main()
