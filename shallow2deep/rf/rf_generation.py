import argparse
import os

from shallow2deep.rf import rf_utils


def main():
    parser = argparse.ArgumentParser(description="RF generation")
    parser.add_argument("--volume_path", required=True, type=str, help="Path to volume with filters")
    parser.add_argument("--rf_save_folder", required=True, type=str, help="Folder where to safe RF's")
    parser.add_argument(
        "--rf_num", type=int, required=True, help="Number of RF's to create",
    )
    parser.add_argument(
        "--min_patch_len", type=int, default=100, help="Minimum Quadratic patch side-length",
    )
    parser.add_argument(
        "--max_patch_len", type=int, default=300, help="Maximum Quadratic patch side-length",
    )
    parser.add_argument("--label_internal_path", type=str, default="label", help="Internal ground truth path")
    args = parser.parse_args()

    volume_path = args.volume_path
    rf_save_folder = args.rf_save_folder

    label_internal_path = args.label_internal_path

    min_patch_len = args.min_patch_len
    max_patch_len = args.max_patch_len
    rf_num = args.rf_num

    assert rf_num > 0
    assert max_patch_len > min_patch_len
    assert min_patch_len > 0

    if not os.path.exists(rf_save_folder):
        os.makedirs(rf_save_folder)

    volume_name = volume_path.split("/")[-1].split(".")[0]

    patch_generator = rf_utils.create_patch_generator(
        path=volume_path,
        label_internal_path=label_internal_path, min_patch_len=min_patch_len,
        max_patch_len=max_patch_len, patch_num=rf_num
    )

    for i, (train_data, label_data, coordinate_data, filter_names) in enumerate(patch_generator):
        save_path = os.path.join(rf_save_folder, f"rf_{i}_" + volume_name + ".pkl")
        print(f"Generating RF {i} with patch from {coordinate_data} and saving to {save_path}...")

        rf_utils.train_and_save_rf(
            train_data=train_data,
            label_data=label_data,
            coordinate_data=coordinate_data,
            filter_names=filter_names,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
