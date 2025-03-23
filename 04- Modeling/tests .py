import os
import random
import shutil


def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits each image class in the source_dir into train, validation, and test directories.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    classes = [d for d in os.listdir(
        source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for class_name in classes:
        class_src = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(
            class_src) if os.path.isfile(os.path.join(class_src, f))]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, file_list in splits.items():
            class_dst = os.path.join(output_dir, split, class_name)
            if not os.path.exists(class_dst):
                os.makedirs(class_dst)
            for file_name in file_list:
                src_file = os.path.join(class_src, file_name)
                dst_file = os.path.join(class_dst, file_name)
                shutil.copy2(src_file, dst_file)
    print(f"Dataset split into train, validation, and test in '{output_dir}'.")


# Paths to dataset directory
Base_Folder = 'D:/Learning/University of sadat/Grade 4/Semester 2/06- Graduation Project/Coding/'
original_dataset_dir = f'{Base_Folder}00- The DataSet/00- Dogs Femur Fracture'
split_dataset_dir = f'{Base_Folder}00- The DataSet/Dataset_split'
