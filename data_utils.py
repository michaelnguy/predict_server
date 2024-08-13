from constants import DATA_DIR, GT_TAGS_FILE_NAME, IMG_PATHS_FILE_NAME
import random
import json
import os


def split_data_indices(
    data_size, data_types, data_probs, random_seed=0, fp_to_write=None
):
    data_idx = {dt: [] for dt in data_types}
    random.seed(random_seed)

    for i in range(data_size):
        chosen_data_type = random.choices(data_types, weights=data_probs)[0]
        data_idx[chosen_data_type].append(i)

    if fp_to_write is not None:
        with open(fp_to_write, "w") as f:
            json.dump(data_idx, f, indent=2)
        print("Wrote data indices split to file:", fp_to_write)
    return data_idx


def get_batch_idx(batch_size, data_size):
    batch_is = range(0, data_size, batch_size)
    batch_js = range(batch_size, data_size + batch_size, batch_size)
    batches = list(zip(batch_is, batch_js))
    return batches


def load_image_paths(file_path=os.path.join("../HARRISON", IMG_PATHS_FILE_NAME)):
    print("Loading image paths from file:", file_path)
    with open(file_path) as f:
        img_fps = f.readlines()
    img_fps = [fp.strip() for fp in img_fps]
    print("Finished loading %d image paths" % len(img_fps))
    return img_fps


def load_image_tags(file_path=os.path.join("../HARRISON", "/tag_list.txt")):
    print("Loading image tags from file:", file_path)
    with open(file_path) as f:
        img_tags = f.readlines()
    img_tags = [tag.strip().split() for tag in img_tags]
    print("Finished loading tags for %d images" % len(img_tags))
    return img_tags
