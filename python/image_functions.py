"""Functions for finding and handling images"""

import nibabel as nib
import cv2
import os
import tensorflow as tf
import numpy as np
import itertools
import random
from sklearn.model_selection import train_test_split

_IMG_DIRS = {
    "MRI-POS": r"C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\positiiviset",
    "MRI-NEG": r"C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\negatiiviset",
    "PET-POS": r"C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\positiiviset",
    "PET-NEG": r"C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\negatiiviset",
    "MASK": r"C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\positiiviset",
    "DATA-ROOT": r"C:\Users\henri\Desktop\Koulu\Gradu\data",
}
_IMG_DIMS = (224, 224)
_BATCH_SIZE = 36


def fetch_imgs(start_dir: str, file_contains: str, ext: str) -> list:

    images = []
    for patient in os.listdir(start_dir):
        folders = os.listdir(start_dir + "\\" + patient)

        try:
            target_folder = [elem for elem in folders if file_contains in elem.lower()][
                0
            ]
            files = os.listdir(start_dir + "\\" + patient + "\\" + target_folder)
            target_file = [elem for elem in files if ext in elem][0]
            images.append(
                start_dir + "\\" + patient + "\\" + target_folder + "\\" + target_file
            )

        except IndexError:
            continue

    return images


def fetch_slices(mri: list, pet: list, masks=None) -> list:
    def _is_consistent(mri, pet, mask=None):
        if mask is not None:
            assert (
                mri.shape[2] == pet.shape[2] and mri.shape[2] == mask.shape[2]
            ), "Image dimensions are not consistent"
        else:
            assert mri.shape[2] == pet.shape[2], "Image dimensions are not consistent"

    def get_global_minmax(image):
        image_max = [np.max(image[:, :, index]) for index in range(image.shape[2])]
        image_min = [np.min(image[:, :, index]) for index in range(image.shape[2])]
        return image_max, image_min

    slices = []
    global_pet = {"max": [], "min": []}
    global_mri = {"max": [], "min": []}

    if masks:
        for image_index in range(len(mri)):
            mri_data = nib.load(mri[image_index]).get_fdata()
            pet_data = nib.load(pet[image_index]).get_fdata()
            mask_data = nib.load(masks[image_index]).get_fdata()
            _is_consistent(mri_data, pet_data, mask_data)

            max_pet, min_pet = get_global_minmax(pet_data)
            global_pet["max"].append(max_pet)
            global_pet["min"].append(min_pet)

            max_mri, min_mri = get_global_minmax(mri_data)
            global_mri["max"].append(max_mri)
            global_mri["min"].append(min_mri)

            slices.append(
                {
                    mri[image_index]: mri_data,
                    pet[image_index]: pet_data,
                    masks[image_index]: mask_data,
                }
            )

    else:
        for image_index in range(len(mri)):
            mri_data = nib.load(mri[image_index]).get_fdata()
            pet_data = nib.load(pet[image_index]).get_fdata()
            _is_consistent(mri_data, pet_data)

            max_pet, min_pet = get_global_minmax(pet_data)
            global_pet["max"].append(max_pet)
            global_pet["min"].append(min_pet)

            max_mri, min_mri = get_global_minmax(mri_data)
            global_mri["max"].append(max_mri)
            global_mri["min"].append(min_mri)

            slices.append(
                {
                    mri[image_index]: mri_data,
                    pet[image_index]: pet_data,
                }
            )

    return slices, global_pet, global_mri


def save_imgs(train: tuple, test: tuple, dir):

    train_dir = dir + r"\train"
    test_dir = dir + r"\test"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        os.mkdir(train_dir + r"\Pos")
        os.mkdir(train_dir + r"\Neg")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        os.mkdir(test_dir + r"\Pos")
        os.mkdir(test_dir + r"\Neg")

    for index, label in enumerate(train[1]):
        if label == "Pos":
            cv2.imwrite(train_dir + f"\\Pos\\mri_pet_{index}.png", train[0][index])
        else:
            cv2.imwrite(train_dir + f"\\Neg\\mri_pet_{index}.png", train[0][index])

    for index, label in enumerate(test[1]):
        if label == "Pos":
            cv2.imwrite(test_dir + f"\\Pos\\mri_pet_{index}.png", test[0][index])
        else:
            cv2.imwrite(test_dir + f"\\Neg\\mri_pet_{index}.png", test[0][index])


def build_img(data: list, mri_names: str, pet_names: str, resize_dims: tuple) -> list:

    tensors = []
    for image_index, dictionary in enumerate(data):
        for slice in range(dictionary[mri_names[image_index]].shape[2]):
            img_array = np.dstack(
                (
                    cv2.resize(
                        dictionary[mri_names[image_index]][:, :, slice],
                        resize_dims,
                    ),
                    np.zeros(resize_dims),
                    cv2.resize(
                        dictionary[pet_names[image_index]][:, :, slice],
                        resize_dims,
                    ),
                    # cv2.resize(
                    #     normalize(dictionary[pet_names[image_index]][:, :, slice], mri=False),
                    #     resize_dims,
                    # ),  
                )
            )
            tensors.append(img_array)

    return tensors


def normalize(mat, mri=True):
    if mri:
        return (
            (mat - min_maxes["global_mri_min"])
            / (min_maxes["global_mri_max"] - min_maxes["global_mri_min"])
            * 255.0
        )
    else:
        return (
            (mat - min_maxes["global_pet_min"])
            / (min_maxes["global_pet_max"] - min_maxes["global_pet_min"])
            * 255.0
        )


# globaalisti?
def zscore(mat):
    return (mat - np.mean(mat)) / np.var(mat)


def extract_positives(data: dict, mris: str, pets: str, masks: str) -> list:

    positive_slices = []

    for image_index, dictionary in enumerate(data):
        placeholder = {
            mris[image_index]: None,
            pets[image_index]: None,
            "z-coordinate": [],
        }
        for slice_index in range(dictionary[mris[image_index]].shape[2]):
            # jos maskin summa > x ota mukaan
            if dictionary[masks[image_index]][:, :, slice_index].any():
                if placeholder[mris[image_index]] is None:
                    placeholder[mris[image_index]] = dictionary[mris[image_index]][
                        :, :, slice_index
                    ]
                    placeholder[pets[image_index]] = dictionary[pets[image_index]][
                        :, :, slice_index
                    ]
                else:
                    placeholder[mris[image_index]] = np.dstack(
                        (
                            placeholder[mris[image_index]],
                            dictionary[mris[image_index]][:, :, slice_index],
                        )
                    )
                    placeholder[pets[image_index]] = np.dstack(
                        (
                            placeholder[pets[image_index]],
                            dictionary[pets[image_index]][:, :, slice_index],
                        )
                    )
                placeholder["z-coordinate"].append(slice_index)
        positive_slices.append(placeholder)

    return positive_slices


def extract_negatives(
    num_slices: int, negatives: list, mris: str, pets: str, seed: int
) -> list:

    random.seed(seed)
    negative_slices = []

    for image_index, dictionary in enumerate(negatives):

        placeholder = {
            mris[image_index]: None,
            pets[image_index]: None,
            "z-coordinate": [],
        }

        # Leikkeiden rajaus?
        z_coordinates = random.sample(
            list(range(round(dictionary[mris[image_index]].shape[2] * 0.8))), num_slices
        )
        placeholder[mris[image_index]] = dictionary[mris[image_index]][
            :, :, z_coordinates
        ]
        placeholder[pets[image_index]] = dictionary[pets[image_index]][
            :, :, z_coordinates
        ]
        placeholder["z-coordinate"] = z_coordinates

        negative_slices.append(placeholder)

    return negative_slices


def create_dataset(set_seed: int):

    mri_pos = fetch_imgs(_IMG_DIRS["MRI-POS"], "mri", ".img")
    mri_neg = fetch_imgs(_IMG_DIRS["MRI-NEG"], "mri", ".img")
    pet_pos = fetch_imgs(_IMG_DIRS["PET-POS"], "pet", ".img")
    pet_neg = fetch_imgs(_IMG_DIRS["PET-NEG"], "pet", ".img")
    masks = fetch_imgs(_IMG_DIRS["MASK"], "maski", ".img")

    positives, global_pos_pet, global_pos_mri = fetch_slices(mri_pos, pet_pos, masks)
    negatives, global_neg_pet, global_neg_mri = fetch_slices(mri_neg, pet_neg)

    global min_maxes
    min_maxes = {
        "global_pet_max": np.max(
            [item for sublist in global_pos_pet["max"] for item in sublist]
            + [item for sublist in global_neg_pet["max"] for item in sublist]
        ),
        "global_pet_min": np.min(
            [item for sublist in global_pos_pet["min"] for item in sublist]
            + [item for sublist in global_neg_pet["min"] for item in sublist]
        ),
        "global_mri_max": np.max(
            [item for sublist in global_pos_mri["max"] for item in sublist]
            + [item for sublist in global_neg_mri["max"] for item in sublist]
        ),
        "global_mri_min": np.min(
            [item for sublist in global_pos_mri["min"] for item in sublist]
            + [item for sublist in global_neg_mri["min"] for item in sublist]
        ),
    }

    positive_slices = extract_positives(positives, mri_pos, pet_pos, masks)
    num_positives = len(
        list(
            itertools.chain(
                *[dictionary["z-coordinate"] for dictionary in positive_slices]
            )
        )
    )

    slice_per_patient = round(num_positives / len(negatives))
    negative_slices = extract_negatives(
        slice_per_patient, negatives, mri_neg, pet_neg, set_seed
    )

    pos_examples = build_img(positive_slices, mri_pos, pet_pos, _IMG_DIMS)
    neg_examples = build_img(negative_slices, mri_neg, pet_neg, _IMG_DIMS)

    examples = pos_examples + neg_examples
    labels = [1] * len(pos_examples) + [0] * len(neg_examples)

    train_examples, test_examples, train_labels, test_labels = train_test_split(
        examples, labels, random_state=set_seed, test_size=0.3
    )

    train_examples, val_examples, train_labels, val_labels = train_test_split(
        train_examples, train_labels, random_state=set_seed, test_size=0.2
    )
    
    num_val = len(val_labels)
    num_test = len(test_labels)

    train_examples = tf.data.Dataset.from_tensor_slices(train_examples)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    val_examples = tf.data.Dataset.from_tensor_slices(val_examples)
    val_labels = tf.data.Dataset.from_tensor_slices(val_labels)
    test_examples = tf.data.Dataset.from_tensor_slices(test_examples)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    
    train_data = tf.data.Dataset.zip((train_examples, train_labels))
    train_data = train_data.batch(36)
    val_data = tf.data.Dataset.zip((val_examples, val_labels))
    val_data = train_data.batch(num_val)
    test_data = tf.data.Dataset.zip((test_examples, test_labels))
    test_data = test_data.batch(num_test)
    return train_data, val_data, test_data

    # save_imgs(
    #     (train_examples, train_labels),
    #     (test_examples, test_labels),
    #     _IMG_DIRS["DATA-ROOT"],
    # )

    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     _IMG_DIRS["DATA-ROOT"] + r"\train",
    #     validation_split=0.25,
    #     subset="training",
    #     seed=set_seed,
    #     image_size=_IMG_DIMS,
    #     batch_size=_BATCH_SIZE,
    # )
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     _IMG_DIRS["DATA-ROOT"] + r"\train",
    #     validation_split=0.25,
    #     subset="validation",
    #     seed=set_seed,
    #     image_size=_IMG_DIMS,
    #     batch_size=_BATCH_SIZE,
    # )

    # return train_ds, val_ds
