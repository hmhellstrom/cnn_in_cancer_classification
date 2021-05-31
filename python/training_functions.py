from math import ceil
from random import sample, shuffle


def train_test_split(labels, test_prcnt, stratify=True):
    num_test = ceil(len(labels) * test_prcnt)
    if stratify:
        positives_prcnt = sum(labels) / len(labels)
        positives = labels > 0
        positives = [index for index, element in enumerate(positives) if element]
        negatives = labels == 0
        negatives = [index for index, element in enumerate(negatives) if element]
        test_indices = sample(positives, ceil(num_test * positives_prcnt)) + sample(
            negatives, num_test - ceil(num_test * positives_prcnt)
        )
        shuffle(test_indices)
    else:
        test_indices = shuffle(sample(range(len(labels)), num_test))
    train_indices = list(set(list(range(len(labels)))).difference(set(test_indices)))
    shuffle(train_indices)
    return train_indices, test_indices
