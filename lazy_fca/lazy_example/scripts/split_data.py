import sys
from typing import Collection


def k_fold_cross_validation(X: Collection, k: int, randomise: bool = False):
    """Generates k (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/k. So each training iterable is of length (k-1)*len(X)/k.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise:
        from random import shuffle
        X = list(X)
        shuffle(X)

    for i in range(k):
        X_train = [x for i, x in enumerate(X) if i % k != i]
        X_test = [x for i, x in enumerate(X) if i % k == i]
        yield X_train, X_test, i


if __name__ == '__main__':
    source = sys.argv[1]
    with open(source, 'r') as data:
        for training, validation, k in k_fold_cross_validation(data.readlines(), k=7):
            with open("%s_train_%d.txt" % (source, k), 'w') as tf:
                tf.writelines(training)
            with open("%s_validation_%d.txt" % (source, k), 'w') as tv:
                tv.writelines(validation)
