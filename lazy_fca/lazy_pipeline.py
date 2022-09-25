import copy
from typing import Iterator

import pandas as pd
from tqdm import tqdm


def load_data() -> pd.DataFrame:
    df_test = pd.read_csv('lazy_example/data/test1.csv', sep=',')
    df_train = pd.read_csv('lazy_example/data/train1.csv', sep=',')
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    df = df.replace(to_replace='positive', value=True)
    df = df.replace(to_replace='negative', value=False)
    return df


def binarize_data(X: pd.DataFrame) -> 'pd.DataFrame[bool]':
    result_data = copy.deepcopy(X)
    for column in X.columns.values:
        result_data = pd.concat([
            result_data,
            pd.get_dummies(result_data[column], prefix=column, prefix_sep=': ')],
            axis=1
        )
        del result_data[column]
    return result_data.astype(bool)


def predict(x: 'pd.Series[bool]', X_train: 'pd.DataFrame[bool]', Y_train: 'pd.Series[bool]') -> bool:
    iterator = ((x_train, y_train) for (_, x_train), y_train in zip(X_train.iterrows(), Y_train))

    pos, neg = 0, 0
    for x_train, y_train in iterator:
        if all(x & x_train == x_train):
            if y_train:
                pos += 1
            else:
                neg += 1

    pos /= Y_train.sum()
    neg /= (~Y_train).sum()

    return pos > neg


def predict_array(
        X: 'pd.DataFrame[bool]', Y: 'pd.Series[bool]',
        n_train: int, update_train: bool = True, use_tqdm: bool = False
) -> Iterator[bool]:
    for i, (_, x) in tqdm(
        enumerate(X[n_train:].iterrows()),
        initial=n_train, total=len(X),
        desc='Predicting step by step',
        disable=not use_tqdm,
    ):
        n_trains = n_train + i if update_train else n_train
        yield predict(x, X[:n_trains], Y[:n_trains])
