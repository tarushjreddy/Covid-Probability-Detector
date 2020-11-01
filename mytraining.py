import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffeled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffeled[:test_set_size]
    train_indices = shuffeled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    df = pd.read_csv("dataone.csv")
    train, test = data_split(df, 0.2)

    x_train = train[['FEVER', 'bodyPain', 'runNose',
                     'Oxylow', 'age', 'InfecProb']].to_numpy()
    x_test = test[['FEVER', 'bodyPain', 'runNose',
                   'Oxylow', 'age', 'InfecProb']].to_numpy()

    y_train = train[['InfecProb']].to_numpy().reshape(2448,)
    y_test = test[['InfecProb']].to_numpy().reshape(612,)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    file = open('model.pkl', 'wb')

    pickle.dump(clf, file)
    # file.close()
