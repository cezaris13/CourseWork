import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def zGate() -> np.array:
    return np.array([[1, 0], [0, -1]])


def xGate() -> np.array:
    return np.array([[0, 1], [1, 0]])


def yGate() -> np.array:
    return np.array([[0, -1j], [1j, 0]])


def identityGate() -> np.array:
    return np.eye(2)


def letterToGate(letter: str) -> np.array:
    if letter == "Z":
        return zGate()
    elif letter == "I":
        return identityGate()
    elif letter == "X":
        return xGate()
    elif letter == "Y":
        return yGate()
    else:
        return None


def createMatrixFromParameters(coefs: list, gates: list) -> np.ndarray:
    matrix: np.ndarray = np.zeros((2 ** len(gates[0]), 2 ** len(gates[0])), dtype=complex)
    for gate in gates:
        tempGate = None
        gateTemp = gate[::-1]
        for letter in gateTemp:
            if tempGate is None:
                tempGate = letterToGate(letter)
            else:
                tempGate = np.kron(tempGate, letterToGate(letter))
        matrix += coefs[gates.index(gate)] * tempGate

    return matrix


def prepareDataset(normalizeValues: bool = False, dataset: str = "iris", subsetSize= 7, classToFilterOut = 2) -> (np.ndarray, np.ndarray, np.array, np.array):
    if dataset == "iris":
        X,y = datasets.load_iris(return_X_y=True)
        X = X[y!=classToFilterOut]
        y = y[y!=classToFilterOut]
    elif dataset == "breastCancer":
        X,y = datasets.load_breast_cancer(return_X_y=True)
    elif dataset == "dummyData":
        X, y = datasets.make_classification(
            n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=45
        )
    elif dataset == "digits":
        X, y = datasets.load_digits(n_class=2, return_X_y=True)
    elif dataset == "wine":
        X, y = datasets.load_wine(return_X_y=True) # 3 classes; filter out some class
    else: 
        raise ValueError("Dataset not yet implemented")
    y: np.array = np.where(y == 0, -1, 1) # might cause some problems in a future

    if normalizeValues: # normalize values returns nan with digits dataset
        max = np.max(X, axis=0)
        min = np.min(X, axis=0)
        X = (2*X - min - max) / (max - min)
    return train_test_split(X, y, test_size=(X.shape[0]-subsetSize)/(X.shape[0]))