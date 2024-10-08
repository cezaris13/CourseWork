import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import List
from qiskit_aer import Aer

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
    matrix: np.ndarray = np.zeros(
        (2 ** len(gates[0]), 2 ** len(gates[0])), dtype=complex
    )
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


def prepareDataset(
    normalizeValues: bool = False,
    dataset: str = "iris",
    subsetSize: int = 7,
    classToFilterOut: int = 2,
) -> (np.ndarray, np.ndarray, np.array, np.array):
    if dataset == "iris":
        X, y = datasets.load_iris(return_X_y=True)
        #generate random number from 0 to 2
        randomClass = np.random.randint(0, 3)
        y: np.array = np.where(y == randomClass, 1, -1)  # might cause some problems in a future
        # X = X[y != classToFilterOut]
        # y = y[y != classToFilterOut]
    elif dataset == "breastCancer":
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif dataset == "dummyData":
        X, y = datasets.make_classification(
            n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=45
        )
    elif dataset == "digits":
        X, y = datasets.load_digits(n_class=2, return_X_y=True)
    elif dataset == "wine":
        X, y = datasets.load_wine(return_X_y=True)  # 3 classes; filter out some class
    else:
        raise ValueError("Dataset not yet implemented")
    # y: np.array = np.where(y == 0, -1, 1)  # might cause some problems in a future

    if normalizeValues:  # normalize values returns nan with digits dataset
        max = np.max(X, axis=0)
        min = np.min(X, axis=0)
        X = (2 * X - min - max) / (max - min)
    return train_test_split(X, y, test_size=(X.shape[0] - subsetSize) / (X.shape[0]),stratify=y)


def generateParams(qubits: int, layers: int) -> List[List[int]]:
    params = []
    if qubits != 3:
        params.append([1 for _ in range(qubits)])
        for _ in range(layers):
            params.append([1 for _ in range(qubits)])
            params.append([1 for _ in range(qubits - 2)])
    else:
        for _ in range(qubits):
            params.append([1 for _ in range(qubits)])
    return params


def appendMatrices(matrices: List[str], qubits: int) -> List[str]:
    return [i + "I" * (qubits - 3) for i in matrices]


def splitParameters(
    array: List[int], qubits: int, alternating=False
) -> List[List[int]]:
    chunks = []
    i = 0
    secondChunk = False

    if alternating == False:
        while i < len(array):
            chunkSize = qubits
            chunks.append(array[i : i + chunkSize])
            i += chunkSize
    else:
        chunks.append(array[i : i + qubits])
        i += qubits
        while i < len(array):
            if secondChunk == False:
                chunkSize = qubits
            else:
                chunkSize = qubits - 2
            chunks.append(array[i : i + chunkSize])
            i += chunkSize
            secondChunk = not secondChunk
    return chunks


def getTotalAnsatzParameters(qubits: int, layers: int) -> int:
    if qubits == 2:
        raise ValueError("2 qubits not yet implemented")
    if qubits == 3:
        return 3 * qubits
    else:
        return qubits + 2 * layers * (qubits - 1)

class TriangleMatrix:
    def __init__(self, length: int, array: np.array = None) -> None:
        self.length = length
        if array is not None:
            if len(array) == int(length * (length + 1) / 2):
                self.array = array
            else:
                raise Exception("Invalid array length")
        else:
            self.array = np.zeros(int(length * (length + 1) / 2))

    def getIndex(self, i: int, j: int) -> int:
        if i > j:
            raise Exception("Invalid index")
        else:
            return int(i*(1-i + 2 * self.length) / 2) + (j - i)

    def setElement(self, i: int, j: int, value: float):
        index = self.getIndex(i, j)
        self.array[index] = value

    def getElement(self, i: int, j: int) -> float:
        index = self.getIndex(i, j)
        return self.array[index]

    def print(self):
        for i in range(self.length):
            for j in range(self.length):
                try:
                    print(self.getElement(i, j), end=" ")
                except:
                    print("*", end=" ")
            print("")


def prepareBackend(threads: int, jobs: int):
    backend = Aer.get_backend("aer_simulator")

    if threads != 1:
        backend.set_options(
            max_parallel_threads=threads,
            max_parallel_experiments=jobs,
            max_parallel_shots=0,
            statevector_parallel_threshold=3
        )

    return backend
