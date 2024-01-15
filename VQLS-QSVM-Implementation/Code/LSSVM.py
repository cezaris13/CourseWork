import numpy as np


def linearKernel(x1: np.array, x2: np.array) -> float:
    return np.dot(x1, x2)


# Calculate the kernel matrix
def kernelMatrix(X: np.ndarray, Y: np.ndarray, kernelFunction: callable) -> np.ndarray:
    nSamples: int = X.shape[0]
    mSamples: int = Y.shape[0]
    K: np.ndarray = np.zeros((nSamples, mSamples))
    for i in range(nSamples):
        for j in range(mSamples):
            K[i, j] = kernelFunction(X[i], Y[j])
    return K


def createTopMatrix(N: int) -> np.ndarray:
    matrix: np.ndarray = np.zeros((N + 1, N + 1))
    matrix[0, 0] = 0
    matrix[0, 1:] = 1
    matrix[1:, 0] = 1
    return matrix


def createBottomMatrix(X: np.ndarray, gamma: float, kernelFunction: callable) -> np.ndarray:
    omega: np.ndarray = kernelMatrix(X, X, kernelFunction)
    return omega + np.identity(X.shape[0]) / gamma


def lssvmMatrix(X: np.ndarray, gamma: float, kernel: str = "linearKernel") -> np.ndarray:
    if kernel == "linearKernel":
        kernelFunction = linearKernel
    matrixTop: np.ndarray = createTopMatrix(X.shape[0])
    matrixBottom: np.ndarray = createBottomMatrix(X, gamma, kernelFunction)
    for i in range(matrixTop.shape[0]):
        for j in range(matrixTop.shape[1]):
            if i == 0 or j == 0:
                continue
            matrixTop[i, j] = matrixBottom[i - 1, j - 1]
    return matrixTop


def weightsAndBiasVector(weights: np.array, bias: float) -> np.array:
    return np.concatenate((np.array([bias]), weights))


def prepareLabels(y: np.array) -> np.array:
    return np.concatenate((np.array([0]), y))


def predict(xTrain: np.ndarray, xTest: np.ndarray, weights: np.array, b:float, kernelFunction: callable = linearKernel) -> np.array:
    ls: np.ndarray = kernelMatrix(xTest, xTrain, kernelFunction)
    predictions: np.array = np.dot(ls, weights) + b
    return predictions


def accuracy(yTest: np.array, predictions: np.array) -> float:
    for i in range(len(predictions)):
        if predictions[i] >= 0:
            predictions[i] = 1
        else:
            predictions[i] = -1

    return np.sum(predictions == yTest) / len(yTest)
