import numpy as np


def linearKernel(x1, x2) -> float:
    return np.dot(x1, x2)


# Calculate the kernel matrix
def kernelMatrix(X, Y, kernelFunction: callable):
    nSamples = X.shape[0]
    mSamples = Y.shape[0]
    K = np.zeros((nSamples, mSamples))
    for i in range(nSamples):
        for j in range(mSamples):
            K[i, j] = kernelFunction(X[i], Y[j])
    return K


def createTopMatrix(N):
    matrix = np.zeros((N + 1, N + 1))
    matrix[0, 0] = 0
    matrix[0, 1:] = 1
    matrix[1:, 0] = 1
    return matrix


def createBottomMatrix(X, gamma, kernelFunction: callable):
    Omega = kernelMatrix(X, X, kernelFunction)
    return Omega + np.identity(X.shape[0]) / gamma


def lssvmMatrix(X, gamma, kernel: str = "linearKernel"):
    if kernel == "linearKernel":
        kernelFunction = linearKernel
    matrixTop = createTopMatrix(X.shape[0])
    matrixBottom = createBottomMatrix(X, gamma, kernelFunction)
    for i in range(matrixTop.shape[0]):
        for j in range(matrixTop.shape[1]):
            if i == 0 or j == 0:
                continue
            matrixTop[i, j] = matrixBottom[i - 1, j - 1]
    return matrixTop


def weightsAndBiasVector(weights, bias):
    return np.concatenate((np.array([bias]), weights))


def prepareLabels(y):
    return np.concatenate((np.array([0]), y))


def predict(xTrain, xTest, weights, b, kernelFunction: callable = linearKernel):
    ls = kernelMatrix(xTest, xTrain, kernelFunction)
    predictions = np.dot(ls, weights) + b
    return predictions


def accuracy(yTest, predictions):
    for i in range(len(predictions)):
        if predictions[i] >= 0:
            predictions[i] = 1
        else:
            predictions[i] = -1

    return np.sum(predictions == yTest) / len(yTest)
