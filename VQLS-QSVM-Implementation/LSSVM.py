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
    matrix_top = np.zeros((N + 1, N + 1))
    matrix_top[0, 0] = 0
    matrix_top[0, 1:] = 1
    matrix_top[1:, 0] = 1
    return matrix_top


def createBottomMatrix(X, gamma, kernelFunction: callable):
    Omega = kernelMatrix(X, X, kernelFunction)
    return Omega + np.identity(X.shape[0]) / gamma


def lssvmMatrix(X, gamma, kernel: str = "linearKernel"):
    if kernel == "linearKernel":
        kernelFunction = linearKernel
    matrix_top = createTopMatrix(X.shape[0])
    matrix_bottom = createBottomMatrix(X, gamma, kernelFunction)
    for i in range(matrix_top.shape[0]):
        for j in range(matrix_top.shape[1]):
            if i == 0 or j == 0:
                continue
            matrix_top[i, j] = matrix_bottom[i - 1, j - 1]
    return matrix_top


def weightsAndBiasVector(weights, bias):
    return np.concatenate((np.array([bias]), weights))


def prepareLabels(y):
    return np.concatenate((np.array([0]), y))


def predict(xTrain, xTest, weights, b, kernelFunction: callable = linearKernel):
    ls = kernelMatrix(xTest, xTrain, kernelFunction)
    predictions = np.sign(np.dot(ls, weights) + b)
    return predictions


def accuracy(yTest, predictions):
    return np.sum(np.sign(predictions) == yTest) / len(yTest)
