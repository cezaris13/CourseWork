from LSSVM import lssvmMatrix, prepareLabels, predict, linearKernel
from qiskit.quantum_info import SparsePauliOp
from VQLS import getMatrixCoeffitients, minimization, ansatzTest, estimateNorm, plotCost
from qiskit import QuantumCircuit
import numpy as np
from itertools import chain


class VQLSSVM:
    def __init__(self, gamma: float, shots: int):
        self.gamma = gamma
        self.shots = shots

    def train(
        self,
        xTrain,
        yTrain,
        quantumSimulation: bool = True,
        iterations: int = 200,
        verbose: bool = False,
    ) -> (np.array, float):
        self.xTrain = xTrain
        self.xTrainSize = xTrain.shape[0]
        inputMatrix = lssvmMatrix(xTrain, self.gamma, "linearKernel")
        yVector = prepareLabels(yTrain)

        if verbose:
            print("LS-SVM Matrix:", inputMatrix)

        pauliOp = SparsePauliOp.from_operator(inputMatrix)
        paulis = pauliOp.paulis
        if verbose:
            print(paulis)

        coefficientSet = getMatrixCoeffitients(pauliOp)
        if verbose:
            print("Pauli matrix coeffitients", coefficientSet)

        outF = minimization(
            paulis=paulis,
            coefficientSet=coefficientSet,
            totalNeededQubits=pauliOp.num_qubits + 2,
            bVector=yVector,
            quantumSimulation=quantumSimulation,
            shots=self.shots,
            iterations=iterations,
        )
        if verbose:
            print("Output Vector:", outF)
        circ = QuantumCircuit(pauliOp.num_qubits, pauliOp.num_qubits)
        estimatedX = ansatzTest(circ, outF)
        if verbose:
            print("Output Vector after ansatz test:", estimatedX)
        estimatedNorm, estimatedVector = estimateNorm(inputMatrix, estimatedX, yVector)
        if verbose:
            print("Estimated norm:", estimatedNorm)
            print("Estimated vector:", estimatedVector)
        weights = estimatedVector * estimatedNorm
        self.b = weights[0]
        self.weights = weights[1:]
        return self.weights, self.b

    def predict(self, xTest, kernelFunction: callable = linearKernel) -> np.array:
        if xTest.shape[0] > self.xTrainSize:
            splitTest = list(self.chunks(xTest, self.xTrainSize))
            predictions = [self.predict(xTestSubset, kernelFunction) for xTestSubset in splitTest]
            predictions = np.array(list(chain.from_iterable(predictions)))
        elif xTest.shape[0] < self.xTrainSize:
            xTestCopy = xTest.copy()
            for i in range(self.xTrainSize - xTest.shape[0]):
                xTestCopy = np.append(xTestCopy, [[0, 0]], axis=0)
            predictions = predict(
                self.xTrain, xTestCopy, self.weights, self.b, kernelFunction
            )
            predictions = predictions[: xTest.shape[0]]
        else:
            predictions = predict(
                self.xTrain, xTest, self.weights, self.b, kernelFunction
            )
        return predictions

    def accuracy(self, xTest, yTest):
        correct = 0
        predictions = self.predict(xTest)
        predictions = [ self.assignClass(i) for i in predictions]
        for i in range(len(predictions)):
            if predictions[i] == yTest[i]:
                correct += 1
        return correct / (xTest.shape[0])

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def plotCost(self):
        plotCost()
    
    def assignClass(self, prediction):
        if prediction >= 0:
            return 1
        else:
            return -1
