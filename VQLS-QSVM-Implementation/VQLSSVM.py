from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit import QuantumCircuit
import numpy as np
from itertools import chain
from typing import List

from VQLS import getMatrixCoeffitients, minimization, ansatzTest, estimateNorm, plotCost
from LSSVM import lssvmMatrix, prepareLabels, predict, linearKernel

class VQLSSVM:
    def __init__(self, gamma: float, shots: int):
        self.gamma = gamma
        self.shots = shots

    def train(
        self,
        xTrain: np.ndarray,
        yTrain: np.ndarray,
        quantumSimulation: bool = True,
        iterations: int = 200,
        verbose: bool = False,
    ) -> (np.array, float):
        self.xTrain = xTrain
        self.xTrainSize = xTrain.shape[0]
        inputMatrix: np.ndarray = lssvmMatrix(xTrain, self.gamma, "linearKernel")
        yVector: np.array = prepareLabels(yTrain)

        if verbose:
            print("LS-SVM Matrix:", inputMatrix)

        pauliOp: SparsePauliOp = SparsePauliOp.from_operator(inputMatrix)
        paulis: PauliList = pauliOp.paulis
        if verbose:
            print(paulis)

        coefficientSet: List[float] = getMatrixCoeffitients(pauliOp)
        if verbose:
            print("Pauli matrix coeffitients", coefficientSet)

        outF: List[List[float]] = minimization(
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
        circ: QuantumCircuit = QuantumCircuit(pauliOp.num_qubits, pauliOp.num_qubits)
        estimatedX: List[complex] = ansatzTest(circ, outF)
        if verbose:
            print("Output Vector after ansatz test:", estimatedX)
        estimatedNorm, estimatedVector = estimateNorm(inputMatrix, estimatedX, yVector)
        if verbose:
            print("Estimated norm:", estimatedNorm)
            print("Estimated vector:", estimatedVector)
        weights: List[float] = estimatedVector * estimatedNorm
        self.b = weights[0]
        self.weights = weights[1:]
        return self.weights, self.b

    def predict(
        self, xTest: np.ndarray, kernelFunction: callable = linearKernel
    ) -> np.array: # optimize this
        if xTest.shape[0] > self.xTrainSize:
            splitTest = list(self.chunks(xTest, self.xTrainSize))
            predictions = [
                self.predict(xTestSubset, kernelFunction) for xTestSubset in splitTest
            ]
            predictions = np.array(list(chain.from_iterable(predictions)))
        elif xTest.shape[0] < self.xTrainSize:
            xTestCopy = xTest.copy()
            for _ in range(self.xTrainSize - xTest.shape[0]):
                zeros = np.zeros((1, xTest.shape[1]))
                xTestCopy = np.append(xTestCopy, zeros, axis=0)
            predictions = predict(
                self.xTrain, xTestCopy, self.weights, self.b, kernelFunction
            )
            predictions = predictions[: xTest.shape[0]]
        else:
            predictions = predict(
                self.xTrain, xTest, self.weights, self.b, kernelFunction
            )
        return predictions

    def accuracy(self, xTest: np.ndarray, yTest: np.array) -> float:
        correct: int = 0
        predictions: np.array = self.predict(xTest)
        predictions = [self.assignClass(i) for i in predictions]
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

    def assignClass(self, prediction: float) -> int:
        if prediction >= 0:
            return 1
        else:
            return -1
