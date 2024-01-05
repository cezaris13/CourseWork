from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit import QuantumCircuit
import numpy as np
from itertools import chain
from typing import List

from VQLS import getMatrixCoeffitients, minimization, ansatzTest, estimateNorm, plotCost, getCostHistory
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
        method: str = "COBYLA",
        verbose: bool = False,
    ) -> (np.array, float):
        self.xTrain = xTrain
        self.xTrainSize = xTrain.shape[0]
        inputMatrix: np.ndarray = lssvmMatrix(xTrain, self.gamma, "linearKernel")
        yVector: np.array = prepareLabels(yTrain)

        if verbose:
            print ("Condition number of the matrix: ", np.linalg.cond(inputMatrix))
        if verbose:
            print("LS-SVM Matrix:\n", inputMatrix)
        pauliOp: SparsePauliOp = SparsePauliOp.from_operator(inputMatrix)
        paulis: PauliList = pauliOp.paulis
        # self.totalNeededQubits = pauliOp.num_qubits + 2
        # self.inputMatrix = inputMatrix
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
            method=method,
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
        if verbose:
            print("Weights:", weights)
        self.b = weights[0]
        self.weights = weights[1:]
        return self.weights, self.b

    def predict(
        self, xTest: np.ndarray, weights: np.array = [], kernelFunction: callable = linearKernel
    ) -> np.array:
        if weights != []:
            return predict(
                self.xTrain, xTest, weights, self.b, kernelFunction
            )
        else:
            return predict(
                self.xTrain, xTest, self.weights, self.b, kernelFunction
            )

    def accuracy(self, xTest: np.ndarray, yTest: np.array, weights: np.array = []) -> float:
        correct: int = 0
        predictions: np.array = self.predict(xTest,weights=weights)
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

    def getCostHistory(self):
        return getCostHistory()

    # def plotAccuracy(self, xTest: np.ndarray, yTest: np.array) -> int:
    #     print("Plotting accuracy")
    #     accuracyList = []
    #     print (self.totalNeededQubits)
    #     parameters = getWeightsValueHistory()
    #     print(parameters)
    #     inputMatrix = self.inputMatrix
    #     print(inputMatrix)
    #     # for parameter in parameters:
    #     parameter = parameters[0]
    #     out = [parameter[0:3], parameter[3:6], parameter[6:9]]
    #     qc = QuantumCircuit(self.totalNeededQubits,self.totalNeededQubits)
    #     weights = ansatzTest(qc,out)
    #     print (weights)
    #     estimatedNorm, estimatedNormVector = estimateNorm(inputMatrix, weights, yTest)
    #     print(estimatedNorm)    
    #     # weightsVector = estimatedNorm * estimatedNormVector
    #     # print(weightsVector)
    #     # accuracyList.append(self.accuracy(xTest, yTest, weights=weightsVector))
    #     return accuracyList
    
    def assignClass(self, prediction: float) -> int:
        if prediction >= 0:
            return 1
        else:
            return -1
