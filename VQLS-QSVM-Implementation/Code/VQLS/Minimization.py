from qiskit.quantum_info import PauliList
from qiskit_aer import Aer
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
import gc
from qiskit_algorithms.optimizers import ADAM, SPSA, GradientDescent

from Code.Utils import splitParameters, getTotalAnsatzParameters, TriangleMatrix
from Code.VQLS.Circuits import prepareCircuits

costHistory = []
# weightsValueHistory = []


def minimization(
    paulis: PauliList,
    coefficientSet: list,
    qubits: int,
    bVector: list,
    quantumSimulation: bool = True,
    method: str = "COBYLA",
    shots: int = 100000,
    iterations: int = 200,
    verbose: bool = True,
    layers: int = 3,
    threads: int = 1,
    jobs: int = 1,
    options: dict = {},
) -> List[List[float]]:
    global costHistory
    costHistory = []
    # global weightsValueHistory
    # weightsValueHistory = []
    totalParamsNeeded = getTotalAnsatzParameters(qubits, layers)
    x: List[float] = [
        float(random.randint(0, 3000)) for _ in range(0, totalParamsNeeded)
    ]
    x = x / np.linalg.norm(x)
    start = time.time()
    (
        triangleMatrixHadamardCircuit,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    ) = prepareCircuits(
        paulis,
        bVector,
        qubits,
        quantumSimulation,
        layers,
        "aer_simulator",
    )
    end = time.time()
    if verbose:
        print("Time to prepare circuits:", end - start)

    # exc = ThreadPoolExecutor(max_workers=threads)

    arguments = [
        coefficientSet,
        triangleMatrixHadamardCircuit,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
        quantumSimulation,
        shots,
        threads,
        jobs,
        # exc,
    ]
    print ("Transpiled circuits length:", len(triangleMatrixHadamardCircuit.array))
    print ("Transpiled special circuits:", len(transpiledSpecialHadamardCircuits))

    start = time.time()
    methods = ["ADAM", "SPSA", "GD"]

    if method in methods:
        funcWrapper = lambda params: calculateCostFunction(params, arguments)

        if method == "ADAM":
            lr = options["lr"] if "lr" in options else 0.05
            optimizer = ADAM(maxiter=iterations, lr=lr)
        elif method == "SPSA":
            learning_rate = (
                options["learning_rate"] if "learning_rate" in options else 0.2
            )
            perturbation = options["perturbation"] if "perturbation" in options else 0.1
            optimizer = SPSA(
                maxiter=iterations,
                learning_rate=learning_rate,
                perturbation=perturbation,
            )
        elif method == "GD":
            learning_rate = (
                options["learning_rate"] if "learning_rate" in options else 0.2
            )
            optimizer = GradientDescent(maxiter=iterations, learning_rate=learning_rate)

        out = optimizer.minimize(funcWrapper, x0=x)
        end = time.time()
        if verbose:
            print("Time to minimize:", end - start)
        return splitParameters(out.x, qubits, alternating=qubits!=3)

    out = minimize(
        calculateCostFunction,
        x0=x,
        args=arguments,
        method=method,
        options={"maxiter": iterations},
    )

    end = time.time()

    if verbose:
        print("Time to minimize:", end - start)
        print(out)

    return splitParameters(out["x"], qubits, alternating=qubits!=3)


def calculateCostFunction(parameters: list, args: list) -> float: # this function has to be parallelized
    cost = 0
    if len(costHistory) > 0:
        cost = costHistory[len(costHistory) - 1]
    print("Iteration:", len(costHistory) + 1, ", cost:", cost, end="\r")
    overallSum1: float = 0
    overallSum2: float = 0
    backend = Aer.get_backend("aer_simulator")

    coefficientSet = args[0]
    triangleMatrixHadamardCircuit = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]
    isQuantumSimulation = args[5]
    shots = args[6]
    threads = args[7]
    jobs = args[8]
    # exc = args[7]

    bindedHadamardGates = [
        i.assign_parameters({parametersHadamard: parameters})
        for i in triangleMatrixHadamardCircuit.array
    ]

    bindedSpecHadamardGates = [
        i.assign_parameters({parametersSpecialHadamard: parameters})
        for i in transpiledSpecialHadamardCircuits
    ]
    lenPaulis = len(bindedSpecHadamardGates)

    # backend.set_options(executor=exc)
    # backend.set_options(max_job_size=8)

    if threads > 1:
        backend.set_options(
            max_parallel_threads=threads,
            max_parallel_experiments = jobs,
            max_parallel_shots = 0,
            statevector_parallel_threshold = 3
        )

    # we have triangular matrix:
    # X X X X X
    # . X X X X
    # . . X X X
    # . . . X X
    # . . . . X
    # lower triangular matrix is calculated using this formula: <0|V(a)^d A_n^d A_m V(a)|0> = (<0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    # c_n conj c_m <0|V(a)^d A_n^d A_m V(a)|0> = ( c_n conj c_m <0|V(a)^d A_m^d A_n V(a)|0>) conjugate

    if isQuantumSimulation:
        results = backend.run(bindedHadamardGates, shots=shots).result()
    else:
        resultVectors = []
        for i in range(lenPaulis*(lenPaulis+1)//2):
            result = backend.run(bindedHadamardGates[i]).result()
            outputstate = np.real(
                result.get_statevector(bindedHadamardGates[i], decimals=100)
            )
            resultVectors.append(outputstate)
        triangleResultVectors = TriangleMatrix(lenPaulis, resultVectors)
    bindedHadamardGatesTriangle = TriangleMatrix(lenPaulis, bindedHadamardGates)

    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            if isQuantumSimulation:
                outputstate = results.get_counts(bindedHadamardGatesTriangle.getElement(i, i+ j))
            else:
                outputstate = triangleResultVectors.getElement(i, i + j)
            m_sum = getMSum(isQuantumSimulation, outputstate, shots)
            multiply = coefficientSet[i] * coefficientSet[i + j]

            if j == 0:  # since the main diagional is not counted twice and  the list first element is the main diagional
                overallSum1 += multiply * (1 - (2 * m_sum))
            else:
                temp = multiply * (1 - (2 * m_sum))
                overallSum1 += np.conjugate(temp) + temp

    # del results
    # del bindedHadamardGates
    # gc.collect()
    if isQuantumSimulation:
        results = backend.run(bindedSpecHadamardGates, shots=shots).result()
    else:
        resultVectors = []
        for i in range(lenPaulis):
            result = backend.run(bindedSpecHadamardGates[i]).result()
            outputstate = np.real(
                result.get_statevector(bindedSpecHadamardGates[i], decimals=100)
            )
            resultVectors.append(outputstate)

    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            mult = 1
            indexArray = [i, j + i]
            for index in indexArray:
                if isQuantumSimulation:
                    outputstate = results.get_counts(bindedSpecHadamardGates[index])
                else:
                    outputstate = resultVectors[index]
                m_sum = getMSum(isQuantumSimulation, outputstate, shots)
                mult = mult * (1 - (2 * m_sum))
            multiply = coefficientSet[i] * coefficientSet[j + i]
            if j == 0:
                overallSum2 += multiply * mult
            else:
                tempSum = multiply * mult
                overallSum2 += tempSum + np.conjugate(tempSum)
    # del results
    # del bindedSpecHadamardGates

    # gc.collect()

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)
    # weightsValueHistory.append(parameters)
    return totalCost


def getMSum(isQuantumSimulation: bool, outputstate, shots: int) -> float:
    if isQuantumSimulation:
        if "1" in outputstate.keys():
            m_sum = float(outputstate["1"]) / shots
        else:
            m_sum = 0
        return m_sum
    else:
        m_sum = 0
        for l in range(len(outputstate)):
            if l % 2 == 1:
                n = outputstate[l] ** 2
                m_sum += n
        return m_sum


def getApproximationValue(A: np.ndarray, b: np.array, o: np.array) -> float:
    return ((b.dot(A.dot(o) / (np.linalg.norm(A.dot(o))))) ** 2).real


def getCostHistory():
    return costHistory


def plotCost():
    plt.style.use("seaborn-v0_8")
    plt.plot(costHistory, "g")
    plt.ylabel("Cost function")
    plt.xlabel("Optimization steps")
    plt.show()

# def calculateWeightsAccuracy(A, bVector, qubits: int) -> float:
#     accuracyList = []
#     parameters = weightsValueHistory
#     for parameter in parameters:
#         out = [parameter[0:3], parameter[3:6], parameter[6:9]]
#         qc = QuantumCircuit(qubits, qubits)
#         weights = ansatzTest(qc, out)
#         estimatedNorm, estimatedNormVector = estimateNorm(A, weights, bVector)
#         weightsVector = estimatedNorm * estimatedNormVector
#         # weights, b = weightsVector[1:], weightsVector[0]
#         predictions = np.dot(A, weightsVector)
#         print(predictions)
#         print(bVector)
#         accuracyList.append(accuracy(bVector, predictions))
#     return accuracyList


# def getWeightsValueHistory():
#     return weightsValueHistory


# def plotAccuracy(listOfAccuracies: List[float]):
#     plt.style.use("seaborn-v0_8")
#     plt.plot(listOfAccuracies, "g")
#     plt.ylabel("Accuracy")
#     plt.xlabel("Optimization steps")
#     plt.show()
