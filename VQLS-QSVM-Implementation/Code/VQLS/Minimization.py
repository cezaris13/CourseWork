from qiskit.quantum_info import PauliList
from qiskit_aer import Aer
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from typing import List
from qiskit_algorithms.optimizers import ADAM, SPSA, GradientDescent

from Code.Utils import splitParameters, getTotalAnsatzParameters, TriangleMatrix
from Code.VQLS.Circuits import prepareCircuits

costHistory = []


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
    totalParamsNeeded = getTotalAnsatzParameters(qubits, layers)
    x: List[float] = [
        float(random.randint(0, 3000)) for _ in range(0, totalParamsNeeded)
    ]
    x = x / np.linalg.norm(x)
    start = time.time()
    (
        transpiledHadamardCircuit,
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

    backend = Aer.get_backend("aer_simulator")

    if threads != 1:
        backend.set_options(
            max_parallel_threads=threads,
            max_parallel_experiments=jobs,
            max_parallel_shots=0,
            statevector_parallel_threshold=3
        )

    arguments = [
        coefficientSet,
        transpiledHadamardCircuit,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
        quantumSimulation,
        shots,
        backend
    ]

    start = time.time()
    methods = ["ADAM", "SPSA", "GD"]

    if method in methods:
        def funcWrapper(params): return calculateCostFunction(
            params, arguments)

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
            optimizer = GradientDescent(
                maxiter=iterations, learning_rate=learning_rate)

        out = optimizer.minimize(funcWrapper, x0=x)
        end = time.time()
        if verbose:
            print("Time to minimize:", end - start)
        return splitParameters(out.x, qubits, alternating=qubits != 3)

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

    return splitParameters(out["x"], qubits, alternating=qubits != 3)


def calculateCostFunction(parameters: list, args: list) -> float:
    cost = 0
    if len(costHistory) > 0:
        cost = costHistory[len(costHistory) - 1]
    print("Iteration:", len(costHistory) + 1, ", cost:", cost, end="\r")
    overallSum1: float = 0
    overallSum2: float = 0

    coefficientSet = args[0]
    transpiledHadamardCircuit = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]
    isQuantumSimulation = args[5]
    shots = args[6]
    backend = args[7]

    bindedHadamardGates = list(map(lambda x: x.assign_parameters({ parametersHadamard: parameters}), transpiledHadamardCircuit))
    bindedSpecHadamardGates = list(map(lambda x: x.assign_parameters({ parametersSpecialHadamard: parameters}), transpiledSpecialHadamardCircuits))
    lenPaulis = len(bindedSpecHadamardGates)
    resultsHadamard, resultsSpecialHadamard = runExperiments(bindedHadamardGates, bindedSpecHadamardGates, isQuantumSimulation, shots, backend)
    bindedHadamardGatesTriangle = TriangleMatrix(lenPaulis, resultsHadamard)

    # we have triangular matrix:
    # X X X X X
    # . X X X X
    # . . X X X
    # . . . X X
    # . . . . X
    # lower triangular matrix is calculated using this formula: <0|V(a)^d A_n^d A_m V(a)|0> = (<0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    # c_n conj c_m <0|V(a)^d A_n^d A_m V(a)|0> = ( c_n conj c_m <0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            outputstate = bindedHadamardGatesTriangle.getElement(i, i + j)
            m_sum = getMSum(isQuantumSimulation, outputstate, shots)
            multiply = coefficientSet[i] * coefficientSet[i + j]

            if j == 0:  # since the main diagional is not counted twice and  the list first element is the main diagional
                overallSum1 += multiply * (1 - (2 * m_sum))
            else:
                temp = multiply * (1 - (2 * m_sum))
                overallSum1 += np.conjugate(temp) + temp

    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            mult = 1
            indexArray = [i, j + i]
            for index in indexArray:
                outputstate = resultsSpecialHadamard[index]
                m_sum = getMSum(isQuantumSimulation, outputstate, shots)
                mult = mult * (1 - (2 * m_sum))
            multiply = coefficientSet[i] * coefficientSet[j + i]
            if j == 0:
                overallSum2 += multiply * mult
            else:
                tempSum = multiply * mult
                overallSum2 += tempSum + np.conjugate(tempSum)

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)
    return totalCost

def runExperiments(bindedHadamardGates, bindedSpecHadamardGates, isQuantumSimulation: bool, shots: int, backend):
    gates = bindedHadamardGates + bindedSpecHadamardGates

    if isQuantumSimulation:
        results = backend.run(gates, shots=shots).result()
        resultsHadamard = list(map(lambda x: results.get_counts(x), bindedHadamardGates))
        resultsSpecialHadamard = list(map(lambda x: results.get_counts(x), bindedSpecHadamardGates))
    else:
        results = backend.run(gates).result()
        resultsHadamard = list(map(lambda x: np.real(results.get_statevector(x, decimals=100)), bindedHadamardGates))
        resultsSpecialHadamard = list(map(lambda x: np.real(results.get_statevector(x, decimals=100)), bindedSpecHadamardGates))

    return resultsHadamard, resultsSpecialHadamard

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