from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import SparsePauliOp, PauliList
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import ceil
from qiskit.circuit import ParameterVector
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
import gc
from itertools import product
from qiskit.algorithms.optimizers import ADAM, SPSA, GradientDescent
import contextlib
import io
from Code.Utils import splitParameters, getTotalAnsatzParameters

costHistory = []
# weightsValueHistory = []


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


def convertMatrixIntoCircuit(
    circuit: QuantumCircuit,
    paulis: PauliList,
    controlled: bool = False,
    auxiliaryQubit: int = 0,
    showBarriers: bool = True,
):
    qubitIndexList: List[int] = []
    qubits: int = circuit.num_qubits
    for i in range(qubits):
        if controlled:
            if i != auxiliaryQubit:
                qubitIndexList.append(i)
        else:
            qubitIndexList.append(i)

    for p in range(len(paulis)):
        for i in range(len(paulis[p])):
            currentGate = paulis[p][i]
            # currentGate = paulis[p][len(paulis[p])-1-i]
            if currentGate.x and currentGate.z == False:
                if controlled:
                    circuit.cx(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.x(i)
            elif currentGate.x and currentGate.z:
                if controlled:
                    circuit.cy(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.y(i)
            elif currentGate.z and currentGate.x == False:
                if controlled:
                    circuit.cz(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.z(i)
        if showBarriers:
            circuit.barrier()


def getMatrixCoeffitients(pauliOp: SparsePauliOp) -> List[float]:
    coeffs: List[float] = []
    paulis: PauliList = pauliOp.paulis
    for p in range(len(paulis)):
        containsIdentity: bool = False
        for i in range(len(paulis[p])):
            currentGate = paulis[p][i]
            # currentGate = paulis[p][len(paulis[p]) - 1 - i]
            if currentGate.x == False and currentGate.z == False:
                containsIdentity = True
        coeffs.append(pauliOp.coeffs[p])
        if containsIdentity == False:
            coeffs.append(pauliOp.coeffs[p])
    return coeffs


# VLQS part
def applyFixedAnsatz(
    circ: QuantumCircuit,
    qubits: int,
    parameters: List[List[float]],
    offset: int = 0,
    layers: int = 3,
    barrier: bool = False,
):  # maybe change to 2local or EfficientSU2
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html
    gates = getFixedAnsatzGates(qubits, parameters, offset=offset, layers=layers)
    gatesToCircuit(circ, gates, barrier=barrier)


# Creates the Hadamard test
def hadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    qubits: int,
    parameters: List[List[float]],
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    applyFixedAnsatz(circ, qubits, parameters, offset=1)

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )  # change to predefined instructions

    circ.barrier()

    circ.h(auxiliaryIndex)


def controlB(
    circ: QuantumCircuit, auxiliaryIndex: int, qubits: int, values: List[float]
):
    qubits = [i + 1 for i in range(qubits)]
    custom = createB(values).to_gate().control()
    circ.append(custom, [auxiliaryIndex] + qubits)


def createB(values: List[float]) -> QuantumCircuit:
    qubits: int = ceil(np.log2(len(values)))
    if len(values) != 2**qubits:
        values = np.pad(values, (0, 2**qubits - len(values)), "constant")
    values = values / np.linalg.norm(values)
    circ: QuantumCircuit = QuantumCircuit(qubits)
    circ.prepare_state(values)
    return circ


def getBArray(values: List[float]) -> np.array:
    qubits: int = ceil(np.log2(len(values)))
    if len(values) != 2**qubits:
        values = np.pad(values, (0, 2**qubits - len(values)), "constant")
    return np.array(values / np.linalg.norm(values))


# Creates controlled anstaz for calculating |<b|psi>|^2 with a Hadamard test
def controlFixedAnsatz(
    circ: QuantumCircuit,
    qubits: int,
    parameters: List[List[float]],
    barrier: bool = False,
):
    gates = getControlledFixedAnsatzGates(qubits, parameters)
    gatesToCircuit(circ, gates, barrier=barrier)


# Create the controlled Hadamard test, for calculating <psi|psi>
def specialHadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    qubits: int,
    parameters: List[List[float]],
    weights: List[float],
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    controlFixedAnsatz(circ, qubits, parameters)

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    controlB(circ, auxiliaryIndex, qubits, weights)

    circ.barrier()

    circ.h(auxiliaryIndex)


def calculateCostFunction(parameters: list, args: list) -> float: # this function has to be parallelized
    cost = 0
    if len(costHistory) > 0:
        cost = costHistory[len(costHistory) - 1]
    print("Iteration:", len(costHistory) + 1, ", cost:", cost, end="\r")
    overallSum1: float = 0
    overallSum2: float = 0
    backend = Aer.get_backend("aer_simulator")

    coefficientSet = args[0]
    transpiledHadamardCircuits = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]
    isQuantumSimulation = args[5]
    shots = args[6]
    exc = args[7]

    bindedHadamardGates = []
    for i in range(len(transpiledHadamardCircuits)):
        bindedHadamardGates.append(
            [
                j.bind_parameters({parametersHadamard: parameters})
                for j in transpiledHadamardCircuits[i]
            ]
        )
    bindedSpecHadamardGates = [
        i.bind_parameters({parametersSpecialHadamard: parameters})
        for i in transpiledSpecialHadamardCircuits
    ]
    lenPaulis = len(bindedSpecHadamardGates)

    # backend.set_options(executor=exc)
    # backend.set_options(max_job_size=1)

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
            if isQuantumSimulation:
                results = backend.run(bindedHadamardGates[i][j], shots=shots).result()
                outputstate = results.get_counts()
            else:
                job = backend.run(bindedHadamardGates[i][j])
                result = job.result()
                outputstate = np.real(
                    result.get_statevector(bindedHadamardGates[i][j], decimals=100)
                )

            m_sum = getMSum(isQuantumSimulation, outputstate, shots)
            multiply = coefficientSet[i] * coefficientSet[i + j]

            if (
                j == 0
            ):  # since the main diagional is not counted twice and  the list first element is the main diagional
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
            job = backend.run(bindedSpecHadamardGates[i])
            result = job.result()
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


# test and minimization functions here
def ansatzTest(circ: QuantumCircuit, qubits: int, outF: list):
    applyFixedAnsatz(circ, qubits, outF)
    circ.save_statevector()

    backend = Aer.get_backend("aer_simulator")

    with contextlib.redirect_stdout(io.StringIO()):
        t_circ = transpile(circ, backend)
    job = backend.run(t_circ)

    result = job.result()
    return result.get_statevector(circ, decimals=10)


def minimization(
    paulis: PauliList,
    coefficientSet: list,
    totalNeededQubits: int,
    bVector: list,
    quantumSimulation: bool = True,
    method: str = "COBYLA",
    shots: int = 100000,
    iterations: int = 200,
    verbose: bool = True,
    layers: int = 3,
    options: dict = {},
) -> List[List[float]]:
    global costHistory
    costHistory = []
    qubits = totalNeededQubits - 2
    # global weightsValueHistory
    # weightsValueHistory = []
    totalParamsNeeded = getTotalAnsatzParameters(qubits, layers)
    x: List[float] = [
        float(random.randint(0, 3000)) for _ in range(0, totalParamsNeeded)
    ]
    x = x / np.linalg.norm(x)
    start = time.time()
    (
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    ) = prepareCircuits(
        paulis,
        bVector,
        totalNeededQubits,
        quantumSimulation,
        layers,
        "aer_simulator",
    )
    end = time.time()
    if verbose:
        print("Time to prepare circuits:", end - start)

    exc = ThreadPoolExecutor(max_workers=4)
    arguments = [
        coefficientSet,
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
        quantumSimulation,
        shots,
        exc,
    ]

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


def prepareCircuits(# circuit contruction has to be rethought, since there are some parts that are repeated
    paulis: PauliList,
    bVector: List[float],
    totalNeededQubits: int,
    isQuantumSimulation: bool,
    layers: int,
    backendStr: str,
) -> (list, ParameterVector, list, ParameterVector):
    qubits = totalNeededQubits - 2
    backend = Aer.get_backend(backendStr)
    totalParamsNeeded = getTotalAnsatzParameters(qubits, layers)
    parametersHadamard: ParameterVector = ParameterVector(
        "parametersHadarmard", totalParamsNeeded
    )  # prone to change
    parametersSpecialHadamard: ParameterVector = ParameterVector(
        "parametersSpecialHadamard", totalParamsNeeded
    )
    parametersHadamardSplit = splitParameters(parametersHadamard, qubits, alternating=qubits!=3)
    parametersSpecialHadamardSplit = splitParameters(
        parametersSpecialHadamard, qubits,alternating=qubits!=3
    )
    hadamardCircuits: List[List[QuantumCircuit]] = []
    specialHadamardCircuits: List[QuantumCircuit] = []
    transpiledHadamardCircuits: List[List[QuantumCircuit]] = []

    for i in range(len(paulis)):
        tempHadamardCircuits: List[QuantumCircuit] = []
        for j in range(i, len(paulis)):
            if isQuantumSimulation:
                circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
                hadamardTest(
                    circ,
                    [paulis[i], paulis[j]],
                    qubits,
                    parametersHadamardSplit,
                )
                circ.measure(0, 0)
            else:
                circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
                hadamardTest(
                    circ,
                    [paulis[i], paulis[j]],
                    qubits,
                    parametersHadamardSplit,
                )
                circ.save_statevector()

            tempHadamardCircuits.append(circ)
        with contextlib.redirect_stdout(io.StringIO()):
            hadamardCircuits = transpile(tempHadamardCircuits, backend=backend)
        transpiledHadamardCircuits.append(hadamardCircuits)

    for i in range(len(paulis)):
        if isQuantumSimulation:
            circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
            specialHadamardTest(
                circ, [paulis[i]], qubits, parametersSpecialHadamardSplit, bVector
            )
            circ.measure(0, 0)
        else:
            circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
            specialHadamardTest(
                circ, [paulis[i]], qubits, parametersSpecialHadamardSplit, bVector
            )
            circ.save_statevector()

        specialHadamardCircuits.append(circ)
    with contextlib.redirect_stdout(io.StringIO()):
        transpiledSpecialHadamardCircuits = transpile(
            specialHadamardCircuits, backend=backend
        )

    return (
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    )


# Quantum normalized vector after ansatztest can have negative or positive values,
# so we need to check all combinations of signs, which one returns the minimum difference between b and bEstimated
# minimum difference between b and bEstimated is the sign combination we are looking for
def bestMatchingSignsVector(
    A: np.ndarray, xEstimated: np.array, b: np.array
) -> List[float]:
    values: List[int] = [-1, 1]
    combos: List[float] = list(
        product(values, repeat=len(xEstimated)) # this has to be rethought, since 2^2^5 is too much
    )  # generates all 8 bit combinations
    minDifference: float = 10000000
    minDifferenceValue: List[float] = []
    for combo in combos:
        vc: List[float] = np.multiply(
            xEstimated, list(combo)
        )  # multiply each element of vector with the corresponding element of combo
        bEstimated: List[float] = A.dot(vc)  # calculate bEst
        difference: float = np.linalg.norm(
            bEstimated - b
        )  # calculate difference between b and bEstimated
        if difference < minDifference:
            minDifference = difference
            minDifferenceValue = vc
    return minDifferenceValue


# estimate norm of vector
# once we got the sign combination, we can calculate the norm of the vector
# norm = b.T * b / b.T * A * v
# check this formula in the paper
def estimateNorm(
    A: np.ndarray, estimatedX: np.array, b: np.array, verbose: bool = False
) -> (float, List[float]):
    v: List[float] = bestMatchingSignsVector(A, estimatedX, b)
    leftSide: float = b.T.dot(A.dot(v))
    rightSide: float = b.T.dot(b)  # maybe test this with \vec{1} vector
    estimatedNorm: float = rightSide / leftSide

    if verbose:
        print("Estimated X:", estimatedX)
        print("Best matching signs vector:", v)
        print("Estimated norm:", estimatedNorm)

    return estimatedNorm, v


def gatesToCircuit(circuit: QuantumCircuit, gateList, barrier: bool = False):
    lastGate = ""
    for i in range(len(gateList)):
        if lastGate != gateList[i][0] and barrier:
            circuit.barrier()
        lastGate = gateList[i][0]
        if gateList[i][0] == "Ry":  # ("Ry", (theta, qubit))
            circuit.ry(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "Rx":  # ("Rx", (theta, qubit))
            circuit.rx(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "Rz":  # ("Rz", (theta, qubit))
            circuit.rz(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CNOT":  # ("CNOT", (control, target))
            circuit.cx(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CZ":  # ("CZ", (control, target))
            circuit.cz(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CCNOT":  # ("CCNOT", (control1, control2, target))
            circuit.ccx(gateList[i][1][0], gateList[i][1][1], gateList[i][1][2])
        elif gateList[i][0] == "CRx":  # ("CRx", (theta, (control, target))
            circuit.crx(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])
        elif gateList[i][0] == "CRy":  # ("CRy", (theta, (control, target))
            circuit.cry(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])
        elif gateList[i][0] == "CRz":  # ("CRz", (theta, (control, target))
            circuit.crz(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])


def getFixedAnsatzGates(
    qubits: int, parameters: List[List[float]], offset: int = 0, layers: int = 3
):  # maybe change to 2local or EfficientSU2
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html
    if qubits < 3:
        raise Exception("Qubits must be at least 3")

    gates = []
    qubitList = [i + offset for i in range(qubits)]
    if qubits == 3:
        for i in range(qubits):
            gates.append(("Ry", (parameters[0][i], qubitList[i])))
        gates.append(("CZ", (qubitList[0], qubitList[1])))
        gates.append(("CZ", (qubitList[2], qubitList[0])))

        for i in range(qubits):
            gates.append(("Ry", (parameters[1][i], qubitList[i])))
        gates.append(("CZ", (qubitList[1], qubitList[2])))
        gates.append(("CZ", (qubitList[2], qubitList[0])))

        for i in range(qubits):
            gates.append(("Ry", (parameters[2][i], qubitList[i])))
    else:
        for i in range(qubits):
            gates.append(("Ry", (parameters[0][i], qubitList[i])))

        layer = 1
        for i in range(layers):
            for i in range(qubits // 2):
                gates.append(("CZ", (qubitList[2 * i], qubitList[2 * i + 1])))
            for i in range(qubits):
                gates.append(("Ry", (parameters[layer][i], qubitList[i])))
            layer = layer + 1
            for i in range(1, qubits, 2):
                if i + 1 < qubits:
                    gates.append(("CZ", (qubitList[i], qubitList[i + 1])))
            for i in range(1, qubits - 1):
                gates.append(("Ry", (parameters[layer][i - 1], qubitList[i])))
            layer = layer + 1
    return gates


def getControlledFixedAnsatzGates(qubits: int, parameters: List[List[float]]):
    gates = getFixedAnsatzGates(qubits, parameters)
    controlledGates = []
    auxiliaryQubit = 0
    auxiliaryQubit2 = qubits + 1
    for i in range(len(gates)):
        if gates[i][0] == "Ry":
            controlledGates.append(("CRy", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "Rx":
            controlledGates.append(("CRx", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "Rz":
            controlledGates.append(("CRz", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "CZ":
            controlledGates.append(
                ("CCNOT", (auxiliaryQubit, gates[i][1][1] + 1, auxiliaryQubit2))
            )
            controlledGates.append(("CZ", (gates[i][1][0] + 1, auxiliaryQubit2)))
            controlledGates.append(
                ("CCNOT", (auxiliaryQubit, gates[i][1][1] + 1, auxiliaryQubit2))
            )

    return controlledGates


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
