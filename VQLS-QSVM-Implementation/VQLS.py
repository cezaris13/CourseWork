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

costHistory = []


def getApproximationValue(A: np.ndarray, b: np.array, o: np.array) -> float:
    return ((b.dot(A.dot(o) / (np.linalg.norm(A.dot(o))))) ** 2).real


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
    circ: QuantumCircuit, qubits: List[int], parameters: List[List[float]]
):  # maybe change to 2local or EfficientSU2
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html
    for i in range(len(qubits)):
        circ.ry(parameters[0][i], qubits[i])

    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])

    for i in range(len(qubits)):
        circ.ry(parameters[1][i], qubits[i])

    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])

    for i in range(len(qubits)):
        circ.ry(parameters[2][i], qubits[i])


# Creates the Hadamard test
def hadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    qubits: List[int],
    auxiliaryIndex: int,
    parameters: List[List[float]],
):
    circ.h(auxiliaryIndex)

    circ.barrier()

    applyFixedAnsatz(circ, qubits, parameters)

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
    circ: QuantumCircuit, auxiliaryIndex: int, qubits: List[int], values: List[float]
):
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
    qubits: List[int],
    parameters: List[List[float]],
    auxiliaryIndex: int,
):
    for i in range(len(qubits)):
        circ.cry(parameters[0][i], auxiliaryIndex, qubits[i])

    circ.ccx(auxiliaryIndex, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(auxiliaryIndex, qubits[1], 4)

    circ.ccx(auxiliaryIndex, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliaryIndex, qubits[0], 4)

    for i in range(len(qubits)):
        circ.cry(parameters[1][i], auxiliaryIndex, qubits[i])

    circ.ccx(auxiliaryIndex, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(auxiliaryIndex, qubits[2], 4)

    circ.ccx(auxiliaryIndex, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliaryIndex, qubits[0], 4)

    for i in range(len(qubits)):
        circ.cry(parameters[2][i], auxiliaryIndex, qubits[i])


# Create the controlled Hadamard test, for calculating <psi|psi>
def specialHadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    qubits: List[int],
    auxiliaryIndex: int,
    parameters: List[List[float]],
    weights: List[float],
):
    circ.h(auxiliaryIndex)

    circ.barrier()

    controlFixedAnsatz(circ, qubits, parameters, auxiliaryIndex)

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


# Now, we are ready to calculate the final cost function. This simply involves us taking the products of all combinations of the expectation outputs from the different circuits,
# multiplying by their respective coefficients, and arranging into the cost function that we discussed previously!
# Implements the entire cost function on the quantum circuit theoretically


### This code may look long and daunting, but it isn't! In this simulation,
# I'm taking a numerical approach, where I'm calculating the amplitude squared of each state corresponding to a measurement of the auxiliary Hadamard test qubit in the $1$ state, then calculating P(0) - P(1)  = 1 - 2P(1) with that information.
# This is very exact, but is not realistic, as a real quantum device would have to sample the circuit many times to generate these probabilities (I'll discuss sampling later).
# In addition, this code is not completely optimized (it completes more evaluations of the quantum circuit than it has to), but this is the simplest way in which the code can be implemented,
# and I will be optimizing it in an update to this tutorial in the near future.
def calculateCostFunctionMatrix(parameters: list, args: list) -> float:
    print("Iteration:", len(costHistory) + 1, end="\r")
    overallSum1: float = 0
    overallSum2: float = 0
    backend = Aer.get_backend("aer_simulator")

    coefficientSet = args[0]
    transpiledHadamardCircuits = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]

    bindedHadamardGates = [
        i.bind_parameters({parametersHadamard: parameters})
        for i in transpiledHadamardCircuits
    ]
    bindedSpecHadamardGates = [
        i.bind_parameters({parametersSpecialHadamard: parameters})
        for i in transpiledSpecialHadamardCircuits
    ]
    lenPaulis = len(bindedSpecHadamardGates)

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            job = backend.run(bindedHadamardGates[i * lenPaulis + j])
            result = job.result()
            outputstate = np.real(
                result.get_statevector(
                    bindedHadamardGates[i * lenPaulis + j], decimals=100
                )
            )

            m_sum = 0
            for l in range(len(outputstate)):
                if l % 2 == 1:
                    n = outputstate[l] ** 2
                    m_sum += n

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - (2 * m_sum))

    resultVectors = []
    for i in range(lenPaulis):
        job = backend.run(bindedSpecHadamardGates[i])
        result = job.result()
        outputstate = np.real(
            result.get_statevector(bindedSpecHadamardGates[i], decimals=100)
        )
        resultVectors.append(outputstate)

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            mult = 1
            for extra in range(2):
                if extra == 0:
                    outputstate = resultVectors[i]
                if extra == 1:
                    outputstate = resultVectors[j]

                m_sum = 0
                for l in range(len(outputstate)):
                    if l % 2 == 1:
                        n = outputstate[l] ** 2
                        m_sum += n
                mult = mult * (1 - (2 * m_sum))

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum2 += multiply * mult

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    # print(totalCost)
    costHistory.append(totalCost)
    return totalCost


# Now, we have found that this algorithm works **in theory**.
# I tried to run some simulations with a circuit that samples the circuit instead of calculating the probabilities numerically.
#  Now, let's try to **sample** the quantum circuit, as a real quantum computer would do!
# For some reason, this simulation would only converge somewhat well for a ridiculously high number of "shots" (runs of the circuit, in order to calculate the probability distribution of outcomes).
# I think that this is mostly to do with limitations in the classical optimizer (COBYLA), due to the noisy nature of sampling a quantum circuit (a measurement with the same parameters won't always yield the same outcome).
# Luckily, there are other optimizers that are built for noisy functions, such as SPSA, but we won't be looking into that in this tutorial.


# Implements the entire cost function on the quantum circuit (sampling, 100000 shots) on the quantum circuit
def calculateCostFunctionQuantumSimulation(parameters: list, args: list) -> float:
    print("Iteration:", len(costHistory) + 1, end="\r")

    overallSum1 = 0
    overallSum2 = 0
    backend = Aer.get_backend("aer_simulator")

    coefficientSet = args[0]
    transpiledHadamardCircuits = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]
    shots = args[5]
    exc = args[6]

    bindedHadamardGates = [
        i.bind_parameters({parametersHadamard: parameters})
        for i in transpiledHadamardCircuits
    ]
    bindedSpecHadamardGates = [
        i.bind_parameters({parametersSpecialHadamard: parameters})
        for i in transpiledSpecialHadamardCircuits
    ]
    lenPaulis = len(bindedSpecHadamardGates)

    backend.set_options(executor=exc)
    backend.set_options(max_job_size=1)

    results = backend.run(bindedHadamardGates, shots=shots).result()

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            outputstate = results.get_counts(bindedHadamardGates[i * lenPaulis + j])

            if "1" in outputstate.keys():
                m_sum = float(outputstate["1"]) / shots
            else:
                m_sum = 0

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - 2 * m_sum)

    del results
    del bindedHadamardGates

    gc.collect()

    results = backend.run(bindedSpecHadamardGates, shots=shots).result()

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            mult = 1
            for extra in range(2):
                if extra == 0:
                    outputstate = results.get_counts(bindedSpecHadamardGates[i])
                if extra == 1:
                    outputstate = results.get_counts(bindedSpecHadamardGates[j])

                if "1" in outputstate.keys():
                    m_sum = float(outputstate["1"]) / shots
                else:
                    m_sum = 0

                mult = mult * (1 - 2 * m_sum)

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum2 += multiply * mult

    del results
    del bindedSpecHadamardGates

    gc.collect()

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)
    return totalCost


# test and minimization functions here
def ansatzTest(circ: QuantumCircuit, outF: list):
    applyFixedAnsatz(circ, [0, 1, 2], outF)
    circ.save_statevector()

    backend = Aer.get_backend("aer_simulator")

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
) -> List[List[float]]:
    global costHistory
    costHistory = []
    x: List[float] = [float(random.randint(0, 3000)) for _ in range(0, 9)]
    x = x / np.linalg.norm(x)
    start = time.time()
    (
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    ) = prepareCircuits(
        paulis, bVector, totalNeededQubits, quantumSimulation, "aer_simulator"
    )
    end = time.time()
    print("Time to prepare circuits:", end - start)
    start = time.time()
    if quantumSimulation:
        exc = ThreadPoolExecutor(max_workers=4)
        out = minimize(
            calculateCostFunctionQuantumSimulation,
            x0=x,
            args=[
                coefficientSet,
                transpiledHadamardCircuits,
                parametersHadamard,
                transpiledSpecialHadamardCircuits,
                parametersSpecialHadamard,
                shots,
                exc,
            ],
            method=method,
            options={"maxiter": iterations},
        )
    else:
        out = minimize(
            calculateCostFunctionMatrix,
            x0=x,
            args=[
                coefficientSet,
                transpiledHadamardCircuits,
                parametersHadamard,
                transpiledSpecialHadamardCircuits,
                parametersSpecialHadamard,
            ],
            method=method,
            options={"maxiter": iterations},
        )

    end = time.time()
    print("Time to minimize:", end - start)
    print(out)
    return [out["x"][0:3], out["x"][3:6], out["x"][6:9]]


def prepareCircuits(
    paulis: PauliList,
    bVector: List[float],
    totalNeededQubits: int,
    isQuantumSimulation: bool,
    backendStr: str,
) -> (list, ParameterVector, list, ParameterVector):
    backend = Aer.get_backend(backendStr)
    parametersHadamard: ParameterVector = ParameterVector("parametersHadarmard", 9)
    parametersSpecialHadamard: ParameterVector = ParameterVector(
        "parametersSpecialHadamard", 9
    )
    parametersHadamardSplit = [
        parametersHadamard[0:3],
        parametersHadamard[3:6],
        parametersHadamard[6:9],
    ]
    parametersSpecialHadamardSplit = [
        parametersSpecialHadamard[0:3],
        parametersSpecialHadamard[3:6],
        parametersSpecialHadamard[6:9],
    ]

    hadamardCircuits: List[QuantumCircuit] = []
    specialHadamardCircuits: List[QuantumCircuit] = []

    for i in range(len(paulis)):
        for j in range(len(paulis)):
            if isQuantumSimulation:
                circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
            else:
                circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
            hadamardTest(
                circ, [paulis[i], paulis[j]], [1, 2, 3], 0, parametersHadamardSplit
            )
            if isQuantumSimulation:
                circ.measure(0, 0)
            else:
                circ.save_statevector()
            hadamardCircuits.append(circ)

    transpiledHadamardCircuits = transpile(hadamardCircuits, backend=backend)

    for i in range(len(paulis)):
        if isQuantumSimulation:
            circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
        else:
            circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
        specialHadamardTest(
            circ, [paulis[i]], [1, 2, 3], 0, parametersSpecialHadamardSplit, bVector
        )
        if isQuantumSimulation:
            circ.measure(0, 0)
        else:
            circ.save_statevector()
        specialHadamardCircuits.append(circ)

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
        product(values, repeat=len(xEstimated))
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
    A: np.ndarray, estimatedX: np.array, b: np.array
) -> (float, List[float]):
    v: List[float] = bestMatchingSignsVector(A, estimatedX, b)
    leftSide: float = b.T.dot(A.dot(v))
    rightSide: float = b.T.dot(b)
    estimatedNorm: float = rightSide / leftSide
    return estimatedNorm, v
