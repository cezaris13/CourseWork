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
            currentGate = paulis[p][i] # figure out type
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
            currentGate = paulis[p][i] # figure out type
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
    circ: QuantumCircuit, qubits: List[int], parameters: List[List[float]], auxiliaryIndex: int
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
    )  # sita pakeisti predefined instructions

    circ.barrier()

    controlB(
        circ, auxiliaryIndex, qubits, weights
    )  # sita pakeisti predefined instructions(turetu sutaupyti laiko)

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
def calculateCostFunctionMatrixOld(parameters: list, args: list) -> float:
    print("Iteration:", len(costHistory) + 1, end="\r")
    overallSum1 = 0
    overallSum2 = 0
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    paulis = args[0]
    coefficientSet = args[1]
    bVector = args[2]
    totalNumberOfQubits = args[3]
    backend = Aer.get_backend("aer_simulator")

    for i, pauliI in enumerate(paulis):
        for j, pauliJ in enumerate(paulis):
            circ = QuantumCircuit(totalNumberOfQubits, totalNumberOfQubits)
            hadamardTest(circ, [pauliI, pauliJ], [1, 2, 3], 0, parameters)
            circ.save_statevector()
            t_circ = transpile(circ, backend)
            job = backend.run(t_circ)

            result = job.result()
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            o = outputstate

            m_sum = 0
            for l in range(len(o)):
                if l % 2 == 1:
                    n = o[l] ** 2
                    m_sum += n

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - (2 * m_sum))

    resultVectors = []
    for i, pauliI in enumerate(paulis):
        circ = QuantumCircuit(totalNumberOfQubits, totalNumberOfQubits)
        specialHadamardTest(circ, [pauliI], [1, 2, 3], 0, parameters, bVector)
        circ.save_statevector()
        t_circ = transpile(circ, backend)
        job = backend.run(t_circ)
        result = job.result()
        outputstate = np.real(result.get_statevector(circ, decimals=100))
        resultVectors.append(outputstate)

    for i in range(len(paulis)):  # optimize it little bit more
        for j in range(len(paulis)):
            mult = 1
            for extra in range(2):
                if extra == 0:
                    o = resultVectors[i]
                if extra == 1:
                    o = resultVectors[j]

                m_sum = 0
                for l in range(len(o)):
                    if l % 2 == 1:
                        n = o[l] ** 2
                        m_sum += n
                mult = mult * (1 - (2 * m_sum))

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum2 += multiply * mult

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)

    return totalCost

def calculateCostFunctionMatrix(parameters: list, args: list) -> float:
    print("Iteration:", len(costHistory) + 1, end="\r")
    overallSum1 = 0
    overallSum2 = 0
    backend = Aer.get_backend("aer_simulator")

    coefficientSet = args[0]
    transpiledHadamardCircuits = args[1]
    parametersHadamard = args[2]
    transpiledSpecialHadamardCircuits = args[3]
    parametersSpecialHadamard = args[4]

    qcrs = [i.bind_parameters({parametersHadamard: parameters}) for i in transpiledHadamardCircuits]
    qcrs1 = [i.bind_parameters({parametersSpecialHadamard: parameters}) for i in transpiledSpecialHadamardCircuits]
    lenPaulis = len(qcrs1)

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            job = backend.run(qcrs[i*lenPaulis + j])
            result = job.result()
            outputstate = np.real(result.get_statevector(qcrs[i*lenPaulis + j], decimals=100))

            m_sum = 0
            for l in range(len(outputstate)):
                if l % 2 == 1:
                    n = outputstate[l] ** 2
                    m_sum += n

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - (2 * m_sum))

    resultVectors = []
    for i in range(lenPaulis):
        job = backend.run(qcrs1[i])
        result = job.result()
        outputstate = np.real(result.get_statevector(qcrs1[i], decimals=100))
        resultVectors.append(outputstate)

    for i in range(lenPaulis):  # optimize it little bit more
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
def calculateCostFunctionQuantumSimulationOld(
    parameters: list, args: list
) -> float:
    print("Iteration:", len(costHistory) + 1, end="\r")
    overallSum1 = 0
    overallSum2 = 0
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    paulis = args[0]
    coefficientSet = args[1]
    bVector = args[2]
    totalNumberOfQubits = args[3]
    shots = args[4]

    backend = Aer.get_backend("aer_simulator")

    for i, pauliI in enumerate(paulis):
        for j, pauliJ in enumerate(paulis):
            circ = QuantumCircuit(totalNumberOfQubits, 1)
            hadamardTest(circ, [pauliI, pauliJ], [1, 2, 3], 0, parameters)
            circ.measure(0, 0)

            t_circ = transpile(circ, backend)
            job = backend.run(t_circ, shots=shots)

            result = job.result()
            outputstate = result.get_counts(circ)

            if "1" in outputstate.keys():
                m_sum = float(outputstate["1"]) / shots
            else:
                m_sum = 0

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - 2 * m_sum)

    resultVectors = []
    for i, pauliI in enumerate(paulis):
        circ = QuantumCircuit(totalNumberOfQubits, 1)
        specialHadamardTest(circ, [pauliI], [1, 2, 3], 0, parameters, bVector)
        circ.measure(0, 0)
        t_circ = transpile(circ, backend)
        job = backend.run(t_circ, shots=shots)
        result = job.result()
        outputstate = result.get_counts(circ)
        resultVectors.append(outputstate)

    for i in range(len(paulis)):
        for j in range(len(paulis)):
            mult = 1
            for extra in range(2):
                if extra == 0:
                    outputstate = resultVectors[i]
                if extra == 1:
                    outputstate = resultVectors[j]

                if "1" in outputstate.keys():
                    m_sum = float(outputstate["1"]) / shots
                else:
                    m_sum = 0

                mult = mult * (1 - 2 * m_sum)

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum2 += multiply * mult

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)
    return totalCost

def calculateCostFunctionQuantumSimulation(
    parameters: list, args: list
) -> float:
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

    qcrs = [i.bind_parameters({parametersHadamard: parameters}) for i in transpiledHadamardCircuits]
    qcrs1 = [i.bind_parameters({parametersSpecialHadamard: parameters}) for i in transpiledSpecialHadamardCircuits]
    lenPaulis = len(qcrs1)

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            job = backend.run(qcrs[i*lenPaulis + j], shots=shots)
            result = job.result()
            outputstate = result.get_counts(qcrs[i*lenPaulis + j])

            if "1" in outputstate.keys():
                m_sum = float(outputstate["1"]) / shots
            else:
                m_sum = 0

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum1 += multiply * (1 - 2 * m_sum)

    resultVectors = []
    for i in range(lenPaulis):
        job = backend.run(qcrs1[i], shots=shots)
        result = job.result()
        outputstate = result.get_counts(qcrs1[i])
        resultVectors.append(outputstate)

    for i in range(lenPaulis):
        for j in range(lenPaulis):
            mult = 1
            for extra in range(2):
                if extra == 0:
                    outputstate = resultVectors[i]
                if extra == 1:
                    outputstate = resultVectors[j]

                if "1" in outputstate.keys():
                    m_sum = float(outputstate["1"]) / shots
                else:
                    m_sum = 0

                mult = mult * (1 - 2 * m_sum)

            multiply = coefficientSet[i] * coefficientSet[j]
            overallSum2 += multiply * mult

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

def minimizationOld(
    paulis: PauliList,
    coefficientSet: list,
    totalNeededQubits: int,
    bVector: list,
    quantumSimulation: bool = True,
    method: str = "COBYLA",
    shots = 100000
) -> list:
    global costHistory
    costHistory = []
    x = [float(random.randint(0, 3000)) for _ in range(0, 9)]
    x = x / np.linalg.norm(x)
    if quantumSimulation:
        out = minimize(
            calculateCostFunctionQuantumSimulationOld,
            x0=x,
            args=[paulis, coefficientSet, bVector, totalNeededQubits, shots],
            method=method,
            options={"maxiter": 200},
        )
    else:
        out = minimize(
            calculateCostFunctionMatrixOld,
            x0=x,
            args=[paulis, coefficientSet, bVector, totalNeededQubits],
            method=method,
            options={"maxiter": 200},
        )
    print(out)
    return [out["x"][0:3], out["x"][3:6], out["x"][6:9]]

def minimization(
    paulis: PauliList,
    coefficientSet: list,
    totalNeededQubits: int,
    bVector: list,
    quantumSimulation: bool = True,
    method: str = "COBYLA",
    shots: int = 100000,
) -> list:
    global costHistory
    costHistory = []
    x = [float(random.randint(0, 3000)) for _ in range(0, 9)]
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
            ],
            method=method,
            options={"maxiter": 200},
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
            options={"maxiter": 200},
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
    parametersSpecialHadamard: ParameterVector = ParameterVector("parametersSpecialHadamard", 9)
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
                circ = QuantumCircuit(totalNeededQubits, 1)
            else:
                circ = QuantumCircuit(totalNeededQubits)
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
            circ = QuantumCircuit(totalNeededQubits, 1)
        else:
            circ = QuantumCircuit(totalNeededQubits)
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

# from qiskit.circuit.library import EfficientSU2
# from qiskit import QuantumCircuit
# qubits = 2
# ansatz = EfficientSU2(qubits, su2_gates=['ry'], entanglement='circular', reps=qubits -1, insert_barriers=True)
# qc = QuantumCircuit(qubits)  # create a circuit and append the RY variational form
# qc.compose(ansatz, inplace=True)
# #decompose whats inside
# qc.decompose().draw(output="mpl")
# # qc.draw(output="mpl")

# from qiskit.circuit.library import TwoLocal
# qubits = 4
# ansatz = TwoLocal(qubits, ['ry','cz'], reps=qubits -1, insert_barriers=True)
# qc = QuantumCircuit(qubits)  # create a circuit and append the RY variational form
# qc.compose(ansatz, inplace=True)
# #decompose whats inside
# qc.decompose().draw(output="mpl")

# import numpy as np
# from qiskit import QuantumCircuit, Aer, transpile, execute
# from qiskit.quantum_info.operators import Operator
# def convertMatrixIntoUnitary(matrix:np.ndarray):
#     Q,_ = np.linalg.qr(matrix)
#     # print(Q)
#     for i in range(Q.shape[1]):
#         Q[:,i] = Q[:,i]/np.linalg.norm(Q[:,i])

#     # print(Q)
#     return Operator(Q)

# def convertUnitaryIntoCircuit(unitary:Operator):
#     return unitary.to_instruction()

# matrix = np.array([[1, 0, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0],
#                      [0, 3, 0, 0]])

# unitary = convertMatrixIntoUnitary(matrix)
# circuit = convertUnitaryIntoCircuit(unitary)

# qc = QuantumCircuit(2,2)
# qc.append(circuit, [0, 1])

# backend = Aer.get_backend('unitary_simulator')

# job = execute(qc, backend, shots=8192)
# result = job.result()

# # print(result.get_unitary(circuit, decimals=3))
# qc.decompose().draw(output="mpl")

# # circuit.draw(output="mpl")1

# from qiskit.quantum_info import Operator

# circuit = QuantumCircuit(2)

# cx = Operator([
#     [1, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, 1, 0],
#     [0, 1, 0, 0]
# ])
# circuit.unitary(cx, [0, 1], label='cx')
# circuit.draw(output="mpl")
# backend = Aer.get_backend('unitary_simulator')
# t_gate = transpile(circuit, backend, basis_gates=['cx', 'u3', 'u1', 'u2', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'ccx', 'cccx', 'cswap'])
# t_gate.draw(output="mpl")
# # execute(circ, backend).result().get_unitary()