from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import SparsePauliOp, PauliList
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import ceil


class QuantumSVM:
    def __init__(self, shots=100000, auxiliaryQubit: int = 0):
        self.auxiliaryQubit = auxiliaryQubit
        self.shots = shots
        self.costHistory = []

    def zGate(self) -> np.array:
        return np.array([[1, 0], [0, -1]])

    def identityGate(self) -> np.array:
        return np.eye(2)

    def getApproximationValue(self, A: np.ndarray, b: np.array, o: np.array):
        print(
            ((b.dot(A.dot(o) / (np.linalg.norm(A.dot(o))))) ** 2).real
        )  # change this to return type

    def plotCost(self):
        plt.style.use("seaborn-v0_8")
        plt.plot(self.costHistory, "g")
        plt.ylabel("Cost function")
        plt.xlabel("Optimization steps")
        plt.show()

    def convertMatrixIntoCircuit(
        self,
        circuit: QuantumCircuit,
        paulis: PauliList,
        controlled=False,
        auxiliaryQubit=0,
        showBarriers=True,
    ):
        qubitIndexList = []
        qubits = circuit.num_qubits
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

    def getMatrixCoeffitients(self, pauliOp: SparsePauliOp) -> list:
        coeffs = []
        paulis = pauliOp.paulis
        for p in range(len(paulis)):
            containsIdentity = False
            for i in range(len(paulis[p])):
                currentGate = paulis[p][len(paulis[p]) - 1 - i]
                if currentGate.x == False and currentGate.z == False:
                    containsIdentity = True
            coeffs.append(pauliOp.coeffs[p])
            if containsIdentity == False:
                coeffs.append(pauliOp.coeffs[p])
        return coeffs

    # VLQS part

    def applyFixedAnsatz(
        self, circ: QuantumCircuit, qubits: list, parameters: list
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
        self,
        circ: QuantumCircuit,
        paulis: PauliList,
        qubits: list,
        auxiliaryIndex: int,
        parameters: list,
    ):
        circ.h(auxiliaryIndex)

        circ.barrier()

        self.applyFixedAnsatz(circ, qubits, parameters)

        circ.barrier()

        self.convertMatrixIntoCircuit(
            circ,
            paulis,
            controlled=True,
            auxiliaryQubit=auxiliaryIndex,
            showBarriers=False,
        )  # change to predefined instructions

        circ.barrier()

        circ.h(auxiliaryIndex)

    def controlB(
        self, circ: QuantumCircuit, auxiliaryIndex: int, qubits: list, values: list
    ):
        custom = self.createB(values).to_gate().control()
        circ.append(custom, [auxiliaryIndex] + qubits)

    def createB(self, values: list) -> QuantumCircuit:
        qubits = ceil(np.log2(len(values)))
        if len(values) != 2**qubits:
            values = np.pad(values, (0, 2**qubits - len(values)), "constant")
        values = values / np.linalg.norm(values)
        circ = QuantumCircuit(qubits)
        circ.prepare_state(values)
        return circ

    def getBArray(self, values: list) -> np.array:
        qubits = ceil(np.log2(len(values)))
        if len(values) != 2**qubits:
            values = np.pad(values, (0, 2**qubits - len(values)), "constant")
        return np.array(values / np.linalg.norm(values))

    # Creates controlled anstaz for calculating |<b|psi>|^2 with a Hadamard test
    def controlFixedAnsatz(
        self, circ: QuantumCircuit, qubits: list, parameters: list, auxiliaryIndex: int
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
        self,
        circ: QuantumCircuit,
        paulis: PauliList,
        qubits: list,
        auxiliaryIndex: int,
        parameters: list,
        weights: list,
    ):
        circ.h(auxiliaryIndex)

        circ.barrier()

        self.controlFixedAnsatz(circ, qubits, parameters, auxiliaryIndex)

        circ.barrier()

        self.convertMatrixIntoCircuit(
            circ,
            paulis,
            controlled=True,
            auxiliaryQubit=auxiliaryIndex,
            showBarriers=False,
        )  # sita pakeisti predefined instructions

        circ.barrier()

        self.controlB(
            circ, auxiliaryIndex, qubits, weights
        )  # sita pakeisti predefined instructions(turetu sutaupyti laiko)

        circ.barrier()

        circ.h(auxiliaryIndex)

    # cost functions here
    # Implements the entire cost function on the quantum circuit theoretically
    def calculateCostFunctionMatrix(self, parameters: list, args: list) -> float:
        print("Iteration:", len(self.costHistory) + 1, end="\r")
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
                self.hadamardTest(circ, [pauliI, pauliJ], [1, 2, 3], 0, parameters)
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
            self.specialHadamardTest(circ, [pauliI], [1, 2, 3], 0, parameters, bVector)
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
        self.costHistory.append(totalCost)

        return totalCost

    # Implements the entire cost function on the quantum circuit (sampling, 100000 shots) on the quantum circuit
    def calculateCostFunctionQuantumSimulation(
        self, parameters: list, args: list
    ) -> float:
        print("Iteration:", len(self.costHistory) + 1, end="\r")
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
                self.hadamardTest(circ, [pauliI, pauliJ], [1, 2, 3], 0, parameters)
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
            self.specialHadamardTest(circ, [pauliI], [1, 2, 3], 0, parameters, bVector)
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
        self.costHistory.append(totalCost)
        return totalCost

    # test and minimization functions here
    def ansatzTest(self, circ: QuantumCircuit, outF: list):
        self.applyFixedAnsatz(circ, [0, 1, 2], outF)
        circ.save_statevector()

        backend = Aer.get_backend("aer_simulator")

        t_circ = transpile(circ, backend)
        job = backend.run(t_circ)

        result = job.result()
        return result.get_statevector(circ, decimals=10)

    def minimization(
        self,
        paulis: PauliList,
        coefficientSet: list,
        totalNeededQubits: int,
        bVector: list,
        quantumSimulation: bool = True,
        method: str = "COBYLA",
    ) -> list:
        self.costHistory = []
        x = [float(random.randint(0, 3000)) for _ in range(0, 9)]
        x = x / np.linalg.norm(x)
        if quantumSimulation:
            out = minimize(
                self.calculateCostFunctionQuantumSimulation,
                x0=x,
                args=[paulis, coefficientSet, bVector, totalNeededQubits, self.shots],
                method=method,
                options={"maxiter": 200},
            )
        else:
            out = minimize(
                self.calculateCostFunctionMatrix,
                x0=x,
                args=[paulis, coefficientSet, bVector, totalNeededQubits],
                method=method,
                options={"maxiter": 200},
            )
        print(out)
        return [out["x"][0:3], out["x"][3:6], out["x"][6:9]]


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