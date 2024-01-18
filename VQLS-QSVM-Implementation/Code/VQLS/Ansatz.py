from typing import List
from qiskit import QuantumCircuit


def fixedAnsatz(
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


# Creates controlled anstaz for calculating |<b|psi>|^2 with a Hadamard test
def controlledFixedAnsatz(
    circ: QuantumCircuit,
    qubits: int,
    parameters: List[List[float]],
    barrier: bool = False,
):
    gates = getControlledFixedAnsatzGates(qubits, parameters)
    gatesToCircuit(circ, gates, barrier=barrier)


def gatesToCircuit(circuit: QuantumCircuit, gateList, barrier: bool = False):# add type
    lastGate: str = ""
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
