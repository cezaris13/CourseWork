from qiskit import QuantumCircuit
from typing import List
from math import ceil
import numpy as np

def controlledLabelVectorCircuit(
    circ: QuantumCircuit, auxiliaryIndex: int, qubits: int, values: List[float]
):
    qubits = [i + 1 for i in range(qubits)]
    custom = labelVectorCircuit(values).to_gate().control()
    circ.append(custom, [auxiliaryIndex] + qubits)


def labelVectorCircuit(values: List[float]) -> QuantumCircuit:
    qubits: int = ceil(np.log2(len(values)))
    if len(values) != 2**qubits:
        values = np.pad(values, (0, 2**qubits - len(values)), "constant")
    values = values / np.linalg.norm(values)
    circ: QuantumCircuit = QuantumCircuit(qubits)
    circ.prepare_state(values)
    return circ


def getPaddedLabelVector(values: List[float]) -> np.array:
    qubits: int = ceil(np.log2(len(values)))
    if len(values) != 2**qubits:
        values = np.pad(values, (0, 2**qubits - len(values)), "constant")
    return np.array(values / np.linalg.norm(values))