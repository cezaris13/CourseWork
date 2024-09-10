from qiskit import QuantumCircuit
from typing import List
from math import ceil
import numpy as np

def controlledLabelVectorCircuit(
    circ: QuantumCircuit, auxiliaryIndex: int, qubits: int, values: List[float]
):
    '''
    Takes vector, converts it to gate, converts to control gate, and applies it to the circuit.

    circ - quantum circuit on which the controlled gate will be applied.

    auxilaryIndex - index of a qubit which will control the label vector gate.

    qubits - number of qubits needed for the quantum circuit.

    values - y vector values.
    '''
    qubits = [i + 1 for i in range(qubits)]
    custom = labelVectorCircuit(values).to_gate().control()
    circ.append(custom, [auxiliaryIndex] + qubits)


def labelVectorCircuit(values: List[float]) -> QuantumCircuit:
    '''
    Takes vector, converts it to gate.
    If the vector length is not power of 2, it is padded with 0s.

    values - y vector values.
    '''
    values: np.array = getPaddedLabelVector(values)
    qubits: int = ceil(np.log2(len(values)))

    circ: QuantumCircuit = QuantumCircuit(qubits)
    circ.prepare_state(values)
    return circ


def getPaddedLabelVector(values: List[float]) -> np.array:
    '''
    If the vector length is not power of 2, it is padded with 0s.

    Returns normalized power of 2 vector.

    values - y vector values.
    '''
    qubits: int = ceil(np.log2(len(values)))
    if len(values) != 2**qubits:
        values = np.pad(values, (0, 2**qubits - len(values)), "constant")
    return np.array(values / np.linalg.norm(values))