from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import PauliList
from typing import List
import contextlib
import io

from Code.VQLS.Ansatz import applyFixedAnsatz, controlFixedAnsatz
from Code.VQLS.LCU import convertMatrixIntoCircuit
from Code.VQLS.LabelVector import controlB


def ansatzTest(circ: QuantumCircuit, qubits: int, outF: list):
    applyFixedAnsatz(circ, qubits, outF)
    circ.save_statevector() # this might be the problem

    backend = Aer.get_backend("aer_simulator")

    with contextlib.redirect_stdout(io.StringIO()):
        t_circ = transpile(circ, backend)
    job = backend.run(t_circ)

    result = job.result()
    return result.get_statevector(circ, decimals=10)


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
