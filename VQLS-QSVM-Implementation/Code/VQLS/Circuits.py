from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import PauliList
from qiskit.circuit import ParameterVector
from typing import List
import contextlib
import io

from Code.VQLS.Ansatz import fixedAnsatz, controlledFixedAnsatz
from Code.VQLS.LCU import convertMatrixIntoCircuit
from Code.VQLS.LabelVector import controlledLabelVectorCircuit
from Code.Utils import getTotalAnsatzParameters, splitParameters


def prepareCircuits(
    paulis: PauliList,
    bVector: List[float],
    qubits: int,
    isQuantumSimulation: bool,
    layers: int,
    backendStr: str,
) -> (list, ParameterVector, list, ParameterVector):
    backend = Aer.get_backend(backendStr)
    parametersHadamard, parametersHadamardSplit = prepareParameterVector(
        "parametersHadarmard", qubits, layers
    )
    parametersSpecialHadamard, parametersSpecialHadamardSplit = prepareParameterVector(
        "parametersSpecialHadamard", qubits, layers
    )

    labelVectorCircuit = QuantumCircuit(qubits + 1)
    controlledLabelVectorCircuit(labelVectorCircuit, 0, qubits, bVector)

    fixedAnsatzCircuit = QuantumCircuit(qubits + 1)
    fixedAnsatz(fixedAnsatzCircuit, qubits, parametersHadamardSplit, offset=1)

    controlledFixedAnsatzCircuit = QuantumCircuit(qubits + 2)
    controlledFixedAnsatz(
        controlledFixedAnsatzCircuit, qubits, parametersSpecialHadamardSplit
    )

    hadamardCircuits: List[List[QuantumCircuit]] = []
    specialHadamardCircuits: List[QuantumCircuit] = []
    transpiledHadamardCircuits: List[List[QuantumCircuit]] = []

    for i in range(len(paulis)):
        tempHadamardCircuits: List[QuantumCircuit] = []
        for j in range(i, len(paulis)):
            hadamardTest1 = lambda circuit: hadamardTest(
                circuit, [paulis[i], paulis[j]], fixedAnsatzCircuit
            )
            circ = constructCircuit(isQuantumSimulation, qubits + 1, hadamardTest1)
            tempHadamardCircuits.append(circ)
        with contextlib.redirect_stdout(io.StringIO()):
            hadamardCircuits = transpile(tempHadamardCircuits, backend=backend)
        transpiledHadamardCircuits.append(hadamardCircuits)

    for i in range(len(paulis)):
        specHadamardTest = lambda circuit: specialHadamardTest(
            circuit,
            [paulis[i]],
            controlledFixedAnsatzCircuit,
            labelVectorCircuit,
        )
        circ = constructCircuit(isQuantumSimulation, qubits + 2, specHadamardTest)
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


def getSolutionVector(circ: QuantumCircuit, qubits: int, outF: list):
    fixedAnsatz(circ, qubits, outF)
    circ.save_statevector()  # this might be the problem

    backend = Aer.get_backend("aer_simulator")

    with contextlib.redirect_stdout(io.StringIO()):
        t_circ = transpile(circ, backend)
    job = backend.run(t_circ)

    result = job.result()
    return result.get_statevector(circ, decimals=10)


def hadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    fixedAnsatzCircuit: QuantumCircuit,
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    circ.append(fixedAnsatzCircuit, range(fixedAnsatzCircuit.num_qubits))

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    circ.h(auxiliaryIndex)


def specialHadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    controlledFixedAnsatzCircuit: QuantumCircuit,
    controlLabelVectorCircuit: QuantumCircuit,
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    circ.append(
        controlledFixedAnsatzCircuit, range(controlledFixedAnsatzCircuit.num_qubits)
    )

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    circ.append(controlLabelVectorCircuit, range(controlLabelVectorCircuit.num_qubits))

    circ.barrier()

    circ.h(auxiliaryIndex)


def constructCircuit(
    isQuantumSimulation: bool, totalNeededQubits: int, circuitLambda: callable
) -> QuantumCircuit:
    if isQuantumSimulation:
        circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
        circuitLambda(circ)
        circ.measure(0, 0)
    else:
        circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
        circuitLambda(circ)
        circ.save_statevector()

    return circ


def prepareParameterVector(name: str, qubits: int, layers: int):
    totalParamsNeeded: int = getTotalAnsatzParameters(qubits, layers)
    parameters: ParameterVector = ParameterVector(name, totalParamsNeeded)
    return parameters, splitParameters(parameters, qubits, alternating=qubits != 3)
