from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import PauliList
from qiskit.circuit import ParameterVector
from typing import List
import contextlib
import io
import threading

from Code.VQLS.Ansatz import fixedAnsatz, controlledFixedAnsatz
from Code.VQLS.LCU import convertMatrixIntoCircuit
from Code.VQLS.LabelVector import controlledLabelVectorCircuit
from Code.Utils import getTotalAnsatzParameters, splitParameters, prepareBackend


def prepareCircuits(
    paulis: PauliList,
    bVector: List[float],
    qubits: int,
    isQuantumSimulation: bool,
    layers: int,
    threads: int,
    jobs: int,
    threading = False,
) -> (list, ParameterVector, list, ParameterVector):
    backend = prepareBackend(threads, jobs)
    if threading:
        parameterVectorThread = ReturnValueThread(target=lambda: prepareParameterVector(
            "parametersHadarmard", qubits, layers
        ))

        parameterSpecialVectorThread = ReturnValueThread(target=lambda: prepareParameterVector(
            "parametersSpecialHadamard", qubits, layers))


        parameterVectorThread.start()
        parameterSpecialVectorThread.start()

        parametersHadamard, parametersHadamardSplit = parameterVectorThread.join()
        parametersSpecialHadamard, parametersSpecialHadamardSplit = parameterSpecialVectorThread.join()
    else:
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

    if threading:
        hadamardCircuitsThread = ReturnValueThread(target=lambda: prepareHadamardTestCircuits(
            paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend
        ))

        specialHadamardCircuitsThread = ReturnValueThread(target=lambda: prepareSpecialHadamardTestCircuits(
            paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend
        ))

        hadamardCircuitsThread.start()
        specialHadamardCircuitsThread.start()
        transpiledHadamardCircuits = hadamardCircuitsThread.join()
        transpiledSpecialHadamardCircuits = specialHadamardCircuitsThread.join()
    else:
        transpiledHadamardCircuits = prepareHadamardTestCircuits(paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend)
        transpiledSpecialHadamardCircuits = prepareSpecialHadamardTestCircuits(paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend)

    return (
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    )


def prepareHadamardTestCircuits(paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend):
    transpiledHadamardCircuits: List[QuantumCircuit] = []
    for i in range(len(paulis)):
        tempHadamardCircuits: List[QuantumCircuit] = []
        for j in range(i, len(paulis)):
            def hadamardTest1(circuit): return hadamardTest(
                circuit, [paulis[i], paulis[j]], fixedAnsatzCircuit
            )
            circ = constructCircuit(isQuantumSimulation, qubits + 1, hadamardTest1)
            tempHadamardCircuits.append(circ)
        with contextlib.redirect_stdout(io.StringIO()):
            hadamardCircuits = transpile(tempHadamardCircuits, backend=backend, optimization_level=1)
        transpiledHadamardCircuits.extend(hadamardCircuits)
    return transpiledHadamardCircuits


def prepareSpecialHadamardTestCircuits(paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend):
    specialHadamardCircuits: List[QuantumCircuit] = []
    for i in range(len(paulis)):
        def specHadamardTest(circuit): return specialHadamardTest(
            circuit,
            [paulis[i]],
            controlledFixedAnsatzCircuit,
            labelVectorCircuit,
        )
        circ = constructCircuit(isQuantumSimulation,
                                qubits + 2, specHadamardTest)
        specialHadamardCircuits.append(circ)

    with contextlib.redirect_stdout(io.StringIO()):
        transpiledSpecialHadamardCircuits = transpile(
            specialHadamardCircuits, backend=backend, optimization_level=1
        )
    return transpiledSpecialHadamardCircuits


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
        controlledFixedAnsatzCircuit, range(
            controlledFixedAnsatzCircuit.num_qubits)
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

    circ.append(controlLabelVectorCircuit, range(
        controlLabelVectorCircuit.num_qubits))

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


import threading
import sys

class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  # could alternatively raise an exception, depends on the use case
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result