from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList
from ThirdParty.TensorizedPauliDecomposition import PauliDecomposition
from typing import List


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


def getLCU(inputMatrix, method: str = "TPD") -> (PauliList, List[float]):
    if method == "TPD":
        paulis, coefficientSet = PauliDecomposition(inputMatrix, sparse=True)
        pauliOp = SparsePauliOp(paulis, coefficientSet)
    elif method == "sparsePauliOp":
        pauliOp: SparsePauliOp = SparsePauliOp.from_operator(inputMatrix)
    else:
        raise ValueError("Method not implemented")
    paulis: PauliList = pauliOp.paulis
    coefficientSet: List[float] = getMatrixCoeffitients(pauliOp)
    return paulis, coefficientSet