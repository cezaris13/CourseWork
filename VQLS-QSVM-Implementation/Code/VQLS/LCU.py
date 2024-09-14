from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList
from ThirdParty.TensorizedPauliDecomposition import PauliDecomposition
from typing import List
from numpy import ndarray


def convertMatrixIntoCircuit(
    circuit: QuantumCircuit,
    paulis: PauliList,
    controlled: bool = False,
    auxiliaryQubit: int = 0,
    showBarriers: bool = True,
):
    """
    Convert a matrix of Pauli operators into a quantum circuit.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to which the Pauli gates will be added.
    - paulis (PauliList): A list of Pauli operators to convert into gates.
    - controlled (bool, optional): Flag indicating whether to use controlled gates. Default is False.
    - auxiliaryQubit (int, optional): The index of the auxiliary qubit for controlled gates. Default is 0.
    - showBarriers (bool, optional): Flag indicating whether to add barriers between gate sequences. Default is True.

    Notes:
    - Controlled gates are added if `controlled` is True; otherwise, standard Pauli gates are added.

    Returns:
    - None: The function modifies the provided `QuantumCircuit` in place.

    Example:
    >>> circuit = QuantumCircuit(3)
    >>> paulis = PauliList([Pauli('X'), Pauli('Z')])
    >>> convertMatrixIntoCircuit(circuit, paulis, controlled=True, auxiliaryQubit=2)
    """

    # Prepare the list of qubit indices for gate application
    qubitIndexList: List[int] = []
    qubits: int = circuit.num_qubits
    for i in range(qubits):
        # If using controlled gates, exclude the auxiliary qubit from the list
        if controlled:
            if i != auxiliaryQubit:
                qubitIndexList.append(i)
        else:
            # Include all qubits if not using controlled gates
            qubitIndexList.append(i)

    # Determine the type of Pauli gate and apply it to the appropriate qubits
    for p in range(len(paulis)):
        for i in range(len(paulis[p])):
            currentGate = paulis[p][i]
            if currentGate.x and currentGate.z == False: # x gate
                if controlled: # Apply controlled-X gate if controlled is True else, regular x gate
                    circuit.cx(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.x(i)
            elif currentGate.x and currentGate.z: # y gate
                if controlled: # Apply controlled-Y gate if controlled is True else, regular y gate
                    circuit.cy(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.y(i)
            elif currentGate.z and currentGate.x == False: # z gate
                if controlled: # Apply controlled-Z gate if controlled is True else, regular z gate
                    circuit.cz(auxiliaryQubit, qubitIndexList[i])
                else:
                    circuit.z(i)
        if showBarriers:
            circuit.barrier()


def getLCU(inputMatrix: ndarray, method: str = "TPD") -> tuple[PauliList, List[float]]:
    '''
    Given input matrix and the method, get PauliList and coefficients whichs make up the matrix

    inputMatrix - the LSSVM matrix which to decompose to Pauli matrices.

    method - TPD, sparsePauliOp, method for Pauli decomposition.
    '''
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

def getMatrixCoeffitients(pauliOp: SparsePauliOp) -> List[float]:
    """
    Extracts coefficients from a SparsePauliOp object, considering the presence of identity operators.

    Parameters:
    - pauliOp (SparsePauliOp): A SparsePauliOp object containing Pauli operators and their associated coefficients.

    Returns:
    - List[float]: A list of coefficients. Each coefficient appears once for each Pauli operator.
      If a Pauli operator contains only identity operators (i.e., all components are identity),
      its coefficient is included twice in the result list.

    Notes:
    - The function checks each Pauli operator to determine if it contains only identity components.
      If so, the corresponding coefficient is appended twice to the result list.
    """
    coeffs: List[float] = []
    paulis: PauliList = pauliOp.paulis
    for p in range(len(paulis)):
        containsIdentity: bool = False
        for i in range(len(paulis[p])):
            currentGate = paulis[p][i]
            # Check if the Pauli operator contains only identity components
            if currentGate.x == False and currentGate.z == False:
                containsIdentity = True
        # Append the coefficient associated with the Pauli operator
        coeffs.append(pauliOp.coeffs[p])
        # If the Pauli operator contains only identity components, append the coefficient again
        if containsIdentity == False:
            coeffs.append(pauliOp.coeffs[p])
    return coeffs