import numpy as np

def zGate() -> np.array:
    return np.array([[1, 0], [0, -1]])

def identityGate() -> np.array:
    return np.eye(2)

def letterToGate(letter:str) -> np.array:
    if letter == "Z":
        return zGate()
    elif letter == "I":
        return identityGate()
    else:
        return None

def createMatrixFromParameters(coefs:list, gates:list) -> np.array:
    matrix = np.zeros((2**len(gates[0]), 2**len(gates[0])), dtype=complex)
    for gate in gates:
        tempGate = None
        gate = gate[::-1]
        for letter in gate:
            if tempGate is None:
                tempGate = letterToGate(letter)
            else:
                tempGate = np.kron(tempGate, letterToGate(letter))
        matrix += coefs[gates.index(gate)] * tempGate

    return matrix
