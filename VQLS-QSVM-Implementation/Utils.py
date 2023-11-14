import numpy as np

def zGate() -> np.array:
    return np.array([[1, 0], [0, -1]])

def identityGate() -> np.array:
    return np.eye(2)

def createMatrixFromParameters(coefs:list, gates:list) -> np.array:
    matrix = np.zeros((len(gates[0]), len(gates[0])), dtype=complex)
    for gate in gates:
        combDate = 
        matrix += coefs[gates.index(gate)] * gate
