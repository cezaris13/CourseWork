from typing import List
import numpy as np


def bestMatchingSignsVector(
    A: np.ndarray, xEstimated: np.array, b: np.array
) -> List[float]:
    '''
    Quantum normalized vector after getSolutionVector can have negative or positive values,
    so we need to check combinations of signs, which one returns the minimum difference between b and bEstimated
    we do it by multiplying by -1 each element of the vector and checking the difference, if it is smaller, we keep the sign of i-th element
    we do it for all elements of the vector

    A * x = b
    '''
    xEstimated = np.array(xEstimated)
    minDifference: float = calculateDifference(A, xEstimated, b)
    for i in range(len(xEstimated)):
        xEstimated[i] = -1 * xEstimated[i]
        difference = calculateDifference(A, xEstimated, b)
        if minDifference < difference:
            xEstimated[i] = -1 * xEstimated[i]
        else:
            minDifference = difference
    return xEstimated


def calculateDifference(A: np.ndarray, xEstimated: np.array, b: np.array) -> float:
    '''
    Calculates norm of the bEstimated and b vector difference
    '''
    bEstimated: List[float] = A.dot(xEstimated)
    return np.linalg.norm(bEstimated - b)


def estimateNorm(
    A: np.ndarray, estimatedX: np.array, b: np.array, verbose: bool = False
) -> (float, List[float]):
    '''
    Estimate norm of vector
    once we got the sign combination, we can calculate the norm of the vector
    norm = b.T * b / b.T * A * v
    '''
    v: List[float] = bestMatchingSignsVector(A, estimatedX, b)

    leftSide: float = b.T.dot(A.dot(v))
    rightSide: float = b.T.dot(b)
    estimatedNorm: float = rightSide / leftSide

    if verbose:
        print("Estimated X:", estimatedX)
        print("Best matching signs vector:", v)
        print("Estimated norm:", estimatedNorm)

    return estimatedNorm, v
