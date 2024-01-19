from typing import List
import numpy as np

# Quantum normalized vector after getSolutionVector can have negative or positive values,
# so we need to check combinations of signs, which one returns the minimum difference between b and bEstimated
# we do it by multiplying by -1 each element of the vector and checking the difference, if it is smaller, we keep the sign of i-th element
# we do it for all elements of the vector
def bestMatchingSignsVector(
    A: np.ndarray, xEstimated: np.array, b: np.array
) -> List[float]:
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
    bEstimated: List[float] = A.dot(xEstimated)  # calculate bEstimated
    return np.linalg.norm(bEstimated - b)


# estimate norm of vector
# once we got the sign combination, we can calculate the norm of the vector
# norm = b.T * b / b.T * A * v
# check this formula in the paper
def estimateNorm(
    A: np.ndarray, estimatedX: np.array, b: np.array, verbose: bool = False
) -> (float, List[float]):
    v: List[float] = bestMatchingSignsVector(A, estimatedX, b)
    # rightSide1: float = np.linalg.norm(b)
    # leftSide1: float = np.linalg.norm(A.dot(v))
    # estimatedNorm2: float = rightSide1 / leftSide1
     
    leftSide: float = b.T.dot(A.dot(v))
    rightSide: float = b.T.dot(b)  # maybe test this with \vec{1} vector
    estimatedNorm: float = rightSide / leftSide

    if verbose:
        print("Estimated X:", estimatedX)
        print("Best matching signs vector:", v)
        print("Estimated norm:", estimatedNorm)

    return estimatedNorm, v
