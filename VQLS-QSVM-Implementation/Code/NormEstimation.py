from typing import List
import numpy as np
from itertools import product

# Quantum normalized vector after ansatztest can have negative or positive values,
# so we need to check all combinations of signs, which one returns the minimum difference between b and bEstimated
# minimum difference between b and bEstimated is the sign combination we are looking for
def bestMatchingSignsVector(
    A: np.ndarray, xEstimated: np.array, b: np.array
) -> List[float]:
    values: List[int] = [-1, 1]
    combos: List[float] = list(
        product(values, repeat=len(xEstimated)) # this has to be rethought, since 2^2^5 is too much
    )  # generates all 8 bit combinations
    minDifference: float = 10000000
    minDifferenceValue: List[float] = []
    for combo in combos:
        vc: List[float] = np.multiply(
            xEstimated, list(combo)
        )  # multiply each element of vector with the corresponding element of combo
        bEstimated: List[float] = A.dot(vc)  # calculate bEst
        difference: float = np.linalg.norm(
            bEstimated - b
        )  # calculate difference between b and bEstimated
        if difference < minDifference:
            minDifference = difference
            minDifferenceValue = vc
    return minDifferenceValue


# estimate norm of vector
# once we got the sign combination, we can calculate the norm of the vector
# norm = b.T * b / b.T * A * v
# check this formula in the paper
def estimateNorm(
    A: np.ndarray, estimatedX: np.array, b: np.array, verbose: bool = False
) -> (float, List[float]):
    v: List[float] = bestMatchingSignsVector(A, estimatedX, b)
    leftSide: float = b.T.dot(A.dot(v))
    rightSide: float = b.T.dot(b)  # maybe test this with \vec{1} vector
    estimatedNorm: float = rightSide / leftSide

    if verbose:
        print("Estimated X:", estimatedX)
        print("Best matching signs vector:", v)
        print("Estimated norm:", estimatedNorm)

    return estimatedNorm, v
