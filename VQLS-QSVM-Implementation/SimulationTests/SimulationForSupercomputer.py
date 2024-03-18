import numpy as np
import csv
import sys
import time
sys.path.append("..")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from Code.VQLSSVM import VQLSSVM
from Code.Utils import prepareDataset

np.set_printoptions(precision=10, suppress=True)


shots: int = 10000
gamma: float = 0.01 # regularization parameter
classToFilterOut: int = 2

# datasets = ["iris","breastCancer"]


vqlssvmVectors: VQLSSVM = VQLSSVM(gamma, shots)

def getListsAverage(data: list):
    maximumLengthList = len(max(data, key=len))
    sumList =[]
    for i in range(maximumLengthList):
        iterationSum = 0
        for j in range(len(data)):
            if i < len(data[j]):
                iterationSum += data[j][i]
            else:
                iterationSum += data[j][-1]
        sumList.append(iterationSum)

    sumList = np.array(sumList)
    return sumList/len(data)

def collectTrainData(dataset: str, isQuantumSimulation: bool):
    xTrain, xTest, yTrain, yTest = prepareDataset(normalizeValues=True, dataset=dataset,subsetSize=subsetSize, classToFilterOut=classToFilterOut)
    vqlssvmVectors.train(xTrain, yTrain, quantumSimulation=isQuantumSimulation, verbose=False, iterations = trainIterations, method="COBYLA")
    cost = vqlssvmVectors.getCostHistory()
    accuracy = vqlssvmVectors.accuracy(xTest, yTest)
    clf = SVC(kernel='linear')
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    return cost, accuracy, accuracy_score(yTest, yPred)

def main(qubits: int, iterations: int, trainIterations: int, datasets: list, subsetSize: int):
    costs = {}
    accuracies = {}
    accuraciesSVM = {}

    for dataset in datasets:
        costs[dataset] = []
        accuracies[dataset] = []
        accuraciesSVM[dataset] = []

    for i in range(iterations):
        print(i,"th iteration")
        for dataset in datasets:
            print("Dataset:",dataset)
            cost,accuracy,svmAccuracy = collectTrainData(dataset, False)
            costs[dataset].append(cost)
            accuracies[dataset].append(accuracy)
            accuraciesSVM[dataset].append(svmAccuracy)
            print("\n")
        print("\n")

    for dataset in datasets:
        with open('../SimulationResults/costDataset' + dataset+'.csv', 'w', newline='') as csvfile:
            avgCosts = {}
            length = costs[dataset].__len__()
            s = np.array([sum(a) for a in zip(*costs[dataset])])
            avgCosts[dataset] = getListsAverage(costs[dataset])
            
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Iteration'] + [dataset])
            combined = [ [i] + [x] for i, x in enumerate(avgCosts[dataset])]
            writer.writerows(combined)

    with open('../SimulationResults/accuracyDataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        titleRow = []
        dataRow = []
        for dataset in datasets:
            titleRow.append(dataset)
            titleRow.append(dataset + "SVM")
            dataRow.append(np.mean(accuracies[dataset]))
            dataRow.append(np.mean(accuraciesSVM[dataset]))
        writer.writerow(titleRow)
        writer.writerow(dataRow)

if __name__ == '__main__':
    qubits = int(sys.argv[1])
    iterations = int(sys.argv[2])
    trainIterations = int(sys.argv[3])
    datasets = [sys.argv[4]]
    subsetSize: int = 2**qubits - 1 # number of training points

    #calculate start and end time
    start = time.time()
    main(qubits, iterations, trainIterations, datasets, subsetSize)
    end = time.time()
    print("Time for",qubits , " qubits :",end - start)