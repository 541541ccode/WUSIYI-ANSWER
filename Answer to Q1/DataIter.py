import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataIter():
	def __init__(self):
		data = input_data.read_data_sets('./data_set', one_hot=False)
		trainData = data.train._images
		trainLabel = data.train._labels
		testData = data.test._images
		testLabel = data.test._labels
		self.trainLabelNew = []
		self.trainDataNew = []
		for i in range(10):
			self.trainDataNew.append(trainData[trainLabel == i])
			self.trainLabelNew.append(trainLabel[trainLabel == i])
		self.testLabelNew = []
		self.testDataNew = []
		for i in range(10):
			self.testDataNew.append(testData[testLabel == i])
			self.testLabelNew.append(testLabel[testLabel == i])

	def getBatch_train(self, batchSize, bagSize=100):
		batch = []
		label = []
		for i in range(batchSize):
			numOfOne = np.random.random_integers(0, bagSize, 1)
			index1 = np.random.choice(len(self.trainDataNew[1]), numOfOne, replace=False)
			index7 = np.random.choice(len(self.trainDataNew[7]), bagSize - numOfOne, replace=False)
			sample1 = self.trainDataNew[1][index1]
			sample7 = self.trainDataNew[7][index7]
			batch.append(np.concatenate([sample1, sample7], 0)[np.random.choice(bagSize, bagSize, False)])
			label.append(numOfOne / bagSize)
		batch = np.reshape(batch, [batchSize, bagSize, 28, 28, 1])
		label = np.reshape(label, [batchSize, 1])
		return batch, label

	def getBatch_test(self, batchSize, bagSize=100):
		batch = []
		label = []
		for i in range(batchSize):
			numOfOne = np.random.random_integers(0, bagSize, 1)
			index1 = np.random.choice(len(self.testDataNew[1]), numOfOne, replace=False)
			index7 = np.random.choice(len(self.testDataNew[7]), bagSize - numOfOne, replace=False)
			sample1 = self.testDataNew[1][index1]
			sample7 = self.testDataNew[7][index7]
			batch.append(np.concatenate([sample1, sample7], 0)[np.random.choice(bagSize, bagSize, False)])
			label.append(numOfOne / bagSize)
		batch = np.reshape(batch, [batchSize, bagSize, 28, 28, 1])
		label = np.reshape(label, [batchSize, 1])
		return batch, label