from DataIter import DataIter
from Network import Network
from utils import saveImages
import tensorflow as tf
import os

def test():
	testDir = './testResult/'
	if not os.path.exists(testDir):
		os.makedirs(testDir)
	_ = [os.remove(os.path.join(testDir, file)) for file in os.listdir(testDir)]

	batchSize = 50
	bagSize = 100
	totalIteration = 10
	dataIter = DataIter()
	network = Network(batchSize, bagSize, inputSize=28)
	ckpt = tf.train.get_checkpoint_state(network.ckptDir)
	network.saver.restore(network.sess, ckpt.model_checkpoint_path)
	for iter in range(totalIteration):
		batch, label = dataIter.getBatch_test(batchSize, bagSize)
		feedDict = {network.inputImg: batch, network.inputLabel: label, network.trainIng: True}
		logit, loss = network.sess.run([network.logit, network.loss], feed_dict=feedDict)
		savePath = testDir + 'image_%d_truth_%2.2f_logit_%2.2f.png' % (iter, label[0], logit[0])
		saveImages(batch[0], [10, 10], savePath)


if __name__ == '__main__':
	test()