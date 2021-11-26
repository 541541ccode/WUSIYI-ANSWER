from DataIter import DataIter
from Network import Network
from utils import saveImages
import tensorflow as tf
import os

def train():
	batchSize = 50
	bagSize = 100
	totalIteration = 5000
	dataIter = DataIter()
	network = Network(batchSize, bagSize, inputSize=28)
	for iter in range(totalIteration):
		batch, label = dataIter.getBatch_train(batchSize, bagSize)
		feedDict = {network.inputImg: batch, network.inputLabel: label, network.trainIng: True}
		_ = network.sess.run(network.step, feed_dict=feedDict)

		if iter % 20 == 0:
			loss = network.sess.run(network.loss, feed_dict=feedDict)
			trainSummary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
			network.trainWriter.add_summary(trainSummary, iter)
			print('%d/%d, loss, %3.3f' % (iter, totalIteration, loss))

		if iter % 100 == 0:
			logit, loss = network.sess.run([network.logit, network.loss], feed_dict=feedDict)
			savePath = network.trainSaveImgDir + 'image_%d_truth_%2.2f_logit_%2.2f.png' % (iter, label[0], logit[0])
			saveImages(batch[0], [10, 10], savePath)

		if iter % 100 == 0:
			batch, label = dataIter.getBatch_test(batchSize, bagSize)
			feedDict = {network.inputImg: batch, network.inputLabel: label, network.trainIng: True}
			logit, loss = network.sess.run([network.logit, network.loss], feed_dict=feedDict)
			validSummary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
			network.valWriter.add_summary(validSummary, iter)
			savePath = network.valSaveImgDir + 'image_%d_truth_%2.2f_logit_%2.2f.png' % (iter, label[0], logit[0])
			saveImages(batch[0], [10, 10], savePath)

		if iter % 1000 == 0:
			network.saver.save(network.sess, network.ckptDir + 'model', iter)

	network.saver.save(network.sess, network.ckptDir + 'model', totalIteration)

if __name__ == '__main__':
	train()