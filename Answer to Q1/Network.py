import tensorflow as tf
import os
import math

class Network():
	def __init__(self, batchSize, bagSize, inputSize):
		self.sess = tf.Session()
		self.batchSize = batchSize
		self.bagSize = bagSize
		self.inputSize = inputSize
		self.trainLogDir = './log/train'
		self.valLogDir = './log/valid'
		self.ckptDir = './ckpt/'
		self.trainSaveImgDir = './saveImg/train/'
		self.valSaveImgDir = './saveImg/valid/'
		self.folderPrepare()
		self.build()

	def folderPrepare(self):
		if not os.path.exists(self.trainLogDir):
			os.makedirs(self.trainLogDir)
		if not os.path.exists(self.valLogDir):
			os.makedirs(self.valLogDir)
		if not os.path.exists(self.ckptDir):
			os.makedirs(self.ckptDir)
		if not os.path.exists(self.trainSaveImgDir):
			os.makedirs(self.trainSaveImgDir)
		if not os.path.exists(self.valSaveImgDir):
			os.makedirs(self.valSaveImgDir)
		_ = [os.remove(os.path.join(self.trainLogDir, file)) for file in os.listdir(self.trainLogDir)]
		_ = [os.remove(os.path.join(self.valLogDir, file)) for file in os.listdir(self.valLogDir)]
		_ = [os.remove(os.path.join(self.trainSaveImgDir, file)) for file in os.listdir(self.trainSaveImgDir)]
		_ = [os.remove(os.path.join(self.valSaveImgDir, file)) for file in os.listdir(self.valSaveImgDir)]

	def batchNorm(self, x, name):
		with tf.variable_scope(name):
			mean, var = tf.nn.moments(x, [0, 1, 2])
			scale = tf.get_variable('scale', [int(x.shape[-1])], tf.float32, tf.ones_initializer, trainable=True)
			offset = tf.get_variable('offset', [int(x.shape[-1])], tf.float32, tf.zeros_initializer, trainable=True)
			x = tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-7)
			return x

	def featureExtractor(self, x, featureNum, training=None, scope='featureExtractor'):
		with tf.variable_scope(scope):
			ch = 8
			maxCh = 512
			convNum = 4
			for i in range(convNum):
				x = tf.layers.conv2d(x, min(ch, maxCh), 3, padding='same', use_bias=False)
				x = self.batchNorm(x, 'BN_' + str(i))
				# x = tf.layers.batch_normalization(x, training=training)
				x = tf.nn.relu(x)
				if i < convNum - 1:
					x = tf.layers.max_pooling2d(x, 2, 2)
					ch *= 2
			x = tf.layers.average_pooling2d(x, int(x.shape[1]), int(x.shape[1]))
			x = tf.reshape(x, [int(x.shape[0]), int(x.shape[-1])])
			x = tf.layers.dense(x, 256, activation=tf.nn.relu)
			x = tf.layers.dense(x, featureNum, activation=tf.nn.sigmoid)  # 此处源码使用relu，运行失败了
		return x

	def distributionPoolingFilter(self, x, binsNum=6, sigma=0.05):
		alpha = 1 / math.sqrt(2 * math.pi * (sigma ** 2))
		beta = -1 / (2 * (sigma ** 2))
		batchSize, bagSize, featureNum = x.get_shape().as_list()
		samplePoints = tf.reshape(tf.range(0, binsNum, 1, tf.float32) / (binsNum - 1), [1, 1, 1, binsNum])
		samplePoints = tf.tile(samplePoints, [batchSize, bagSize, featureNum, 1])
		x = tf.tile(tf.reshape(x, [batchSize, bagSize, featureNum, 1]), [1, 1, 1, binsNum])
		diff = samplePoints - x
		diff2 = diff ** 2
		result = alpha * tf.exp(beta * diff2)
		outUnnormalized = tf.reduce_sum(result, 1)  # [batchSize, featureNum, binsNum]
		normCoeff = tf.stop_gradient(tf.reduce_sum(outUnnormalized, 2, keepdims=True))  # [batchSize, featureNum, 1]
		out = outUnnormalized / normCoeff
		return out

	def representationTransformation(self, x, scope='representationTransformation'):
		with tf.variable_scope(scope):
			x = tf.reshape(x, [int(x.shape[0]), -1])
			x = tf.layers.dense(x, 512, activation=tf.nn.relu)
			x = tf.layers.dense(x, 128, activation=tf.nn.relu)
			x = tf.layers.dense(x, 1, activation=None)
			return x

	def model(self, x, training, scope='model'):
		with tf.variable_scope(scope):
			batchSize = int(x.shape[0])
			bagSize = int(x.shape[1])
			featureNum = 128
			x = tf.reshape(x, [-1, int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1])])
			x = self.featureExtractor(x, featureNum, training)
			x = tf.reshape(x, [batchSize, bagSize, featureNum])
			x = self.distributionPoolingFilter(x)
			x = self.representationTransformation(x)
		return x

	def build(self):
		self.inputImg = tf.placeholder(tf.float32, [self.batchSize, self.bagSize, self.inputSize, self.inputSize, 1])
		self.inputLabel = tf.placeholder(tf.float32, [self.batchSize, 1])
		self.trainIng = tf.placeholder(tf.bool)
		self.logit = self.model(self.inputImg, self.trainIng)
		self.loss = tf.losses.absolute_difference(self.inputLabel, self.logit)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
			self.step = optimizer.minimize(self.loss)
		self.saver = tf.train.Saver(max_to_keep=2)
		self.trainWriter = tf.summary.FileWriter(self.trainLogDir, self.sess.graph)
		self.valWriter = tf.summary.FileWriter(self.valLogDir)

		init = tf.global_variables_initializer()
		self.sess.run(init)
