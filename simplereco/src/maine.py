from __future__ import division
from __future__ import print_function
import sys, argparse, cv2, editdistance, os, random, tensorflow as tf, random, numpy as np

def preprocess(image, imageSize, dataAugmentation=False):
	"place the image in target image having size imageSize, transpose for tensor flow TF and normalize gray-value"
	if image is None:
		image = np.zeros([imageSize[1], imageSize[0]])
	if dataAugmentation:
		stretch = (random.random() - 0.5)
		wStretched = max(int(image.shape[1] * (1 + stretch)), 1)
		image = cv2.resize(image, (wStretched, image.shape[0]))
	(width, height) = imageSize
	(h, w) = image.shape
	fx = w / width
	fy = h / height
	f = max(fx, fy)
	newSize = (max(min(width, int(w / f)), 1), max(min(height, int(h / f)), 1))
	image = cv2.resize(image, newSize)
	target = np.ones([height, width]) * 255
	target[0:newSize[1], 0:newSize[0]] = image
	#cv2.imshow("ste-",image)
	#cv2.waitKey(10000)
	image = cv2.transpose(target)
	(m, s) = cv2.meanStdDev(image)
	m = m[0][0]
	s = s[0][0]
	image = image - m
	image = image / s if s>0 else image
	print(len(image),len(image[0]))
	return image

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class Model:
	"minimalistic tensor flow model for handwritten text recognition"
	batchSize = 50
	imageSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		"init model: adding CNN, RNN and CTC and initializing TF"
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0
		self.is_train = tf.placeholder(tf.bool, name='is_train')
		self.inputimages = tf.placeholder(tf.float32, shape=(None, Model.imageSize[0], Model.imageSize[1]))
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
		(self.sess, self.saver) = self.setupTF()
	def setupCNN(self):
		"this creates CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=self.inputimages, axis=3)
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)
		pool = cnnIn4d
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
		self.cnnOut4d = pool

	def setupRNN(self):
		"this creates RNN layers and return output of these layers"
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

	def setupCTC(self):
		"creates and calculate CTC loss and decoder and return them"
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)
		sess=tf.Session()
		saver = tf.train.Saver(max_to_keep=1)
		modelDir = 'model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir)
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())
		return (sess,saver)

	def toSparse(self, texts):
		"putting ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0]
		for (batchElement, text) in enumerate(texts):
			labelStr = [self.charList.index(c) for c in text]
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)
		return (indices, values, shape)

	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		encodedLabelStrs = [[] for i in range(batchSize)]
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)
		else:
			decoded=ctcOutput[0][0]
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0]
				encodedLabelStrs[batchElement].append(label)
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.images)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputimages : batch.images, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal

	def dumpNNOutput(self, rnnOutput):
		"dump the output of the NN to CSV file(s)"
		dumpDir = '../dump/'
		if not os.path.isdir(dumpDir):
			os.mkdir(dumpDir)
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)
			with open(fn, 'w') as f:
				f.write(csv)

	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		numBatchElements = len(batch.images)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputimages : batch.images, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)
		if self.dump:
			self.dumpNNOutput(evalRes[1])
		return (texts, probs)

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)

class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath

class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, images):
		self.images = np.stack(images, axis=0)
		self.gtTexts = gtTexts

class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"
	def __init__(self, filePath, batchSize, imageSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"
		assert filePath[-1]=='/'
		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imageSize = imageSize
		self.samples = []
		f=open(filePath+'words.txt')
		chars = set()
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			if not line or line[0]=='#':
				continue
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue
			self.samples.append(Sample(gtText, fileName))
		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)
		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]
		self.numTrainSamplesPerEpoch = 25000
		self.trainSet()
		self.charList = sorted(list(chars))

	def truncateLabel(self, text, maxTextLen):
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples

	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)

	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		images = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imageSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, images)

def capture():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check)
            print(frame)
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='saved_image.jpg', image=frame)
                webcam.release()
                image_new = cv2.imread('saved_image.jpg', cv2.IMREAD_GRAYSCALE)
                image_new = image_new[60:420,0:640]
                print(len(image_new),len(image_new[0]))
                image_new = cv2.imshow("Captured Image", image_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                image_ = cv2.imread('saved_image.jpg', cv2.IMREAD_ANYCOLOR)
                print("Converting RGB image to grayscale...")
                gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
                print("Converted RGB image to grayscale...")
                print("Resizing image to 28x28 scale...")
                image_ = image_[60:420,0:640]
                cv2.imshow("saving",image_)
                image_ = cv2.resize(gray,(173,93))
                print("Resized...")
                image_ = image_[13:80,0:173]
                cv2.imshow("saving",image_)
                image_resized = cv2.imwrite(filename='tesst.png', image=image_)
                print("Image saved!")
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

class FilePaths:
	"filenames and paths to data"
	fnCharList = 'model/charList.txt'
	fnAccuracy = 'model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = 'static/tesst.png'
	fnCorpus = '../data/corpus.txt'

def train(model, loader):
	"train NN"
	epoch = 0
	bestCharErrorRate = float('inf')
	noImprovementSince = 0
	earlyStopping = 5
	while True:
		epoch += 1
		print('Epoch:', epoch)
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
		charErrorRate = validate(model, loader)
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break
def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		print('Ground truth -> Recognized')
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate

def infer(model, fnimage):
	"recognize text in image provided by file path"
	image = preprocess(cv2.imread(fnimage, cv2.IMREAD_GRAYSCALE), Model.imageSize)
	batch = Batch(None, [image])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])
	return ( (str(recognized[0]),str(probability[0])) )

def main():
	"main function"
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
	args = parser.parse_args()
	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch
	if args.train or args.validate:
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imageSize, Model.maxTextLen)
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)

	else:
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
		val = infer(model, FilePaths.fnInfer)
		return str(val)
if __name__ == '__main__':
	main()
