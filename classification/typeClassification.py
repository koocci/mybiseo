import numpy as np
from konlpy.tag import Twitter
from gensim.models import Word2Vec
import classification.makeMatrix as makeMatrix

class typeClassification:
    def __init__(self):
        self.queries = ['날씨/Noun', '음악/Noun', '재생/Noun', '메모/Noun', '메시지/Noun', '메세지/Noun']
        self.weightMat = 0
        self.sentences = []
        self.w2vModel = 0

    # 문장 토큰화 처리
    def tokenize(self, sentence):
        twitter = Twitter()
        words = twitter.pos(sentence, norm=True, stem=True)
        tokenizedWords = []

        for word in words:
            tokenizedWords.append(word[0] + '/' + word[1])
        return {
            'realWords': sentence,
            'tokenizedWords': tokenizedWords
        }

    # 파일 읽기
    def readDataFiles(self, filePath):
        with open(filePath, 'r') as rf:
            sentences = []
            for line in rf:
                sentences.append(self.tokenize(line)['tokenizedWords'])
            return sentences

    # 원핫인코딩
    # #def makeOnehot(sentences):
    # def getAllWords(sentences):
    #     allWords = []
    #     # onehot = []
    #     for sentence in sentences:
    #         allWords = list(set(allWords).union(set(sentence)))
    #
    #     # wordDict = {w: i for i, w in enumerate(allWords)}
    #     # wordIndex = [wordDict[word] for word in allWords]
    #
    #     return allWords #, wordDict, wordIndex

    # SkipGram 만들기
    # def makeSkipGram(wordIndex):
    #     skipGrams = []
    #
    #     for i in range(1, len(wordIndex) - 1):
    #         target = wordIndex[i]
    #         context = [wordIndex[i - 1], wordIndex[i + 1]]
    #
    #         for w in context:
    #             skipGrams.append([target, w])
    #     return skipGrams
    #
    # # skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
    # def randomBatch(data, size):
    #     randomInputs = []
    #     randomLabels = []
    #     randomIndex = np.random.choice(range(len(data)), size, replace=False)
    #
    #     for i in randomIndex:
    #         randomInputs.append(data[i][0])  # target
    #         randomLabels.append([data[i][1]])  # context word
    #
    #     return randomInputs, randomLabels

    # def getTdmMat(sentences, queries):
    #
    #     tdmMat = np.zeros((len(queries), len(sentences)))
    #     # 0이 나오지 않게하기 위해
    #     tdmMat += 0.01
    #
    #     for idx, query in enumerate(queries):
    #         for idx2, sentence in enumerate(sentences):
    #             if query in sentence:
    #                 tdmMat[idx][idx2] += 1
    #     return tdmMat

    def learning(self, filePath):

        # 새롭게 할 시
        self.sentences = self.readDataFiles(filePath)
        self.w2vModel = Word2Vec(self.sentences, size=100, window=4, min_count=1, workers=4, iter=300, sg=1)
        self.w2vModel.save('w2vTypeModel')

        # 저장되어 있을 시
        # self.w2vModel = Word2Vec.load('./w2vTypeModel')

        # 거리 행렬, query index
        distMat, queriesIdx, self.queries = makeMatrix.getDistMat(self.w2vModel, self.queries)
        # 가중치 행렬
        self.weightMat = makeMatrix.getWeightMat(distMat, queriesIdx)



        # 단어문서행렬 Term-Document Matrix)
        # tdmMat = makeMatrix.getTdmMat(self.w2vModel, self.sentences)
        # tdmMat = getTdmMat(sentences, queries)

        # # 내적
        # scores = makeMatrix.getDot(self.weightMat, tdmMat.T)
        #
        # maxScoreMat = np.zeros(len(self.sentences))
        # maxScoreIdxMat = np.zeros(len(self.sentences))
        #
        # for idx, score in enumerate(scores):
        #     for idx2, key in enumerate(score):
        #         if maxScoreMat[idx2] < key:
        #             maxScoreMat[idx2] = key
        #             maxScoreIdxMat[idx2] = idx

        # print(maxScoreIdxMat)
        # for idx, sentence in enumerate(self.sentences):
        #     print(self.queries[int(maxScoreIdxMat[idx])])
        #     print(sentence)

        print('-' * 100)
        print('type learning complete')
        print('-'*100)

    def predict(self, newSentence):
        '''

        :param newSentence: 입력된 문장
        :return: TYPE
        '''
        tokenizedWords = self.tokenize(newSentence)['tokenizedWords']

        # 단어문서행렬
        tdmMat = makeMatrix.getNewTdmMat(self.w2vModel, tokenizedWords)

        # 내적
        score = makeMatrix.getDot(self.weightMat, tdmMat.T)

        maxScore = -1
        maxScoreIdx = -1

        for idx, key in enumerate(score):
            if maxScore < key:
                maxScore = key
                maxScoreIdx = idx

        return self.queries[int(maxScoreIdx)]

    def testing(self, newSentence):
        '''

        :param newSentence: 입력된 문장
        :return: TYPE
        '''
        tokenizedWords = self.tokenize(newSentence)['tokenizedWords']

        # 단어문서행렬
        tdmMat = makeMatrix.getNewTdmMat(self.w2vModel, tokenizedWords)

        # 내적
        score = makeMatrix.getDot(self.weightMat, tdmMat.T)

        maxScore = -1
        maxScoreIdx = -1

        for idx, key in enumerate(score):
            if maxScore < key:
                maxScore = key
                maxScoreIdx = idx

        return self.queries[int(maxScoreIdx)]


        # allWords = makeOnehot(readDataFiles(filePath))
        #
        # # 학습을 반복할 횟수
        # trainingEpoch = 300
        # # 학습률
        # learningRate = 0.1
        # # 한 번에 학습할 데이터의 크기
        # batchSize = 20
        # # 단어 벡터를 구성할 임베딩 차원의 크기
        # embeddingSize = 100
        # # word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
        # # batch_size 보다 작아야 합니다.
        # numSampled = 15
        # # 총 단어 갯수
        # wordCnt = len(allWords)
        #
        # inputs = tf.placeholder(tf.int32, shape=[batchSize])
        # # tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야합니다.
        # labels = tf.placeholder(tf.int32, shape=[batchSize, 1])
        #
        # # 총 단어 갯수와 임베딩 갯수
        # embeddings = tf.Variable(tf.random_uniform([wordCnt, embeddingSize], -1.0, 1.0))
        # selectedEmbed = tf.nn.embedding_lookup(embeddings, inputs)
        #
        #
        # nceWeights = tf.Variable(tf.random_uniform([wordCnt, embeddingSize], -1.0, 1.0))
        # nceBiases = tf.Variable(tf.zeros([wordCnt]))
        #
        #
        # loss = tf.reduce_mean(
        #     tf.nn.nce_loss(nceWeights, nceBiases, labels, selectedEmbed, numSampled, wordCnt))
        #
        # train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)
        #
        # with tf.Session() as sess:
        #     init = tf.global_variables_initializer()
        #     sess.run(init)
        #
        #     for step in range(1, trainingEpoch + 1):
        #         batchInputs, batchLabels = randomBatch(makeSkipGram(wordIndex), batchSize)
        #
        #         _, loss_val = sess.run([train_op, loss],
        #                                feed_dict={inputs: batchInputs,
        #                                           labels: batchLabels})
        #
        #         if step % 10 == 0:
        #             print("loss at step ", step, ": ", loss_val)
        #
        #     trainedEmbeddings = embeddings.eval()
        #
        # for i, label in enumerate(allWords):
        #     x, y= trainedEmbeddings[i]
        #     print(x, y)
