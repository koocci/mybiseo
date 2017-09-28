from konlpy.tag import Twitter
from gensim.models import Word2Vec
import classification.makeMatrix as makeMatrix

'''

    {
        weather: {
            intent: [find]
        },
        music: {
            intent: [play, find]
        },
        memo: {
            intent: [insert]
        },
        alarm: {
            intent: [send]
        }
    }


'''

class intentClassification:
    def __init__(self):
        self.types = ['날씨', '음악', '메모', '알람']
        self.intentObj = {}

    # 문장 intent 도출
    def intentTokenWords(self, sentence):
        twitter = Twitter()
        words = twitter.pos(sentence, norm=True, stem=True)
        intentTokenWords = []

        for idx, word in enumerate(words):
            intentWord = ''
            if word[1] == 'Verb':
                if idx > 0:
                    intentWord = words[idx-1][0] + ' ' + word[0]
                else:
                    intentWord = word[0]
                intentTokenWords.append(intentWord)

        if intentTokenWords is None:
            if words[len(words) - 2] is not None:
                intentTokenWords = words[words[len(words) - 2]] + words[words[len(words) - 1]]
            else:
                intentTokenWords = words[words[len(words) - 1]]
        return {
            'realWords': sentence,
            'intentTokenWords': intentTokenWords
        }

    # intent tokenize
    def intentTokenize(self, filePath):
        twitter = Twitter()
        intentWord = []
        with open(filePath, 'r') as rf:
            for line in rf:
                words = twitter.pos(line, norm=True, stem=True)
                tmpArr = []
                for word in words:
                    tmpArr.append(word[0])
                intentWord.append(' '.join(tmpArr))
            return intentWord

    def getTypeIntentFiles(self, type):
        weatherDataSet = ['classification//dict/weatherData/intent/find.txt']
        musicDataSet = ['classification//dict/musicData/intent/find.txt', 'classification/dict/musicData/intent/play.txt']
        memoDataSet = ['classification//dict/memoData/intent/insert.txt']
        alarmDataSet = ['classification//dict/alarmData/intent/send.txt']
        if type == '날씨':
            return weatherDataSet, ['find']
        elif type == '음악':
            return musicDataSet, ['find', 'play']
        elif type == '메모':
            return memoDataSet, ['insert']
        else: # 알람
            return alarmDataSet, ['send']

    def learning(self):
        # 모든 타입 학습

        # intent dic 부분
        for type in self.types:
            typeIntentFilePathes, sepQueries = self.getTypeIntentFiles(type)
            tmpArr = []
            for typeIntentFilePath in typeIntentFilePathes:
                tmpArr.append(self.intentTokenize(typeIntentFilePath))
            self.intentObj[type] = {}
            self.intentObj[type]['intentArr'] = tmpArr
            self.intentObj[type]['queries'] = sepQueries

            # 새롭게 할 시
            # self.intentObj[type]['w2vModel'] = Word2Vec(self.intentObj[type]['intentArr'], size=100, window=4, min_count=1, workers=4, iter=300, sg=1)
            # self.intentObj[type]['w2vModel'].save(type + '_w2vIntentModel')

            # 저장되어 있을 시
            self.intentObj[type]['w2vModel'] = Word2Vec.load('classification/' + type + '_w2vIntentModel')

            # 거리 행렬, query index
            distMat, queriesIdx, self.intentObj[type]['queries'] = makeMatrix.getDistMat(self.intentObj[type]['w2vModel'], self.intentObj[type]['queries'])

            # 가중치 행렬
            self.intentObj[type]['weightMat'] = makeMatrix.getWeightMat(distMat, queriesIdx)

            # 단어문서행렬 Term-Document Matrix)
            # tdmMat = makeMatrix.getTdmMat(self.intentObj[type]['w2vModel'], self.intentObj[type]['intentArr'])

            # # 내적
            # scores = makeMatrix.getDot(self.intentObj[type]['weightMat'], tdmMat.T)
            #
            # maxScoreMat = np.zeros(len(self.intentObj[type]['intentArr']))
            # maxScoreIdxMat = np.zeros(len(self.intentObj[type]['intentArr']))
            #
            # # print("type : ", type)
            # for idx, score in enumerate(scores):
            #     for idx2, key in enumerate(score):
            #         if maxScoreMat[idx2] < key:
            #             maxScoreMat[idx2] = key
            #             maxScoreIdxMat[idx2] = idx

                # print("maxScoreIdxMat : ", maxScoreIdxMat)
                # for idx, sentence in enumerate(self.intentObj[type]['intentArr']):
                #     print("query : ", self.intentObj[type]['queries'][int(maxScoreIdxMat[idx])])
                #     print("sentence : ", sentence)

        print('-' * 100)
        print('intent learning complete')
        print('-' * 100)

    def predict(self, sentence, type):
        # type을 보고 접근

        # dic 문장의 intent 부분
        sentIntentWords = self.intentTokenWords(sentence)['intentTokenWords']

        # 단어문서행렬
        tdmMat = makeMatrix.getNewTdmMat(self.intentObj[type]['w2vModel'], sentIntentWords)

        # 내적
        score = makeMatrix.getDot(self.intentObj[type]['weightMat'], tdmMat.T)

        maxScore = -1
        maxScoreIdx = -1

        for idx, key in enumerate(score):
            if maxScore < key:
                maxScore = key
                maxScoreIdx = idx

        return self.intentObj[type]['queries'][int(maxScoreIdx)]

    def testing(self, sentence, type):
        # type을 보고 접근

        # dic 문장의 intent 부분
        sentIntentWords = self.intentTokenWords(sentence)['intentTokenWords']

        # 단어문서행렬
        tdmMat = makeMatrix.getNewTdmMat(self.intentObj[type]['w2vModel'], sentIntentWords)

        # 내적
        score = makeMatrix.getDot(self.intentObj[type]['weightMat'], tdmMat.T)

        maxScore = -1
        maxScoreIdx = -1

        for idx, key in enumerate(score):
            if maxScore < key:
                maxScore = key
                maxScoreIdx = idx

        return self.intentObj[type]['queries'][int(maxScoreIdx)]
