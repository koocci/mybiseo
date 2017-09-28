import numpy as np
import math

# 거리행렬
def getDistMat(w2vModel, queries):
    '''

    :param w2vModel: w2v 모델
    :param queries: 구현 기능 List
    :return: 거리행렬, 선택된 기능 idx, 기능 List(배열 순서가 바뀜)
    '''
    distMat = []
    queriesIdx = []
    newQueries = []
    for idx, key in enumerate(w2vModel.wv.vocab):
        tmpArr = []
        if key in queries:
            queriesIdx.append(idx)
            newQueries.append(key)
        for key2 in w2vModel.wv.vocab:
            wordVector1 = w2vModel[key]
            wordVector2 = w2vModel[key2]
            # 유클리드 계산
            tmpArr.append(math.sqrt(sum((wordVector1 - wordVector2) ** 2)))
        distMat.append(tmpArr)
    return distMat, queriesIdx, newQueries


# 가중치 행렬
def getWeightMat(distMat, queriesIdx):
    '''

    :param distMat: 거리행렬
    :param queriesIdx: 쿼리 idx
    :return: 가중치 행렬
    '''
    # 분산
    var = np.var(distMat)

    weightMat = []
    tmpWeightMat = (-1 * (np.power(distMat, 2) / (2 * var)))
    for idx in queriesIdx:
        row = tmpWeightMat[idx]
        tmpArr = []
        for wordWeight in row:
            tmpArr.append(math.exp(wordWeight))
        weightMat.append(tmpArr)
    return weightMat


# 단어문서 행렬
def getTdmMat(w2vModel, sentences):
    tdmMat = np.zeros((len(sentences), len(w2vModel.wv.vocab)))
    for idx, sentence in enumerate(sentences):
        for idx2, key in enumerate(w2vModel.wv.vocab):
            if key in sentence:
                tdmMat[idx][idx2] = 1
            else:
                tdmMat[idx][idx2] = 0
    return tdmMat

# 단어문서 행렬
def getNewTdmMat(w2vModel, newSentence):
    '''

    :param w2vModel: w2v 모델
    :param newSentence: 새로운 문장
    :return: 단어문서 행렬
    '''
    tdmMat = np.zeros((len(w2vModel.wv.vocab)))
    for idx, key in enumerate(w2vModel.wv.vocab):
        if key in newSentence:
            tdmMat[idx] = 1
        else:
            tdmMat[idx] = 0
    return tdmMat

# 내적
def getDot(weightMat, tdmMat):
    return np.dot(weightMat, tdmMat)