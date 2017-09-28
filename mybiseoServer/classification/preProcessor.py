import classification.typeClassification as typeClassification
import classification.intentClassification as intentClassification
import classification.paramClassification as paramClassification
from konlpy.tag import Twitter

def paramTokenize(text):
    twitter = Twitter()

    return twitter.pos(text, norm=True)

def preProcessor():
    # locData()
    typeClassTest = typeClassification.typeClassification()
    typeClassTest.learning('classification/dict/allData.txt')
    intentClassTest = intentClassification.intentClassification()
    intentClassTest.learning()
    paramsClassTest = paramClassification.paramClassification()
    paramsClassTest.loading()
    return typeClassTest, intentClassTest, paramsClassTest

def locData():
    readFilePath = 'classification./dict/weatherData/locDic/address.txt'
    writeFilePath = 'classification./dict/weatherData/params/find/address.txt'
    addArr = []
    with open(readFilePath, 'r') as rf:
        for line in rf:
            lineArr = line.split('|')
            newAddr = lineArr[1]
            sido = lineArr[4]
            gu = lineArr[6]
            dong = lineArr[8]
            if sido[-3:] == '광역시' or  sido[-3:] == '특별시':
                newSidos = [sido[:-3], sido[:-3] + '시']
                for newSido in newSidos:
                    addArr.append((newSido).strip())
                    addArr.append((newSido + ' ' + gu).strip())
                    addArr.append((newSido + ' ' + gu + ' ' + dong).strip())
                    addArr.append((newSido + ' ' + dong).strip())
                    addArr.append((newSido + ' ' + newAddr).strip())
            addArr.append((sido).strip())
            addArr.append((sido + ' ' + gu).strip())
            addArr.append((sido + ' ' + gu + ' ' + dong).strip())
            addArr.append((sido + ' ' + dong).strip())
            addArr.append((gu + ' ' + dong).strip())
            addArr.append((sido + ' ' + newAddr).strip())
            addArr.append((gu).strip())
            addArr.append((dong).strip())
            addArr.append((newAddr).strip())
        addArr = list(set(addArr))
        with open(writeFilePath, 'w') as wf:
            wf.write('\n'.join(addArr))

def makeWeatherTag():
    dates = ['']
    addresses = ['']
    additionals = []

    # with open('./dict/weatherData/params/find/date.txt', 'r') as rdf:
    with open('classification./dict/weatherData/params/find/test_date.txt', 'r') as rdf:
        for line in rdf:
            tmpArr = []
            for idx, word in enumerate(line.split(' ')):
                word = word.replace('\n', '')
                if idx == 0:
                    tmpArr.append(word + ' B-DTE')
                else:
                    tmpArr.append(word + ' I-DTE')
            dates.append('\n'.join(tmpArr))
    # with open('./dict/weatherData/params/find/address.txt', 'r') as raf:
    with open('classification./dict/weatherData/params/find/test_address.txt', 'r') as raf:
        for line in raf:
            tmpArr = []
            for idx, word in enumerate(line.split(' ')):
                word = word.replace('\n', '')
                if idx == 0:
                    tmpArr.append(word + ' B-LOC')
                else:
                    tmpArr.append(word + ' I-LOC')
            addresses.append('\n'.join(tmpArr))

    with open('classification./dict/weatherData/params/find/weatherData.txt', 'r') as rf:
        for line in rf:
            words = paramTokenize(line)
            tmpArr = []
            for word in words:
                tmpArr.append(word[0] + ' O')
            additionals.append('\n'.join(tmpArr))
    with open('classification./dict/weatherData/params/find/taggingData.txt', 'w') as wf:
        for date in dates:
            if date != '':
                date = date + '\n'
            for address in addresses:
                if address != '':
                    address = address + '\n'
                for additional in additionals:
                    additional = additional + '\n'
                    taggingData = date + address + additional + '\n'
                    wf.write(taggingData)
                    taggingData = address + date + additional + '\n'
                    wf.write(taggingData)

