from django.shortcuts import render
from django.http import HttpResponse
from mybiseo.settings.base import TYPE_INSTANCE, INTENT_INSTANCE, PARAMS_INSTANCE
import jpype
import re
import json
from django.utils.datastructures import MultiValueDictKeyError
from konlpy.tag import Twitter
import datetime

# Create your views here.

def removeJosa(wordArr):
    twitter = Twitter()
    try:
        newWords = twitter.pos(wordArr[-1], norm=True)
        p = re.compile('(.*)이라고|(.*)라고|(.*)고')
        m = p.match(wordArr[-1])
        if m:
            wordArr[-1] = m.group(1) or m.group(2) or m.group(3)
            return wordArr
        else:
            newWord = []
            for word in newWords:
                if word[1] != "Josa":
                    newWord.append(word[0])
            wordArr[-1] = ''.join(newWord)
            return wordArr
    except IndexError:
        return ''

def timeUtil(words):
    hour = ''
    min = ''
    twelveRe = re.compile('오후')
    flag = False
    hourRe = re.compile('[0-9]{1,2}시')
    minRe = re.compile('[0-9]{1,2}분')
    wordArr = words.split(' ')
    for word in wordArr:
        if twelveRe.match(word) is not None:
            flag = True
        elif hourRe.match(word) is not None:
            hour = hourRe.match(word).group()[:-1]
        elif minRe.match(word) is not None:
            min = minRe.match(word).group()[:-1]

    if flag:
        hour = int(hour) + 12

    return hour, min

def dateUtil(words):
    now = datetime.datetime.now()
    joogan = False
    year = ''
    month = ''
    day = ''

    if words == '오늘':
        year = now.strftime('%Y')
        month = now.strftime('%m')
        day = now.strftime('%d')
    elif words == '내일':
        tomorrow = now + datetime.timedelta(days=1)
        year = tomorrow.strftime('%Y')
        month = tomorrow.strftime('%m')
        day = tomorrow.strftime('%d')
    elif words == '주간' or words == '이번주' or words == '이번 주':
        joogan = True
        year = now.strftime('%Y')
        month = now.strftime('%m')
        day = now.strftime('%d')
    elif words == '어제':
        yesterday = now - datetime.timedelta(days=1)
        year = yesterday.strftime('%Y')
        month = yesterday.strftime('%m')
        day = yesterday.strftime('%d')
    elif words == '모레':
        after2day = now + datetime.timedelta(days=2)
        year = after2day.strftime('%Y')
        month = after2day.strftime('%m')
        day = after2day.strftime('%d')
    elif words == '그저께':
        before2day = now - datetime.timedelta(days=2)
        year = before2day.strftime('%Y')
        month = before2day.strftime('%m')
        day = before2day.strftime('%d')
    elif words == '그끄저께':
        before3day = now - datetime.timedelta(days=3)
        year = before3day.strftime('%Y')
        month = before3day.strftime('%m')
        day = before3day.strftime('%d')
    elif words == '글피':
        after3day = now + datetime.timedelta(days=3)
        year = after3day.strftime('%Y')
        month = after3day.strftime('%m')
        day = after3day.strftime('%d')
    else:
        yearRe = re.compile('[0-9]{4}년')
        monthRe = re.compile('[0-9]{1,2}월')
        dayRe = re.compile('[0-9]{1,2}일')
        wordArr = words.split(' ')
        for word in wordArr:
            if yearRe.match(word) is not None:
                year = yearRe.match(word).group()[:-1]
            elif monthRe.match(word) is not None:
                month = monthRe.match(word).group()[:-1]
            elif dayRe.match(word) is not None:
                day = dayRe.match(word).group()[:-1]

    return joogan, year, month, day

def getSentence(request):
    # JVM Error Control
    jpype.attachThreadToJVM()
    try:
        sentence = request.GET['sentence']
        type = TYPE_INSTANCE.predict(sentence).split('/')[0]
        if type == '메세지' or type == '메시지':
            type = '메시지'
        if type == '재생':
            type = '음악'

        intent = INTENT_INSTANCE.predict(sentence, type)
        NER_sent , NER_tag = PARAMS_INSTANCE.predict(sentence)
        print(NER_sent)
        print(NER_tag)
        dateWords = []
        locateWords = []
        phoneWords = []
        msgWords = []
        timeWords = []
        singerWords = []
        albumWords = []
        songWords = []
        manWords = []

        for idx, tag in enumerate(NER_tag):
            if tag == 'B-DTE' or tag == 'I-DTE':
                dateWords.append(NER_sent[idx])
            elif tag == 'B-LOC' or tag == 'I-LOC':
                locateWords.append(NER_sent[idx])
            elif tag == 'B-PNM' or tag == 'I-PNM':
                phoneWords.append(NER_sent[idx])
            elif tag == 'B-MSG' or tag == 'I-MSG':
                msgWords.append(NER_sent[idx])
            elif tag == 'B-TME' or tag == 'I-TME':
                timeWords.append(NER_sent[idx])
            elif tag == 'B-SGR' or tag == 'I-SGR':
                singerWords.append(NER_sent[idx])
            elif tag == 'B-ALB' or tag == 'I-ALB':
                albumWords.append(NER_sent[idx])
            elif tag == 'B-SNG' or tag == 'I-SNG':
                songWords.append(NER_sent[idx])
            elif tag == 'B-MAN' or tag == 'I-MAN':
                manWords.append(NER_sent[idx])

        joogan, year, month, day = dateUtil(' '.join(removeJosa(dateWords)) or "")
        hour, min = timeUtil(' '.join(removeJosa(timeWords)) or "")
        phoneNum = ''.join(removeJosa(phoneWords)) or ""
        man = ' '.join(removeJosa(manWords)) or ""
        phoneFlag = False
        if phoneNum != '':
            phoneFlag = True

        timeFlag = False
        if hour != '' or min != '':
            timeFlag = True

        memo = ''
        if type == '메모':
            p = re.compile('(.*)이라고|(.*)라고|(.*)고')
            m = p.match(sentence)
            if m:
                memo = m.group(1) or m.group(2) or m.group(3)
            else:
                memo = sentence

        result = {'code': 200,
                  'message': "성공",
                  'data':
                      {
                          'type': type,
                          'intent': intent,
                          'params':
                              {
                                  'phoneFlag': phoneFlag,
                                  'timeFlag': timeFlag,
                                  'joogan': joogan,
                                  'year': year,
                                  'month': month,
                                  'day': day,
                                  'hour': hour,
                                  'min': min,
                                  'location': ' '.join(removeJosa(locateWords)) or "",
                                  'phoneNum': phoneNum,
                                  'message': ' '.join(removeJosa(msgWords)) or "",
                                  'singer': ' '.join(removeJosa(singerWords)) or "",
                                  'album': ' '.join(removeJosa(albumWords)) or "",
                                  'song': ' '.join(removeJosa(songWords)) or "",
                                  'man': man,
                                  'memo': memo
                              }
                      }
                  }

        return HttpResponse(json.dumps(result), content_type="application/json")
    except MultiValueDictKeyError as e:
        res = {'code': 400, 'message': '요청한 문장이 없습니다.'}
        return HttpResponse(json.dumps(res), content_type="application/json")