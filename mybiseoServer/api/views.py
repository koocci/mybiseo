from django.shortcuts import render
from django.http import HttpResponse
from mybiseo.settings.base import TYPE_INSTANCE, INTENT_INSTANCE, PARAMS_INSTANCE
import jpype
import json
from django.utils.datastructures import MultiValueDictKeyError

# Create your views here.

def getSentence(request):
    # JVM Error Control
    jpype.attachThreadToJVM()
    try:
        sentence = request.GET['sentence']
        type = TYPE_INSTANCE.predict(sentence).split('/')[0]
        intent = INTENT_INSTANCE.predict(sentence, type)
        NER_sent , NER_tag = PARAMS_INSTANCE.predict(sentence)
        dateWords = []
        locateWords = []
        for idx, tag in enumerate(NER_tag):
            if tag == 'B-DTE' or tag == 'I-DTE':
                dateWords.append(NER_sent[idx])
            elif tag == 'B-LOC' or tag == 'I-LOC':
                locateWords.append(NER_sent[idx])

        result = {'code': 200,
                  'message': "성공",
                  'data':
                      {
                          'type': type,
                          'intent': intent,
                          'params':
                              {
                                  'date': ' '.join(dateWords),
                                  'location': ' '.join(locateWords)
                              }
                      }
                  }

        return HttpResponse(json.dumps(result), content_type="application/json")
    except MultiValueDictKeyError as e:
        res = {'code': 400, 'message': '요청한 문장이 없습니다.'}
        return HttpResponse(json.dumps(res), content_type="application/json")