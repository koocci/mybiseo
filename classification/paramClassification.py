from classification.nerTagging.model.data_utils import CoNLLDataset
from classification.nerTagging.model.ner_model import NERModel
from classification.nerTagging.model.config import Config
from classification.nerTagging.evaluate import align_data

# 러닝 후, 데이터 확인
# https://github.com/guillaumegenthial/sequence_tagging
#
#
# dataset config 파일 바꿔서 확인
# preProcessor에 build랑 main 등록

# memo는 정규식
# 알람은 시간 정규식
# 날씨는 param 확인
# 음악도 데이터 만든 후 확인

# create instance of config
class paramClassification:
    def __init__(self):
        config = 0
        model = 0

    def loading(self):
        self.config = Config()
        self.model = NERModel(self.config)
        self.model.build()

        # # create dataset
        # test = CoNLLDataset(self.config.filename_test, self.config.processing_word,
        #                     self.config.processing_tag, self.config.max_iter)
        # # evaluate and interact
        # self.model.evaluate(test)
    def predict(self, sentence):
        self.model.restore_session(self.config.dir_model)
        words_raw = sentence.strip().split(" ")
        preds = self.model.predict(words_raw)

        return words_raw, preds