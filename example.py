import json
import numpy as np
from assembler import assembled_pipeline
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSeq2SeqLM, BertTokenizer

data = ['我爱你。',
        '我恨你。',
        '我不爱你。',
        '我不恨你。']

translation_model_path = './opus-mt-zh-en'
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)

classification_model_path = './bert-base-personality'
classification_tokenizer = BertTokenizer.from_pretrained(classification_model_path)
classification_model = BertForSequenceClassification.from_pretrained(classification_model_path).cuda()

prob = assembled_pipeline(data,
                          translation_tokenizer,
                          translation_model,
                          classification_tokenizer,
                          classification_model, )
lexicon = {'Extroversion': 0, 'Neuroticism': 1, 'Agreeableness': 2, 'Conscientiousness': 3, 'Openness': 4}
print('prob:\n', prob)
print('lexicon:\n', lexicon)
np.savetxt('analysis_prob.csv', prob, delimiter=',', fmt='%f')
json_string = json.dumps(lexicon)
with open('lexicon.json', 'w') as file:
    file.write(json_string)
