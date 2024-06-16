import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from assembler import assembled_pipeline
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSeq2SeqLM, BertTokenizer

data_path = './data/douban_movie_short_comment.csv'
data = pd.read_csv(data_path)['Comment'].tolist()
chunk_size, prob = 32, None

translation_model_path = './opus-mt-zh-en'
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)

classification_model_path = './bert-base-personality'
classification_tokenizer = BertTokenizer.from_pretrained(classification_model_path)
classification_model = BertForSequenceClassification.from_pretrained(classification_model_path).cuda()

for i in tqdm(range(len(data) // chunk_size)):
    result = assembled_pipeline(data[i*chunk_size: i*chunk_size+chunk_size],
                                translation_tokenizer,
                                translation_model,
                                classification_tokenizer,
                                classification_model,)
    prob = np.vstack((prob, result)) if prob is not None else result

lexicon = {'Extroversion': 0, 'Neuroticism': 1, 'Agreeableness': 2, 'Conscientiousness': 3, 'Openness': 4}
np.savetxt('analysis_prob.csv', prob, delimiter=',', fmt='%f')
json_string = json.dumps(lexicon)
with open('lexicon.json', 'w') as file:
    file.write(json_string)
