import json
import numpy as np
import pandas as pd
from assembler import assembled_pipeline
data_path = './data/douban_movie_short_comment.csv'
data = pd.read_csv(data_path)['Comment'].tolist()
prob, lexicon = assembled_pipeline(data)
np.savetxt('analysis_prob.csv', prob, delimiter=',', fmt='%f')
json_string = json.dumps(lexicon)
with open('lexicon.json', 'w') as file:
    file.write(json_string)
