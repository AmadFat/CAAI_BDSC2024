import json
import numpy as np
from assembler import assembled_pipeline

data = ['我爱你。',
        '我恨你。',
        '我不爱你。',
        '我不恨你。']

prob, lexicon = assembled_pipeline(data)
print('prob:\\n', prob)
print('lexicon:\\n', lexicon)
np.savetxt('analysis_prob.csv', prob, delimiter=',', fmt='%f')
json_string = json.dumps(lexicon)
with open('lexicon.json', 'w') as file:
    file.write(json_string)
