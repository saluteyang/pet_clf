import pandas as pd
import glob
import json

# test_sentiment = pd.DataFrame()
lst_ids = []
for f in glob.glob('data/test_sentiment/*.json'):
    lst_ids.append(f)

lst_ids = [x.replace('.', '/').split('/') for x in lst_ids]

clean_ids = []
for item in lst_ids:
    for subitem in item:
        if subitem not in ['data', 'test_sentiment', 'json']:
            clean_ids.append(subitem)

sent_dict = dict.fromkeys(clean_ids)

for f, i in enumerate(glob.glob('data/test_sentiment/*.json')):
    with open(f) as json_data:
        data = json.load(json_data)
        sent_dict[clean_ids[i]] = data['documentSentiment']

