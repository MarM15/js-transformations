
#!/usr/bin/python

import pickle
import gib_train_py3

model_data = pickle.load(open('./src/Gibberish-Detector/gib_model.pki', 'rb'))
model_mat = model_data['mat']
threshold = model_data['thresh']

def detect(l):

    return(gib_train_py3.avg_transition_prob(l, model_mat) > threshold)
