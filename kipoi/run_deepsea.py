import time

import kipoi
import sklearn.metrics
import numpy as np

from load_data import make_data_iterator

t = time.time()

path = '../../deepsea_train/test.mat'
batch_size = 100

iter = make_data_iterator(path, batch_size, 
	seperate_dnase=False, 
	num_repeat=1,
    tfrecords=False)

model = kipoi.get_model('DeepSEA/predict')

outputs = []
truth = []

for i, (data, target) in enumerate(iter):
	# if i < 100:
	# 	continue
	data = np.swapaxes(data, 1, 2)
	data = np.expand_dims(data, 2)
	data = data[:,[0,2,1,3]] #why tho
	outputs.append(model.predict_on_batch(data.astype(np.float32)))
	truth.append(target)

	if i != 0 and i % 10 == 0:
		print(i * batch_size / 10000.)

	if i == 100:
		break
		

outputs = np.concatenate(outputs)
truth = np.concatenate(truth)

aucs = []
for i in range(919):
	try:
		aucs.append(sklearn.metrics.roc_auc_score(truth[:,i], outputs[:,i]))
	except:
		pass

print(time.time() - t)
print(np.mean(aucs))