import sklearn
import numpy as np
import kipoi

from load_data import make_data_iterator

# Running this converts the the data in path into two npy files, one with the DeepSea ground truth, and one
# with the outputs of Kipoi's DeepSea model. To get logits, I've been modifying the DeepSea source in .kipoi/
# to not apply sigmoid, and then running this file again.

path = '../../deepsea_train/train.mat'
batch_size = 100

gen = make_data_iterator(path, batch_size, 
	seperate_dnase=False, 
	num_repeat=1,
    tfrecords=False)

def kipoi_gen(gen):
	for data, target in gen:
		data = np.swapaxes(data, 1, 2)
		data = np.expand_dims(data, 2)
		data = data[:,[0,2,1,3]] #why tho
		yield data.astype(np.float32), target

gen = kipoi_gen(gen)

print("Loading model")

model = kipoi.get_model('DeepSEA/predict')

indices = [53, 128, 210, 216, 223, 240, 413, 420, 421, 423, 428, 430, 435, 436, 442, 444, 725, 727, 781]

all_targets = []
all_outputs = []

print("Running model")
for i, (data, target) in enumerate(gen):
	all_targets.append(target[:,indices])
	all_outputs.append(model.predict_on_batch(data)[:,indices])
	if i % 1000 == 0:
		print(i * batch_size)

print("Concating")
targets = np.concatenate(all_targets)
outputs = np.concatenate(all_outputs)

print("Saving")
np.save('train_targets.npy', targets)
np.save('train_outputs.npy', outputs)
