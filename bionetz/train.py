import tensorflow as tf
import glob
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


import logz

def train(ops, log_freq, save_freq, save_path, DNAse, iterations,
 train_iterator, valid_iterator, valid_size = 1000 // 64, num_logits = 919 - 126):
    with tf.Session() as sess:
        # If a model exists, restore it. Otherwise, initialize a new one
        if glob.glob(save_path + '*'):
            ops["saver"].restore(sess, save_path)
            print("Model restored.")
        else:
            sess.run(ops["init_op"])
            print("Model initialized.")
        
        # Calculate and print the number of parameters
        n_parameters = np.sum([np.prod(v.shape) 
                        for v in tf.trainable_variables()])
        print("Number of parameters: %i" % n_parameters)

        # The training iterator
        
        training_losses = []
        valid_losses = []

        
        for i in range(iterations):
            input_, dnase, target = next(train_iterator)

            # One step of training
            _loss, _ = sess.run([ops["loss"], ops["optimizer"]], feed_dict={
                ops["input_placeholder"]: input_,
                ops["dnase_placeholder"]: dnase,
                ops["target_placeholder"]: target,
            })
            training_losses.append(_loss)

            # Save the model
            if i % save_freq == 0:
                save_path = ops["saver"].save(sess, save_path)
                print("Model saved in file: %s" % save_path)

            # Validate and log results
            if i % log_freq == 0:
                all_logits = np.array([[0] * num_logits])
                all_targets = np.array([[0] * num_logits])

                # The validation loop
                for _ in range(valid_size):
                    b, d, t = next(valid_iterator)
                    _logits, _valid_loss = sess.run([ops["logits"], ops["loss"]],
                     feed_dict={
                        ops["input_placeholder"]: b,
                        ops["dnase_placeholder"]: d,
                        ops["target_placeholder"]: t})
                    valid_losses += [_valid_loss]
                    all_logits = np.append(all_logits, _logits, axis = 0)
                    all_targets = np.append(all_targets, t, axis = 0)

                # Log relevant statistics
                log(i, training_losses, valid_losses, all_logits, all_targets)
                training_losses = []
                valid_losses = []



def log(i, training_losses, valid_losses, valid_logits, valid_targets):
    aucs = []
    for j in np.arange(919 - 126):
        try:
            aucs += [roc_auc_score(valid_targets[:, j],valid_logits[:, j])]
        except ValueError:
            continue

    logz.log_tabular('Iteration', i)
    logz.log_tabular('Loss', np.mean(training_losses))
    logz.log_tabular('Valid Loss', np.mean(valid_losses))
    logz.log_tabular('Average AUPRC', np.mean(aucs))
    logz.log_tabular('80th percentile AUPRC', np.percentile(aucs, 80))
    logz.dump_tabular()