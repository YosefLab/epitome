"""
Training utilities.
"""
import glob
import logz
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf


def train(ops,
          log_freq,
          save_freq,
          save_path,
          DNAse,
          iterations,
          train_iterator,
          valid_iterator,
          valid_size=1000//64,
          num_logits=816 - 126,
          ):
    """"Main training loop.

    Args:
        ops: Dictionary mapping keys to Tensors.
        log_freq: Int. How many gradient steps before logging summaries.
        save_freq: Int. How many gradient steps before saving a checkpoint.
        save_path: String. Path to save checkpoints and logs to.
        DNAse: Boolean. If this is true, use the DNAse labels as inputs to the 
            model. Otherwise, leave them in the training labels.
        iterations: Int. Number of iterations/gradient steps to train for.
        train_iterator: Iterator of the training data. See `load_data.py`.
        valid_iterator: Iterator of the validation data. See `load_data.py`.
        valid_size: Int. The number of examples in the validation set floor
            divided by the batch size.
        num_logits: Int. Number of dimensions in the output logits.
    """
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
                log(i, training_losses, valid_losses, all_logits, all_targets,
                    num_logits)
                training_losses = []
                valid_losses = []


def log(i,
        training_losses,
        valid_losses,
        valid_logits,
        valid_targets,
        num_logits
        ):
    """Logging a single gradient step to outfile.

    Args:
        i: Int. Current gradient step iteration.
        training_losses: Float. The training loss.
        validation_losses: Float. The validation loss.
        valid_logits: To comput ROC AUC.
        valid_targets: To comput ROC AUC.

    Returns:
        Nothing. Logs get dumped to outfile.
    """
    aucs = []
    for j in np.arange(num_logits):
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