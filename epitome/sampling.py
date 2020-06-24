import numpy as np

class MLSMOTE:
    """ Method modfied from https://simidat.ujaen.es/sites/default/files/biblio/2015-KBS-MLSMOTE.pdf
    Charte et al., Knowledge-Based Systems, 2015.
    
    We are not generating synthetic samples, just oversampling. However, using the mean imbalance scores
    works better than simply undersampling or oversampling.
    """
    
    def __init__(self, labels):
        self.labels = labels

        # set initial indices to all spots that have at least 2 positives
        self.indices = np.where(np.sum(self.labels, axis = 1) > 1)[0]
        
        # set mean ir and irs for individual labels
        self.lblsums = 0
        self.set_mean_lbl_sums()
        
    def fit_resample(self):
        """
        Oversamples sparse multiclass dataset with pseudo-labels y.
        In our case, each column in y corresponds to 1 cell type specific ChIP-seq experiment.

        Args:
            :param y: matrix of shape (samples, labels)

        """
        indices = self.indices

        mean_ir = self.get_mean_imbalance_ratio()
        
        # run the most common cases first to balance out the least balanced labels last
        for label_idx in (-self.lblsums).argsort():
            
            irlbl = self.get_imbalance_ratio_per_label(label_idx)
            
            if irlbl > mean_ir:
                # For all labels, check if imbalance ratio is greater than the mean
                # If it is, oversample for this label by choosing number of points proportional to the
                # imbalance. 
                min_bag_idx=self.get_all_instances_of_label(label_idx)

                # for each sample, we resample
                k = int(irlbl/mean_ir * 10) * min_bag_idx.shape[0] 
                r = np.random.choice(min_bag_idx, size = k)
                # randomly choose k label and append to indices
                indices = np.concatenate((indices, r))

        self.indices = indices
        return indices

    def create_new_sample(self, sample_ids, ref_neigh_idx, neighbour_idxs):
        raise NotImplementedError

    def get_all_instances_of_label(self, l):
        """ Get indices where label is true. 

        Args:
            :param l: index for label l

        Returns:
            indices in dataset where this label is true

        """
        return np.where((self.labels[:,l] == 1))[0]
    
    
    def set_mean_lbl_sums(self):
        """ Calculate the mean imbalance ratio for a given label.
        Mean Imbalance ratio = mean(argmax(number of positives of all labels)/number of positives for this label) for all labels

        Returns:
            mean imbalance ratio
        """
        self.lblsums = np.sum(self.labels[self.indices, :], axis = 0)
    
    def get_mean_imbalance_ratio(self):
        """ Return the mean imbalance ratio for a given label.
        Mean Imbalance ratio = mean(argmax(number of positives of all labels)/number of positives for this label) for all labels

        Returns:
            mean imbalance ratio
        """
        return np.mean(np.max(self.lblsums)/self.lblsums)

    def get_imbalance_ratio_per_label(self, l):
        """ Return the imbalance ratio for a given label.
        Imbalance ratio = argmax(number of positives of all labels)/number of positives for this label

        Args:
            :param l: column index for label l

        Returns:
            imbalance ratio = = argmax(number of positives of all labels)/number of positives for this label
        """

        return (np.max(self.lblsums)/self.lblsums)[l]
