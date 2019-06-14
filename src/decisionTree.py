from import_data import *
from print_tree import *
import matplotlib.pyplot as plt

class ID3_decision_tree(object):
    def __init__(self):
        pass
    
    def chooseBestFeature(self, X, Y):
        """
        Choose the best feature as the root for the tree or subtree

        X: A numpy array of shape (train_batch_size, D), each 
        data has a dimension of D.
        Y: A numpy array of shape (train_batch_size,) containing the training
        labels, Y[i] is the corresponding label of X[i]

        return: The dimension index that has the max entropy divergence
        """
        sys_entropy = self.__ShannonEnt(Y)
        train_dim = X.shape[1] 
        label_entropy = np.zeros((train_dim,), dtype=np.float32)
        label_entropy[:] = sys_entropy

        for i in range(train_dim):
            distinct_labels = np.unique(X[:, i])
            for label in distinct_labels:
                label_entropy[i] -= self.__ShannonEnt(Y[X[:, i] == label])
        
        return np.argmax(label_entropy)

    def __ShannonEnt(self, labels):
        """
        Calculate the Shannon Entropy for the input labels

        Input:
        labels: A numpy array of shape (train_batch_size, ), each 
        data has a dimension of D.
        """
        # Obtain the distinct labels that occur in the batch
        distinct_labels = np.unique(labels)
        train_num = labels.shape[0]
        entropy = 0
        for label in distinct_labels:
            p = float(labels[labels == label].shape[0]) / train_num
            logp = np.log2(p)
            entropy -= p * logp
        
        return entropy

    def __majority(self, Y):
        label_cnt = np.zeros(Y.shape[0], dtype=mp.int32)
        label_distinct = np.unique(Y)
        for i, label in enumerate(label_distinct):
            label_cnt[i] = Y[Y[:] == label].shape[0]
        return np.argmax(label_cnt)

    def train(self, X, Y, features=np.arange(6, dtype=np.int32)):
        # if there is only one type of label left, return this label
        if np.unique(Y).shape[0] == 1:
            return Y[0]
        
        # if there is no feature left to build the subtree, return a label by voting
        if X.shape[1] == 0:
            return self.__majority(Y)

        best_feature_index = self.chooseBestFeature(X, Y)
        best_label = features[best_feature_index]
        tree = {best_label: {}}

        # obtain the different values that occur in the best feature
        best_feature_values = np.unique(X[:, best_feature_index])
        
        for value in best_feature_values:
            subX = X[X[:, best_feature_index] == value, :]
            subY = Y[X[:, best_feature_index] == value]
            subfeatures = np.delete(features, best_feature_index)
            subX = np.delete(subX, best_feature_index, axis=1)
            tree[best_label][value] = self.train(subX, subY, subfeatures)
        
        return tree



    def predict(self, X_test):
        pass

    def measure(self, y_pred, testlabels):
        """
        Calculate and return accuracy, Macro F1 and Micro F1 according to 
        y_pred and testlabels

        Inputs:
        y_pred: A numpy array of shape (test_batch_size, ) which is the prediction result
        and is turned into onehot vectors in this function.
        testlabels: A numpy array of shape (test_batch_size, ) which is the ground truth labels
        and is turned into onehot vectors in this function
        """
        mask = np.zeros(18, dtype=np.int32)
        
        # Tranform the prediction and ground truth labels to one hot vectors
        y_pred = np.eye(18, dtype=np.int32)[y_pred]
        testlabels = np.eye(18, dtype=np.int32)[testlabels]
        
        # Calculate TP 
        TP = y_pred * testlabels
        
        # Sum the result(label based)
        cnt_pred = np.sum(y_pred, axis=0)
        cnt_GT = np.sum(testlabels, axis=0)
        cnt_TP = np.sum(TP, axis=0)

        # Generate mask to only consider the labels exist in the current fold
        mask[:] = cnt_GT
        mask[mask > 0] = 1
        cnt_pred = cnt_pred * mask

        precision = cnt_TP / (cnt_pred + 0.000001)
        recall = cnt_TP / (cnt_GT + 0.000001)

        print(mask)
        # Calculate accuracy
        num_correct = np.sum(cnt_TP)
        accuracy = float(num_correct) / y_pred.shape[0]
        print('Got %d / %d correct => accuracy: %f' % (num_correct, testlabels.shape[0], accuracy))
        
        # Calculate Macro F1
        f1 = 2 * precision * recall / (precision + recall + 0.000001)
        # Only consider the labels that exists in current batch
        Macro_F1 = np.mean(f1[np.where(mask > 0)])
        print("Macro F1:", Macro_F1)
        
        # Calculate Micro F1
        micro_pred = np.sum(cnt_pred)
        micro_GT = np.sum(cnt_GT)
        micro_TP = np.sum(cnt_TP)
        micro_precision = micro_TP / micro_pred
        micro_recall = micro_TP / micro_GT
        Micro_F1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 0.000001)
        print("Micro F1:", Micro_F1)

        return accuracy, Macro_F1, Micro_F1
    




if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data_chess()
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    classifier = ID3_decision_tree()
    tree = classifier.train(X_train[1901:6000], Y_train[1901:6000])
    main(tree)