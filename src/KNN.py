from import_data import *
np.set_printoptions(threshold=np.inf)
class KNearestNeighbour(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        """
        Train the linear classifier, for KNN this step is just
        memorizing the training data.

        Inputs:
        X: A numpy array of shape (train_batch_size, D), each 
        data has a dimension of D.
        Y: A numpy array of shape (train_batch_size,) containing the training
        labels, Y[i] is the corresponding label of X[i]
        """
        self.X_train = X
        self.Y_train = Y

    def predict(self, testset, k=1):
        """
        Predict labels for the testset

        Inputs:
        testset: A numpy array of shape (test_batch_size, D), each 
        data has a dimension of D.

        Returns:
        Y: A numpy array of shape (test_batch_size,) containing predicted labels for the
        test data, where Y[i] is the predicted label for the test point X[i]. 
        """
        num_test = testset.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        # Calculate the l2 distance between all the test points and training points(vectorized)
        dists = np.sqrt(np.sum(testset ** 2, axis=1, keepdims=True) + np.sum(self.X_train ** 2, axis=1) - 2 * testset.dot(self.X_train.T))
        
        # Do the prediction based on the dists matrix
        Y = self.predict_labels(dists, k=k)
        return Y

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=np.int32)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y. 
            closest_y = self.Y_train[np.argsort(dists[i])][:k]

            # Find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i].
            y_pred[i] = np.argmax(np.bincount(closest_y))
        
        return y_pred
    
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
    classifier = KNearestNeighbour()
    X_train, Y_train, X_test, Y_test = load_data()
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)
    
    # Set up training data
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    X_train_folds = []
    Y_train_folds = []

    X_train_folds = np.array(np.array_split(X_train, num_folds))
    Y_train_folds = np.array(np.array_split(Y_train, num_folds))

    k_to_Micro_F1 = []

    for k in k_choices:
        temp = []
        for i in range(num_folds):
            selected_folds = [x for x in range(num_folds) if x != i]
            selected_X_train = np.concatenate(X_train_folds[selected_folds])
            selected_Y_train = np.concatenate(Y_train_folds[selected_folds])
            classifier.train(selected_X_train, selected_Y_train)

            y_predicts = classifier.predict(X_train_folds[i], k)
            accuracy, Macro_F1, Micro_F1 = classifier.measure(y_predicts, Y_train_folds[i])
            temp.append(Micro_F1)
        k_to_Micro_F1.append(temp)
    
    # Print all the cross validation results
    for key, value in enumerate(k_to_Micro_F1):
        for Micro_F1 in value:
            print('k = %d, accuracy = %f' % (k_choices[key], Micro_F1))
    
    # Print the mean results and select the best resulted k as the final parameter set up
    k_to_Micro_F1 = np.mean(k_to_Micro_F1, axis=1)
    print(k_to_Micro_F1)
    best_k = k_choices[np.argmax(k_to_Micro_F1)]
    print("Max k choice:",best_k)
    
    # Evaluate using the testset
    classifier.train(X_train, Y_train)
    y_pred = classifier.predict(X_test, k=best_k)
    accur, Macro_F1, Micro_F1 = classifier.measure(y_pred, Y_test)
    print(accur, Macro_F1, Micro_F1)