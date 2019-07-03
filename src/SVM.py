import numpy as np
from import_data import *
from SVMKernel import *
import cvxopt.solvers

class SVM(object):
    def __init__(self, sigma=0, C=0.5, gamma=5):
        self.sigma = sigma
        self.C = C
        self.gamma = gamma

        self.bias = None
        self.support_labels = None
        self.support_multipliers = None
        self.support_vec = None

        self.K = None

    def __gram_matrix(self, X_train):
        """
        :param X_train: A numpy matrix of N * D, N is the count of training set,
        D is the dimension of the training data
        :return: A matrix of N * N, each element stores the result of kernel(Xi, Xj)
        """
        N, D = X_train.shape

        # K = np.zeros((N, N), dtype=np.float32)
        if self.sigma == 0:
            K = linear(X_train, X_train)
        else:
            K = gaussian_vec(X_train, X_train, self.gamma)

        return K

    def __compute_multipliers(self, X, y):
        """
        :param X: A numpy array of shape (N, D), meaning the input of the training set
        :param y: A numpy array of shape (N, ), meaning the labels of the training set
        :return: The solution of the QP problem
        """
        N, D = X.shape
        self.K = self.__gram_matrix(X)
        print("Matrix K computation completed.")

        # The goal problem is to minimize 1/2 X.T * P * X + q.T * X
        # s.t.
        # G * X <= h
        # A * X = b
        # In the SVM case the X above in the annotation above is the array of α

        # * is referred to element wise multiplication, not matrix multiplication
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-1 * np.ones(N))

        # For the standard problem(without soft margins)
        # The constraints are α_i >= 0 which equals to -α_i <= 0
        G_std = cvxopt.matrix(-np.eye(N))
        h_std = cvxopt.matrix(np.zeros(N))

        # For the SVM with soft margins, the additional constraints are
        # α_i <= C
        if self.C > 0:
            G_soft = cvxopt.matrix(np.eye(N))
            h_soft = cvxopt.matrix(np.ones(N) * self.C)

        else:
            G_soft = None
            h_soft = None

        # Combine the two types of constraints above together using np.vstack()
        G = cvxopt.matrix(np.vstack((G_std, G_soft)))
        h = cvxopt.matrix(np.vstack((h_std, h_soft)))

        A = cvxopt.matrix(y.astype(float), (1, N))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def train(self, X_train, Y_train):
        alpha = self.__compute_multipliers(X_train, Y_train)
        MIN_MULTIPLIERS_THRESHOLD = 1e-5
        print("Support Vectors Count:", np.sum(alpha > MIN_MULTIPLIERS_THRESHOLD))
        print("b:", self.bias)

        self.support_multipliers = alpha[alpha > MIN_MULTIPLIERS_THRESHOLD]
        self.support_vec = X_train[alpha > MIN_MULTIPLIERS_THRESHOLD]
        self.support_labels = Y_train[alpha > MIN_MULTIPLIERS_THRESHOLD]

        index = alpha > MIN_MULTIPLIERS_THRESHOLD
        supportK = self.K[index, :]
        supportK = supportK[:, index]

        self.bias = np.mean(1 / self.support_labels - np.sum((self.support_multipliers * self.support_labels).reshape(-1, 1) * supportK, axis=0))

    def predict(self, X_test):
        """

        :param X_test: A numpy array of shape (N, D)
        :return: The prediction on the test set, the value should be 1 or -1, the shape is (N, )
        """
        if self.sigma == 0:
            K = linear(self.support_vec, X_test)
        else:
            K = gaussian_vec(self.support_vec, X_test)

        y_pred = np.sum((self.support_multipliers * self.support_labels).reshape(-1, 1) * K, axis=0) + self.bias
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = -1
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
        mask = np.zeros(2, dtype=np.int32)

        # Transform the labels -1 to 0 to perform the one hot calculation
        y_pred[y_pred == -1] = 0
        testlabels[testlabels == -1] = 0

        # Tranform the prediction and ground truth labels to one hot vectors
        y_pred = np.eye(2, dtype=np.int32)[y_pred.astype(int)]
        testlabels = np.eye(2, dtype=np.int32)[testlabels]

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
    X_train, Y_train, X_test, Y_test = load_data_chess("0")
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    A = Y_train[Y_train == 1]
    B = Y_train[Y_train == -1]

    Asize = (int)(5000 * A.shape[0] / X_train.shape[0])
    Bsize = 5000 - Asize

    Aindex = np.sort(np.random.randint(0, A.shape[0], size=Asize, dtype="I"))
    Bindex = np.sort(np.random.randint(0, B.shape[0], size=Bsize, dtype="I"))

    X = np.vstack((X_train[Aindex], X_train[Bindex]))
    Y = np.append(Y_train[Aindex], Y_train[Bindex])

    softSVM = SVM(sigma=1, gamma=15)
    softSVM.train(X, Y)
    y_pred = softSVM.predict(X_test)
    accuracy, MacroF1, MicroF1 = softSVM.measure(y_pred, Y_test)
    print(accuracy, MacroF1, MicroF1)
