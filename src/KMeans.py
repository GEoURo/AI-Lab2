from import_data import *


class KMeans(object):
    def __init__(self):
        pass
    
    def randCenter(self, X, k):
        """
        Randomly initialize k centers using the information from X

        Inputs:
        X: A numpy array of shape (N, D), N is the number of the items in the dataset,
        D is the dimension of the data
        k: The number of centers, assigned by user.
        """
        dim = X.shape[1]
        # k centers each with dim of dimensions
        centers = np.zeros((k, dim))
        for j in range(dim):
            minj = np.min(X[:, j])
            maxj = np.max(X[:, j])
            rangej = float(maxj - minj)
            centers[:, j:j + 1] = minj + rangej * np.random.rand(k, 1)
        
        return centers

    def train(self, k, X):
        """
        Perform KMeans cluster on the input X

        Inputs:
        k: The number of cluster centers, assigned by user
        X: A numpy array of shape (N, D), N is the number of the items in the dataset,
        D is the dimension of the data
        """
        N = X.shape[0]
        # Store the cluster info for each data point.
        # The first is which center to be assigned to.
        # The second is its distance to the center.
        step = 0

        clusterInfo = np.zeros((N, 2))
        centers = self.randCenter(X, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            dists = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(centers ** 2, axis=1) - 2 * X.dot(centers.T)
            cent_choice = np.argmin(dists, axis=1)

            clusterChanged = not (abs((clusterInfo[:, 0] - cent_choice)) < 1e-5).all()
            clusterInfo[:, 0:1] = cent_choice.reshape(-1, 1)
            clusterInfo[:, 1:2] = dists[np.arange(N), cent_choice].reshape(-1, 1)

            for i in range(k):
                ptsinCluster = X[clusterInfo[:, 0] == i, :]
                if ptsinCluster.any():
                    centers[i, :] = np.mean(ptsinCluster, axis=0)

            step += 1
            '''
            if step % 10 == 0:
                print("Step: %d" % step)
                for i in range(k):
                    print("Center %d:" % i, centers[i])
                print()
            '''
        print("Total iteration step:", step)
        return clusterInfo[:, 0]
    
    def measure(self, pred, k, real):
        cluster_label = []
        real_label = []
        for i in range(k):
            cluster_label.append(np.argwhere(pred[:] == i).reshape(-1, ))
            real_label.append(np.argwhere(real[:] == i).reshape(-1, ))
        purity = 0
        for i in range(k):
            max = 0
            for j in range(k):
                tmp = np.intersect1d(cluster_label[i], real_label[j]).shape[0]
                if tmp > max:
                    max = tmp
            purity += max
        purity /= real.shape[0]

        RI_a = 0
        RI_d = 0
        for i in range(len(pred)):
            real_intsec = np.argwhere(real[:] == real[i])
            pred_intsec = np.argwhere(pred[:] == pred[i])
            RI_a += np.intersect1d(real_intsec, pred_intsec).shape[0]

            real_intsec = np.argwhere(real[:] != real[i])
            pred_intsec = np.argwhere(pred[:] != pred[i])
            RI_d += np.intersect1d(real_intsec, pred_intsec).shape[0]

        RI = (RI_a + RI_d) / (real.shape[0] * (real.shape[0] - 1))
        return purity, RI

if __name__ == "__main__":
    dict1 = ["Bufonidae", "Dendrobatidae", "Hylidae", "Leptodactylidae"]
    dict2 = ["Rhinella", "Osteocephalus", "Scinax", "Leptodactylus", "Dendropsophus", "Ameerega", "Hypsiboas", "Adenomera"]
    dict3 = [
    "Rhinellagranulosa",
    "OsteocephalusOophagus",
    "ScinaxRuber"
    "LeptodactylusFuscus",
    "HylaMinuta",
    "HypsiboasCinerascens",
    "Ameeregatrivittata",
    "AdenomeraAndre",
    "HypsiboasCordobae",
    "AdenomeraHylaedactylus",  
    ]
    np.set_printoptions(threshold=np.inf)
    X, Y = load_data_frog()
    print(X.shape)
    print(Y.shape)
    Yk = []
    Y1 = Y[:, 0]
    Y2 = Y[:, 1]
    Y3 = Y[:, 2]
    for i, key in enumerate(dict1):
        Y1[Y1[:] == key] = i
        Yk.append(Y1)
        
    for i, key in enumerate(dict2):
        Y2[Y2[:] == key] = i
        Yk.append(Y2)
        
    for i, key in enumerate(dict3):
        Y3[Y3[:] == key] = i
        Yk.append(Y3)
    
    k_choices = [4, 8, 10]
    classifier = KMeans()
    purity_choice = []
    RI_choice = []
    for i, k in enumerate(k_choices):
        Y_test = Yk[i]
        purity = 0
        RI = 0
        for step in range(3):
            result = classifier.train(k, X)
            for i in range(k):
                print(result[result[:] == i].shape[0], end='\t')
            print()
            
            purity_, RI_ = classifier.measure(result, k, Y_test)
            print(RI_)
            purity += purity_
            RI += RI_
        
        purity_choice.append(purity / 3)
        RI_choice.append(RI / 3)

    print(purity_choice)
    print(RI_choice)
    k_optimize = k_choices[np.argmax(np.array(RI_choice))]
    print("Optimize k choice:", k_optimize)

    result = classifier.train(k_optimize, X)
    save_buf = np.append([k_optimize], result.astype(int).tolist()).tolist()

    save_df = pd.DataFrame(save_buf)
    save_df.to_csv('./KMeans.csv', index=False, header=False)
    