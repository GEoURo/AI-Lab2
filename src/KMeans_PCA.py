import KMeans
from import_data import *
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        pass

    def train(self, X, theshold=0.8):
        """
        Perform PCA on the input X and returns the matrix after being reduced on dimension

        Inputs:
        X: A numpy array of shape (N, D), N is the number of the items in the dataset,
        D is the dimension of the data
        theshold: The theshold which control the number of eigenvalue to be selected.
        """
        average = np.mean(X, axis=0)
        X -= average
        covX = np.cov(X, rowvar=False)
        eigenValue, eigenVec = np.linalg.eig(covX)
        # Sort the eigenvalue
        # The eigenvectors are column vectors, so there is a trasposition
        eigenIndex = np.argsort(eigenValue)[::-1]
        eigenValue = np.sort(eigenValue)[::-1]
        eigenVec = eigenVec.T[eigenIndex]
        eigenValue_cum = np.cumsum(eigenValue) / np.sum(eigenValue)
        
        # Select the number of eigenvalue to be used according to theshold
        m = np.min(np.argwhere(eigenValue_cum > theshold))
        selectedVec = eigenVec[:m, :]
        return X.dot(selectedVec.T)

if __name__ == "__main__":
    dict1 = ["Bufonidae", "Dendrobatidae", "Hylidae", "Leptodactylidae"]
    dict2 = ["Rhinella", "Osteocephalus", "Scinax", "Leptodactylus", "Dendropsophus", "Ameerega", "Hypsiboas",
             "Adenomera"]
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


    pca = PCA()
    kmeans = KMeans.KMeans()
    reduced_X = pca.train(X, 0.7)
    print(reduced_X.shape)

    x = reduced_X[:, 0]
    y = reduced_X[:, 1]
    plt.scatter(x, y, s=1, marker=".")
    plt.show()

    k_choices = [4, 8, 10]
    purity_choice = []
    RI_choice = []

    for i, k in enumerate(k_choices):
        Y_test = Yk[i]
        purity = 0
        RI = 0
        for step in range(3):
            result = kmeans.train(k, reduced_X)
            for i in range(k):
                print(result[result[:] == i].shape[0], end='\t')
            print()

            purity_, RI_ = kmeans.measure(result, k, Y_test)
            print(RI_)
            purity += purity_
            RI += RI_

        purity_choice.append(purity / 3)
        RI_choice.append(RI / 3)

    print(purity_choice)
    print(RI_choice)
    k_optimize = k_choices[np.argmax(np.array(RI_choice))]
    print("Optimize k choice:", k_optimize)

    result = kmeans.train(k_optimize, reduced_X)
    save_buf = np.append([k_optimize], result.astype(int).tolist()).tolist()

    save_df = pd.DataFrame(save_buf)
    save_df.to_csv('./PCA.csv', index=False, header=False)