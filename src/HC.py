from import_data import *

class HierarchicalClustering(object):
    def __init__(self):
        pass

    def compute_dist_vec(self, X1, X2):
        dists = np.sum(X1 ** 2, axis=1, keepdims=True) + np.sum(X2 ** 2, axis=1) - 2 * X1.dot(X2.T)

        # Avoid Error
        dists[dists < 0] = 0
        dists = np.sqrt(dists)

        return dists

    def HC(self, dataset, k):
        self.dataset = dataset
        self.k = k

        num_dataset = dataset.shape[0]
        C = [dataset[i].reshape(1, -1) for i in range(dataset.shape[0])]
        # last column is original index
        # The last but one column is contains the labels
        dists = self.compute_dist_vec(dataset[:, :-2], dataset[:, :-2])
        dists[np.arange(num_dataset), np.arange(num_dataset)] = 9999
        q = dataset.shape[0]

        while q > k:
            min_index = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            i_temp, j_temp = min(min_index), max(min_index)

            C[i_temp] = np.vstack((C[i_temp], C[j_temp]))
            del C[j_temp]
            dists = np.delete(dists, j_temp, axis=0)
            dists = np.delete(dists, j_temp, axis=1)
            for j in range(q - 1):
                X1, X2 = C[i_temp][:, :-2], C[j][:, :-2]
                temp_dist = self.compute_dist_vec(X1, X2)
                dists[i_temp, j] = np.sum(temp_dist) / (C[i_temp].shape[0] * C[j].shape[0])
                dists[j, i_temp] = dists[i_temp, j]
                dists[i_temp, i_temp] = 9999
            q = q - 1
        self.cluster_result = C

    def measure(self):
        # purity
        intersection = 0
        sample = self.dataset[:, :-2]
        labels = self.dataset[:, -2]
        sample_num = sample.shape[0]

        for i in range(self.k):
            temp_label = self.cluster_result[i][:, -2].astype(np.int32)
            intersection += np.max(np.bincount(temp_label))
        purity = intersection / sample_num

        # RI
        dot_num = sample_num*(sample_num-1)/2
        a = 0
        cluster_record = {}
        for i in range(self.k):
            temp_label = self.cluster_result[i][:, -2].astype(np.int32)
            cluster_cnt = np.array(np.bincount(temp_label))
            cluster_record[i] = cluster_cnt.copy()
            cluster_cnt = cluster_cnt*(cluster_cnt-1)/2
            a += np.sum(cluster_cnt)

        d = 0
        label_num = np.unique(labels).shape[0]
        temp=[0 for i in range(label_num)]
        for i in range(self.k):
            for j in range(label_num):
                if(j < cluster_record[i].shape[0]):
                    spec_num = labels[labels == j].shape[0]
                    temp[j] += np.sum(cluster_record[i])-cluster_record[i][j]
                    d += cluster_record[i][j]*(sample_num-spec_num-temp[j])
        RI = (a + d) / dot_num
        return (purity, RI)

if __name__ == "__main__":
    df = pd.read_csv('../datasets/青蛙聚类/frog.csv')
    dataset = np.array(df.iloc[:, :])

    rand_batch_size = 2500
    indices = np.random.choice(np.arange(dataset.shape[0]), rand_batch_size)

    dataset = dataset[indices]
    dataset = np.column_stack((dataset, np.arange(dataset.shape[0])))
    HC = HierarchicalClustering()

    HC.HC(dataset, 4)

    purity, RI = HC.measure()
    print(purity, RI)

    save_buf = [0 for i in range(dataset.shape[0] + 1)]
    save_buf[0] = HC.k

    for i in range(HC.k):
        temp_num = HC.cluster_result[i].shape[0]
        for j in range(temp_num):
            save_buf[int(HC.cluster_result[i][j, -1]) + 1] = int(HC.cluster_result[i][j, -2])

    save_df = pd.DataFrame(save_buf)
    save_df.to_csv('./HC.csv', index=False, header=False)