import numpy as np

from data import GetData
from method import Model

class Experiments(object):
    def __init__(self, mir_dis_data, model_name='TDRC', **kwargs):
        super().__init__()
        self.mir_dis_data = mir_dis_data
        self.model = Model(model_name)
        self.parameters = kwargs

    def CV_triplet(self):
        k_folds = 5
        index_matrix = np.array(np.where(self.mir_dis_data.type_tensor == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics_tensor = np.zeros((1, 7))
        # metrics_CP = np.zeros((1, 7))
        for k in range(k_folds):

            train_tensor = np.array(self.mir_dis_data.type_tensor, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[train_index] = 0
            train_matrix = train_tensor.sum(2)
            train_matrix[np.where(train_matrix > 0)] = 1
            miRNA_func_similarity_matrix = np.mat(self.mir_dis_data.get_functional_sim(train_matrix))
            predict_tensor = self.model()(train_tensor, self.mir_dis_data.dis_sim, miRNA_func_similarity_matrix,
                                             r=self.parameters['r'], alpha=self.parameters['alpha'],
                                             beta=self.parameters['beta'], lam=self.parameters['lam'],
                                             tol=1e-6, max_iter=500)

            for i in range(10):
                metrics_tensor = metrics_tensor + self.cv_tensor_model_evaluate(self.mir_dis_data.type_tensor, predict_tensor,
                                                                           train_index, i)

        # print(metrics_tensor / (k + 1))
        result = np.around(metrics_tensor / 50, decimals=4)
        return result

    def cv_tensor_model_evaluate(self, association_tensor, predict_tensor, train_index, seed):
        test_po_num = np.array(train_index).shape[1]
        test_index = np.array(np.where(association_tensor == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        # print(np.where((negative_index-test_index)!=0))
        test_ne_index = tuple(test_index[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_tensor[test_ne_index].flatten()), np.mat(association_tensor[train_index].flatten())))
        predict_score = np.column_stack(
            (np.mat(predict_tensor[test_ne_index].flatten()), np.mat(predict_tensor[train_index].flatten())))
        # real_score and predict_score are array
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]
        return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]








if __name__ == '__main__':

    root = ''
    mir_dis_data = GetData(root)
    experiment = Experiments(mir_dis_data, model_name='TDRC', r=4, alpha=0.125, beta=0.25, lam=0.001, tol=1e-6, max_iter=500)
    print(experiment.CV_triplet())
