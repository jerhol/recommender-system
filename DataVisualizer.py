import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataVisualizer:
    def __init__(self, recommender_system):
        self.recommender_system = recommender_system

    def plot_ratings_distribution_unfiltered(self):
        ratings_df = pd.read_csv("data/Ratings.csv")

        plt.figure(figsize=(10, 6))
        plt.hist(ratings_df['Book-Rating'], bins=10, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Book Ratings (Unfiltered Data)')
        plt.xlabel('Book Rating')
        plt.ylabel('Frequency')
        plt.xticks(range(1, 11))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()

    def plot_ratings_distribution_filtered(self):
        ratings = self.recommender_system.user_item_matrix.values.flatten()
        non_zero_ratings = ratings[ratings > 0]

        plt.figure(figsize=(10, 6))
        plt.hist(non_zero_ratings, bins=10, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Book Ratings (Filtered Data)')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

    def plot_rmse_mae(self):
        _, _, predictions_com = self.recommender_system.test_model()

        precision_svd, recall_svd, svd_rmse, svd_mae = self.recommender_system.get_performance_metrics(_, k=10,
                                                                                                       threshold=7.0)
        precision_com, recall_com, hybrid_rmse, hybrid_mae = self.recommender_system.get_performance_metrics(
            predictions_com, k=10, threshold=7.0)

        labels = ['SVD Model', 'Hybrid Model']
        rmse_values = [svd_rmse, hybrid_rmse]
        mae_values = [svd_mae, hybrid_mae]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width / 2, rmse_values, width, label='RMSE')
        ax.bar(x + width / 2, mae_values, width, label='MAE')

        ax.set_ylabel('Error')
        ax.set_title('RMSE and MAE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.ylim(0, 1)
        plt.show()


    def plot_precision_recall(self):
        _, predictions_knn, _ = self.recommender_system.test_model()
        precision, recall, _, _ = self.recommender_system.get_performance_metrics(predictions_knn, k=10, threshold=7.0)

        labels = ['Precision', 'Recall']
        values = [precision, recall]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=['blue', 'green'])
        plt.ylabel('Scores')
        plt.title(f'Precision and Recall @ 10 for KNN')
        plt.ylim(0, 1)
        plt.show()

    def plot_weighted_performance(self):
        weights = [round(i * 0.1, 1) for i in range(11)]
        precision_list, recall_list, rmse_list, mae_list = [], [], [], []

        for weight_svd in weights:
            weight_knn = 1 - weight_svd

            predictions_svd, predictions_knn, predictions_com = self.recommender_system.test_model()

            for i in range(len(predictions_svd)):
                avg_est = (weight_svd * predictions_svd[i][3]) + (weight_knn * predictions_knn[i][3])
                predictions_com[i] = (
                predictions_svd[i][0], predictions_svd[i][1], predictions_svd[i][2], avg_est, None)

            precision, recall, rmse, mae = self.recommender_system.get_performance_metrics(predictions_com)
            precision_list.append(precision)
            recall_list.append(recall)
            rmse_list.append(rmse)
            mae_list.append(mae)

        plt.figure(figsize=(10, 6))
        plt.plot(weights, precision_list, label="Precision@10", marker="o")
        plt.plot(weights, recall_list, label="Recall@10", marker="o")
        plt.plot(weights, rmse_list, label="RMSE", marker="o")
        plt.plot(weights, mae_list, label="MAE", marker="o")
        plt.title("Performance Metrics vs. Weight Distribution (SVD vs. KNN)")
        plt.xlabel("SVD Weight")
        plt.ylabel("Performance Metric Value")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
