import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict

class RecommenderSystem:
    def __init__(self, n_components=50, min_ratings_per_user=10, min_ratings_per_book=25):
        self.filtered_df = pd.read_csv('data/Filtered.csv', low_memory=False)
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_book = min_ratings_per_book
        self.n_components = n_components
        self.user_item_matrix = None
        self.reconstructed_matrix = None
        self.cosine_similarity_matrix = None
        self.knn = None

        self.process_data()
        self.train_svd()
        self.train_knn()

    def process_data(self):
        self.filtered_df = self.filtered_df.sample(frac=0.03, random_state=42)
        self.user_item_matrix = self.filtered_df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
        self.cosine_similarity_matrix = cosine_similarity(self.user_item_matrix)

    def train_svd(self):
        svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        U = svd_model.fit_transform(self.user_item_matrix)
        Vt = svd_model.components_
        self.reconstructed_matrix = np.dot(U, Vt)

    def train_knn(self):
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn.fit(self.user_item_matrix.values)

    def svd_predict(self, user_id, book_title):
        try:
            book_index = self.user_item_matrix.index.get_loc(book_title)
            user_index = self.user_item_matrix.columns.get_loc(user_id)
            return self.reconstructed_matrix[book_index, user_index]
        except KeyError:
            return 0

    def knn_predict(self, user_id, book_title):
        try:
            book_index = self.user_item_matrix.index.get_loc(book_title)
            similarity_scores = self.cosine_similarity_matrix[book_index]
            nearest_neighbors = np.argsort(similarity_scores)[::-1][:11]

            if user_id in self.user_item_matrix.columns:
                user_col_index = self.user_item_matrix.columns.get_loc(user_id)
                avg_rating = self.user_item_matrix.iloc[nearest_neighbors[1:], user_col_index].mean()
                return avg_rating if not pd.isna(avg_rating) else 0
            else:
                return 0
        except KeyError:
            return 0

    def recommend_books(self, rated_books, user_id, n=10, weight_svd=0.7, weight_knn=0.3):
        combined_predictions = defaultdict(float)

        for book_title, rating in rated_books:
            try:
                input_book_index = self.user_item_matrix.index.get_loc(book_title)
                distances, indices = self.knn.kneighbors([self.user_item_matrix.iloc[input_book_index].values],
                                                         n_neighbors=n + 1)
                similar_books_knn = [(self.user_item_matrix.index[indices.flatten()[i]], distances.flatten()[i])
                                     for i in range(1, len(distances.flatten()))]
            except KeyError:
                continue

            for similar_book, distance in similar_books_knn:
                svd_pred = self.svd_predict(user_id, similar_book)
                knn_pred = 1 - distance
                combined_pred = (weight_svd * svd_pred) + (weight_knn * knn_pred)
                combined_predictions[similar_book] += combined_pred * rating

        sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)

        recommendations = [
            (book_title, combined_score, self.filtered_df[self.filtered_df['Book-Title'] == book_title]['Image-URL-M'].values[0],
             self.filtered_df[self.filtered_df['Book-Title'] == book_title]['Book-Author'].values[0])
            for book_title, combined_score in sorted_predictions[:n]
        ]

        return recommendations

    def get_books(self):
        return self.user_item_matrix.index.tolist()

    def test_model(self):
        train_indices, test_indices = train_test_split(range(len(self.user_item_matrix)), test_size=0.2, random_state=42)
        test_set = [(self.user_item_matrix.columns[i], self.user_item_matrix.index[j], self.user_item_matrix.values[j, i])
                    for i in test_indices for j in range(self.user_item_matrix.shape[0])]

        predictions_svd = [(user_id, book_title, true_rating, self.svd_predict(user_id, book_title), None)
                           for user_id, book_title, true_rating in test_set]
        predictions_knn = [(user_id, book_title, true_rating, self.knn_predict(user_id, book_title), None)
                           for user_id, book_title, true_rating in test_set]
        predictions_com = [(predictions_svd[i][0], predictions_svd[i][1], predictions_svd[i][2],
                            (predictions_svd[i][3] + predictions_knn[i][3]) / 2, None)
                           for i in range(len(predictions_svd))]

        return predictions_svd, predictions_knn, predictions_com

    @staticmethod
    def get_performance_metrics(predictions, k=10, threshold=7.0):
        precisions = dict()
        recalls = dict()
        user_est_true = defaultdict(list)

        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        precision = sum(precisions.values()) / len(precisions)
        recall = sum(recalls.values()) / len(recalls)
        rmse = np.sqrt(np.mean([float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]))
        mae = np.mean([float(abs(true_r - est)) for (_, _, true_r, est, _) in predictions])

        return precision, recall, rmse, mae
    
recommender = RecommenderSystem()
