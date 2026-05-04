import numpy as np

class SimpleLinearTS:
    def __init__(self, n_items, n_features=10):
        self.n_items = n_items
        self.n_features = n_features
        
        # Векторы для каждого item
        self.item_vectors = np.random.randn(n_items, n_features) * 0.1
        
        # Параметры модели
        self.B = np.eye(n_features)
        self.theta = np.zeros(n_features)
        self.B_inv = np.eye(n_features)  # кэшируем обратную матрицу
        
        # Популярность
        self.popularity = np.zeros(n_items)
        
    def fit(self, df):
        for item_id in df['item_id_enc']:
            if item_id < self.n_items:
                self.popularity[item_id] += 1
        
        if self.popularity.max() > 0:
            self.popularity = self.popularity / self.popularity.max()
        
        print(f"Обучено {len(df)} записей, {self.n_items} уникальных item'ов")
        return self
    
    def _get_context(self, state):
        if state is None or len(state) == 0:
            return np.zeros(self.n_features)
        
        context = np.zeros(self.n_features)
        count = 0
        
        for item_id in state[-5:]:
            if 0 < item_id < self.n_items:
                context += self.item_vectors[item_id]
                count += 1
        
        if count > 0:
            context = context / count
        
        return context
    
    def recommend(self, user_id=None, state=None, k=1):
        context = self._get_context(state)
        
        # Семплируем theta (один раз за вызов!)
        try:
            theta_sample = np.random.multivariate_normal(self.theta, self.B_inv)
        except:
            theta_sample = self.theta + np.random.randn(self.n_features) * 0.1
        
        # Векторизованное вычисление scores для всех item'ов
        # Берём только первые n_features компонент item_vectors
        scores = self.item_vectors @ theta_sample + self.popularity * 0.3
        
        return int(np.argmax(scores))
    
    def update(self, item, reward):
        if item >= self.n_items:
            return
        
        if reward > 0.7:
            self.theta += self.item_vectors[item] * 0.01
        
        # Обновляем B и её обратную
        outer = np.outer(self.item_vectors[item], self.item_vectors[item])
        self.B += outer
        
        # Обновление обратной матрицы по формуле Шермана-Моррисона
        try:
            v = self.B_inv @ self.item_vectors[item]
            self.B_inv -= np.outer(v, v) / (1 + np.dot(self.item_vectors[item], v))
        except:
            self.B_inv = np.linalg.inv(self.B)