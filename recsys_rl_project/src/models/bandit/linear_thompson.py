import numpy as np
import random

class SimpleLinearTS:
    """
    Очень простая версия Linear Thompson Sampling
    """
    
    def __init__(self, n_items, n_features=10):
        self.n_items = n_items
        self.n_features = n_features
        
        # Векторы для каждого item (простые случайные)
        self.item_vectors = np.random.randn(n_items, n_features) * 0.1
        
        # Параметры модели
        self.B = np.eye(n_features)  # Матрица 10x10
        self.theta = np.zeros(n_features)  # Вектор 10
        
        # Для подсчета популярности
        self.popularity = np.zeros(n_items)
        
    def fit(self, df):
        """Просто считаем популярность item'ов"""
        for item_id in df['item_id_enc']:
            self.popularity[item_id] += 1
        
        # Нормализуем популярность
        if self.popularity.max() > 0:
            self.popularity = self.popularity / self.popularity.max()
        
        print(f"Обучено {len(df)} записей, {self.n_items} уникальных item'ов")
        return self
    
    def _get_context(self, state):
        """Создаем контекст из истории пользователя"""
        if state is None or len(state) == 0:
            return np.zeros(self.n_features)
        
        # Берем среднее векторов последних item'ов
        context = np.zeros(self.n_features)
        count = 0
        
        for item_id in state[-5:]:  # последние 5 item'ов
            if item_id > 0 and item_id < self.n_items:
                context += self.item_vectors[item_id]
                count += 1
        
        if count > 0:
            context = context / count
        
        return context
    
    def recommend(self, user_id=None, state=None, k=1):
        """Рекомендуем item"""
        
        # Получаем контекст
        context = self._get_context(state)
        
        # Сэмплируем theta из нормального распределения
        try:
            theta_sample = np.random.multivariate_normal(self.theta, np.linalg.inv(self.B))
        except:
            theta_sample = self.theta + np.random.randn(self.n_features) * 0.1
        
        # Вычисляем score для каждого item
        scores = np.zeros(self.n_items)
        
        for item_id in range(self.n_items):
            # Комбинируем контекст и вектор item'a
            features = np.concatenate([context, self.item_vectors[item_id]])
            
            # Обрезаем до нужной размерности
            if len(features) > self.n_features:
                features = features[:self.n_features]
            elif len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))
            
            # Score = theta * features + бонус популярности
            score = np.dot(theta_sample, features)
            score += self.popularity[item_id] * 0.3  # добавляем популярность
            
            scores[item_id] = score
        
        # Возвращаем лучший item
        best_item = np.argmax(scores)
        return int(best_item)
    
    def update(self, item, reward):
        """Обновляем модель после каждого шага"""
        
        # Простое обновление: если reward хороший, усиливаем связь
        if reward > 0.7:
            self.theta += self.item_vectors[item] * 0.01
        
        # Обновляем матрицу B
        self.B += np.outer(self.item_vectors[item], self.item_vectors[item])