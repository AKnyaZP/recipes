import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer
import logging 
 
logging.basicConfig(level=logging.INFO)
 
class RecipeEmbedder:
    """
    Класс для создания векторных представлений рецептов.
    Использует многоязычные модели для поддержки русского языка.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased"):
        """
        Инициализирует модель эмбеддингов.
        
        Args:
            model_name: Название модели из HuggingFace
        """
        logging.info(f"model_name: {model_name}")
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logging.info(f"device: {device}")
        
        # Загружаем модель
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Модель загружена. Размерность эмбеддингов: {self.embedding_dim}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Кодирует список текстов в векторные представления.
        
        Args:
            texts: Список текстов для векторизации
            batch_size: Размер батча для обработки
            
        Returns:
            Матрица эмбеддингов shape=(len(texts), embedding_dim)
        """
        print(f"🔄 Векторизуем {len(texts)} текстов...")
        
        # Предобработка текстов
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("[EMPTY]")  # Плейсхолдер для пустых текстов
            else:
                processed_texts.append(text.strip())
        
        # Генерируем эмбеддинги
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Нормализуем для косинусного сходства
            convert_to_numpy=True
        )
        
        print(f"✅ Создано {embeddings.shape[0]} эмбеддингов")
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Кодирует поисковый запрос.
        
        Args:
            query: Текст запроса
            
        Returns:
            Вектор запроса
        """
        if not query.strip():
            query = "[EMPTY]"
        
        embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embedding[0]
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет семантическое сходство между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Коэффициент сходства (0-1)
        """
        emb1 = self.encode_query(text1)
        emb2 = self.encode_query(text2)
        
        # Косинусное сходство (векторы уже нормализованы)
        similarity = np.dot(emb1, emb2)
        return float(similarity)