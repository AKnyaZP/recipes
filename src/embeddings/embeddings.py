import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer
import logging
import asyncio
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeEmbedder:
    """
    Класс для создания векторных представлений рецептов.
    Использует многоязычные модели для поддержки русского языка.
    Поддерживает автоматическое переключение sync/async режимов.
    """

    def __init__(self, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased"):
        """
        Инициализирует модель эмбеддингов.

        Args:
            model_name: Название модели из HuggingFace
        """
        logger.info(f"model_name: {model_name}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"device: {device}")

        # Загружаем модель
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"✅ Модель загружена. Размерность эмбеддингов: {self.embedding_dim}")

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Предобработка текстов перед векторизацией.

        Args:
            texts: Список текстов

        Returns:
            Обработанные тексты
        """
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("[EMPTY]")  # Плейсхолдер для пустых текстов
            else:
                processed_texts.append(text.strip())
        return processed_texts

    def _encode_texts_sync(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Внутренний синхронный метод кодирования текстов.

        Args:
            texts: Список текстов для векторизации
            batch_size: Размер батча для обработки

        Returns:
            Матрица эмбеддингов shape=(len(texts), embedding_dim)
        """
        logger.info(f"🔄 Векторизуем {len(texts)} текстов...")

        # Предобработка текстов
        processed_texts = self._preprocess_texts(texts)

        # Генерируем эмбеддинги
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Нормализуем для косинусного сходства
            convert_to_numpy=True
        )

        logger.info(f"✅ Создано {embeddings.shape[0]} эмбеддингов")
        return embeddings

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Кодирует список текстов в векторные представления.
        Автоматически определяет sync/async контекст.

        Args:
            texts: Список текстов для векторизации
            batch_size: Размер батча для обработки

        Returns:
            Матрица эмбеддингов shape=(len(texts), embedding_dim)
        """
        try:
            # Проверяем наличие running event loop
            loop = asyncio.get_running_loop()
            # Если есть loop, выполняем в executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._encode_texts_sync, texts, batch_size)
                return future.result()
        except RuntimeError:
            # Нет event loop - синхронное выполнение
            return self._encode_texts_sync(texts, batch_size)

    def _encode_query_sync(self, query: str) -> np.ndarray:
        """
        Внутренний синхронный метод кодирования запроса.

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

    def encode_query(self, query: str) -> np.ndarray:
        """
        Кодирует поисковый запрос.
        Автоматически определяет sync/async контекст.

        Args:
            query: Текст запроса

        Returns:
            Вектор запроса
        """
        try:
            # Проверяем наличие running event loop
            loop = asyncio.get_running_loop()
            # Если есть loop, выполняем в executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._encode_query_sync, query)
                return future.result()
        except RuntimeError:
            # Нет event loop - синхронное выполнение
            return self._encode_query_sync(query)

    def _get_similarity_sync(self, text1: str, text2: str) -> float:
        """
        Внутренний синхронный метод вычисления сходства.

        Args:
            text1: Первый текст
            text2: Второй текст

        Returns:
            Коэффициент сходства (0-1)
        """
        emb1 = self._encode_query_sync(text1)
        emb2 = self._encode_query_sync(text2)

        # Косинусное сходство (векторы уже нормализованы)
        similarity = np.dot(emb1, emb2)
        return float(similarity)

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет семантическое сходство между двумя текстами.
        Автоматически определяет sync/async контекст.

        Args:
            text1: Первый текст
            text2: Второй текст

        Returns:
            Коэффициент сходства (0-1)
        """
        try:
            # Проверяем наличие running event loop
            loop = asyncio.get_running_loop()
            # Если есть loop, выполняем в executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._get_similarity_sync, text1, text2)
                return future.result()
        except RuntimeError:
            # Нет event loop - синхронное выполнение
            return self._get_similarity_sync(text1, text2)

