import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss
import os
import dotenv
from loguru import logger


class FAISSVectorStore:
    """
    Векторное хранилище на основе FAISS для поиска похожих рецептов.
    """

    def __init__(self, embedding_dim: int):
        """
        Инициализирует векторное хранилище.

        Args:
            embedding_dim: Размерность векторов эмбеддингов
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.metadata = []

        logger.info(f"Создаем FAISS индекс (размерность: {embedding_dim})")

    def build_index(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Строит FAISS индекс из эмбеддингов и документов.

        Args:
            embeddings: Матрица эмбеддингов shape=(n_docs, embedding_dim)
            documents: Список документов с метаданными
        """
        logger.info(f"Строим индекс из {len(documents)} документов...")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Размерность эмбеддингов ({embeddings.shape[1]}) "
                f"не совпадает с ожидаемой ({self.embedding_dim})"
            )

        self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(embeddings.astype(np.float32))
        self.documents = documents.copy()
        self.metadata = [
            {
                'id': doc.get('id', str(i)),
                'name': doc.get('name', ''),
                'url': doc.get('url', ''),
                'ingredients_text': doc.get('ingredients_text', '')
            }
            for i, doc in enumerate(documents)
        ]

        logger.info(f"Индекс построен. Всего документов: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Выполняет поиск похожих документов.

        Args:
            query_embedding: Вектор запроса shape=(embedding_dim,)
            k: Количество возвращаемых результатов

        Returns:
            Список кортежей (документ, скор сходства)
        """
        if self.index is None:
            raise ValueError("Индекс не построен. Вызовите build_index() сначала.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0: 
                doc = self.documents[idx].copy()
                doc.update(self.metadata[idx])
                results.append((doc, float(score)))

        return results

    def save(self, index_path: str = "data/faiss_index"):
        """
        Сохраняет индекс и метаданные на диск.

        Args:
            index_path: Путь для сохранения индекса
        """
        if self.index is None:
            raise ValueError("Нечего сохранять - индекс не построен.")

        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss_file = index_path / "faiss.index"
        faiss.write_index(self.index, str(faiss_file))

        # Сохраняем документы и метаданные
        data_file = index_path / "documents.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)

        print(f"Индекс сохранен в {index_path}")

    def load(self, index_path: str = "data/faiss_index"):
        """
        Загружает индекс и метаданные с диска.

        Args:
            index_path: Путь к сохраненному индексу
        """
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Индекс не найден: {index_path}")

        # Загружаем FAISS индекс
        faiss_file = index_path / "faiss.index"
        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS индекс не найден: {faiss_file}")

        self.index = faiss.read_index(str(faiss_file))

        # Загружаем документы и метаданные
        data_file = index_path / "documents.pkl"
        if data_file.exists():
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                stored_dim = data.get('embedding_dim')

                if stored_dim and stored_dim != self.embedding_dim:
                    logger.warning(f"Предупреждение: размерность изменилась {stored_dim} -> {self.embedding_dim}")

        logger.info(f"Индекс загружен из {index_path}. Документов: {self.index.ntotal}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику индекса.

        Returns:
            Словарь со статистикой
        """
        if self.index is None:
            return {"status": "empty", "total_docs": 0}

        return {
            "status": "ready",
            "total_docs": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, 'is_trained', True)
        }