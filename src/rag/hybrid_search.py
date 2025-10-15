import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import re

class HybridSearch:
    """
    Реализует гибридный поиск, комбинируя:
    1. Векторный поиск (семантическое сходство)
    2. BM25 поиск (лексическое сходство по ключевым словам)
    """
    
    def __init__(self, vector_store, documents: List[Dict[str, Any]]):
        """
        Инициализирует гибридный поиск.
        
        Args:
            vector_store: FAISS векторное хранилище
            documents: Список документов для построения BM25 индекса
        """
        self.vector_store = vector_store
        self.documents = documents
        
        print("🔍 Инициализируем гибридный поиск...")
        
        # Готовим тексты для BM25
        self.corpus_tokens = []
        for doc in documents:
            # Используем полный текст рецепта для BM25
            text = doc.get('full_text', '') + ' ' + doc.get('name', '')
            tokens = self._tokenize(text)
            self.corpus_tokens.append(tokens)
        
        # Строим BM25 индекс
        print(f"📝 Строим BM25 индекс из {len(self.corpus_tokens)} документов...")
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        print("✅ Гибридный поиск готов!")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст для BM25.
        Простая токенизация с учетом русского языка.
        
        Args:
            text: Входной текст
            
        Returns:
            Список токенов
        """
        if not text:
            return []
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Заменяем дефисы и подчеркивания на пробелы
        text = re.sub(r'[-_]', ' ', text)
        
        # Оставляем только буквы, цифры и пробелы (поддержка кириллицы)
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        
        # Разбиваем на токены
        tokens = text.split()
        
        # Фильтруем слишком короткие токены
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def vector_search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Выполняет векторный поиск.
        
        Args:
            query_embedding: Эмбеддинг запроса
            k: Количество результатов
            
        Returns:
            Список (документ, скор)
        """
        return self.vector_store.search(query_embedding, k)
    
    def bm25_search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Выполняет BM25 поиск.
        
        Args:
            query: Текстовый запрос
            k: Количество результатов
            
        Returns:
            Список (документ, скор)
        """
        # Токенизируем запрос
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Получаем скоры BM25
        scores = self.bm25.get_scores(query_tokens)
        
        # Сортируем по убыванию скора
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-k результатов
        results = []
        for i, (doc_idx, score) in enumerate(scored_docs[:k]):
            if score > 0:  # Только документы с положительным скором
                doc = self.documents[doc_idx].copy()
                results.append((doc, float(score)))
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        query_embedding: np.ndarray, 
        k: int = 10,
        alpha: float = 0.6
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Выполняет гибридный поиск, комбинируя векторный и BM25 поиск.
        
        Args:
            query: Текстовый запрос
            query_embedding: Эмбеддинг запроса
            k: Общее количество результатов
            alpha: Вес векторного поиска (0.0-1.0), BM25 весит (1-alpha)
            
        Returns:
            Список (документ, комбинированный_скор) отсортированный по убыванию скора
        """
        print(f"🔍 Гибридный поиск: '{query}' (α={alpha})")
        
        # Получаем результаты от обоих методов
        # Берем больше результатов для лучшего покрытия
        search_k = min(k * 2, len(self.documents))
        
        vector_results = self.vector_search(query_embedding, search_k)
        bm25_results = self.bm25_search(query, search_k)
        
        print(f"  📊 Векторный поиск: {len(vector_results)} результатов")
        print(f"  📊 BM25 поиск: {len(bm25_results)} результатов")
        
        # Нормализуем скоры для комбинирования
        combined_scores = {}
        
        # Нормализуем векторные скоры (косинусное сходство уже в диапазоне [-1, 1])
        # Приводим к диапазону [0, 1]
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
            min_vector_score = min(score for _, score in vector_results)
            vector_range = max_vector_score - min_vector_score if max_vector_score > min_vector_score else 1.0
            
            for doc, score in vector_results:
                doc_id = doc.get('id', str(hash(doc['name'])))
                normalized_score = (score - min_vector_score) / vector_range
                combined_scores[doc_id] = {
                    'document': doc,
                    'vector_score': normalized_score,
                    'bm25_score': 0.0
                }
        
        # Нормализуем BM25 скоры
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            
            for doc, score in bm25_results:
                doc_id = doc.get('id', str(hash(doc['name'])))
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0.0
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]['bm25_score'] = normalized_score
                else:
                    combined_scores[doc_id] = {
                        'document': doc,
                        'vector_score': 0.0,
                        'bm25_score': normalized_score
                    }
        
        # Вычисляем комбинированные скоры
        final_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (
                alpha * scores['vector_score'] + 
                (1 - alpha) * scores['bm25_score']
            )
            final_results.append((scores['document'], combined_score))
        
        # Сортируем по убыванию комбинированного скора
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ Итого: {len(final_results)} уникальных результатов")
        
        return final_results[:k]