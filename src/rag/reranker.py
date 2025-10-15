"""
Модуль для переранжирования результатов поиска в RAG системе.
Использует cross-encoder модели для более точного ранжирования.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
import concurrent.futures
import re
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """
    Классификатор намерений пользовательских запросов.
    Определяет тип запроса: одно блюдо, несколько блюд или общий.
    """

    def __init__(self):
        # Паттерны для определения типа запроса
        self.single_dish_patterns = [
            r'что в ([а-яё\s]+)',
            r'ингредиенты ([а-яё\s]+)',
            r'состав ([а-яё\s]+)',
            r'из чего ([а-яё\s]+)',
            r'рецепт ([а-яё\s]+)',
        ]

        self.multi_dish_patterns = [
            r'([а-яё\s]+) и ([а-яё\s]+)',
            r'([а-яё\s]+), ([а-яё\s]+)',
            r'для ([а-яё\s]+) и ([а-яё\s]+)',
            r'несколько',
            r'разные',
        ]

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Классифицирует запрос и извлекает названия блюд.

        Returns:
            {
                'intent': 'single_dish' | 'multi_dish' | 'general',
                'dish_names': List[str],
                'confidence': float
            }
        """
        query_lower = query.lower().strip()

        # Проверяем паттерны для одного блюда
        for pattern in self.single_dish_patterns:
            match = re.search(pattern, query_lower)
            if match:
                dish_name = match.group(1).strip()
                return {
                    'intent': 'single_dish',
                    'dish_names': [dish_name],
                    'confidence': 0.8
                }

        # Проверяем паттерны для нескольких блюд
        for pattern in self.multi_dish_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'несколько' in query_lower or 'разные' in query_lower:
                    return {
                        'intent': 'multi_dish',
                        'dish_names': [],
                        'confidence': 0.7
                    }
                else:
                    dish_names = [g.strip() for g in match.groups() if g]
                    return {
                        'intent': 'multi_dish',
                        'dish_names': dish_names,
                        'confidence': 0.8
                    }

        # По умолчанию - общий запрос
        return {
            'intent': 'general',
            'dish_names': [],
            'confidence': 0.5
        }


class RecipeReranker:
    """
    Класс для переранжирования результатов поиска рецептов.
    Использует cross-encoder модель для более точного определения релевантности.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Инициализирует реранкер.

        Args:
            model_name: Название cross-encoder модели
        """
        logger.info(f"Загружаем реранкер: {model_name}")
        self.model = CrossEncoder(model_name)
        self.intent_classifier = QueryIntentClassifier()
        logger.info("✅ Реранкер готов")

    def _calculate_name_similarity(self, query: str, dish_name: str) -> float:
        """
        Вычисляет семантическое сходство между запросом и названием блюда.
        """
        query_words = set(query.lower().split())
        name_words = set(dish_name.lower().split())

        if not query_words or not name_words:
            return 0.0

        # Jaccard similarity + bonus for exact matches
        intersection = query_words.intersection(name_words)
        union = query_words.union(name_words)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Bonus for exact substring match
        exact_bonus = 0.3 if dish_name.lower() in query.lower() else 0.0

        return min(1.0, jaccard + exact_bonus)

    def _rerank_sync(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Внутренний синхронный метод переранжирования.
        """
        if not docs:
            return []

        # Подготавливаем пары (query, document) для cross-encoder
        pairs = []
        for doc in docs:
            # Используем название + ингредиенты для переранжирования
            doc_text = f"{doc.get('name', '')} {doc.get('ingredients_text', '')}"
            pairs.append([query, doc_text])

        # Получаем скоры от cross-encoder
        scores = self.model.predict(pairs)

        # Добавляем bonus за точное совпадение названий
        for i, doc in enumerate(docs):
            name_sim = self._calculate_name_similarity(query, doc.get('name', ''))
            scores[i] = 0.7 * scores[i] + 0.3 * name_sim

        # Сортируем по убыванию скора
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Возвращаем top_k документов
        return [doc for doc, _ in scored_docs[:top_k]]

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Переранжирует документы по релевантности к запросу.
        Автоматически определяет sync/async контекст.

        Args:
            query: Пользовательский запрос
            docs: Список документов для переранжирования
            top_k: Количество документов для возврата

        Returns:
            Список переранжированных документов
        """
        try:
            # Проверяем наличие running event loop
            loop = asyncio.get_running_loop()
            # Если есть loop, выполняем в executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._rerank_sync, query, docs, top_k)
                return future.result()
        except RuntimeError:
            # Нет event loop - синхронное выполнение
            return self._rerank_sync(query, docs, top_k)

    def filter_by_intent(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Фильтрует документы в соответствии с намерением запроса.

        Returns:
            {
                'intent_info': Dict,
                'filtered_docs': List[Dict],
                'search_strategy': str
            }
        """
        intent_info = self.intent_classifier.classify(query)
        intent = intent_info['intent']
        dish_names = intent_info['dish_names']

        if intent == 'single_dish' and dish_names:
            # Для одного блюда - строгая фильтрация по названию
            target_name = dish_names[0].lower()
            filtered = []

            for doc in docs:
                doc_name = doc.get('name', '').lower()
                # Проверяем точное совпадение или вхождение
                if target_name in doc_name or doc_name in target_name:
                    filtered.append(doc)

            # Если точных совпадений нет, берем все документы
            if not filtered:
                filtered = docs

            return {
                'intent_info': intent_info,
                'filtered_docs': filtered[:3],  # Максимум 3 для одного блюда
                'search_strategy': 'single_dish_focused'
            }

        elif intent == 'multi_dish':
            # Для нескольких блюд - группируем по названиям
            if dish_names:
                # Если есть конкретные названия, ищем по каждому
                result_docs = []
                for target_name in dish_names:
                    target_lower = target_name.lower()
                    dish_docs = []

                    for doc in docs:
                        doc_name = doc.get('name', '').lower()
                        if target_lower in doc_name or doc_name in target_lower:
                            dish_docs.append(doc)

                    # Берем лучший документ для каждого блюда
                    if dish_docs:
                        result_docs.extend(dish_docs[:2])

                return {
                    'intent_info': intent_info,
                    'filtered_docs': result_docs[:6],  # Максимум 6 для нескольких блюд
                    'search_strategy': 'multi_dish_targeted'
                }
            else:
                # Общий запрос о нескольких блюдах
                return {
                    'intent_info': intent_info,
                    'filtered_docs': docs[:5],
                    'search_strategy': 'multi_dish_general'
                }

        else:
            # Общий запрос
            return {
                'intent_info': intent_info,
                'filtered_docs': docs[:5],
                'search_strategy': 'general'
            }
