from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import os
from loguru import logger
import asyncio
import concurrent.futures

from data.data_loader import (
    load_povarenok_data,
    prepare_documents,
    save_processed_data,
    load_processed_data,
)
from embeddings.embeddings import RecipeEmbedder
from store.vector_store import FAISSVectorStore
from rag.hybrid_search import HybridSearch
from llm.llm import RecipeLLM
from rag.reranker import RecipeReranker  # Новый импорт




def _to_list_of_dicts(obj) -> Optional[List[Dict[str, Any]]]:
    """
    Попытка привести объект к списку словарей.
    Поддерживаются: list[dict], dict (ключ -> list), HF Dataset/DatasetDict, pandas DataFrame и т.д.
    Возвращает None если не удалось привести.
    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        list_values = [v for v in obj.values() if isinstance(v, list)]
        if list_values:
            length = len(list_values[0])
            try:
                keys = list(obj.keys())
                records = []
                for i in range(length):
                    rec = {}
                    for k in keys:
                        val = obj[k]
                        rec[k] = val[i] if isinstance(val, list) and i < len(val) else None
                    records.append(rec)
                return records
            except Exception:
                pass
        return None

    try:
        if hasattr(obj, "to_dict"):
            d = obj.to_dict()
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return v
                list_values = [v for v in d.values() if isinstance(v, list)]
                if list_values:
                    length = len(list_values[0])
                    keys = list(d.keys())
                    records = []
                    for i in range(length):
                        rec = {}
                        for k in keys:
                            val = d[k]
                            rec[k] = val[i] if isinstance(val, list) and i < len(val) else None
                        records.append(rec)
                    return records

        if hasattr(obj, "to_records") or hasattr(obj, "to_dict"):
            try:
                recs = obj.to_dict(orient="records")
                if isinstance(recs, list):
                    return recs
            except Exception:
                pass

        try:
            lst = list(obj)
            if lst and isinstance(lst[0], dict):
                return lst
        except Exception:
            pass
    except Exception:
        pass

    return None


class RecipeRAGPipeline:
    """
    Основной класс RAG пайплайна для поиска рецептов.
    Объединяет все компоненты: загрузку данных, векторизацию, поиск и генерацию.
    Поддерживает автоматическое определение sync/async контекста.
    Включает реранкер для улучшения релевантности результатов.
    """

    def __init__(self):
        """
        Инициализирует RAG пайплайн.
        """
        logger.info("🚀 Инициализируем RAG пайплайн для рецептов")

        # Компоненты пайплайна
        self.embedder: Optional[RecipeEmbedder] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        self.hybrid_search: Optional[HybridSearch] = None
        self.llm: Optional[RecipeLLM] = None
        self.reranker: Optional[RecipeReranker] = None  # Новый компонент
        self.documents: List[Dict[str, Any]] = []

        logger.info("✅ RAG пайплайн создан")

    def setup_embeddings(self):
        """
        Настраивает компонент векторизации.
        """
        logger.info("\nШаг 1: Настройка векторизации")
        self.embedder = RecipeEmbedder()

    def _load_and_process_data_sync(self, max_recipes: int = None):
        """
        Внутренний синхронный метод загрузки и обработки данных.
        """
        logger.info(f"\n Шаг 2: Загрузка данных (лимит: {max_recipes or 'без ограничений'})")

        # compute processed file path at repo root /data/processed_recipes.json
        processed_file = Path(__file__).resolve().parents[2] / "data" / "processed_recipes.json"

        # 1) Попытка загрузить готовый processed файл (если он есть)
        processed = None
        if processed_file.exists():
            try:
                logger.info(f" Найден файл с обработанными данными: {processed_file}")
                processed = load_processed_data(str(processed_file))
            except Exception as e:
                logger.warning(f"Ошибка при загрузке processed_file: {e}")
                processed = None

        if not processed:
            logger.info("ℹprocessed data отсутствуют/пусты — загружаем и обрабатываем исходные данные...")
            hf_token = os.getenv("HF_TOKEN")

            raw_recipes = None
            try:
                raw_recipes = load_povarenok_data(max_recipes=max_recipes, use_auth_token=hf_token)
            except TypeError:
                try:
                    raw_recipes = load_povarenok_data("rogozinushka/povarenok-recipes", max_recipes=max_recipes, use_auth_token=hf_token)
                except TypeError:
                    try:
                        raw_recipes = load_povarenok_data()
                    except Exception as e:
                        logger.error(f"Не удалось вызвать load_povarenok_data автоматически: {e}")
                        raw_recipes = None
                except Exception as e:
                    logger.error(f"Ошибка при загрузке данных (second attempt): {e}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных (first attempt): {e}")

            if not raw_recipes:
                raise RuntimeError("Не удалось загрузить исходные рецепты (raw_recipes пустые). Проверьте доступность датасета или HF_TOKEN.")

            logger.info("Обрабатываем сырые данные в документы...")
            try:
                try:
                    processed = prepare_documents(raw_recipes, max_recipes)
                except TypeError:
                    processed = prepare_documents(raw_recipes)
            except Exception as e:
                raise RuntimeError(f"Ошибка при prepare_documents: {e}")

            if not processed:
                raise RuntimeError("prepare_documents вернул пустой результат")

            try:
                try:
                    save_processed_data(processed, str(processed_file))
                except TypeError:
                    save_processed_data(str(processed_file), processed)
                logger.info(f"Обработанные данные сохранены в {processed_file}")
            except Exception as e:
                logger.warning(f" Не удалось сохранить processed data на диск: {e}")

        normalized = _to_list_of_dicts(processed)
        if normalized is None:
            if not processed_file.exists():
                raise RuntimeError("Processed data не удалось преобразовать в список словарей и кэш отсутствует.")
            else:
                try:
                    reloaded = load_processed_data(str(processed_file))
                    normalized = _to_list_of_dicts(reloaded)
                except Exception:
                    normalized = None

            if normalized is None:
                raise RuntimeError(f"Не удалось привести processed data к списку документов. Тип исходного объекта: {type(processed)}")

        if max_recipes and len(normalized) > max_recipes:
            normalized = normalized[:max_recipes]
            logger.info(f" Ограничиваем до {max_recipes} рецептов")

        self.documents = normalized
        logger.info(f"Готово к работе с {len(self.documents)} рецептами")

    def load_and_process_data(self, max_recipes: int = None):
        """
        Загружает и обрабатывает данные рецептов.
        Автоматически определяет sync/async контекст.

        Args:
            max_recipes: Ограничение количества рецептов для тестирования
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._load_and_process_data_sync, max_recipes)
                return future.result()
        except RuntimeError:
            return self._load_and_process_data_sync(max_recipes)

    def _build_vector_index_sync(self, force_rebuild: bool = False):
        """
        Внутренний синхронный метод построения векторного индекса.
        """
        logger.info("\n Шаг 3: Построение векторного индекса")

        if not self.embedder:
            raise ValueError("Сначала настройте векторизацию с помощью setup_embeddings()")

        if not self.documents or len(self.documents) == 0:
            raise ValueError("Сначала загрузите и обработайте данные с помощью load_and_process_data(); найдено 0 документов")

        self.vector_store = FAISSVectorStore(self.embedder.embedding_dim)

        index_path = Path(__file__).resolve().parents[2] / "data" / "faiss_index"

        if index_path.exists() and not force_rebuild:
            logger.info("Найден существующий индекс, загружаем...")
            try:
                self.vector_store.load(str(index_path))
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки индекса: {e}")
                logger.info(" Пересоздаем индекс...")

        # Создаем новый индекс
        logger.info("Векторизуем документы...")
        texts = [doc.get("full_text", "") for doc in self.documents]
        embeddings = self.embedder.encode_texts(texts)

        logger.info(" Строим FAISS индекс...")
        self.vector_store.build_index(embeddings, self.documents)

        logger.info("Сохраняем индекс...")
        self.vector_store.save(str(index_path))

        logger.info("Векторный индекс готов")

    def build_vector_index(self, force_rebuild: bool = False):
        """
        Строит векторный индекс FAISS.
        Автоматически определяет sync/async контекст.

        Args:
            force_rebuild: Принудительное пересоздание индекса
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._build_vector_index_sync, force_rebuild)
                return future.result()
        except RuntimeError:
            return self._build_vector_index_sync(force_rebuild)

    def setup_hybrid_search(self):
        """
        Настраивает гибридный поиск.
        """
        logger.info("\n Шаг 4: Настройка гибридного поиска")

        if not self.vector_store or not self.documents:
            raise ValueError("Сначала постройте векторный индекс")

        self.hybrid_search = HybridSearch(self.vector_store, self.documents)
        logger.info("Гибридный поиск готов")

    def setup_reranker(self):
        """
        Настраивает реранкер для улучшения качества поиска.
        """
        logger.info("\n Шаг 4.5: Настройка реранкера")

        try:
            self.reranker = RecipeReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Реранкер готов")
        except Exception as e:
            logger.warning(f"Не удалось загрузить реранкер: {e}. Продолжаем без реранкера.")
            self.reranker = None

    def setup_llm(self):
        """
        Настраивает языковую модель.
        """
        logger.info("\nШаг 5: Настройка языковой модели")

        self.llm = RecipeLLM("Qwen/Qwen2.5-1.5B-Instruct")

        # Выводим информацию о модели
        model_info = self.llm.get_model_info()
        logger.info(f"Модель: {model_info.get('model_name', 'unknown')}")
        logger.info(f"Параметры: ~{model_info.get('num_parameters', 0):,}")
        logger.info("LLM готова")

    def _initialize_full_pipeline_sync(self, max_recipes: int = None, force_rebuild: bool = False):
        """
        Внутренний синхронный метод полной инициализации.
        """
        logger.info("Полная инициализация RAG пайплайна")

        start_time = time.time()

        # Последовательно настраиваем все компоненты
        self.setup_embeddings()
        self.load_and_process_data(max_recipes)
        self.build_vector_index(force_rebuild)
        self.setup_hybrid_search()
        self.setup_reranker()  
        self.setup_llm()

        elapsed = time.time() - start_time
        logger.info(f"\nПолная инициализация завершена за {elapsed:.1f}с")
        logger.info("RAG пайплайн готов к работе!")

    def initialize_full_pipeline(self, max_recipes: int = None, force_rebuild: bool = False):
        """
        Инициализирует весь пайплайн за один вызов.
        Автоматически определяет sync/async контекст.

        Args:
            max_recipes: Ограничение количества рецептов
            force_rebuild: Принудительное пересоздание индекса
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._initialize_full_pipeline_sync, max_recipes, force_rebuild)
                return future.result()
        except RuntimeError:
            return self._initialize_full_pipeline_sync(max_recipes, force_rebuild)

    def _search_recipes_sync(self, query: str, k: int = 5) -> List[tuple]:
        """
        Внутренний синхронный метод поиска рецептов с реранкингом.
        """
        if not self.hybrid_search or not self.embedder:
            raise ValueError("Пайплайн не инициализирован")

        query_embedding = self.embedder.encode_query(query)

        initial_k = min(k * 3, 20)  # Берем больше кандидатов для реранкинга
        raw_results = self.hybrid_search.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=initial_k,
            alpha=0.6,
        )

        if self.reranker and raw_results:
            docs = [doc for doc, _ in raw_results]

            filter_result = self.reranker.filter_by_intent(query, docs)
            filtered_docs = filter_result['filtered_docs']
            intent_info = filter_result['intent_info']

            logger.info(f" Определено намерение: {intent_info['intent']} (confidence: {intent_info['confidence']:.2f})")

            # Применяем реранкинг
            if filtered_docs:
                reranked_docs = self.reranker.rerank(query, filtered_docs, top_k=k)
                # Возвращаем в формате (doc, score)
                return [(doc, i) for i, doc in enumerate(reranked_docs)]
            else:
                logger.warning(" Фильтрация по намерению не вернула результатов")

        # Fallback: возвращаем исходные результаты
        return raw_results[:k]

    def search_recipes(self, query: str, k: int = 5) -> List[tuple]:
        """
        Выполняет поиск рецептов по запросу с реранкингом.
        Автоматически определяет sync/async контекст.
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._search_recipes_sync, query, k)
                return future.result()
        except RuntimeError:
            return self._search_recipes_sync(query, k)

    def _ask_sync(self, question: str) -> Dict[str, Any]:
        """
        Внутренний синхронный метод ответа на вопрос.
        """
        if not all([self.hybrid_search, self.embedder, self.llm]):
            raise ValueError("Пайплайн не полностью инициализирован")

        logger.info(f"\nВопрос: {question}")

        start_time = time.time()

        search_results = self._search_recipes_sync(question, k=5)

        search_time = time.time() - start_time

        response = self.llm.generate_response(question, search_results)

        total_time = time.time() - start_time

        # Формируем результат
        result = {
            "question": question,
            "answer": response,
            "found_recipes": len(search_results) if search_results is not None else 0,
            "search_results": [
                {
                    "name": doc.get("name", ""),
                    "ingredients": doc.get("ingredients_text", ""),
                    "url": doc.get("url", ""),
                    "relevance_score": float(score) if isinstance(score, (int, float)) else 0.0,
                }
                for doc, score in (search_results or [])[:3]
            ],
            "timing": {"search_time": round(search_time, 3), "total_time": round(total_time, 3)},
        }

        logger.info(f" Ответ: {response}")
        logger.info(f"Время: {total_time:.3f}с (поиск: {search_time:.3f}с)")

        return result

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Отвечает на вопрос пользователя о рецептах.
        Автоматически определяет sync/async контекст.
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._ask_sync, question)
                return future.result()
        except RuntimeError:
            return self._ask_sync(question)
