from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import os
from logging import getLogger
import asyncio


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
from rag.reranker import RecipeReranker


logger = getLogger(__name__)
logger.setLevel("DEBUG")


def _to_list_of_dicts(obj) -> Optional[List[Dict[str, Any]]]:
    """
    Попытка привести объект к списку словарей.
    Поддерживаются: list[dict], dict (ключ -> list), HF Dataset/DatasetDict, pandas DataFrame и т.д.
    Возвращает None если не удалось привести.
    """
    if obj is None:
        return None

    # already list of dicts?
    if isinstance(obj, list):
        return obj

    # dict of lists -> convert by zipping
    if isinstance(obj, dict):
        # если словарь вида {'train': [...]}
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        # если dict of lists (columns) -> zip to records
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

    # objects with to_pandas / to_dict / to_list behaviour (HF Dataset, pandas.DataFrame)
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
    Полностью асинхронная реализация через async/await.
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
        self.reranker: Optional[RecipeReranker] = None
        self.documents: List[Dict[str, Any]] = []

        logger.info("✅ RAG пайплайн создан")

    async def setup_embeddings(self):
        """
        Настраивает компонент векторизации (асинхронно).
        """
        logger.info("\n📝 Шаг 1: Настройка векторизации")
        # Выполняем в executor, так как инициализация модели блокирующая
        loop = asyncio.get_event_loop()
        self.embedder = await loop.run_in_executor(
            None,
            RecipeEmbedder,
            "sentence-transformers/distiluse-base-multilingual-cased"
        )

    async def load_and_process_data(self, max_recipes: int = None):
        """
        Асинхронно загружает и обрабатывает данные рецептов.

        Args:
            max_recipes: Ограничение количества рецептов для тестирования
        """
        logger.info(f"\n📂 Шаг 2: Загрузка данных (лимит: {max_recipes or 'без ограничений'})")

        # compute processed file path at repo root /data/processed_recipes.json
        processed_file = Path(__file__).resolve().parents[2] / "data" / "processed_recipes.json"

        # 1) Попытка загрузить готовый processed файл (если он есть)
        processed = None
        if processed_file.exists():
            try:
                logger.info(f"📋 Найден файл с обработанными данными: {processed_file}")
                loop = asyncio.get_event_loop()
                processed = await loop.run_in_executor(
                    None,
                    load_processed_data,
                    str(processed_file)
                )
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при загрузке processed_file: {e}")
                processed = None

        # 2) Если processed пуст или None -> загрузим сырые данные и обработаем
        if not processed:
            logger.info("ℹ️ processed data отсутствуют/пусты — загружаем и обрабатываем исходные данные...")
            hf_token = os.getenv("HF_TOKEN")

            # Вызов load_povarenok_data асинхронно
            loop = asyncio.get_event_loop()
            raw_recipes = None

            try:
                raw_recipes = await loop.run_in_executor(
                    None,
                    lambda: load_povarenok_data(max_recipes=max_recipes, use_auth_token=hf_token)
                )
            except TypeError:
                try:
                    raw_recipes = await loop.run_in_executor(
                        None,
                        lambda: load_povarenok_data("rogozinushka/povarenok-recipes", max_recipes=max_recipes, use_auth_token=hf_token)
                    )
                except TypeError:
                    try:
                        raw_recipes = await loop.run_in_executor(
                            None,
                            load_povarenok_data
                        )
                    except Exception as e:
                        logger.error(f"⚠️ Не удалось вызвать load_povarenok_data автоматически: {e}")
                        raw_recipes = None
                except Exception as e:
                    logger.error(f"⚠️ Ошибка при загрузке данных (second attempt): {e}")
            except Exception as e:
                logger.error(f"⚠️ Ошибка при загрузке данных (first attempt): {e}")

            if not raw_recipes:
                raise RuntimeError("Не удалось загрузить исходные рецепты (raw_recipes пустые). Проверьте доступность датасета или HF_TOKEN.")

            logger.info("⚙️ Обрабатываем сырые данные в документы...")
            try:
                try:
                    processed = await loop.run_in_executor(
                        None,
                        prepare_documents,
                        raw_recipes,
                        max_recipes
                    )
                except TypeError:
                    processed = await loop.run_in_executor(
                        None,
                        prepare_documents,
                        raw_recipes
                    )
            except Exception as e:
                raise RuntimeError(f"Ошибка при prepare_documents: {e}")

            if not processed:
                raise RuntimeError("prepare_documents вернул пустой результат")

            # Сохраняем обработанные данные асинхронно
            try:
                try:
                    await loop.run_in_executor(
                        None,
                        save_processed_data,
                        processed,
                        str(processed_file)
                    )
                except TypeError:
                    await loop.run_in_executor(
                        None,
                        save_processed_data,
                        str(processed_file),
                        processed
                    )
                logger.info(f"✅ Обработанные данные сохранены в {processed_file}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить processed data на диск: {e}")

        # 3) Нормализуем processed в список словарей
        normalized = _to_list_of_dicts(processed)
        if normalized is None:
            if not processed_file.exists():
                raise RuntimeError("Processed data не удалось преобразовать в список словарей и кэш отсутствует.")
            else:
                try:
                    loop = asyncio.get_event_loop()
                    reloaded = await loop.run_in_executor(
                        None,
                        load_processed_data,
                        str(processed_file)
                    )
                    normalized = _to_list_of_dicts(reloaded)
                except Exception:
                    normalized = None

            if normalized is None:
                raise RuntimeError(f"Не удалось привести processed data к списку документов. Тип исходного объекта: {type(processed)}")

        # применяем лимит, если нужно
        if max_recipes and len(normalized) > max_recipes:
            normalized = normalized[:max_recipes]
            logger.info(f"🔬 Ограничиваем до {max_recipes} рецептов")

        self.documents = normalized
        logger.info(f"✅ Готово к работе с {len(self.documents)} рецептами")

    async def build_vector_index(self, force_rebuild: bool = False):
        """
        Асинхронно строит векторный индекс FAISS.

        Args:
            force_rebuild: Принудительное пересоздание индекса
        """
        logger.info("\n🗄️ Шаг 3: Построение векторного индекса")

        if not self.embedder:
            raise ValueError("Сначала настройте векторизацию с помощью setup_embeddings()")

        if not self.documents or len(self.documents) == 0:
            raise ValueError("Сначала загрузите и обработайте данные с помощью load_and_process_data(); найдено 0 документов")

        # Инициализируем векторное хранилище
        self.vector_store = FAISSVectorStore(self.embedder.embedding_dim)

        # Проверяем существующий индекс
        index_path = Path(__file__).resolve().parents[2] / "data" / "faiss_index"

        if index_path.exists() and not force_rebuild:
            logger.info("📂 Найден существующий индекс, загружаем...")
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.vector_store.load,
                    str(index_path)
                )
                return
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки индекса: {e}")
                logger.info("🔨 Пересоздаем индекс...")

        # Создаем новый индекс асинхронно
        logger.info("🔄 Векторизуем документы...")
        texts = [doc.get("full_text", "") for doc in self.documents]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.embedder.encode_texts,
            texts
        )

        logger.info("🏗️ Строим FAISS индекс...")
        await loop.run_in_executor(
            None,
            self.vector_store.build_index,
            embeddings,
            self.documents
        )

        logger.info("💾 Сохраняем индекс...")
        await loop.run_in_executor(
            None,
            self.vector_store.save,
            str(index_path)
        )

        logger.info("✅ Векторный индекс готов")

    async def setup_hybrid_search(self):
        """
        Асинхронно настраивает гибридный поиск.
        """
        logger.info("\n🔍 Шаг 4: Настройка гибридного поиска")

        if not self.vector_store or not self.documents:
            raise ValueError("Сначала постройте векторный индекс")

        loop = asyncio.get_event_loop()
        self.hybrid_search = await loop.run_in_executor(
            None,
            HybridSearch,
            self.vector_store,
            self.documents
        )
        logger.info("✅ Гибридный поиск готов")

    async def setup_reranker(self):
        """
        Асинхронно настраивает реранкер для улучшения качества поиска.
        """
        logger.info("\n🎯 Шаг 4.5: Настройка реранкера")

        try:
            loop = asyncio.get_event_loop()
            self.reranker = await loop.run_in_executor(
                None,
                RecipeReranker,
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("✅ Реранкер готов")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить реранкер: {e}. Продолжаем без реранкера.")
            self.reranker = None

    async def setup_llm(self):
        """
        Асинхронно настраивает языковую модель.
        """
        logger.info("\n🤖 Шаг 5: Настройка языковой модели")

        loop = asyncio.get_event_loop()
        self.llm = await loop.run_in_executor(
            None,
            RecipeLLM,
            "Qwen/Qwen2.5-1.5B-Instruct"
        )

        # Выводим информацию о модели
        model_info = self.llm.get_model_info()
        logger.info(f"📋 Модель: {model_info.get('model_name', 'unknown')}")
        logger.info(f"📋 Параметры: ~{model_info.get('num_parameters', 0):,}")
        logger.info("✅ LLM готова")

    async def initialize_full_pipeline(self, max_recipes: int = None, force_rebuild: bool = False):
        """
        Асинхронно инициализирует весь пайплайн за один вызов.

        Args:
            max_recipes: Ограничение количества рецептов
            force_rebuild: Принудительное пересоздание индекса
        """
        logger.info("🚀 Полная инициализация RAG пайплайна")

        start_time = time.time()

        # Последовательно настраиваем все компоненты
        await self.setup_embeddings()
        await self.load_and_process_data(max_recipes)
        await self.build_vector_index(force_rebuild)
        await self.setup_hybrid_search()
        await self.setup_reranker()
        await self.setup_llm()

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Полная инициализация завершена за {elapsed:.1f}с")
        logger.info("🎉 RAG пайплайн готов к работе!")

    async def search_recipes(self, query: str, k: int = 5) -> List[tuple]:
        """
        Асинхронно выполняет поиск рецептов по запросу с реранкингом.
        """
        if not self.hybrid_search or not self.embedder:
            raise ValueError("Пайплайн не инициализирован")

        # Векторизуем запрос асинхронно
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            self.embedder.encode_query,
            query
        )

        # Выполняем гибридный поиск с увеличенным k для реранкинга
        initial_k = min(k * 3, 20)
        raw_results = await loop.run_in_executor(
            None,
            self.hybrid_search.hybrid_search,
            query,
            query_embedding,
            initial_k,
            0.6
        )

        # Применяем реранкинг если доступен
        if self.reranker and raw_results:
            # Извлекаем документы из результатов
            docs = [doc for doc, _ in raw_results]

            # Фильтруем по намерению асинхронно
            filter_result = await loop.run_in_executor(
                None,
                self.reranker.filter_by_intent,
                query,
                docs
            )
            filtered_docs = filter_result['filtered_docs']
            intent_info = filter_result['intent_info']

            logger.info(f"🎯 Определено намерение: {intent_info['intent']} (confidence: {intent_info['confidence']:.2f})")

            # Применяем реранкинг асинхронно
            if filtered_docs:
                reranked_docs = await loop.run_in_executor(
                    None,
                    self.reranker.rerank,
                    query,
                    filtered_docs,
                    k
                )
                # Возвращаем в формате (doc, score)
                return [(doc, i) for i, doc in enumerate(reranked_docs)]
            else:
                logger.warning("⚠️ Фильтрация по намерению не вернула результатов")

        # Fallback: возвращаем исходные результаты
        return raw_results[:k]

    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Асинхронно отвечает на вопрос пользователя о рецептах.
        """
        if not all([self.hybrid_search, self.embedder, self.llm]):
            raise ValueError("Пайплайн не полностью инициализирован")

        logger.info(f"\n❓ Вопрос: {question}")

        start_time = time.time()

        # Выполняем поиск релевантных рецептов с реранкингом асинхронно
        search_results = await self.search_recipes(question, k=3)

        search_time = time.time() - start_time

        # Генерируем ответ асинхронно
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.llm.generate_response,
            question,
            search_results
        )

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

        logger.info(f"💬 Ответ: {response}")
        logger.info(f"⏱️ Время: {total_time:.3f}с (поиск: {search_time:.3f}с)")

        return result
