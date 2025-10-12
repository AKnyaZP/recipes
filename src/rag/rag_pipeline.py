
# ...existing code...
"""
Основной RAG пайплайн - связывает все компоненты системы.
"""

from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import os
from logging import getLogger

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
        # пытаемся найти первую подходящую запись со списком элементов словарей
        # или ключ со списком длинной > 0
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
        # HF Dataset / DatasetDict -> try iterate or use to_dict
        if hasattr(obj, "to_dict"):
            d = obj.to_dict()
            # if to_dict returns dict of lists -> convert
            if isinstance(d, dict):
                # DatasetDict returns {split: Dataset}, handle that
                # if values are lists -> zip to records
                for v in d.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return v
                # if d itself is columns -> zip
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

        # try pandas
        if hasattr(obj, "to_records") or hasattr(obj, "to_dict"):
            try:
                # pandas.DataFrame -> to_dict(orient='records')
                recs = obj.to_dict(orient="records")
                if isinstance(recs, list):
                    return recs
            except Exception:
                pass

        # fallback: try to iterate and coerce to list
        try:
            lst = list(obj)
            # ensure elements are dicts
            if lst and isinstance(lst[0], dict):
                return lst
            # if elements are tuples/lists and correspond to columns, try to convert
        except Exception:
            pass
    except Exception:
        pass

    return None


class RecipeRAGPipeline:
    """
    Основной класс RAG пайплайна для поиска рецептов.
    Объединяет все компоненты: загрузку данных, векторизацию, поиск и генерацию.
    """

    def __init__(self):
        """
        Инициализирует RAG пайплайн.
        """
        print("🚀 Инициализируем RAG пайплайн для рецептов")

        # Компоненты пайплайна
        self.embedder: Optional[RecipeEmbedder] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        self.hybrid_search: Optional[HybridSearch] = None
        self.llm: Optional[RecipeLLM] = None
        self.documents: List[Dict[str, Any]] = []

        print("✅ RAG пайплайн создан")

    def setup_embeddings(self):
        """
        Настраивает компонент векторизации.
        """
        print("\n📝 Шаг 1: Настройка векторизации")
        self.embedder = RecipeEmbedder("sentence-transformers/distiluse-base-multilingual-cased")

    def load_and_process_data(self, max_recipes: int = None):
        """
        Загружает и обрабатывает данные рецептов.

        Args:
            max_recipes: Ограничение количества рецептов для тестирования
        """
        print(f"\n📂 Шаг 2: Загрузка данных (лимит: {max_recipes or 'без ограничений'})")

        # compute processed file path at repo root /data/processed_recipes.json
        processed_file = Path(__file__).resolve().parents[2] / "data" / "processed_recipes.json"

        # 1) Попытка загрузить готовый processed файл (если он есть)
        processed = None
        if processed_file.exists():
            try:
                print(f"📋 Найден файл с обработанными данными: {processed_file}")
                processed = load_processed_data(str(processed_file))
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке processed_file: {e}")
                processed = None

        # 2) Если processed пуст или None -> загрузим сырые данные и обработаем
        if not processed:
            print("ℹ️ processed data отсутствуют/пусты — загружаем и обрабатываем исходные данные...")
            hf_token = os.getenv("HF_TOKEN")

            # Вызов load_povarenok_data: поддерживаем несколько возможных сигнатур
            raw_recipes = None
            try:
                # try signature with max_recipes kwarg
                raw_recipes = load_povarenok_data(max_recipes=max_recipes, use_auth_token=hf_token)
            except TypeError:
                try:
                    # common alternative: load_povarenok_data(dataset_id, max_recipes=...)
                    raw_recipes = load_povarenok_data("rogozinushka/povarenok-recipes", max_recipes=max_recipes, use_auth_token=hf_token)
                except TypeError:
                    try:
                        # try simple call without args
                        raw_recipes = load_povarenok_data()
                    except Exception as e:
                        print(f"⚠️ Не удалось вызвать load_povarenok_data автоматически: {e}")
                        raw_recipes = None
                except Exception as e:
                    print(f"⚠️ Ошибка при загрузке данных (second attempt): {e}")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке данных (first attempt): {e}")

            if not raw_recipes:
                raise RuntimeError("Не удалось загрузить исходные рецепты (raw_recipes пустые). Проверьте доступность датасета или HF_TOKEN.")

            print("⚙️ Обрабатываем сырые данные в документы...")
            try:
                # prepare_documents может принимать параметр max_recipes
                try:
                    processed = prepare_documents(raw_recipes, max_recipes)
                except TypeError:
                    processed = prepare_documents(raw_recipes)
            except Exception as e:
                raise RuntimeError(f"Ошибка при prepare_documents: {e}")

            if not processed:
                raise RuntimeError("prepare_documents вернул пустой результат")

            # Сохраняем обработанные данные
            try:
                # try signature save_processed_data(processed, path) or save_processed_data(path, processed)
                try:
                    save_processed_data(processed, str(processed_file))
                except TypeError:
                    save_processed_data(str(processed_file), processed)
                print(f"✅ Обработанные данные сохранены в {processed_file}")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить processed data на диск: {e}")

        # 3) Нормализуем processed в список словарей
        normalized = _to_list_of_dicts(processed)
        if normalized is None:
            # если обработанные данные всё ещё не в нужном формате, пробуем дополнительно load_processed_data если не делали ранее
            if not processed_file.exists():
                raise RuntimeError("Processed data не удалось преобразовать в список словарей и кэш отсутствует.")
            else:
                # попытка ещё раз загрузить через load_processed_data
                try:
                    reloaded = load_processed_data(str(processed_file))
                    normalized = _to_list_of_dicts(reloaded)
                except Exception:
                    normalized = None

            if normalized is None:
                raise RuntimeError(f"Не удалось привести processed data к списку документов. Тип исходного объекта: {type(processed)}")

        # применяем лимит, если нужно
        if max_recipes and len(normalized) > max_recipes:
            normalized = normalized[:max_recipes]
            print(f"🔬 Ограничиваем до {max_recipes} рецептов")

        self.documents = normalized
        print(f"✅ Готово к работе с {len(self.documents)} рецептами")

    def build_vector_index(self, force_rebuild: bool = False):
        """
        Строит векторный индекс FAISS.

        Args:
            force_rebuild: Принудительное пересоздание индекса
        """
        print("\n🗄️ Шаг 3: Построение векторного индекса")

        if not self.embedder:
            raise ValueError("Сначала настройте векторизацию с помощью setup_embeddings()")

        if not self.documents or len(self.documents) == 0:
            raise ValueError("Сначала загрузите и обработайте данные с помощью load_and_process_data(); найдено 0 документов")

        # Инициализируем векторное хранилище
        self.vector_store = FAISSVectorStore(self.embedder.embedding_dim)

        # Проверяем существующий индекс
        index_path = Path(__file__).resolve().parents[2] / "data" / "faiss_index"

        if index_path.exists() and not force_rebuild:
            print("📂 Найден существующий индекс, загружаем...")
            try:
                self.vector_store.load(str(index_path))
                return
            except Exception as e:
                print(f"⚠️ Ошибка загрузки индекса: {e}")
                print("🔨 Пересоздаем индекс...")

        # Создаем новый индекс
        print("🔄 Векторизуем документы...")
        texts = [doc.get("full_text", "") for doc in self.documents]
        embeddings = self.embedder.encode_texts(texts)

        print("🏗️ Строим FAISS индекс...")
        self.vector_store.build_index(embeddings, self.documents)

        print("💾 Сохраняем индекс...")
        self.vector_store.save(str(index_path))

        print("✅ Векторный индекс готов")

    def setup_hybrid_search(self):
        """
        Настраивает гибридный поиск.
        """
        print("\n🔍 Шаг 4: Настройка гибридного поиска")

        if not self.vector_store or not self.documents:
            raise ValueError("Сначала постройте векторный индекс")

        self.hybrid_search = HybridSearch(self.vector_store, self.documents)
        print("✅ Гибридный поиск готов")

    def setup_llm(self):
        """
        Настраивает языковую модель.
        """
        print("\n🤖 Шаг 5: Настройка языковой модели")

        self.llm = RecipeLLM("Qwen/Qwen2.5-1.5B-Instruct")

        # Выводим информацию о модели
        model_info = self.llm.get_model_info()
        print(f"📋 Модель: {model_info.get('model_name', 'unknown')}")
        print(f"📋 Параметры: ~{model_info.get('num_parameters', 0):,}")
        print("✅ LLM готова")

    def initialize_full_pipeline(self, max_recipes: int = None, force_rebuild: bool = False):
        """
        Инициализирует весь пайплайн за один вызов.

        Args:
            max_recipes: Ограничение количества рецептов
            force_rebuild: Принудительное пересоздание индекса
        """
        print("🚀 Полная инициализация RAG пайплайна")

        start_time = time.time()

        # Последовательно настраиваем все компоненты
        self.setup_embeddings()
        self.load_and_process_data(max_recipes)
        self.build_vector_index(force_rebuild)
        self.setup_hybrid_search()
        self.setup_llm()

        elapsed = time.time() - start_time
        print(f"\n✅ Полная инициализация завершена за {elapsed:.1f}с")
        print("🎉 RAG пайплайн готов к работе!")

    def search_recipes(self, query: str, k: int = 1) -> List[tuple]:
        """
        Выполняет поиск рецептов по запросу.
        """
        if not self.hybrid_search or not self.embedder:
            raise ValueError("Пайплайн не инициализирован")

        # Векторизуем запрос
        query_embedding = self.embedder.encode_query(query)

        # Выполняем гибридный поиск
        results = self.hybrid_search.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=k,
            alpha=0.6,
        )

        return results

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Отвечает на вопрос пользователя о рецептах.
        """
        if not all([self.hybrid_search, self.embedder, self.llm]):
            raise ValueError("Пайплайн не полностью инициализирован")

        print(f"\n❓ Вопрос: {question}")

        start_time = time.time()

        # Выполняем поиск релевантных рецептов
        search_results = self.search_recipes(question, k=5)

        search_time = time.time() - start_time

        # Генерируем ответ
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
                    "relevance_score": float(score),
                }
                for doc, score in (search_results or [])[:3]
            ],
            "timing": {"search_time": round(search_time, 3), "total_time": round(total_time, 3)},
        }

        print(f"💬 Ответ: {response}")
        print(f"⏱️ Время: {total_time:.3f}с (поиск: {search_time:.3f}с)")

        return result

# ...existing code...
# if __name__ == "__main__":
#     # Демонстрация работы пайплайна
#     print("🧪 Демонстрация RAG пайплайна")
    
#     try:
#         # Создаем и инициализируем пайплайн
#         rag = RecipeRAGPipeline()
#         rag.initialize_full_pipeline(
#             max_recipes=100,  # Ограничиваем для быстрого тестирования
#             force_rebuild=False
#         )
        
#         # Примеры вопросов
#         test_questions = [
#             "Как приготовить борщ?",
#             "Рецепт салата с курицей",
#             "Что можно приготовить из картофеля?",
#             "Простой десерт с яблоками"
#         ]
        
#         print("\n" + "="*50)
#         print("🔍 ТЕСТИРОВАНИЕ RAG ПАЙПЛАЙНА")
#         print("="*50)
        
#         for question in test_questions:
#             result = rag.ask(question)
            
#             print(f"\n📋 Найдено рецептов: {result['found_recipes']}")
#             print("🥘 Топ рецептов:")
#             for i, recipe in enumerate(result['search_results'], 1):
#                 print(f"  {i}. {recipe['name']} (релевантность: {recipe['relevance_score']:.3f})")
            
#             print("-" * 50)
        
#         print("\n✅ Демонстрация завершена!")
        
#     except Exception as e:
#         print(f"❌ Ошибка: {e}")
#         print("Убедитесь что:")
#         print("1. Датасет recipes.json находится в data/raw/")
#         print("2. Установлены все зависимости")
#         print("3. Достаточно памяти для модели")