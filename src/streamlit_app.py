"""
Streamlit UI для асинхронного RAG-пайплайна рецептов.
Запуск: streamlit run streamlit_app.py --server.port 8501
"""

import os
import traceback
import asyncio
from typing import Optional
import streamlit as st
from rag.rag_pipeline import RecipeRAGPipeline
import logging


st.set_page_config(page_title="Recipes RAG", layout="centered")


# ----- Async helper для Streamlit -----
def run_async(coro):
    """
    Запускает асинхронную функцию в синхронном контексте Streamlit.
    Создает новый event loop если необходимо.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если loop уже запущен, создаём новый
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


# ----- Pipeline helper -----
@st.cache_resource
def get_pipeline():
    """Возвращает экземпляр пайплайна (кэшируется)."""
    return RecipeRAGPipeline()


def ensure_init_state():
    """Инициализирует состояние в session_state."""
    if "pipeline_initialized" not in st.session_state:
        st.session_state["pipeline_initialized"] = False
    if "init_error" not in st.session_state:
        st.session_state["init_error"] = None
    if "auto_init_done" not in st.session_state:
        st.session_state["auto_init_done"] = False


def initialize_pipeline():
    """Инициализирует пайплайн с UI feedback (синхронная обертка)."""
    pipeline = get_pipeline()

    try:
        progress_bar = st.progress(0, text="Настройка эмбеддера...")
        run_async(pipeline.setup_embeddings())

        progress_bar.progress(20, text="Загрузка данных...")
        run_async(pipeline.load_and_process_data(max_recipes=200))

        progress_bar.progress(40, text="Построение векторного индекса...")
        run_async(pipeline.build_vector_index(force_rebuild=False))

        progress_bar.progress(60, text="Настройка гибридного поиска...")
        run_async(pipeline.setup_hybrid_search())

        progress_bar.progress(70, text="Настройка реранкера...")
        run_async(pipeline.setup_reranker())

        progress_bar.progress(90, text="Загрузка LLM...")
        run_async(pipeline.setup_llm())

        progress_bar.progress(100, text="✅ Инициализация завершена!")

        st.session_state["pipeline_initialized"] = True
        st.session_state["init_error"] = None
        st.success("🎉 Система готова к работе!")

    except Exception as e:
        st.session_state["init_error"] = traceback.format_exc()
        st.error(f"❌ Ошибка инициализации: {str(e)}")
        with st.expander("Подробности ошибки"):
            st.code(st.session_state["init_error"])


# ----- Main UI -----
ensure_init_state()

st.title("🍽️ Поиск рецептов с RAG")

# Автоматическая инициализация при первом запуске
if not st.session_state.get("auto_init_done"):
    st.info("⏳ Инициализация системы (это произойдёт один раз при запуске)...")
    initialize_pipeline()
    st.session_state["auto_init_done"] = True
    st.rerun()

pipeline = get_pipeline()
pipeline_ready = (
    st.session_state.get("pipeline_initialized", False) and
    hasattr(pipeline, 'embedder') and pipeline.embedder and
    hasattr(pipeline, 'hybrid_search') and pipeline.hybrid_search and
    hasattr(pipeline, 'llm') and pipeline.llm
)

st.markdown("---")

query = st.text_input(
    "Введите запрос:",
    value="Как приготовить борщ?",
    placeholder="Например: рецепт пиццы с грибами",
    disabled=not pipeline_ready
)

search_button = st.button(
    "🔎 Найти рецепты",
    type="primary",
    disabled=not pipeline_ready,
    use_container_width=True
)

if not pipeline_ready:
    st.warning("⚠️ Система ещё инициализируется. Пожалуйста, подождите...")

if search_button and pipeline_ready:
    if query.strip():
        with st.spinner("🔍 Обработка запроса..."):
            try:
                # Асинхронный вызов через run_async
                result = run_async(pipeline.ask(query))

                # Ответ LLM
                st.markdown("### 💬 Ответ системы")
                st.write(result.get("answer", ""))

                # Найденные рецепты
                st.markdown("### 📚 Найденные рецепты")
                search_results = result.get("search_results", [])

                if search_results:
                    for i, item in enumerate(search_results, start=1):
                        with st.expander(
                            f"{i}. {item.get('name', '(без названия)')} — релевантность: {item.get('relevance_score', 0):.3f}"
                        ):
                            st.markdown(f"**Ингредиенты:**")
                            st.text(item.get('ingredients', 'Не указаны'))

                            if item.get('url'):
                                st.markdown(f"**Ссылка:** [{item.get('url')}]({item.get('url')})")
                else:
                    st.info("Рецепты не найдены")

                # Метаданные
                with st.expander("⏱️ Время обработки"):
                    timing = result.get("timing", {})
                    col1, col2 = st.columns(2)
                    col1.metric("Общее время", f"{timing.get('total_time', 0):.2f}с")
                    col2.metric("Время поиска", f"{timing.get('search_time', 0):.2f}с")

            except Exception as e:
                st.error(f"❌ Ошибка при выполнении запроса: {str(e)}")
                with st.expander("Подробности ошибки"):
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Введите запрос")

# Статус в футере
st.markdown("---")
with st.expander("ℹ️ Информация о системе"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if hasattr(pipeline, 'documents') and pipeline.documents:
            st.metric("Документов", len(pipeline.documents))
        else:
            st.metric("Документов", "0")

    with col2:
        if hasattr(pipeline, 'vector_store') and pipeline.vector_store:
            st.metric("FAISS индекс", "✅")
        else:
            st.metric("FAISS индекс", "❌")

    with col3:
        if hasattr(pipeline, 'reranker') and pipeline.reranker:
            st.metric("Реранкер", "✅")
        else:
            st.metric("Реранкер", "❌")

    with col4:
        if hasattr(pipeline, 'llm') and pipeline.llm:
            st.metric("LLM", "✅")
        else:
            st.metric("LLM", "❌")

# Кнопка переинициализации (опционально)
with st.sidebar:
    st.header("⚙️ Настройки")

    if st.button("🔄 Переинициализировать систему", use_container_width=True):
        st.session_state["pipeline_initialized"] = False
        st.session_state["auto_init_done"] = False
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Статус компонентов")

    status_items = [
        ("Эмбеддер", hasattr(pipeline, 'embedder') and pipeline.embedder),
        ("Векторный индекс", hasattr(pipeline, 'vector_store') and pipeline.vector_store),
        ("Гибридный поиск", hasattr(pipeline, 'hybrid_search') and pipeline.hybrid_search),
        ("Реранкер", hasattr(pipeline, 'reranker') and pipeline.reranker),
        ("LLM", hasattr(pipeline, 'llm') and pipeline.llm),
    ]

    for name, status in status_items:
        st.markdown(f"{'✅' if status else '❌'} {name}")
