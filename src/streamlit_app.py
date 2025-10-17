"""
Streamlit UI для RAG-пайплайна рецептов.
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ - быстрая инициализация.
Запуск: streamlit run streamlit_app.py --server.port 8501
"""

import traceback
import streamlit as st
from rag.rag_pipeline import RecipeRAGPipeline

st.set_page_config(page_title="Recipes RAG", layout="centered")


# ----- Pipeline helper с кэшированием -----
@st.cache_resource(show_spinner="🚀 Инициализируем систему (только первый раз)...")
def get_initialized_pipeline(max_recipes: int = 200, force_rebuild: bool = False):
    """
    Создаёт и инициализирует пайплайн.
    Кэшируется между перерендерами - ВЫПОЛНЯЕТСЯ ТОЛЬКО ОДИН РАЗ!
    """
    pipeline = RecipeRAGPipeline()
    pipeline.initialize_full_pipeline(max_recipes=max_recipes, force_rebuild=force_rebuild)
    return pipeline


# ----- Main UI -----
st.title("Поиск рецептов с RAG")

# Получаем инициализированный pipeline (кэшируется автоматически)
try:
    pipeline = get_initialized_pipeline(max_recipes=2000, force_rebuild=False)
    pipeline_ready = True
    st.success("Система готова к работе!")
except Exception as e:
    pipeline_ready = False
    st.error(f"Ошибка инициализации: {str(e)}")
    with st.expander("Подробности ошибки"):
        st.code(traceback.format_exc())

st.markdown("---")

# ----- Поисковая форма -----
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

# ----- Обработка поиска -----
if search_button and pipeline_ready:
    if query.strip():
        with st.spinner("🔍 Обработка запроса..."):
            try:
                # Прямой синхронный вызов (автоопределение sync/async внутри)
                result = pipeline.ask(query)

                # Ответ LLM
                st.markdown("### 💬 Ответ системы")
                st.write(result.get("answer", ""))

                # Найденные рецепты
                st.markdown("### 📚 Найденные рецепты")
                search_results = result.get("search_results", [])

                if search_results:
                    for i, item in enumerate(search_results, start=1):
                        with st.expander(
                            f"{i}. {item.get('name', '(без названия)')} — "
                            f"релевантность: {item.get('relevance_score', 0):.3f}"
                        ):
                            st.markdown("**Ингредиенты:**")
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

# ----- Статус в футере -----
st.markdown("---")
with st.expander("ℹ️ Информация о системе"):
    if pipeline_ready:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            docs_count = len(pipeline.documents) if hasattr(pipeline, 'documents') else 0
            st.metric("Документов", docs_count)

        with col2:
            faiss_status = "✅" if hasattr(pipeline, 'vector_store') and pipeline.vector_store else "❌"
            st.metric("FAISS индекс", faiss_status)

        with col3:
            reranker_status = "✅" if hasattr(pipeline, 'reranker') and pipeline.reranker else "❌"
            st.metric("Реранкер", reranker_status)

        with col4:
            llm_status = "✅" if hasattr(pipeline, 'llm') and pipeline.llm else "✅"
            st.metric("LLM", llm_status)
    else:
        st.warning("Система не инициализирована")

# ----- Sidebar с настройками -----
with st.sidebar:
    st.header("⚙️ Настройки")

    if st.button("🔄 Переинициализировать систему", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Статус компонентов")

    if pipeline_ready:
        status_items = [
            ("Эмбеддер", hasattr(pipeline, 'embedder') and pipeline.embedder),
            ("Векторный индекс", hasattr(pipeline, 'vector_store') and pipeline.vector_store),
            ("Гибридный поиск", hasattr(pipeline, 'hybrid_search') and pipeline.hybrid_search),
            ("Реранкер", hasattr(pipeline, 'reranker') and pipeline.reranker),
            ("LLM", hasattr(pipeline, 'llm') and pipeline.llm),
        ]

        for name, status in status_items:
            st.markdown(f"{'✅' if status else '❌'} {name}")
    else:
        st.markdown("❌ Система не готова")

    st.markdown("---")
    st.markdown("### 🔧 Параметры")
    st.text(f"Max рецептов: 200")
    st.text(f"Модель: all-MiniLM-L6-v2")
