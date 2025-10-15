"""
Streamlit UI для RAG рецептов с T-one ASR от Т-Банка.
Запуск: streamlit run streamlit_app.py
"""

import traceback
import streamlit as st
from rag.rag_pipeline import RecipeRAGPipeline
from streamlit_mic_recorder import mic_recorder

st.set_page_config(
    page_title="Рецепты с голосовым поиском 🎙️",
    layout="centered",
    page_icon="🍽️"
)


# ===== Pipeline кэширование =====
@st.cache_resource(show_spinner="🚀 Инициализация RAG системы...")
def get_pipeline(max_recipes: int = 200):
    """Создаёт и инициализирует RAG pipeline (один раз)."""
    pipeline = RecipeRAGPipeline()
    pipeline.initialize_full_pipeline(max_recipes=max_recipes, force_rebuild=False)
    return pipeline


# ===== T-one ASR кэширование =====
@st.cache_resource(show_spinner="🎤 Загрузка T-one модели (70M)...")
def get_asr():
    """Загружает T-one ASR модель от Т-Банка."""
    try:
        from asr.asr import ToneASR
        asr = ToneASR(device="auto")
        return asr
    except Exception as e:
        st.warning(f"⚠️ T-one недоступен: {e}")
        st.info("Установите: pip install transformers torch torchaudio soundfile")
        return None


# ===== Функция обработки запроса =====
def process_query(pipeline, query_text):
    """Обрабатывает текстовый запрос и показывает результаты."""
    with st.spinner("🔍 Поиск рецептов..."):
        try:
            result = pipeline.ask(query_text)

            # Ответ системы
            st.markdown("### 💬 Ответ")
            st.write(result.get("answer", ""))

            # Найденные рецепты
            st.markdown("### 📚 Найденные рецепты")
            search_results = result.get("search_results", [])

            if search_results:
                for i, item in enumerate(search_results, start=1):
                    relevance = item.get('relevance_score', 0)
                    name = item.get('name', 'Без названия')

                    with st.expander(f"{i}. {name} — релевантность: {relevance:.3f}"):
                        st.markdown("**Ингредиенты:**")
                        st.text(item.get('ingredients', 'Не указаны'))

                        url = item.get('url')
                        if url:
                            st.markdown(f"🔗 [Перейти к рецепту]({url})")
            else:
                st.info("Рецепты не найдены")

            # Время обработки
            with st.expander("⏱️ Время обработки"):
                timing = result.get("timing", {})
                col1, col2 = st.columns(2)
                col1.metric("Общее время", f"{timing.get('total_time', 0):.2f}с")
                col2.metric("Время поиска", f"{timing.get('search_time', 0):.2f}с")

        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            with st.expander("Подробная информация"):
                st.code(traceback.format_exc())


# ===== MAIN UI =====
st.title("🍽️ Поиск рецептов")
st.caption("💬 Текстовый ввод | 🎙️ Голосовой поиск с T-one от Т-Банка")

# Инициализация компонентов
try:
    pipeline = get_pipeline(max_recipes=200)
    asr = get_asr()
    pipeline_ready = True

    # Статус в sidebar
    with st.sidebar:
        st.success("✅ RAG система готова")
        if asr:
            st.success("✅ T-one ASR готов")
        else:
            st.warning("⚠️ ASR недоступен")

except Exception as e:
    pipeline_ready = False
    st.error(f"❌ Ошибка инициализации: {str(e)}")
    with st.expander("Подробности ошибки"):
        st.code(traceback.format_exc())

st.markdown("---")

# ===== ВКЛАДКИ =====
if pipeline_ready:
    tab1, tab2 = st.tabs(["💬 Текстовый ввод", "🎙️ Голосовой ввод"])

    # ===== ВКЛАДКА 1: Текст =====
    with tab1:
        query = st.text_input(
            "Введите запрос:",
            value="",
            placeholder="Например: как приготовить борщ",
            key="text_query"
        )

        if st.button("🔎 Найти рецепты", type="primary", use_container_width=True):
            if query.strip():
                process_query(pipeline, query)
            else:
                st.warning("⚠️ Введите запрос")

    # ===== ВКЛАДКА 2: Голос =====
    with tab2:
        if not asr:
            st.warning("⚠️ T-one ASR недоступен")
            st.info(
                "Установите зависимости:\n"
                "```bash\n"
                "pip install transformers torch torchaudio soundfile librosa\n"
                "```"
            )
        else:
            st.markdown("### 🎙️ Запись голосового запроса")
            st.caption("Используется T-one от Т-Банка (70M параметров, open source)")

            # Кнопка записи
            audio = mic_recorder(
                start_prompt="🎤 Начать запись",
                stop_prompt="⏹️ Остановить",
                format="wav",
                use_container_width=True,
                key='voice_recorder'
            )

            if audio:
                st.audio(audio['bytes'], format='audio/wav')

                col1, col2 = st.columns([3, 1])

                with col1:
                    if st.button("🔊 Распознать и найти", type="primary", use_container_width=True):
                        with st.spinner("🎧 Распознаём речь через T-one..."):
                            try:
                                # Распознаём через T-one
                                result_asr = asr.transcribe_bytes(
                                    audio['bytes'],
                                    sample_rate=audio.get('sample_rate', 16000)
                                )

                                if result_asr.get('success') and result_asr.get('text'):
                                    recognized_text = result_asr['text']
                                    st.success(f"📝 Распознано: **{recognized_text}**")

                                    # Автоматический поиск
                                    process_query(pipeline, recognized_text)

                                else:
                                    error_msg = result_asr.get('error', 'Речь не распознана')
                                    st.error(f"❌ {error_msg}")

                            except Exception as e:
                                st.error(f"❌ Ошибка ASR: {str(e)}")
                                with st.expander("Подробности"):
                                    st.code(traceback.format_exc())

                with col2:
                    if st.button("🗑️ Очистить", use_container_width=True):
                        st.rerun()

# ===== FOOTER =====
st.markdown("---")

with st.expander("ℹ️ Информация о системе"):
    if pipeline_ready:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            docs = len(pipeline.documents) if hasattr(pipeline, 'documents') else 0
            st.metric("Рецептов", docs)

        with col2:
            faiss = "✅" if hasattr(pipeline, 'vector_store') and pipeline.vector_store else "❌"
            st.metric("FAISS индекс", faiss)

        with col3:
            asr_status = "✅" if asr else "❌"
            st.metric("T-one ASR", asr_status)

        with col4:
            llm = "✅" if hasattr(pipeline, 'llm') and pipeline.llm else "❌"
            st.metric("LLM", llm)
    else:
        st.warning("Система не инициализирована")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")

    if st.button("🔄 Переинициализировать систему", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Статус компонентов")

    if pipeline_ready:
        components = [
            ("🔤 Эмбеддер", hasattr(pipeline, 'embedder') and pipeline.embedder),
            ("📊 Векторный индекс", hasattr(pipeline, 'vector_store') and pipeline.vector_store),
            ("🔍 Гибридный поиск", hasattr(pipeline, 'hybrid_search') and pipeline.hybrid_search),
            ("🎯 Реранкер", hasattr(pipeline, 'reranker') and pipeline.reranker),
            ("🤖 LLM", hasattr(pipeline, 'llm') and pipeline.llm),
            ("🎙️ T-one ASR", asr is not None),
        ]

        for name, status in components:
            st.markdown(f"{'✅' if status else '❌'} {name}")
    else:
        st.error("❌ Система не готова")

    st.markdown("---")
    st.markdown("### 🎙️ О T-one")
    st.caption("**T-one** — открытая ASR модель от Т-Банка")
    st.caption("• 70M параметров")
    st.caption("• WER < 10% для русского")
    st.caption("• Работает локально")
    st.caption("• Open source (Apache 2.0)")

    st.markdown("🔗 [GitHub](https://github.com/voicekit-team/T-one)")
    st.markdown("🤗 [HuggingFace](https://huggingface.co/t-tech/T-one)")

    st.markdown("---")
    st.markdown("### 📦 Параметры")
    st.text("Макс. рецептов: 200")
    st.text("Модель: all-MiniLM-L6-v2")
    st.text("LLM: Qwen2.5-1.5B-Instruct")
