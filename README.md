Recipe RAG

Этот репозиторий содержит небольшой конвейер RAG (гибридный поиск FAISS + BM25) и оболочку LLM. Приложение работает как CLI, создаёт вложения/индексы и принимает запросы в интерактивном режиме.

1. Клонировать репозиторий

```bash
git clone https://github.com/AKnyaZP/recipes.git
```

2. В директорию `temp/` добавить .env c `HF_TOKEN = ""` (опционально)

3. Установить зависимости
```bash
pip install -r requirements.txt
```

4. запуск приложения
```bash
streamlit urn src/streamlit_app.py --server.port 8501
```
