"""
Точка входа для запуска RAG системы.
Демонстрирует работу всего пайплайна.
"""

from rag.rag_pipeline import RecipeRAGPipeline

def main():
    """
    Основная функция для демонстрации RAG пайплайна.
    """
    print("🍽️ СИСТЕМА RAG ДЛЯ ПОИСКА РЕЦЕПТОВ")
    print("=" * 50)
    print("Технологии:")
    print(f"📊 Векторная БД: FAISS")  
    print(f"🔤 Эмбеддинги: {"sentence-transformers/distiluse-base-multilingual-cased"}")
    print(f"🤖 LLM: {"Qwen/Qwen2.5-1.5B-Instruct"}")
    print(f"🔍 Гибридный поиск: Векторный + BM25")
    print("=" * 50)
    
    try:
        # Инициализируем RAG пайплайн
        rag = RecipeRAGPipeline()
        
        # Полная инициализация (может занять несколько минут)
        print("\n⏳ Инициализация может занять несколько минут...")
        rag.initialize_full_pipeline(
            max_recipes=200,      # Ограничиваем для тестирования  
            force_rebuild=False   # Используем кэш если есть
        )
        
        print("\n🎉 Система готова к работе!")
        print("\nПримеры запросов:")
        print("• 'Как приготовить борщ?'")
        print("• 'Рецепт салата с курицей'") 
        print("• 'Что можно сделать из картофеля?'")
        print("• 'Простой десерт'")
        print("\nВведите 'выход' для завершения")
        print("-" * 50)
        
        # Интерактивный режим
        while True:
            try:
                # Получаем запрос от пользователя
                question = input("\n❓ Ваш вопрос: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                # Обрабатываем запрос
                result = rag.ask(question)
                
                # Выводим расширенную информацию
                print(f"\n💬 Ответ системы:")
                print(f"{result['answer']}")
                
                if result['search_results']:
                    print(f"\n📚 Найдено {result['found_recipes']} релевантных рецептов:")
                    
                    for i, recipe in enumerate(result['search_results'], 1):
                        print(f"\n{i}. **{recipe['name']}**")
                        print(f"   🥕 Ингредиенты: {recipe['ingredients'][:100]}...")
                        if recipe['url']:
                            print(f"   🔗 Ссылка: {recipe['url']}")
                        print(f"   📊 Релевантность: {recipe['relevance_score']:.3f}")
                
                # Статистика
                timing = result['timing'] 
                print(f"\n⏱️ Время обработки: {timing['total_time']}с")
                print(f"   (поиск: {timing['search_time']}с)")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Завершение по Ctrl+C")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                continue
    
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("\nВозможные причины:")
        print("1. Отсутствует файл данных data/raw/recipes.json")
        print("2. Недостаточно памяти для загрузки моделей")
        print("3. Отсутствуют зависимости (см. requirements)")
        print("4. Проблемы с доступом к HuggingFace Hub")
        
        return 1
    
    return 0


def run_batch_demo():
    """
    Запускает демонстрацию в пакетном режиме с предустановленными вопросами.
    """
    print("🧪 ДЕМО РЕЖИМ - ПАКЕТНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 50)
    
    try:
        # Инициализируем пайплайн
        rag = RecipeRAGPipeline()
        rag.initialize_full_pipeline(max_recipes=100, force_rebuild=False)
        
        # Тестовые вопросы
        demo_questions = [
            "Как приготовить красный борщ?",
            "Простой рецепт салата цезарь",
            "Что можно приготовить из курицы быстро?",
            "Десерт с яблоками для детей",
            "Вегетарианский суп с овощами"
        ]
        
        print(f"\n🔍 Тестируем {len(demo_questions)} вопросов:")
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n--- Тест {i}/{len(demo_questions)} ---")
            
            result = rag.ask(question)
            
            print(f"🥘 Топ-3 рецепта:")
            for j, recipe in enumerate(result['search_results'], 1):
                print(f"  {j}. {recipe['name']} ({recipe['relevance_score']:.3f})")
        
        print(f"\n✅ Демонстрация завершена!")
        
    except Exception as e:
        print(f"❌ Ошибка демо: {e}")


if __name__ == "__main__":
    import sys
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_batch_demo()
    else:
        exit_code = main()
        sys.exit(exit_code)