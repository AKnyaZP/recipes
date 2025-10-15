import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig
)
from typing import List, Dict, Any
import logging
import asyncio
import concurrent.futures

logging.basicConfig(level=logging.INFO)


class RecipeLLM:
    """
    Класс для работы с компактной языковой моделью.
    Генерирует ответы на основе контекста рецептов.
    Поддерживает улучшенное форматирование для разных типов запросов.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Инициализирует LLM модель.

        Args:
            model_name: Название модели из HuggingFace
        """
        logging.info(f"model_name: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )

        # Настраиваем токены padding'а если нужно
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Определяем тип данных для модели
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Загружаем модель
        logging.info("downloading LLM...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Создаем пайплайн генерации
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            do_sample=True
        )

        logging.info("LLM initialized")

    def _detect_query_intent(self, query: str) -> str:
        """
        Определяет тип запроса для адаптации форматирования контекста.

        Returns:
            'single_dish', 'multi_dish' или 'general'
        """
        query_lower = query.lower()

        # Паттерны для одного блюда
        single_patterns = ['что в', 'ингредиенты', 'состав', 'из чего']
        if any(pattern in query_lower for pattern in single_patterns):
            return 'single_dish'

        # Паттерны для нескольких блюд
        multi_patterns = [' и ', ', ', 'несколько', 'разные']
        if any(pattern in query_lower for pattern in multi_patterns):
            return 'multi_dish'

        return 'general'

    def format_context(self, search_results: List[tuple], query: str = "") -> str:
        """
        Форматирует результаты поиска в контекст для модели с учетом типа запроса.

        Args:
            search_results: Список (документ, скор) из поиска
            query: Пользовательский запрос для адаптации форматирования

        Returns:
            Отформатированный контекст
        """
        if not search_results:
            return "Контекст не найден."

        intent = self._detect_query_intent(query) if query else 'general'
        context_parts = []

        for i, (doc, score) in enumerate(search_results, 1):
            doc_id = doc.get('id', f'recipe_{i}')
            doc_name = doc.get('name', 'Без названия')
            ingredients = doc.get('ingredients_text', 'Не указаны')

            if intent == 'single_dish':
                # Для одного блюда - компактный формат с акцентом на точность
                recipe_text = f"""РЕЦЕПТ #{doc_id}: {doc_name}
ИНГРЕДИЕНТЫ: {ingredients}
РЕЛЕВАНТНОСТЬ: {score:.3f} из 1.000"""
            elif intent == 'multi_dish':
                # Для нескольких блюд - структурированный формат по блюдам
                recipe_text = f"""=== БЛЮДО {i}: {doc_name} ===
ID: {doc_id}
Ингредиенты: {ingredients}
Совпадение с запросом: {score:.1%}"""
            else:
                # Общий формат с описанием
                description = doc.get('full_text', 'Описание отсутствует')[:200]
                recipe_text = f"""Рецепт {i}: {doc_name}
Ингредиенты: {ingredients}
Описание: {description}...
Релевантность: {score:.3f}"""

            context_parts.append(recipe_text)

        return "\n\n".join(context_parts)

    def create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для модели с адаптивными правилами под тип запроса.

        Args:
            query: Вопрос пользователя
            context: Контекст из поиска рецептов

        Returns:
            Готовый промпт
        """
        intent = self._detect_query_intent(query)

        # Базовый шаблон
        RAG_PROMPT_TEMPLATE = """Контекст рецептов:
{context}

Вопрос пользователя: {question}

Ответ:"""

        main_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        # Адаптивные системные правила
        if intent == 'single_dish':
            SYSTEM_PROMPT = """Ты помощник в кулинарии. Отвечай строго по одному блюду из контекста.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Отвечай только на русском языке
2. Если пользователь спрашивает об одном конкретном блюде - отвечай ТОЛЬКО о нём
3. НЕ смешивай ингредиенты разных блюд
4. Выбери САМОЕ релевантное блюдо из контекста
5. Структурируй ответ: Название блюда → Ингредиенты → Краткое описание
6. Если нет точного совпадения - честно скажи об этом"""

        elif intent == 'multi_dish':
            SYSTEM_PROMPT = """Ты помощник в кулинарии. Отвечай по нескольким блюдам структурированно.

ПРАВИЛА:
1. Отвечай только на русском языке
2. Для каждого блюда - отдельный блок с названием
3. НЕ смешивай ингредиенты разных блюд
4. Формат: "Блюдо 1: ... Блюдо 2: ..." 
5. Если блюда нет в контексте - не добавляй его
6. Используй только предоставленную информацию"""
        else:
            SYSTEM_PROMPT = """Ты помощник в кулинарии. Твоя задача - помогать пользователям находить рецепты и отвечать на вопросы о готовке.

Правила:
1. Отвечай только на русском языке
2. Используй только информацию из предоставленного контекста
3. Если в контексте нет нужной информации, честно скажи об этом
4. Структурируй ответ: название блюда, ингредиенты, краткое описание
5. Будь полезным и дружелюбным"""

        # Комбинируем с системным промптом
        # Разные модели требуют разного форматирования
        if "Qwen" in self.model.config.name_or_path or "qwen" in self.model.config.name_or_path.lower():
            # Формат для Qwen модели
            full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{main_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Общий формат
            full_prompt = f"{SYSTEM_PROMPT}\n\n{main_prompt}"

        return full_prompt

    def _generate_sync(self, prompt: str) -> str:
        """
        Внутренний метод для синхронной генерации.

        Args:
            prompt: Готовый промпт

        Returns:
            Сгенерированный текст
        """
        outputs = self.generator(
            prompt,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False  # Возвращаем только новый текст
        )

        return outputs[0]['generated_text'].strip()

    def generate_response(self, query: str, search_results: List[tuple]) -> str:
        """
        Генерирует ответ на запрос пользователя на основе найденных рецептов.
        ВАЖНО: Этот метод теперь может работать как синхронно, так и асинхронно.
        - Если вызывается из async контекста, работает асинхронно
        - Если вызывается из sync контекста, работает синхронно

        Args:
            query: Вопрос пользователя
            search_results: Результаты поиска рецептов

        Returns:
            Сгенерированный ответ
        """
        logging.info(f"Генерируем ответ для запроса: '{query}'")

        # Форматируем контекст с учетом типа запроса
        context = self.format_context(search_results, query)

        # Создаем адаптивный промпт
        prompt = self.create_prompt(query, context)

        # Генерируем ответ
        try:
            # Проверяем, находимся ли мы в async контексте
            try:
                loop = asyncio.get_running_loop()
                # Если есть running loop, используем асинхронный подход
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._generate_sync, prompt)
                    response = future.result()
            except RuntimeError:
                # Нет running loop - работаем синхронно
                response = self._generate_sync(prompt)

            # Очищаем ответ от артефактов
            response = self._clean_response(response)

            logging.info("✅ Ответ сгенерирован")
            return response

        except Exception as e:
            logging.error(f"❌ Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте переформулировать вопрос."

    def _clean_response(self, response: str) -> str:
        """
        Очищает сгенерированный ответ от нежелательных артефактов.

        Args:
            response: Сырой ответ модели

        Returns:
            Очищенный ответ
        """
        # Удаляем специальные токены если они остались
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")

        # Обрезаем на точке если ответ слишком длинный
        sentences = response.split('.')
        if len(sentences) > 4:  # Увеличено с 3 до 4 для более полных ответов
            response = '. '.join(sentences[:4]) + '.'

        # Удаляем лишние пробелы
        response = ' '.join(response.split())

        return response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о загруженной модели.

        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_name": self.model.config.name_or_path,
            "device": self.device,
            "torch_dtype": str(self.model.dtype),
            "vocab_size": self.model.config.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'Unknown')
        }
