import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    GenerationConfig
)
from typing import List, Dict, Any

import logging

logging.basicConfig(level=logging.INFO)

class RecipeLLM:
    """
    Класс для работы с компактной языковой моделью.
    Генерирует ответы на основе контекста рецептов.
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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
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
            max_new_tokens = 512,
            temperature = 0.1,
            top_p = 0.001,
            repetition_penalty = 1.05,
            do_sample = True
        )
        
        logging.info("LLM initialized")
    
    def format_context(self, search_results: List[tuple]) -> str:
        """
        Форматирует результаты поиска в контекст для модели.
        
        Args:
            search_results: Список (документ, скор) из поиска
            
        Returns:
            Отформатированный контекст
        """
        if not search_results:
            return "Контекст не найден."
        
        context_parts = []
        
        for i, (doc, score) in enumerate(search_results, 1):
            recipe_text = f"""Рецепт {i}: {doc['name']}
Ингредиенты: {doc.get('ingredients_text', 'Не указаны')}
Описание: {doc.get('full_text', 'Описание отсутствует')[:300]}..."""
            
            context_parts.append(recipe_text)
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для модели на основе системного промпта, контекста и запроса.
        
        Args:
            query: Вопрос пользователя
            context: Контекст из поиска рецептов
            
        Returns:
            Готовый промпт
        """

        RAG_PROMPT_TEMPLATE = """Контекст рецептов:
        {context}

        Вопрос пользователя: {question}

        Ответ:"""

        main_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        SYSTEM_PROMPT = """Ты помощник в кулинарии. Твоя задача - помогать пользователям находить рецепты и отвечать на вопросы о готовке на основе предоставленной информации.

        Правила:
        1. Отвечай только на русском языке
        2. Используй только информацию из предоставленного контекста
        3. Если в контексте нет нужной информации, честно скажи об этом
        4. Структурируй ответ: название блюда, ингредиенты, краткое описание
        5. Будь полезным и дружелюбным"""

        if "Qwen" in self.model.config.name_or_path or "qwen" in self.model.config.name_or_path.lower():
            full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{main_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{main_prompt}"
        
        return full_prompt
    
    def generate_response(self, query: str, search_results: List[tuple]) -> str:
        """
        Генерирует ответ на запрос пользователя на основе найденных рецептов.
        
        Args:
            query: Вопрос пользователя  
            search_results: Результаты поиска рецептов
            
        Returns:
            Сгенерированный ответ
        """
        logging.info(f"Генерируем ответ для запроса: '{query}'")
        
        context = self.format_context(search_results)
        
        prompt = self.create_prompt(query, context)

        try:
            outputs = self.generator(
                prompt,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = outputs[0]['generated_text'].strip()
            
            # Очищаем ответ от артефактов
            response = self._clean_response(response)
            
            print("Ответ сгенерирован")
            return response
            
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте переформулировать вопрос."
    
    def _clean_response(self, response: str) -> str:
        """
        Очищает сгенерированный ответ от нежелательных артефактов.
        
        Args:
            response: Сырой ответ модели
            
        Returns:
            Очищенный ответ
        """
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
        
        sentences = response.split('.')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'

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
