"""
Загрузка и обработка датасета рецептов Поваренок.
Поддержка async/sync режимов без изменения API.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging
import asyncio
import concurrent.futures
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datasets import load_dataset


def _load_povarenok_data_sync(dataset_name: str = "rogozinushka/povarenok-recipes"):
    """Внутренняя синхронная функция загрузки рецептов из HuggingFace"""
    dataset = load_dataset(dataset_name)

    # Объединение train и test
    recipes = []
    for item in dataset['train']:
        recipes.append({
            'name': item['name'],
            'ingredients': item['ingredients'],
        })

    logger.info(f"Загружено {len(recipes)} рецептов из HuggingFace")
    return recipes


def load_povarenok_data(dataset_name: str = "rogozinushka/povarenok-recipes"):
    """
    Загрузка рецептов из HuggingFace.
    Автоматически определяет sync/async контекст.

    Args:
        dataset_name: Название датасета на HuggingFace

    Returns:
        Список рецептов
    """
    try:
        # Проверяем наличие running event loop
        loop = asyncio.get_running_loop()
        # Если есть loop, выполняем в executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_load_povarenok_data_sync, dataset_name)
            return future.result()
    except RuntimeError:
        # Нет event loop - синхронное выполнение
        return _load_povarenok_data_sync(dataset_name)


def process_recipe(recipe: Dict[str, Any]) -> Dict[str, str]:
    """
    Обрабатывает один рецепт, создавая структурированное представление.
    Функция легковесная, не требует async обёртки.

    Args:
        recipe: Сырой рецепт из датасета

    Returns:
        Обработанный рецепт или None если невалидный
    """
    # Валидация обязательных полей
    if not recipe.get('name') or not recipe.get('ingredients'):
        return None

    # Форматируем ингредиенты
    ingredients = recipe.get('ingredients', {})
    if isinstance(ingredients, dict):
        ingredients_text = '; '.join([
            f"{ingredient} - {amount}" if amount else ingredient
            for ingredient, amount in ingredients.items()
        ])
    else:
        ingredients_text = str(ingredients)

    # Создаем полный текст для поиска
    full_text = f"""Название: {recipe['name']}

Ингредиенты: {ingredients_text}

Рецепт: {recipe['name']} - это блюдо, которое готовится из следующих ингредиентов: {ingredients_text}."""

    return {
        'id': recipe.get('id', ''),
        'name': recipe['name'].strip(),
        'ingredients_text': ingredients_text,
        'full_text': full_text.strip()
    }


def _prepare_documents_sync(recipes: List[Dict[str, Any]], max_recipes: int = None) -> List[Dict[str, str]]:
    """
    Внутренняя синхронная функция обработки рецептов.

    Args:
        recipes: Список сырых рецептов
        max_recipes: Максимальное количество рецептов для обработки

    Returns:
        Список обработанных документов
    """
    logger.info(f"Обрабатываем рецепты...")

    if max_recipes:
        recipes = recipes[:max_recipes]
        logger.info(f"Ограничиваем до {max_recipes} рецептов для тестирования")

    documents = []
    skipped = 0

    for i, recipe in enumerate(recipes):
        processed = process_recipe(recipe)

        if processed is None:
            skipped += 1
            continue

        # Добавляем порядковый номер если нет ID
        if not processed['id']:
            processed['id'] = str(i)

        documents.append(processed)

    logger.info(f"Обработано {len(documents)} рецептов")
    if skipped > 0:
        logger.info(f"Пропущено {skipped} невалидных рецептов")

    return documents


def prepare_documents(recipes: List[Dict[str, Any]], max_recipes: int = None) -> List[Dict[str, str]]:
    """
    Обрабатывает список рецептов и готовит документы для индексации.
    Автоматически определяет sync/async контекст.

    Args:
        recipes: Список сырых рецептов
        max_recipes: Максимальное количество рецептов для обработки (для тестирования)

    Returns:
        Список обработанных документов
    """
    try:
        # Проверяем наличие running event loop
        loop = asyncio.get_running_loop()
        # Если есть loop, выполняем в executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_prepare_documents_sync, recipes, max_recipes)
            return future.result()
    except RuntimeError:
        # Нет event loop - синхронное выполнение
        return _prepare_documents_sync(recipes, max_recipes)


def _save_processed_data_sync(documents: List[Dict[str, str]], output_path: str = "data/processed_recipes.json"):
    """
    Внутренняя синхронная функция сохранения документов.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    logger.info(f"Сохранено {len(documents)} обработанных рецептов в {output_path}")


def save_processed_data(documents: List[Dict[str, str]], output_path: str = "data/processed_recipes.json"):
    """
    Сохраняет обработанные документы в JSON файл.
    Автоматически определяет sync/async контекст.

    Args:
        documents: Список обработанных документов
        output_path: Путь для сохранения
    """
    try:
        # Проверяем наличие running event loop
        loop = asyncio.get_running_loop()
        # Если есть loop, выполняем в executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_save_processed_data_sync, documents, output_path)
            return future.result()
    except RuntimeError:
        # Нет event loop - синхронное выполнение
        _save_processed_data_sync(documents, output_path)


def _load_processed_data_sync(file_path: str = "../data/processed_recipes.json") -> List[Dict[str, str]]:
    """
    Внутренняя синхронная функция загрузки документов.
    """
    if not Path(file_path).exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    logger.info(f"Загружено {len(documents)} обработанных рецептов")
    return documents


def load_processed_data(file_path: str = "../data/processed_recipes.json") -> List[Dict[str, str]]:
    """
    Загружает ранее обработанные документы.
    Автоматически определяет sync/async контекст.

    Args:
        file_path: Путь к файлу с обработанными данными

    Returns:
        Список документов или None если файл не существует
    """
    try:
        # Проверяем наличие running event loop
        loop = asyncio.get_running_loop()
        # Если есть loop, выполняем в executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_load_processed_data_sync, file_path)
            return future.result()
    except RuntimeError:
        # Нет event loop - синхронное выполнение
        return _load_processed_data_sync(file_path)

