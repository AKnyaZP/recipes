"""
Загрузка и обработка датасета рецептов Поваренок.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_dataset

def load_povarenok_data(dataset_name: str = "rogozinushka/povarenok-recipes"):
    """Загрузка рецептов из HuggingFace"""
    dataset = load_dataset(dataset_name)

    # Объединение train и test
    recipes = []
    for item in dataset['train']:
        recipes.append({
            'name': item['name'],
            'ingredients': item['ingredients'],
        })

    print(f"Загружено {len(recipes)} рецептов из HuggingFace")
    return recipes



def process_recipe(recipe: Dict[str, Any]) -> Dict[str, str]:
    """
    Обрабатывает один рецепт, создавая структурированное представление.

    Args:
        recipe: Сырой рецепт из датасета

    Returns:
        Обработанный рецепт
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


def prepare_documents(recipes: List[Dict[str, Any]], max_recipes: int = None) -> List[Dict[str, str]]:
    """
    Обрабатывает список рецептов и готовит документы для индексации.

    Args:
        recipes: Список сырых рецептов
        max_recipes: Максимальное количество рецептов для обработки (для тестирования)

    Returns:
        Список обработанных документов
    """
    print(f"Обрабатываем рецепты...")

    if max_recipes:
        recipes = recipes[:max_recipes]
        print(f"Ограничиваем до {max_recipes} рецептов для тестирования")

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

    print(f"Обработано {len(documents)} рецептов")
    if skipped > 0:
        print(f"Пропущено {skipped} невалидных рецептов")

    return documents


def save_processed_data(documents: List[Dict[str, str]], output_path: str = "data/processed_recipes.json"):
    """
    Сохраняет обработанные документы в JSON файл.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(documents)} обработанных рецептов в {output_path}")


def load_processed_data(file_path: str = "../data/processed_recipes.json") -> List[Dict[str, str]]:
    """
    Загружает ранее обработанные документы.
    """
    if not Path(file_path).exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Загружено {len(documents)} обработанных рецептов")
    return documents