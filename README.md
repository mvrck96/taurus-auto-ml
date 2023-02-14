# taurus-auto-ml

Пакет для автоматического решения задач классификации на табличных данных. Попытка создать к AutoML-like решение.

## Функиональность

Что можно сделать с помощью этого пакета:
- Заполнить пропуски в категориальных и числовых признаках
- Провести скейлинг числовых фичей
- Задать список параметров, которые не надо обрабатывать
- Получить модель, которая лучше всего решает задачу
- Указать список моделей, среди которых надо выбрать лучшую


## Архитектура

В пакете есть несколько классов, каждый из которых отвечает за свою функциональность:
- `DataProvider` -- валидация и загрузка данных, для польнейшего использования
- `Preprocessor` -- обработка данных, заполнение пропусков, скейлинг, энкодинг
- `AutoClassifier` -- подбор лучше модели на кросс валидации, метрика f1
- Отдельные классы для ошибок

## Использование

Флоу использования следующий:

```python
from classifier import AutoClassifier # Импорт базового класса

# Инициализация классификатора
cls = AutoClassifier(models=["all"], data_source="../data/hotels.csv", target="booking_status")

# Получение модели и скора на данных
model, score = cls.fit()
```

## Ограничения

1. Поддерживаемые типы данных для data файла: `csv` и `parquet`
2. В используемом файле с данными, не стоит оставлять идентификаторы любого рода. Например идентификатор ббронирования, его лучше удалить
3. Сейчас используются три модели, это `DecisionTreeClassifier`, `LogisticRegression`, `SVC`
4. При разработке и в отладке использовася этот датасет: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset


## Дистрибуция

Для использования этого репозитория как python пакета необходимо:
1. Скачать код: `git clone https://github.com/mvrck96/taurus-auto-ml.git && cd taurus-auto-ml`
2. Создать виртуальное окружение: `python3 -m venv venv`, после чего активировать его: `source venv/bin/activate`
3. Установить все зависимости: `pip install -r requirements.txt`
4. Установить утилиту для сборки: `pip install build`
5. Собрать пакет: `python -m build`
6. После чего будет создана директория `dist` а в ней будет файл с раширением `.whl` и архив `.tar.gz`
7. Любой из этих файлов можно установить с помощью `pip install`

## Бэклог

Что стоит улучшить ?
- Добавить другие форматы инициализации данных, например просто передавать датафрейм
- Детальнее обрабатывать параметры, для этого надо лучше понимать природу признака и что он описывает
- Более глубокий инжиниринг фичей
- Запуск препроцессинга обернуть в `Pandas Pipeline`
- Расширить пул используемых моделей
- Добавить тюнинг параметров для каждой модели, реализовать можно через `RandomSearchCV`
- Валидировать модели в паралельных процессах, для ускорения работы
- Реализовать метод предсказания
- Добавить тесты: юнит, смок
- Добавить логирование, например через `tqdm` чтобы пользователи понимали что происходит
