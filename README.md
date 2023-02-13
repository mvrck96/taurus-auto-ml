# taurus-auto-ml

Короткое описание

Надо ограничить круг решаемых задач чтобы не утонуть, например классификация (можно  и регрессию сводить тоже к классификации группы результата).

Основная идея авто МЛ: есть большой пайплан обучения модели, в котором можно варьировать не только параметры модели, но и параметры обработки данных. Задача найти такой набор всех этих параметров, который бы давал наибольший скор конечной модели на имеющихся данных.

Поскольку параметров очень много стоит использовать RandomSearch а не GridSearch.

## TODO:

- Работа с источниками данных, прочитать, записать (csv, parquete, etc.) обертка над pandas
- Работа с фичами (посмотреть в лекциях Дьяконова):
    - Скейлинг
    - Запуолнение пропущенных значений
- Работа с моделями
- Оценка результата работы моделей

## Архитектура

Не надо писать много оберток над склерном или пандасом. Нужен класс, в котором можно задать пулл решений и он все сделает автоматически. Прочитает, обработает, зафитит модели, даст метрики.

Если пользователь указывает пулл возможных трансформаций или моделей, то испольщуем его. Если нет, то вычисляем сами или делаем все и смотрим результат.

Флоу работы нужен такой. Инициализируем один класс, у которого есть несколько методов. После чего дергаем нужный метод.

### `Preprocessor`

Выполняет весь препроцессинг. Можно исключать фичи из процессинга или этапы.

Запускается внутри `AutoClassifier`

Нельзя отправлять идентификаторы

1. Получить типы всех фичей
2. Проверить есть ли пропуски
3. Заполнить пропуски, если они есть
4. Сделать скейлинг


### `AutoClassifier`

5. Закинуть данные в модель, обучить
6. Потюнить параметры чтобы улучшить скор
