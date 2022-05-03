# TP-ml-prod
Репозиторий с домашними заданиями по курсу "Машинное обучение в продакшене"

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage logistic regression:
~~~
python src/train_pipeline.py configs/train_config_log_reg.yaml
~~~
Usage random forest:
~~~
python src/train_pipeline.py configs/train_config_random_forest.yaml
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md               <- Правила Использования проекта.
    ├── data
    │   ├── predicted           <- Предстазанные метки для predict_pipeline.py
    │   └── raw                 <- Реальные данные.
    │
    ├── models                  <- Модели, трансформеры и метрики.
    │
    ├── notebooks               <- Jupyter notebooks с предварительным анализом данных.
    │
    ├── requirements.txt        <- Необходимые пакеты для запуска обучения и предсказния.
    │
    ├── setup.py                <- Возможность установки проекта через менеджер pip.
    │
    ├── src                     <- Код для запуска пайплана.
    │   ├── __init__.py         <- Делает src Python модулем.
    │   │
    │   ├── data                <- Работа с данными.
    │   │
    │   ├── entity              <- Структуры с параметрами для работы модели.
    │   │
    │   ├── features            <- Преобразование сырых данных к признакам дял модели.
    │   │
    │   ├── models              <- Тренировки модели и использование готовой модели.
    │   │
    │   ├── predict_pipeline.py <- Пайплайн для прогноза на данных
    │   │
    │   ├── train_pipeline.py   <- Пайплайн для тренировки модели
    │
    ├── tests                   <- Тесты

--------