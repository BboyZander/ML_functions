Данный репозиторий содержит мои функции для автоматизации различных процессов при моделировании.

### Доступные модули и их функции

1. feature_checker.py
    - get_vif (Расчет VIF (variance inflation factor))
    - get_iv (Расчет IV признака после WOE преобразования)
    - calculate_psi (Расчет PSI (population stability index))
    - add_value_order (Сортировка признаков по их add value к модели)
    - two_bucket_dr (Расчет default rate модели при разбиении признака на 2 бакета)
    - get_gini (Расчет значения GINI для каждого фактора для дальнейшего фильтра)
    
2. woe_checker.py
    - get_bad_features
    - woe_rebinning
    - split_from_rebinned
    
3. modeling_function.py
    - validation (Метод кросс валидации)
    - delete_correlated_features (Удаление коррелированных признаков по заданному cut_off)
    - quality_dynamic (Расчет Delta GINI при удалении признаков на каждой итерации)
    
4. plot_functions.py
    - default_rate (Построение дефолт рейта выбранной выборки)
    - gini_distribution (Построение распределения Джини по заданным отрезкам)
    - dr_distribution (Построение дефолт рейта в каждом бакете признака после WOE трансформации)
    - add_value_plot (График add value)
    - plot_roc_auc_curve (График roc_auc_score)
    
5. python_tricks.py
    - deep_flatten (Раскрывает внутренние списки в один)
    - count_occurences (Считает количество вхождений объекта в лист)
    - chunk (Разрезает лист на на части, каждая из которых содержит n объектов)

- [ ] Cоздать класс, объединяющие написанные методы.