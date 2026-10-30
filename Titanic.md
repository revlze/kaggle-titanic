# Titanic

## Notes

- [Черновик EDA](notebooks/archive/draft_marimo_original.py) → [01_eda](notebooks/01_eda.py): смотрел пропуски, баланс классов и связи с выживанием. Числовой heatmap было недостаточно: `Sex` и `Embarked` смотрел отдельно. Редкий возраст сам по себе не ошибка.
- [Исходный ноутбук](notebooks/archive/experiments_original.py) и [experiments.py](notebooks/experiments.py): сначала собрал в одном месте классику, признаки, DNN и Optuna. Дальше разделил это по этапам и вынес общие функции в `src/`.
- [02_classical_models](notebooks/02_classical_models.py): перебрал линейные модели, KNN, дерево, RF и бустинги. В основной YAML-runner вошли логрег, KNN, Gradient Boosting, CatBoost, XGBoost и LightGBM; остальные остались в ноутбуках.
- [03_feature_engineering](notebooks/03_feature_engineering.py): `Name` использовал для `Title`, добавил `FamilySize = SibSp + Parch + 1`. После их добавления заметил небольшой буст, поэтому оставил в базовом наборе.
- [04_dnn](notebooks/04_dnn.py): пробовал MLP, BatchNorm, dropout, разные оптимизаторы и embeddings. `MyModel`/`MyModelEmb` и обучение на PyTorch перенёс в пайплайн через `custom/`. Преимущество DNN на LB не подтверждено.
- [05_nested_comparison](notebooks/05_nested_comparison.py): общий набор признаков, одинаковые внешние фолды, отдельные подбор и оценка. Сохранённый запуск этого ноутбука был короткой проверкой протокола, а не финальным сравнением четырёх моделей.

В итоге `PassengerId` оставил для submission, числовые пропуски заполняю медианой, категориальные — модой. В классике использую one-hot; в DNN с embeddings — ordinal. `Ticket`/`Cabin` не входят в базовый набор, но остаются в варианте `all`.

## Questions

- Нужен ли nested CV без Optuna? В итоговом пайплайне без подбора использую обычную CV, с подбором — nested. У DNN отдельно выделяется внутренний stopping split.
- Почему CV выше LB? Это разные данные. Одну конкретную причину разрыва в этих экспериментах не изолировал.

## Solution

Здесь запуски итогового пайплайна. Accuracy: CV в процентах, STD в процентных пунктах, LB на шкале 0–1. `—` означает, что public score не записан.

| № | Approach | CV, % | CV STD, п.п. | LB | Date |
|---|---|---:|---:|---:|---|
| 1 | Logistic regression (baseline) | 82.72 | 2.15 | **0.77272** | 2026-10-29 |
| 2 | Logistic regression + Optuna | 82.30 | 2.44 | **0.77272** | 2026-10-29 |
| 3 | Gradient boosting (baseline) | 82.86 | 1.71 | 0.76076 | 2026-10-29 |
| 4 | Gradient boosting + Optuna | 83.00 | 2.10 | 0.75837 | 2026-10-29 |
| 5 | XGBoost + Optuna | 83.56 | 3.24 | 0.75837 | 2026-10-30 |
| 6 | XGBoost ensemble (5 models) | 82.87 | 2.62 | 0.75598 | 2026-10-30 |
| 7 | LightGBM + Optuna | 82.58 | 2.81 | 0.76076 | 2026-10-30 |
| 8 | K-nearest neighbors + Optuna | 81.32 | 3.13 | 0.75837 | 2026-10-30 |
| 9 | Gradient boosting + Optuna (50 trials) | 83.57 | 3.26 | — | 2026-10-29 |
| 10 | Gradient boosting + Optuna (60 trials) | 83.85 | 3.14 | — | 2026-10-29 |

## Annotations

Номера совпадают с `Solution`. Общее: seed `0xFACED`, признаки `base_title_family`, 712 development / 179 исключённых holdout-строк. CV: 5 фолдов без Optuna или 5×3 с ней. Scaling включён для логрега и KNN; preprocessing обучается внутри каждого fit.

Ниже основные параметры финальных моделей, а не стартовые настройки поиска. Длинные дроби округлены; полные конфиги и параметры — в [снимке результатов](docs/experiment-results.json). Бюджет trials указан на один поиск: пять внешних поисков и один отдельный финальный. Финальный refit — на всех 891 размеченных строках.

| № | Параметры | Optuna trials |
|---|---|---:|
| 1 | `C=1`; `max_iter=2000` | выкл. |
| 2 | `C=5.6661`; `max_iter=2000` | 60 |
| 3 | `loss=log_loss`; `n_estimators=100`; `learning_rate=0.1`; `max_depth=3` | выкл. |
| 4 | `loss=exponential`; `n_estimators=157`; `learning_rate=0.0473944`; `max_depth=10`; `min_samples_leaf=8`; `min_samples_split=35` | 20 |
| 5 | `n_estimators=250`; `learning_rate=0.0719916`; `max_depth=5`; `min_child_weight=5`; `subsample=0.984965`; `colsample_bytree=0.748772`; `reg_lambda=6.22022` | 60 |
| 6 | `n_models=5`; `method=soft`; `n_estimators=400`; `learning_rate=0.04543`; `max_depth=4`; `min_child_weight=4`; `subsample=0.966773`; `colsample_bytree=0.819121`; `reg_lambda=7.73836` | 60 |
| 7 | `n_estimators=250`; `learning_rate=0.0839242`; `num_leaves=14`; `max_depth=4`; `min_child_samples=19`; `subsample=0.971186`; `subsample_freq=1`; `colsample_bytree=0.729278`; `reg_lambda=6.79023` | 60 |
| 8 | `n_neighbors=23`; `weights=uniform`; `metric=minkowski`; `p=2` | 60 |
| 9 | `loss=exponential`; `n_estimators=200`; `learning_rate=0.0644557`; `max_depth=5`; `min_samples_leaf=17`; `min_samples_split=47` | 50 |
| 10 | `loss=exponential`; `n_estimators=156`; `learning_rate=0.0479104`; `max_depth=5`; `min_samples_leaf=18`; `min_samples_split=36` | 60 |

## Final Ensemble

Из ансамблей до submission дошли пять XGBoost с soft voting (№6): **LB 0.75598**. Они не превзошли одиночный логрег №1/№2 с **LB 0.77272**. Лучший записанный результат получился без ансамбля; private scores нет.

## Processed ideas

### Good

- `Title` и `FamilySize` дали небольшой буст в моих экспериментах. Получилось использовать информацию из `Name` и размер семьи без сложного feature engineering.
- Простой логрег как baseline: на public LB оказался не хуже более сложных отправленных моделей.
- Общее разбиение и preprocessing внутри фолдов. Подбор параметров и оценку разделил; у DNN scheduler и выбор эпохи не используют внешний validation.
- Сохранять параметры вместе с метриками. У DNN размеры входа и embedding-словарей беру из обученного encoder, а не задаю руками.

### Neutral

- Возврат `Ticket`/`Cabin` и варианты preprocessing есть в коде, но отдельного сохранённого сравнения их эффекта нет.
- Optuna не улучшила public score логрега. У Gradient Boosting с 50/60 trials записаны CV 83.57%/83.85%, но public scores для этих запусков нет.
- DNN и embeddings остались полезной частью проекта для практики PyTorch. Старые CV 83.29%/82.44% получены при другом протоколе и на признаках `all`; с таблицей `Solution` их не смешиваю. По заметкам «оптимизатор нормал» нельзя выбрать лучший optimizer.

### Bad

- Обычное дерево переобучалось: в старом снимке train 99.75% против CV 82.03%. Подбор снизил train accuracy, но тот же holdout остался 81.56%.
- Пять XGBoost в моём запуске дали результат хуже одной модели. Параметры подбирались отдельно, поэтому это не доказательство, что усреднение само по себе ухудшает модель.
- В ранних экспериментах выбор и оценка использовали одни CV-фолды, а у DNN оценочный фолд управлял эпохами. Старые максимумы holdout вроде 90.50% не считаю независимым итоговым качеством.

### Не вошло в итоговые прогоны

- `log1p(Fare)` остался графиком в EDA, в признаки модели его не переносил.
- Нативные категории CatBoost и отдельный bootstrap строк для ансамбля целых моделей не использовал. Hard voting и stacking поддерживаются, но подтверждённого полного сравнения с LB для них нет.
