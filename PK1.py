import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

# Загрузка набора данных
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Применение преобразования Yeo-Johnson к одному числовому признаку (первый признак)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = X.copy()
X_transformed.iloc[:, 0] = pt.fit_transform(X.iloc[:, [0]])

# Вывод данных до и после преобразования
print("Исходные данные:\n", X.head())
print("Данные после преобразования Yeo-Johnson:\n", X_transformed.head())

# Использование класса SelectPercentile и метода на основе взаимной информации для отбора признаков
selector = SelectPercentile(mutual_info_classif, percentile=5)
X_selected = selector.fit_transform(X_transformed, y)

# Получение индексов выбранных признаков
selected_features = selector.get_support(indices=True)

# Вывод выбранных признаков
print("Индексы выбранных признаков:", selected_features)
print("Выбранные признаки:\n", X_transformed.iloc[:, selected_features].head())
