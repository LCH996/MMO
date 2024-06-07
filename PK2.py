import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Загрузка набора данных SMS Spam Collection
data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Преобразование меток в бинарные значения

X = data['message']
y = data['label']

# Разделение набора данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Извлечение признаков с использованием CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Извлечение признаков с использованием TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Определение и обучение классификатора Random Forest
rf_classifier_counts = RandomForestClassifier(random_state=42)
rf_classifier_counts.fit(X_train_counts, y_train)
y_pred_rf_counts = rf_classifier_counts.predict(X_test_counts)

rf_classifier_tfidf = RandomForestClassifier(random_state=42)
rf_classifier_tfidf.fit(X_train_tfidf, y_train)
y_pred_rf_tfidf = rf_classifier_tfidf.predict(X_test_tfidf)

# Определение и обучение логистической регрессии
lr_classifier_counts = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier_counts.fit(X_train_counts, y_train)
y_pred_lr_counts = lr_classifier_counts.predict(X_test_counts)

lr_classifier_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier_tfidf.fit(X_train_tfidf, y_train)
y_pred_lr_tfidf = lr_classifier_tfidf.predict(X_test_tfidf)

# Оценка производительности классификаторов
print("Random Forest с CountVectorizer:")
print("Точность:", accuracy_score(y_test, y_pred_rf_counts))
print(classification_report(y_test, y_pred_rf_counts))

print("Random Forest с TfidfVectorizer:")
print("Точность:", accuracy_score(y_test, y_pred_rf_tfidf))
print(classification_report(y_test, y_pred_rf_tfidf))

print("Логистическая регрессия с CountVectorizer:")
print("Точность:", accuracy_score(y_test, y_pred_lr_counts))
print(classification_report(y_test, y_pred_lr_counts))

print("Логистическая регрессия с TfidfVectorizer:")
print("Точность:", accuracy_score(y_test, y_pred_lr_tfidf))
print(classification_report(y_test, y_pred_lr_tfidf))

# Сравнение результатов различных методов и вывод о лучшей комбинации
