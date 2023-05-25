import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('test.csv')

data = data.iloc[2:]

data.columns = ['Homeowner', 'Credit Score', 'Years of Credit History', 'Revolving Balance', 'Revolving Utilization', 'Decision']

data['Revolving Balance'] = data['Revolving Balance'].str.replace('$', '').str.replace(',', '')
data['Revolving Utilization'] = data['Revolving Utilization'].str.replace('%', '')

data = data.apply(pd.to_numeric, errors='coerce')

data = data.dropna()

X = data[['Homeowner', 'Credit Score', 'Years of Credit History', 'Revolving Balance', 'Revolving Utilization']]
y = data['Decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

class_names = ["Not Approved", "Approved"]
y_pred_labels = [class_names[label] for label in y_pred]
y_pred_proba = model.predict_proba(X_test)

for i in range(len(X_test)):
    print(f"Instance {i+1}: Predicted label - {y_pred_labels[i]}, Probability - {y_pred_proba[i][1] * 100:.2f}%")