from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle

pd.set_option('display.max_columns', None)

df = pd.read_csv('../data/master_with_15.csv')
df_without = pd.read_csv('../data/master_without_15.csv')

for col in df.columns:
    if col in ['date', 'exit']:
        continue
    df[col] = df[col].round(2)

df_without['date'] = pd.to_datetime(df_without['date'])
df['date'] = pd.to_datetime(df['date'])

df2 = df_without.drop('exit', axis=1)
dates = df_without['date']
x = df2.drop('date', axis=1)
y = df_without['exit']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

cb = CatBoostClassifier(n_estimators=100, max_depth=5, random_state=42)
cb.fit(X_train, y_train)
filename = './catboost.sav'
pickle.dump(cb, open(filename, 'wb'))
y_pred = cb.predict(X_test)

y_pred = [x == 'True' for x in y_pred]

results = pd.DataFrame(columns=['date', 'timestamp', 'actual', 'predicted'])

count = 0
for i, row in X_test.iterrows():
    if y_pred[count]:
        if dates[i].date() not in results['date'].values or dates[i] < results['timestamp'][results['date'] == dates[i].date()].iloc[0]:
            results = results[results['date'] != dates[i].date()]
            new_df = pd.DataFrame({'date': [dates[i].date()], 'timestamp': [dates[i]], 'actual': [df['pl_points'][df['date'] == dates[i].replace(hour=15)]][0], 'predicted': [df['pl_points'][i]]})
            results = pd.concat([results, new_df])
    count += 1


results = results.sort_values('date')
results['actual_portfolio'] = results['actual'].cumsum()
results['predicted_portfolio'] = results['predicted'].cumsum()

results.reset_index(inplace=True, drop=True)

print(results['actual'])

print(sum(results['actual']), sum(results['predicted']))

# Plot the results actual portfolio vs predicted portfolio in the same graph
results.plot(y=['actual_portfolio', 'predicted_portfolio'], kind='line', figsize=(20, 10))
# plt.plot(results.index, results['actual_portfolio'], label='Actual')
# plt.plot(results.index, results['predicted_portfolio'], label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Portfolio')
plt.ylabel('Portfolio')
plt.show()

# print(sum(results['actual']), sum(results['predicted']))
print(results)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion matrix: {confusion_matrix(y_test, y_pred)}')
print(f'Classification report: {classification_report(y_test, y_pred)}')

# Plot the confusion matrix
plt.matshow(confusion_matrix(y_test, y_pred))
plt.show()

# Print the Classification Report as a table in the terminal
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

