import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reading the csv and converting student_id into a numeric index
student_predict_df = pd.read_csv('student_habits_performance.csv')
student_predict_df = student_predict_df.dropna()
student_predict_df['index'] = pd.factorize(student_predict_df['student_id'])[0]
student_predict_df.set_index('index')
student_predict_df = student_predict_df.drop('student_id', axis=1)

# Encoding the dataframe in order to read categorical values.
encoded_df = pd.get_dummies(student_predict_df)

X = encoded_df.drop(columns=['index', 'exam_score'])
y = student_predict_df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Force all predictions to be within the range of 0-100
y_pred = np.clip(y_pred, 0, 100)

# Calculate mean squared error and root mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Plotting the results
plt.figure(figsize=(10,10))
sb.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Actual vs Predicted Exam Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Exporting the results into a csv.
result_file = X_test.copy()
result_file['exam_score'] = y_test
result_file['predicted_score'] = y_pred
result_file['error'] = np.abs(y_test - y_pred)
result_file = result_file.sort_values(by='error', ascending=False)
result_file.to_csv('exam_score_prediction_errors.csv', index=True)