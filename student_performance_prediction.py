import pandas as pd

# Load the dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Show basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n")
print(df.head())
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode all object (categorical) columns
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Confirm encoding
print("\nData Types after encoding:\n", df.dtypes)
# Define features and target
X = df.drop('G3', axis=1)  # Features (input)
y = df['G3']               # Target (output)
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed.")
from sklearn.metrics import mean_squared_error, r2_score

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
import matplotlib.pyplot as plt

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel("Actual G3 Grades")
plt.ylabel("Predicted G3 Grades")
plt.title("Actual vs Predicted Final Grades (G3)")
plt.grid(True)
plt.plot([0, 20], [0, 20], color='red', linestyle='--')  # ideal line
plt.show()
