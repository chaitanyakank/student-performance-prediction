# student-performance-prediction
Predict student performance using machine learning

Overview
This project aims to predict the final grades (G3) of students based on various features like their family background, school-related activities, and previous academic performance. The dataset used contains student data with various attributes such as age, family support, health status, and study time.

Features
Predictive Model: The project uses machine learning techniques to predict student performance.

Dataset: The dataset contains 395 samples with 33 features including personal and educational information of the students.

Evaluation Metrics: The model performance is evaluated using Mean Squared Error and R² Score.

Technologies Used
Python: The programming language used for building the predictive model.

Libraries:

pandas: For data manipulation and analysis.

matplotlib: For plotting and visualizations.

scikit-learn: For machine learning algorithms and metrics.

numpy: For numerical operations.

Machine Learning:

Linear Regression: Used to predict the student's final grade based on the input features.

How to Run

Clone this repository to your local machine using:
git clone https://github.com/chaitanyakank/student-performance-prediction.git

Navigate to the project folder:
cd student-performance-prediction

Install the necessary libraries (if not already installed):
pip install -r requirements.txt

Run the Python script student_performance_prediction.py to train and test the model:
python student_performance_prediction.py

The model will be trained on the student data and the performance will be printed (MSE and R² Score).

Dataset
The dataset used is the "Student Performance Dataset," which includes information such as:

School (GP or MS)

Sex (M or F)

Age

Address (U or R)

Family size

Study time

Previous failures

Health status

Family support

Internet access

Final grades (G1, G2, G3)

Results
Mean Squared Error: 5.032

R² Score: 0.7545

These results show that the model explains about 75% of the variance in student performance.

Contributing
Feel free to fork this repository and submit pull requests. Any contributions are welcome, whether it's improving the model, fixing bugs, or adding new features!
