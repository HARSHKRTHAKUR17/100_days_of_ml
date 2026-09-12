Complete Case Analysis (CCA)

This repository demonstrates Complete Case Analysis (CCA) for handling missing values in a machine learning dataset.

CCA removes rows containing missing values from the selected features and allows us to examine how much data is lost as a result.

📌 What This Project Covers

Identifying missing values

Calculating missing-value percentages

Selecting features with a small proportion of missing values

Applying Complete Case Analysis

Comparing dataset size before and after CCA

Measuring the percentage of data retained and removed

Examining how CCA affects numerical feature statistics

📊 Dataset

The project uses the Data Science Job dataset.

The dataset contains features such as:

city

city_development_index

gender

relevent_experience

enrolled_university

education_level

major_discipline

company_size

company_type

training_hours

target

🔍 Complete Case Analysis

The implementation identifies columns with a relatively small amount of missing data and then performs CCA on those columns.

new_df = df[cols].dropna()

The amount of data retained is calculated using:

retention_ratio = len(new_df) / len(df)

This quantifies the impact of removing incomplete observations.

⚠️ Important Consideration

CCA is simple, but removing observations can lead to data loss.

Before using CCA in a real ML project, consider:

How much data will be removed?

Is the missingness random?

Could removing observations introduce bias?

Would imputation be a better alternative?

CCA should therefore be treated as a preprocessing decision rather than simply applying dropna() blindly.

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib

📁 Project Structure

Complete-Case-Analysis/
│
├── missing_value_imputation_vscode.py
├── data_science_job.csv
└── README.md

▶️ How to Run

Clone the repository and navigate into the project directory:

git clone <repository-url>
cd Complete-Case-Analysis

Install the required libraries:

pip install numpy pandas matplotlib

Run the script:

python missing_value_imputation_vscode.py

🎯 Learning Objective

The main objective is to understand how Complete Case Analysis works, how much information it can remove, and why the amount of missing data alone isn't enough to decide whether rows should be deleted.