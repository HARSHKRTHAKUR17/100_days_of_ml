# Pandas Profiling

This folder contains work with Pandas Profiling for automated exploratory
data analysis.

## What I Learned

- Automated dataset inspection
- Generating an EDA report
- Understanding dataset structure
- Inspecting variable statistics
- Identifying missing values
- Exploring distributions
- Detecting relationships between variables
- Exporting an EDA report as HTML

## Core Workflow

Load Dataset
→ Create ProfileReport
→ Generate Automated EDA
→ Export HTML Report

## Main Code

```python
from pandas_profiling import ProfileReport

prof = ProfileReport(df)

prof.to_file(output_file="output.html")