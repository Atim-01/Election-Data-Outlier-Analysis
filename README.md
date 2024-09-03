# Outlier Detection in Anambra Election Data

## Table of Contents

1.  [Project Overview](#project-overview)
    -   [Tools and Technologies](#tools-and-technologies)
    -   [Dataset](#dataset)
2.  [Installation](#installation)
3.  [Approach](#approach)
    -   [Data Cleaning](#data-cleaning)
    -   [Neighbor Identification](#neighbor-identification)
    -   [Outlier Score Calculation](#outlier-score-calculation)
    -   [Visualization of Outlier Scores](#visualization-of-outlier-scores)
    -   [Sorting and Reporting](#sorting-and-reporting)
4.  [Key Insights](#key-insights)
    -   [Notable Outliers](#notable-outliers)
5.  [Conclusion](#conclusion)
6.  [Visualization Outputs](#visualization-outputs)

## Project Overview

This project involves detecting outliers in the voting patterns of polling units in Anambra State, Nigeria, during the most recent elections. The goal was to identify polling units with voting results that significantly deviate from their neighboring units, which could indicate potential irregularities.

### **Tools and Technologies**

-   **Programming Language**: Python 3.0
-   **Libraries and Packages**:
    -   **Pandas**: Data manipulation and analysis.
    -   **NumPy**: Numerical computations.
    -   **Geopy**: Geocoding and geospatial calculations.
    -   **Matplotlib** & **Seaborn**: Data visualization.
-   **Development Environment**: Jupyter Notebook for interactive analysis and documentation.

### Dataset

The dataset used for this project is included in the repository under the filename `Anambra_Election_Data.xlsx`. It contains voting data from Anambra State, Nigeria, and includes the following columns:

-   `Polling Unit`
-   `Latitude`
-   `Longitude`
-   `APC`
-   `LP`
-   `PDP`
-   `NNPP`

### Installation

To set up this project on your local machine, follow these steps:

1. **Ensure you have Python installed (version 3.7 or higher):**

   - Download and install Python from the official [Python website](https://www.python.org/downloads/).

2. **Install the required Python libraries:**

   Open your terminal and run the following command to install the necessary libraries:

   ```bash
   pip install pandas geopy matplotlib openpyxl
   ```

3. **Dataset:**

   The dataset (`Anambra_Election_Data.xlsx`) is included in the repository. There is no need to download it separately.

## Approach
### 1. **Data Cleaning**:

-   Handled missing or incomplete entries by imputing or removing as appropriate.
-   Standardized naming conventions for consistency.

### 2. **Neighbor Identification**

- **Objective:** Identify neighboring polling units within a 1 km radius for each unit.
- **Method:** I used geodesic distance to calculate the proximity between polling units. Each unit's neighbors were identified based on their geographical coordinates.

### 3. **Outlier Score Calculation**

- **Objective:** Calculate an outlier score for each political party in every polling unit by comparing the votes with those of its neighbors.
- **Method:** The outlier score was determined by the absolute difference between the votes in a polling unit and the average votes of its neighboring units.

### 4. **Visualization of Outlier Scores**

- **Objective:** Visualize the distribution of outlier scores for each party to gain insights into voting patterns.
- **Method:** Histograms and bar charts were used to illustrate the frequency and magnitude of outlier scores for different parties across the polling units.

### 5. **Sorting and Reporting**

- **Objective:** Identify the top polling units with the highest outlier scores.
- **Method:** The dataset was sorted by outlier scores, and the top outliers were selected for further analysis. These units exhibited significant deviations in voting patterns, prompting further investigation.

## Key Insights

- **APC:** The average outlier score was relatively low, suggesting consistent voting patterns across polling units.
- **LP:** Exhibited a higher average outlier score, indicating more varied voting patterns compared to other parties.
- **PDP & NNPP:** Both parties showed low average outlier scores, similar to APC.

### Notable Outliers

- **Community Secondary School Odekpe I:** Had the highest outlier scores for APC and LP, indicating unusual voting behavior.
- **Okoti Market Square I:** Showed significant deviations for LP and PDP.
- **Community Secondary School Odekpe II:** Also displayed notable outlier scores, though to a lesser extent.

## Conclusion

The analysis highlights specific polling units in Anambra State that require further scrutiny due to their anomalous voting patterns. These findings could be useful for electoral monitoring and ensuring the integrity of election results.

## Visualization Outputs

- **Histograms:** Distribution of outlier scores for APC, LP, PDP, and NNPP.
- **Bar Charts:** Top 3 polling units with the highest outlier scores for each party.


```python

```
