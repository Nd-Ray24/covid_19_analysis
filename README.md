Project Name: Global Covid-19 Statistics and Trends
Overview:
This Project provides a comprehensive overview of COVID-19 statistics across various countries and regions. It included key metrics such as the total number of confirmed cases, deaths, and recoveries, as well as active cases. Additionally, it tracks new cases, deaths, and recoveries reported recently.
Project Workflow
1.	Data collection
2.	Data cleaning and preparation
3.	Descriptive statistics
4.	Exploratory Data Analysis
5.	Feature Transformation/Engineering
1. Data Collection
The dataset for this project was collected from Kaggle, which includes COVID-19 case counts, demographic data, and various health metrics. The dataset is structured with 187 rows, 15 columns`.
The necessary libraries were imported and dataset loaded. 
2. Data Cleaning and Preparation
The dataset was check for missing and duplicate values. The data was free from duplicates and missing values before further use of the dataset.
The column headers with conflicting names were adjusted.
3. Descriptive/Exploratory Data Analysis
This section contains the analysis of covid-19 across various Countries and Regions.
a.	Analysis of Confirmed Cases of Covid-19 per Region:
The analysis showed that the American Region recorded the highest number of confirmed covid-19 cases with 8,299,528 cases. This is followed by Europe with 3,299523 cases, South-East Asia with 1,835297, Eastern Mediterranean with 1,490744, Africa with 723, 205 while Western Pacific 292,428
b.	 Analysis of Covid-19 Deaths per Region.
The analysis of deaths from Covid-19 showed that followed confirmed cases trends with America recording the highest deaths with 342,732 followed by Europe with 211,144, South-East Asia, 41, 349, Eastern Mediterranean 38,339, Africa 12,223 and Western Pacific record the least with 8,249 cases.
c.	
Correlation:  
Correlation heatmap (chart) was used to examine the relationships between the variables. The chart showed strong relationship between most variables, such as Confirmed, Deaths, Recovered, Active, New Cases, New Deaths, New Recovered, Confirmed last week and one week change.

Feature Transformation/Engineering: 
Feature Engineering involves the process of transforming data into feature (feature vectors) that are suitable for machine learning algorithm. Feature vector is an n-dimensional vector or numerical features that represent objects. 
Feature transformation was carried out on string categorical data such as Country and WHO Region making them suitable for machine learning and prediction.
Random forest classifier model:
The linear Regression model was used to predict covid-19 cases by region. The data was divided into training and testing dataset in the ratio of 80% : 20%.

