#import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv('country_wise_latest.csv')
#view dataset - first 5 rows
df.head()
# view data - last 5 rows
df.tail()
#check shape of data - number of rows and columns 
print('The number of rows and columns in the dataset are:')
print('Rows:', df.shape[0])
print('Columns:', df.shape[1])
#view data info - data types and non-null count
df.info()
#check for missing values
df.isnull().sum()
#check for missing values
df.isnull().sum()
# Renaming Countr/Region column to Country
df = df.rename(columns={'Country/Region' : 'Country'})
# Confirming the name changes made.
df.head()
#statistical description of data
df.describe()
# create table for confirmed cases across different regions using groupby function
Region_Confirmed = df.groupby('WHO Region')['Confirmed'].sum().reset_index()

# view table
print(Region_Confirmed)

#plot the confirmed cases across regions
plt.figure(figsize=(10,6))
sns.barplot(x='WHO Region',y='Confirmed', data=Region_Confirmed)
sns.set_style('darkgrid')
plt.title('Confirmed Case Per Region')
plt.xlabel('Region')
plt.ylabel('Count');
# create table for death cases across different regions using groupby function
Region_Death = df.groupby('WHO Region')['Deaths'].sum().reset_index()

#view table
print(Region_Death)

#plot the confirmed Death across regions
plt.figure(figsize=(10,6))
sns.barplot(x='WHO Region',y='Deaths', data=Region_Death)
sns.set_style('darkgrid')
plt.title('Confirmed Death Per Region')
plt.xlabel('Region')
plt.ylabel('Count');
# create table for Recovered  covid-19 patients across different regions using groupby function
Region_Recovered = df.groupby('WHO Region')['Recovered'].sum().reset_index()

#view table
print(Region_Recovered)

#plot the Recovery across regions
plt.figure(figsize=(10,6))
sns.barplot(x='WHO Region',y='Recovered', data=Region_Recovered)
sns.set_style('darkgrid')
plt.title('Recovered Cases Per Region')
plt.xlabel('Region')
plt.ylabel('Count');
# create table for Active cases across different regions using groupby function
Region_Active = df.groupby('WHO Region')['Active'].sum().reset_index()

#view table
print(Region_Active)

#plot the Actve cases of Covid-19 across the regions
plt.figure(figsize=(10,6))
sns.barplot(x='WHO Region',y='Active', data=Region_Active)
sns.set_style('darkgrid')
plt.title('Active Cases Per Region')
plt.xlabel('Region')
plt.ylabel('Count');
# Corrolation heatmap with the data
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
# Get the unique countries for binning
df['Country'].unique()
Country_map = {
        'Afghanistan':0, 'Albania':1, 'Algeria':2, 'Andorra':3, 'Angola':4,
       'Antigua and Barbuda':5, 'Argentina':6, 'Armenia':7, 'Australia':8,
       'Austria':9, 'Azerbaijan':10, 'Bahamas':11, 'Bahrain':12, 'Bangladesh':13,
       'Barbados':14, 'Belarus':15, 'Belgium':16, 'Belize':17, 'Benin':18, 'Bhutan':19,
       'Bolivia':20, 'Bosnia and Herzegovina':21, 'Botswana':22, 'Brazil':23,
       'Brunei':24, 'Bulgaria':25, 'Burkina Faso':26, 'Burma':27, 'Burundi':28,
       'Cabo Verde':29, 'Cambodia':30, 'Cameroon':31, 'Canada':32,
       'Central African Republic':33, 'Chad':34, 'Chile':35, 'China':36, 'Colombia':37,
       'Comoros':38, 'Congo (Brazzaville)':39, 'Congo (Kinshasa)':40, 'Costa Rica':41,
       "Cote d'Ivoire":42, 'Croatia':43, 'Cuba':44, 'Cyprus':45, 'Czechia':46, 'Denmark':47,
       'Djibouti':48, 'Dominica':49, 'Dominican Republic':50, 'Ecuador':51, 'Egypt':52,
       'El Salvador':53, 'Equatorial Guinea':54, 'Eritrea':55, 'Estonia':56,
       'Eswatini':57, 'Ethiopia':58, 'Fiji':59, 'Finland':60, 'France':61, 'Gabon':62,
       'Gambia':63, 'Georgia':64, 'Germany':65, 'Ghana':66, 'Greece':67, 'Greenland':68,
       'Grenada':69, 'Guatemala':70, 'Guinea':71, 'Guinea-Bissau':72, 'Guyana':73,
       'Haiti':74, 'Holy See':75, 'Honduras':76, 'Hungary':77, 'Iceland':78, 'India':79,
       'Indonesia':80, 'Iran':81, 'Iraq':82, 'Ireland':83, 'Israel':84, 'Italy':85,
       'Jamaica':86, 'Japan':87, 'Jordan':88, 'Kazakhstan':89, 'Kenya':90, 'Kosovo':91,
       'Kuwait':92, 'Kyrgyzstan':93, 'Laos':94, 'Latvia':95, 'Lebanon':96, 'Lesotho':97,
       'Liberia':98, 'Libya':99, 'Liechtenstein':100, 'Lithuania':101, 'Luxembourg':102,
       'Madagascar':103, 'Malawi':104, 'Malaysia':105, 'Maldives':106, 'Mali':107, 'Malta':108,
       'Mauritania':109, 'Mauritius':110, 'Mexico':111, 'Moldova':112, 'Monaco':113,
       'Mongolia':114, 'Montenegro':115, 'Morocco':116, 'Mozambique':117, 'Namibia':118,
       'Nepal':119, 'Netherlands':120, 'New Zealand':121, 'Nicaragua':122, 'Niger':123,
       'Nigeria':124, 'North Macedonia':125, 'Norway':126, 'Oman':127, 'Pakistan':128,
       'Panama':129, 'Papua New Guinea':130, 'Paraguay':131, 'Peru':132, 'Philippines':133,
       'Poland':134, 'Portugal':135, 'Qatar':136, 'Romania':137, 'Russia':138, 'Rwanda':139,
       'Saint Kitts and Nevis':140, 'Saint Lucia':141,
       'Saint Vincent and the Grenadines':142, 'San Marino':143,
       'Sao Tome and Principe':144, 'Saudi Arabia':145, 'Senegal':146, 'Serbia':147,
       'Seychelles':148, 'Sierra Leone':149, 'Singapore':150, 'Slovakia':151, 'Slovenia':152,
       'Somalia':153, 'South Africa':154, 'South Korea':155, 'South Sudan':156, 'Spain':157,
       'Sri Lanka':158, 'Sudan':159, 'Suriname':160, 'Sweden':161, 'Switzerland':162, 'Syria':163,
       'Taiwan*':164, 'Tajikistan':165, 'Tanzania':166, 'Thailand':167, 'Timor-Leste':168,
       'Togo':169, 'Trinidad and Tobago':170, 'Tunisia':171, 'Turkey':172, 'US':173, 'Uganda':174,
       'Ukraine':175, 'United Arab Emirates':176, 'United Kingdom':177, 'Uruguay':178,
       'Uzbekistan':179, 'Venezuela':180, 'Vietnam':181, 'West Bank and Gaza':182,
       'Western Sahara':183, 'Yemen':184, 'Zambia':185, 'Zimbabwe':186}
# Map the Country columns with unique numbers
df['Country'] = df['Country'].map(Country_map)
# Confirming the country mapping
df.head()
# Get the Who Regions names for binnig 
df['WHO Region'].unique()
# WHO Region mapping
WHO_Region_map = {'Eastern Mediterranean': 0, 'Europe': 1, 'Africa': 2, 'America':3,
       'Western Pacific': 4, 'South-East Asia': 5}
# WHO Region mapping
df['WHO Region'] = df['WHO Region'].map(WHO_Region_map)
# Selecting features and target
x = df[['Country', 'Deaths','Confirmed', 'Recovered', 'Active', 'New cases','New deaths', 'New recovered', 'Deaths / 100 Cases', 'Recovered / 100 Cases', 'Confirmed last week', '1 week change', '1 week % increase']]  # Features
y = df['WHO Region']                                                                              # Target
# Shape of training and testing data 
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Shape of Training and Test data
print('The shape of the train data is', x_train.shape)
print('The shape of the test data is', x_test.shape)
from sklearn.ensemble import RandomForestClassifier
# Instatiating the model
model = RandomForestClassifier()
from sklearn.metrics import mean_squared_error
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(predictions[:15])
mse = mean_squared_error(y_test, predictions)
#import sk;earn metrics for Classification report
from sklearn.metrics import classification_report
#Model evaluation
print(classification_report(y_test, predictions))
#Check model performance score
print('Training set score: {:.2f}'.format(model.score(x_train, y_train)))
print('Testing set score: {:.2f}'.format(model.score(x_test, y_test)))

