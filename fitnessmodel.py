import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

#LOADING CSV FILES
try:
    df1 = pd.read_csv(r'C:\Users\lenovo\OneDrive\Documents\Gargi\python\MentalFitnessTracker\mental-and-substance-use-as-share-of-disease -AI.csv')
    df2 = pd.read_csv(r'C:\Users\lenovo\OneDrive\Documents\Gargi\python\MentalFitnessTracker\prevalence-by-mental-and-substance-use-disorder _AI.csv')
    print("file loaded")
    #print(df1.head())
    #print(df2.head())
except FileNotFoundError:
    print("File not found. Check the path.")


#MERGING DATASETS
data=pd.merge(df1,df2)



#DATA CLEANING
print(data.isnull().sum())              #total missing values in each column
data.drop("Code",axis=1,inplace=True)   #dropping column 'code' as not required for training 
print(data.size,data.shape)
#renaming column names
data.columns = ['Country','Year','MentalFitness','Schizophrenia','Bipolar','Eating','Anxiety','Drug','Depressive','Alcohol']

# Check the distribution of your target variable
print("Mental Fitness Distribution:")
print(f"Min: {data['MentalFitness'].min()}")
print(f"Max: {data['MentalFitness'].max()}")
print(f"Mean: {data['MentalFitness'].mean()}")
print(f"Std: {data['MentalFitness'].std()}")




#PREPROCESSING DATA
#tranfering non-numerical labels to numerical label
from sklearn.preprocessing import LabelEncoder    #labelencoder is used to normalize labels
l=LabelEncoder()                                  #creating encoder
for i in data.columns:
    if data[i].dtype=='object':
        data[i]=l.fit_transform(data[i])
print(data.shape)

print("-----describe data before scaled")
print(data.describe())  #to get the statistical summary of the dataset




#Normalization of data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for col in data.columns:
    data[col]=scaler.fit_transform(data[[col]])  # Fit scaler on each column of training data

print("-------scaled data")
print(data)
print("-----describe scaled data")
print(data.describe())  #to get the statistical summary of the dataset




#SPLITTING DATA
feature_columns = ['Country', 'Year', 'Schizophrenia', 'Bipolar', 'Eating', 'Anxiety', 'Drug', 'Depressive', 'Alcohol']
X = data[feature_columns]
y = data['MentalFitness']
print(X)
print(y)

from sklearn.model_selection import train_test_split 
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.20, random_state=2)
print(xtrain)
print(xtest)
print("------describe train")
print(xtrain.describe())  #to get the statistical summary of the dataset
print("xtrain:",xtrain.shape)
print("xtest:",xtest.shape)
print("ytrain:",ytrain.shape)
print("ytest:",ytest.shape)




#TRAINING MODEL- RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rf=RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(xtrain,ytrain)         #fit training data



#model evaluation for training set
y_train_pred = rf.predict(xtrain)
y_test_pred = rf.predict(xtest)

#mean square error is the average of the square of the difference b/w observed n predicted values of a variable
mse= mean_squared_error(ytrain, y_train_pred)   

#root mean square error is the average difference b/w values predicted by model n the actual value
rmse= (np.sqrt(mean_squared_error(ytrain, y_train_pred)))

#r2 or coefficient of determination is a measure that provide info about the goodness of fit of a model
r2=r2_score(ytrain, y_train_pred)

# Test metrics
test_mse = mean_squared_error(ytest, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(ytest, y_test_pred)

print("Random Forest Regressor Model for training set:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 score:", r2)

print("Random Forest Regressor Model for test set:")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R2 score:", test_r2)




#SAVING MODEL FOR LATER USE
import joblib
joblib.dump(rf, 'mental_fitness_model.pkl')
joblib.dump(l, 'country_encoder.pkl')

#Save the scaler for later use in prediction (e.g., Flask app)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')
print("succefully trained model!!!")



# Feature importance analysis
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)