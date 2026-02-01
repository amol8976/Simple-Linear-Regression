import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# prepare the data
data = {
    'Hours_Studied' : [1,2,3,4,5,6,7,8,9,10],
    'Score' : [12,25,32,40,50,55,65,72,80,90]
}
df = pd.DataFrame(data)

#features and target
x = df[['Hours_Studied']] # Input (DataFrame)
y = df['Score']

# Split the data
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

#Train the Model 
model = LinearRegression()
model.fit(x_train , y_train)

#Make Predictions
y_pred = model.predict(x_test)
print("predictions : " , y_pred)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squarred Error : " , mse)

#Visualize results
#Convert x to 1D array for plotting
x_1d = x['Hours_Studied'].values
y_pred_full = model .predict(x).flatten()

plt.scatter(x_1d, y, color = 'blue', label = 'Actual Scores')
plt.plot(x_1d, y_pred_full, color = 'red', label = 'Predicted Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score ')
plt.title('Hours Studied vs Score Prediction')
plt.legend()
plt.show()


