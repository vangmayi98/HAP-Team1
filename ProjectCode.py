import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("./data.csv")
df = pd.DataFrame(dataset)

df = df.convert_objects(convert_numeric=True)
f = df.infer_objects()
x = df.iloc[:, :-1].values
y = df.iloc[:, 13].values

df2 = pd.DataFrame(y)
df2 = df2.convert_objects(convert_numeric=True)
df2 = df2.infer_objects()


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 0:13])
x[:, 0:13] = imputer.transform(x[:, 0:13])


sc_X = StandardScaler()
x = sc_X.fit_transform(x)

#data splitting#
x_train = x[:int((len(x)+1)*.80)]
x_test = x[int(len(x)*.80+1):]
y_train = y[:int((len(y)+1)*.80)]
y_test = y[int(len(y)*.80+1):]

print("No of train values in x_train", len(x_train))
print("No of train values in y_train", len(y_train))
print("No of train values in x_test", len(x_test))
print("No of train values in y_test", len(y_test))


result = RandomForestRegressor(n_estimators=20, random_state=0)
result.fit(x_train, y_train)
y_predict = result.predict(x_test)

print("y_predict values: \n", y_predict)

a = 1
print("Testing dataset results here!")
print("Heart attack probability percentages ")

for i in y_predict:
    print("for testcase ", a, "is: ", round(i*100, 2), "%")
    a += 1



print('\nMean Absolute Error is:\n', (metrics.mean_absolute_error(y_test, y_predict))*100 , "%\n")
print('Mean Squared Error is:\n', (metrics.mean_squared_error(y_test, y_predict))*100 , "%\n")
print('Root Mean Squared Error is:\n', (np.sqrt(metrics.mean_squared_error(y_test, y_predict)))*100 , "%\n")