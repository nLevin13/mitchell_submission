import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# reads MLCC data into a pandas Dataframe
data = pd.read_csv('MLCC.csv', header='infer')
feature_ids = ['6054','6055','6056','6057','6058',
                         '6060','6061','6062','6063','6064',
                         '6065','6066','6067','6068','6069',
                         '6070','6071','6072','6073','6074',
                         '6075','6076','6077','6078','6079',
                         '6080','6081','6082','6083','6084',
                         '6085','6087','6088','6089','7778',
                         '7779','7781','7782','7783','7786',
                         '7788','7789','7791','7792','7925',
                         '7926','7927','7928','7929','7930',
                         '7931','8445','8446','8447','8448',
                         '8449','8450']

# X is the list of feature vectors, and y is the result vector
X = data[feature_ids]
y = data.EvalResult

# automatically splits the data into training and test rows:
# X and y train will be used to form our Sigmoid function.
# X and y test will be used later to evaluate the accuracy
# of this model.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# fitting the Logistic Function. I'm using a logistic function
# becuase it's better than a linear function at representing 
# binary classification.
model = LogisticRegression(multi_class='ovr', max_iter=2000)
model.fit(X_train, y_train)

# we'll compare predicted_results to y_test to determine how
# good our model is.
# CAN REPLACE X_test WITH ANY DATAFRAME CONTAINING THE SAME
# FEATURE COLUMNS AS IN THE ORIGINAL CSV FILE
predicted_results = model.predict(X_test)

""" ------------EVALUATION METRIC EXPLANATION-------------
evaluation metrics based on the confusion matrix. The confusion matrix
is a 2 X 2 matrix in this case, since we only have two values that
EvalResult can take on. The diagonal of the matrix has the number of
predictions our model got right, while the other cells have the wrong
predictions. Accuracy measures the total right predictions / total
predictions. Precision measures the (total actual EvalResult=1 given 
predicted EvalResult=1) / (total predictions EvalResult=1). Similar to
precision, the recall value measures the (total actual EvalResult=1 given 
predicted EvalResult=1) / (total actual EvalResult=1).
"""
print("Accuracy:", metrics.accuracy_score(y_test, predicted_results))
print("Precision:", metrics.precision_score(y_test, predicted_results))
print("Recall:", metrics.recall_score(y_test, predicted_results))

""" ------------DIMENSIONALITY REDUCTION----------------
I wanted to also try to run Principal Component Analysis, which only
keeps the feature columns with the most variance. In effect, it removes
the columns which have a smaller effect on determining the EvalResult.
Using PCA has the advantage of reducing computing power, since the data
matrix we work with is much smaller. However, since we lose data, it may 
reduce the overall accuracy of the model. There should be a tradeoff between
these two qualities.
"""
