#imports
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#training data
X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40]])
y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female'])

#test data
test = np.array([[159, 55, 37], [171, 75, 42], [181, 85, 43]])
expected = np.array(['female', 'male', 'male'])

#function to verify predictions
def predictModel(model):
    model = model.fit(X,y)
    prediction = model.predict(test)
    if np.array_equal(expected, prediction):
        print(model, "Passed")
    else:
        print(model, "Failed with this resul:", prediction)
   
#decition tree classifier
clf = tree.DecisionTreeClassifier()
predictModel(clf)

#support vector classifier
svc = SVC(gamma="auto")
predictModel(svc)

#k-nearest classifier
neigh = KNeighborsClassifier(n_neighbors=3)
predictModel(neigh)


#random forest classifier
randomF = RandomForestClassifier(max_depth=2, random_state=0)
predictModel(randomF)
