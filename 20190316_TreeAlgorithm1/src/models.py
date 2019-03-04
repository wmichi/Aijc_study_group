from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(modelname, X_train, X_val, y_train, y_val):
  if modelname == 'RandomForest':
    None
  else:
    model = DecisionTreeClassifier()

  model.fit(X_train, y_train)
  y_pred_train = model.predict(X_train)
  y_pred_val = model.predict(X_val)

  train_acc = accuracy_score(y_train, y_pred_train)
  validation_acc = accuracy_score(y_val, y_pred_val)

  return model, train_acc, validation_acc
