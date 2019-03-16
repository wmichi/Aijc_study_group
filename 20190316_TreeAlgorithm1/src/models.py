from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, X_val, y_train, y_val, args):
  if args.modelname == 'RandomForest':
    model = RandomForestClassifier(n_estimators=args.n_estimators, oob_score=args.oob_score, random_state=args.randomseed)
  elif args.modelname == 'GradientBoosting':
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, random_state=args.randomseed)
  elif args.modelname == 'AdaBoost':
    model = AdaBoostClassifier(random_state=args.randomseed)
  elif args.modelname == 'Bagging':
    model = BaggingClassifier(random_state=args.randomseed)
  else:
    model = DecisionTreeClassifier(random_state=args.randomseed)

  model.fit(X_train, y_train)
  y_pred_train = model.predict(X_train)
  y_pred_val = model.predict(X_val)

  train_acc = accuracy_score(y_train, y_pred_train)
  validation_acc = accuracy_score(y_val, y_pred_val)

  # TODO: add oob_score and feature_importance

  return model, train_acc, validation_acc
