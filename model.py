import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import data_processing
import json

#Load data
'''
with open('credentials.json') as json_data:
    credentials = json.load(json_data)

pr.connect_to_redshift(dbname = 'muni',
                    host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                    port = '5439',
                    user = credentials['user'],
                    password = credentials['password'])

df = pr.redshift_to_pandas("select * from vehicle_monitoring")

df.to_csv('data/vehicle_monitoring.csv', index=False)
'''

df = pd.read_csv('data/vehicle_monitoring.csv')

# Params to pass to the GridSearchCV
param_grid = {
    'loss': ['log'],
    'penalty': ['elasticnet'],
    'alpha': [10 ** x for x in range(-6, 1)],
    'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
}


# Create the pipeline to GridSearch over
pipeline = Pipeline(steps=[
                ('raw_to_stops',data_processing.RawToStops()),
                ('stops_to_durations',data_processing.StopsToDurations()),
                ('durations_to_distributions',data_processing.DurationsToDistributions()),
                ('create_features',data_processing.CreateFeatures()),
                ('sgd', SGDRegressor())])


# Create the GridSearch model
model = GridSearchCV(pipeline,
                   param_grid,
                   cv=5,
                   scoring=None
                   n_jobs=-1,
                   verbose=3)


# Fit the GridSearch model
model.fit(df.drop('fraud', axis=1), df['fraud'])

# Save the best model to a pickle file
pickle.dump(model.best_estimator_, open('best_model_2.p', 'wb'))

print('The ROC AUC score was: {}'.format(model.best_score_))
print()
print('The best parameters were:')
print(model.best_params_)

print(pd.Series(model.best_estimator_.feature_imporances_, index=model.best_params_['keep_columns__keep_columns']).sort_values())

pr.close_up_shop()
