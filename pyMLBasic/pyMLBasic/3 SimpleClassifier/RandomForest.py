import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Common.LoadIris import LoadIris

forest = RandomForestClassifier(criterion = 'entropy' , n_estimators = 10 , random_state = 0 , n_jobs = 2)
 
