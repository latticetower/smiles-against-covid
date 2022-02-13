# Code solutions 
These can be run from the current folder, or converted to jupyter notebooks and saved in `notebooks` folder (for now haven't checked conversion yet).

1. CatboostClassifier+Pubchem fingerprints
This solution is run from this folder like this:
```
python solution_catboost_v0.py
```
Currently no parameters (all parameters are hardcoded).
Train data balancing is made now via catboost's auto_class_weights="Balanced" parameter.