
# Code solutions 
These can be run from the current folder, or converted to jupyter notebooks and saved in `notebooks` folder (for now haven't checked conversion yet).

1. CatboostClassifier+Pubchem fingerprints
This solution is run from this folder like this:
```
python solution_catboost_v0.py
```
Currently no parameters (all parameters are hardcoded).
Train data balancing is made now via catboost's auto_class_weights="Balanced" parameter.

# Additional scripts
1. `code/feature_extraction/prepare_molecules.py`: this scripts embeds molecules from train and test datasets, saves them for future use

2. `code/feature_extraction/prepare_alignit.py`: from sdf with molecules computes (with align-it wrapper software) pharmacophores and saves them to the files.

2. `code/feature_extraction/prepare_pmapper.py`: from sdf with molecules computes (with pmapper wrapper software) pharmacophores and saves them to the files.