# smiles-against-covid

Our team's solutions for https://globalai.innopolis.university/eng are collected here.

## Meaningful submits only are listed here - both with descriptions

In the following list item number == submit number. Current best score should be **emphasized with bold font like this**. To teammates: if you submit something, please add score+description+code to the end of this list. This way we could remember what was submitted and when.

2. "Lazy" baseline score (all 1): 0.068222621184919. This means that both train and test are imbalanced, test set has the same proportion of 1 and 0 as in the train test. Possible approaches to solve this using supervised methods include outlier detection algorithms or require data balancing.

4. `submission_pubchem_fingerprints.csv`: this uses my old solution from Learning to smell challenge, full code is given at `notebooks/eda_v1.ipynb`. (unsupervised, uses good precomputed fingerprints+k-nearest neighbours).

    Score on challenge's leaderboard: 0.21455938697318

5. `submission_pubchem_fingerprints-2.csv` Require two active neighbours to classify as active. Except that uses the same notebook as was in the previous submit.

   Score on challenge's leaderboard: 0.29787234042553

6. `catboost_predictions_v0.csv`: this was made with `code/solution_catboost_v0.py`. CatboostClassifier with auto class balancing+pubchem fingerprints as a features.

    Score on challenge's leaderboard: **0.37096774193548**
