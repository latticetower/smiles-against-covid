# smiles-against-covid

Our team's solutions for https://globalai.innopolis.university/eng are collected here.

## Meaningful submits only are listed here - both with descriptions

In the following list item number == submit number. Current best score should be **emphasized with bold font like this**. To teammates: if you submit something, please add score+description+code to the end of this list. This way we could remember what was submitted and when.

2. "Lazy" baseline score (all 1): 0.068222621184919. This means that both train and test are imbalanced, test set has the same proportion of 1 and 0 as in the train test. Possible approaches to solve this using supervised methods include outlier detection algorithms or require data balancing.

4. `submission_pubchem_fingerprints.csv`: this uses my old solution from Learning to smell challenge, full code is given at `notebooks/eda_v1.ipynb`. (unsupervised, uses good precomputed fingerprints+k-nearest neighbours).

    Score on challenge's leaderboard: **0.21455938697318**