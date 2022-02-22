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

    Score on challenge's leaderboard: 0.37096774193548

13. `catboost_predictions_v0_1.csv`: used pubchem's fingerprints computed by DeepPurpose.utils.smiles2pubchem function instead of precomputed ones (yes, there was a non-critical bug in my old code, and yes, these fingerprints differ). Using the same parameters as before.

    Score on a challenge's leaderboard: 0.3577

14. `catboost_predictions_v0_1.csv`: set depth to 6 and rerun
    Score on a challenge's leaderboard: 0.3654

15-16. 2 submits with DeepPurpose. Latest version of code is at notebooks/deep_purpose_v0.ipynb
    Scores: 0.2326 and 0.2158 (more like terrible than great)

    The 16th submit differs in terms of best model selection (I'm using PR AUC score for this) and data preprocessing (converting everything to canonical SMILES before processing).

17. `catboost_predictions_v0_1.csv`: previous catboost submit was buggy, need to fix it
    Score on a challenge's leaderboard: **0.3740**

18. `catboost_predictions_v0_2.csv`: tweak catboost params
    Score on a challenge's leaderboard: 0.2555

19. `catboost_predictions_v0_3.csv`: tweak params, morgan+cactvs
    Score on a challenge's leaderboard: 0.2596

20. depth=4, score=0.3125

21. use stratifiedgroupkfold with tweaked params, score=0.2921

22. tweak params 0.3607 (don't use standartise yet)

23. with standartise on: 0.3594

24. use canonical, nfolds=11, 0.3387

25. remove duplicates from train: didn't helped, 0.2887

26. Full fragment library based on train+MultinomialNB: 0.2637

27. add opt theshold (the same idea for this as in the deeppurpose code): 0.3068

28. use 1-mean active as a threshold. 0.3609

29. `notebooks/ensemble_submission.ipynb`: took 13 solutions and decide by voting, active or not. Score=0.3256

30. `notebooks/ensemble_submission.ipynb`: remove several bad submissions(based on f1 score, even number of values), score = 0.3559

31. Add balancing to solution with fragment descriptors. Score=0.3673

32. Ensemble of everything, score=0.1692

33. Only 2 best, score=0.3647

34. Add filtering of fragment library to notebook with custom fragment library. Score=0.3135

35. Using only top-3000 fragments, nFolds=11. score=0.3684

36. Nfolds=5. 0.3129

37. want top try without fragments filtering, with nfolds=11. Score=0.3234
38. top2000, nfolds=11. score=0.3429