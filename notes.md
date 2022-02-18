# TODOs
Papers and code to check, ideas to try, etc.

## Papers/repos to check

- [ ] https://github.com/oxpig/DEVELOP - not directly related, however some ideas might be useful
- [ ] https://github.com/oxpig/learning-from-the-ligand
- [ ] Look at other https://github.com/oxpig projects
- [ ] https://github.com/gnina/gnina check time/memory requirements. If it is ok for our case, then try it. ignore otherwise.
- [ ] https://www.mdpi.com/1420-3049/26/23/7201/pdf Pharmacophore alignment method paper (G3PS)
- [ ] https://github.com/czodrowskilab/5minfame

# Approaches/ideas to try:
- [ ] docking
- [x] EDA: group molecules in train dataset by Murcko scaffold and look at them (added `notebooks/datacomparison.ipynb` with this)
- [ ] EDA: try to group pharmacophores I've generated and look at the results
- [ ] make custom fingerprint with rdkit and use them a) independently; b) with cactvs implemented at DeepPurpose
- [ ] finish solution similar to the one published on aicrowd's discussions as top3 winning in Learning to smell challenge
- [ ] make basic script with SentenceTransformers
- [ ] ChemBERTa: cookbook has smiles enumeration recipe for this+SentenceTransformer library
- [ ] anonymize molecule to graph (one of the Hashes at the Cookbook, build GNN using it
- [ ] check and clean data using this package https://github.com/flatkinson/standardiser/blob/master/standardiser/docs/06_alternative.ipynb

## Particular todos:
- [ ] finish sentencetransformers code
- [ ] read how to use smarts
- [ ] add smarts-based fingerprint to custom ones and try this solution
- [ ] use autocatboost?
