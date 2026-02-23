# DieAreaRegressor

An [interactive web interface](https://anncollin.github.io/DieAreaPrediction/) is available at the link above to perform silicon die area predictions without running the Jupyter notebooks locally.

---

## Table of Contents

- [Context](#context)
- [Environment Setup](#environment-setup)
- [Important Note About Data](#important-note-about-data)
- [Repository Structure](#repository-structure)
  - [00_extractDataset.ipynb](#00_extractdatasetipynb)
  - [01_chooseBestModel.ipynb](#01_choosebestmodelipynb)
  - [02_observeBestModel.ipynb](#02_observebestmodelipynb)
- [Citation](#citation)

---

## Context

This repository contains the code associated with our work on silicon die area estimation for integrated circuits used in Life-Cycle Assessment (LCA) modeling. The full paper will be available soon on ArXiv. 

---

## Environment Setup

To create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate ossda
```

---

## Important Note About Data

The OSSDA dataset will publicly available soon. The data files are not yet included in this repository.

As a consequence:

- The Jupyter notebooks are not directly runnable at this stage.  
- All notebook outputs are stored and can be viewed.  
- A `data/` folder will be added in a future update.

---

## Repository Structure

This repository contains multiple Jupyter notebooks corresponding to the analysis workflow presented in the paper.

> ### 00_extractDataset.ipynb
>
> Provides utilities to:
>
> - Load and preprocess the OSSDA dataset  
> - Perform exploratory data analysis  
> - Compute and display descriptive statistics  
>
>
>
>### 01_chooseBestModel.ipynb
>
>Implements the model selection pipeline:
>
> - K-fold cross-validation  
> - Comparison of regression models  
> - Selection of the recommended log-log group-wise linear regression model  
>
> Cross-validation runs used in the paper are stored in the `CVRuns/` folder.
>
> You may run new experiments if desired.
>
>
>
> ### 02_observeBestModel.ipynb
>
> Provides utilities to:
>
> - Load the selected regression model  
> - Evaluate prediction performance  
> - Visualize regression results  
> - Compare estimation approaches  
>
---

## Citation

The full citation of the associated publication will be added soon.
