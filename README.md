# DieAreaRegressor

<p align="center">
  <a href="#about">About</a> &nbsp;&bull;&nbsp;
  <a href="#environment-setup">Environment Setup</a> &nbsp;&bull;&nbsp;
  <a href="#important-note-about-data">Important Note About Data</a> &nbsp;&bull;&nbsp;
  <a href="#repository-structure">Repository Structure</a> &nbsp;&bull;&nbsp;
  <a href="#citation">Citation</a>
</p>

<br>
<br>

<table align="center">
<tr>
<td align="center">

<em style="color: steelblue;">
💡 An <a href="https://anncollin.github.io/DieAreaPrediction/">interactive web interface</a> is available to perform silicon die area predictions without any installation. 
</em>

<br>
<img src="assets/demo.gif" width="780">

</td>
</tr>
</table>

<br>
<br>

## About

gitThis repository contains the code associated with the paper "Advancing silicon die area estimation for integrated circuit modeling in life-cycle assessments". The full paper is available here: https://link.springer.com/article/10.1007/s11367-026-02754-6

---

## Environment Setup

To create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate ossda
```

---

## Important Note About Data

The OSSDA dataset is publicly available on Zenodo: https://zenodo.org/records/18739861

The OSSDA_021726_FRAUNHOFER_IZM.xlsx and OSSDA_021726_UCLOUVAIN.xlsx files (which together form OSSDA) are not included in this repository and need to be downloaded from Zenodo and copied and pasted manually into the data/ folder. 

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

Wattiez, A., Collin, AS., Dethienne, R. et al. Advancing silicon die area estimation for integrated circuit modeling in life-cycle assessments. Int J Life Cycle Assess 31, 186 (2026). https://doi.org/10.1007/s11367-026-02754-6
