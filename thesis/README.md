## DECODING POLITICAL BIAS THROUGH LINGUISTIC INSIGHTS OF LARGE LANGUAGE MODELS

**This repository contains the Supplementary Materials for the Honors Thesis submitted by Su Doga Karaca to the university of Chicago Department of Psychology**

You can find the contents of the directories in this repository below.

## Contact
Su Doga Karaca
sudogakaraca@uchicago.edu

## Repository Structure

### base_regression
This directory contains the project analysis code for the regression results reported in the Results section of the paper. There are multiple directories and files locacted in this directory each with their individual goal. Model_Trials Excel file contains the training, validation, and test results of multiple fine-tuning procedures with different parameters used. only_test_data directory contains all the analysis code reported in the paper. You can access the pre-processing code, the R script used to run the regression analyses, and the liwc_data obtained from the model used and LIWC. The regression_analysis.html links to the regression results and tables reported in the paper. dataframes directory leads to all the data obtained from the preprocessing and used in the analyses.

### correlation
This directory contains the project analysis code for the correlation results reported in the Results section of the paper. There are multiple directories and files locacted in this directory each with their individual goal. In the notebook you can see the correlation matrix codes and the correlation tables reported. In the correlation analysis, normalized version of the liwc_data reported in the base_regression directory is used. This data is contained in the z_score_extremity.csv file.

### regression_trials
This directory contains the project analysis code for the various regression results **NOT** reported in the Results section of the paper. There are multiple directories and files locacted in this directory each with their individual goal. logistic_regression directory contains the code for the logistic regression analyses we ran and all the data used in the process. The three notebooks located in the main directory contain the trials we had for various regression techniques, including linear, multiple, ridge, and lasso models. regression_by_liwc contains analysis conducted with all semantic categories. regression_moral_emotional contains analysis conducted with only the moral and emotional semantic categories. correlation_liwc-ENTITY notebook contains analysis of the data obtained from the main dataset with organization or person names removed.

### semantic_distance
This directory contains the project analysis code for the various follow up results reported in the Results section of the paper. There are multiple directories and files locacted in this directory each with their individual goal. The data directory contains the data used in the analysis. semantic_distance_processing script contains the code we used to obtain the cossine dissimilarity results. regression_analysis.html report the regression coefficients and tables reported in the paper.

### data
This directory contains the data used in the model building and the semantic score analysis by LIWC. created_data directory has the two sets of data used in the model fine-tuning: 
1. the liberal and conservative articles and their original datasets
2. the liberal and conservative articles and their original datasets -- with any organization or person name removed


dictionaries directory contains the moral-emotional dictionaries used in the process. version_txt_dict directory contains the raw txt versions of the dictionaries. liwc_compatible_dict contains the liwc formatted versions of the dictionaries.
model_used directory contains the snapshot of the model fine-tuned at its checkpoint state. model_code contains the model architecture and slurm scripts. outputs dircetory contains the output results we used to run our analyses. output_modified directory contains the test stage proababilities processed to be used for the main analyses. The model checkpoint is located in the AI-cluster of the University of Chicago High-performance computer system. On request, the checkpoint could be shared with anyone.