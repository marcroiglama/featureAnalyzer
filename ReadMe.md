# Feature Analyzer

Feature -> Independent Variable // Label -> Dependent Variable

| Feature \ Label 	|         NUMERIC         	| CATEGORIC                            	| ALL 	|
|-----------------	|:-----------------------:	|--------------------------------------	|-----	|
| NUMERIC         	| Pearson's R             	| Corr Ratio / Dummy Corr / TSNE / PCA 	|   -  	|
| CATEGORIC       	| Anova / Kruscall        	| Cramer's V / Theil's U               	|   -  	|
| ALL             	| RF Regresor Importances 	| RF Classifier Feature Importances     |     	|

## Numeric vs Numeric
1. Correlation (Pearson's R)
* Intuition: how affect the variation on one variable to other. Based on covariance.
* Range: [-1,1]
* Properties: symmetrical / for linear relations.
* Assumptions: variables are linar dependent.

## Categoric vs Numeric
1.Analysis of variance (one way ANOVA)
* Intuition: 
* Range:
* Properties:
* Assumptions:  The samples are independent. / Each sample is from a normally distributed population. / 
The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

2.(Kruskall Wallis H)
* Intuition: non-parametric version of ANOVA
* Range:
* Properties:
* Assumptions:

## Numeric vs Categoric
1.Correlation Ratio (eta)
* Intuition: Given a continuous number, how well can you know to which category it belongs to?
* Range: [0,1]
* Properties: asymmetrical / for non-linear relations / equals to Pearson's R if linear
* Assumptions:  required that the interval level of the variables should be grouped into ranges in order to make sure 
that that there exists sufficient numbers of the categorical values that correspond to each of the interval level of values.

3. TSNE
* Intuition: visualization by non-linear dimensionality reduction 
* Range: [-inf, inf]
* Properties: keeps intravariable original distances
* Assumptions:  None

3. PCA
* Intuition: visualization by linear dimensionality reduction 
* Range: [-inf, inf]
* Properties: minimize information loss exploiting covariance
* Assumptions: features are linealy correlated

## Categoric vs Categoric
1.Association (Cramer's V)
* Intuition: value of coocurrence of the different variables. Based on Pearson’s Chi-Square Test
* Range: [0,1]
* Properties: symmetrical
* Assumptions:

2.Uncertainty Coefficient (Theil's U)
* Intuition: Given the value of x, how many possible states does y have, and how often do they occur. Based on conditional Entropy
* Range: [0,1]
* Properties: asymmetric
* Assumptions:

3. Dummy Correlation
* Intuition: Use dummy encoding (one hot encoding) to compute Pearson's R correlation
* Range: [-1,1]
* Properties: allow to compute linear correlation on categorical data. Sparse results. 
* Assumptions: variables aren't encoded yet.


## All VS ALL
1. Random Forest Feature Importances
* Intuition: train a ML model to get which features are relevant to predict the label value.
* Range: [0, 1]
* Properties:
* Assumptions: 