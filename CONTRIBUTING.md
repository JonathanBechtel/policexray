# Contributing to Police X-ray

If you would like to contribute to this project you are more than welcome to.  For the time being, if you would like to contribute to the project you can do so by contributing
a jupyter notebook in the `notebooks` folder with your analysis.  

For your analysis to be considered and used in production, it must meet the following requirements:

 - It needs to substantially outperform existing results for the category that it falls under
 - It must be able to be analyzed using a SHAP tree explainer, which is used to break predictions down into their individual components for each column
 - It must be created from the current published data source used for the project.
 - The model you use in your results needs to be able to be imported into a flask application and used in an API
 - Your notebook should be able to be run continuously from beginning to end without errors

In the event that there is modeling to be done outside of what's currently inside the application it will be listed in the `issues` tab of this repo.  

# Current Benchmarks

You can create models for the following categories, with the following target variables:

 - `pulled - over / searched`: evaluate the probability of being searched given information that's available at the time of being pulled over
 - `pulled - over / arrested`: evaluate the probability of being arrested given information that's available at the time of being pulled over
 - `being - searched / arrested`:  given all the information available at the time of being pulled over + the reason why you are searched, find the probability of being arrested
 - `search - completed / arrested`:  once a search is conducted and we know whether or not any contraband was found, find the probability of being arrested

Models are evaluated using the Area Under the Precision-Recall Curve, or Average Precision Score.

Current benchmarks are as follows:

 `pulled - over / searched`: 0.215
 `pulled - over / arrested`: 0.255
 `being - searched / arrested`: 0.551
 `search - completed / arrested`: 0.714
 
These values came from using the average_precision_score metric on a 10% holdout set that was stratified by the target variable.  You can find the metric here:  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
