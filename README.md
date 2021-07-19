# Introduction
Thank you for visiting this github repo.  It's the public repository for https://www.policexray.com.  Here you'll find code on how to setup the repo, re-create results, and contribute to the project if you'd like.

# About
The police xray project was designed to combine machine learning, policing outcomes, open data, interactive visualizations and reproducible code to give both experts and laypersons the ability to explore how factors like someone's age, location, race and behavior interact to determine how someone gets searched or arrested.  

The project has three different interfaces:

 - A public website at https://www.policexray.com, that allows people to do scenario analysis to see how a person's characteristics and behavior influence police outcomes
 - An api at https://www.police-project-test.xyz/api that allows developers to access the project's statistical model as a service, either to explore it programmatically or re-use it in their own ways
 - This github repo, which will allow you to re-run our analysis from the command line and explore project artifacts directly

# Installation

To install this repo on your own computer, follow these directions:

 - `git clone my_code_repo`
 - `cd my_code_repo/src`
 - `python app.py`

## Installing With a Development Environment

It is recommended that you setup a development environment to make sure there are no discrepancies between different versions of the packages used to recreate the results.  The packages can be installed from either the `requirements.txt` or the  `environment.yml` file listed in the root directory.  Once you are inside the virtual environment with the correct packages installed you should be able to run `app.py` and re-create everything.

## Successfully Running app.py

`app.py` is designed to execute a single file that will allow you to re-create all of the models used to generate the data on the main page of www.policexray.com.  However, to get it to run correctly you must supply four different arguments.  They are as follows:

 - `--p`: this is the location on your hard drive where the data is located.  You need to download this data from here.
 - `--d`: this is the directory where you would like your results exported to
 - `--s`: this is the scenario you are going to model.  It can be one of three values:  `pulled-over`, `being-searched`, and `search-completed`
 - `--o`: this is the outcome you are going to model.  It can be one of `search` or `arrest`.  Note that the `search` outcome can only be used with the `pulled-over` scenario.  Also note that the `being-searched` and `search-completed` scenarios can only be used with the `arrest` outcome.

So, as an example, if your data file was located at `C:/Users/YourUsername/police-data.csv`, and you had a folder at `C:/Users/YourUsername/Results` where you wanted to download your results, you would run the command in the following way:

`python app.py --p C:/Users/YourUsername/police-data.csv --d C:/Users/YourUsername/Results --s pulled-over --o search`

Depending on your computer hardware and the scenario / outcome combination that you choose, this script will take anywhere from 90 minutes to 6 hours to run.

## Evaluating The Results of app.py

After running `app.py` with the correct arguments, the file will automatically create a folder in the directory provided in the `--d` argument named with the outcome and scenario you are modeling as well as a timestamp.  It contains three different items:

 - a `results.txt` file that lists the model parameters that were found by running the file.
 - a serialized version of the model pipeline that was created during the execution of the script.  This is the same model that's used in the website
 - a file csv file called `cv_results` that lists the outcomes of different versions of parameter searches that were done during fitting
