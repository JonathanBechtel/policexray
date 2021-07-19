# -*- coding: utf-8 -*-
"""
Object to analyze policing data in an automatic way
"""
import os
import pickle
from datetime import datetime
import xgboost as xgb
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

class PolicingAnalyzer():
    
    def __init__(self, filepath: str, output_dir: str,
                 outcome: str          = 'search', 
                 scenario: str         = 'pulled-over',
                 boosting_rounds: int  = 10000,
                 learning_rate: float  = .05,
                 max_depth: int        = 3,
                 min_samples_leaf: int = 30,
                 random_state: int     = 42,
                 fit_method: str       = 'hist'):
        """
        Initializes object that will be used to conduct modeling automatically
        for policing project
        
        Arguments:
            
        filepath: str, location of file to be loaded in for analysis
        output_dir: str, directory where analysis artificats will be placed
        outcome: str, outcome you are evaluating for, should be one of 'arrest',
        or 'search'
        scenario: str, circumstances of situation, should be one of 'pulled-over',
        'being-searched', 'search-completed'
        boosting_rounds: int, number of boosting rounds to use for xgboost,
        learning_rate: float, learning rate to use for xgboost during fitting
        max_depth: int, tree depth to use during model fitting
        min_samples_leaf: int, threshold to use for label mean for category
        encoding
        """
        self.filepath         = filepath
        self.outcome          = outcome
        self.scenario         = scenario
        self.output_dir       = output_dir
        self.boosting_rounds  = boosting_rounds
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.fit_method       = fit_method 
        
    def _validate_inputs(self):
        """Validates all arguments provided at initialization to make sure 
        there are no errors"""
        try:
            self.data = pd.read_csv(self.filepath)
        except:
            raise Exception("File could not be loaded in.  Please double check filepath")
            
        if not os.path.isdir(self.output_dir):
            raise Exception(f"""Directory provided for model output
                            {self.output_dir} is not a valid directory""")
                            
        assert self.outcome in ['search', 'arrest'], """Please make sure value
        provided for outcome is one of 'search' or 'arrest' """
        
        assert self.scenario in ['pulled-over', 'being-searched',\
                                 'search-completed'], """Please make sure value
            provided for scenario is one of 'pulled-over', 'being-searched', or
            'search-completed' """
            
        if self.scenario in ['being-searched', 'search-completed'] and \
            self.outcome != 'arrest':
                raise Exception("""For scenario chosen you may only use
                                'arrest' as an outcome""")
                                
    def _validate_data(self):
        """Double checks that data contains correct columns within dataset"""
        assert len(self.data.columns) == 18, """Your data should have 18
        columns inside of it"""
        
        correct_cols = ['city', 'subject_age', 'subject_race','subject_sex',
                        'arrest_made', 'contraband_found', 'search_conducted',
                        'reason_for_search', 'reason_for_stop',
                        'Observation of Suspected Contraband', 'Informant Tip',
                        'Suspicious Movement', 'Witness Observation',
                        'Erratic/Suspicious Behavior',
                        'Other Official Information', 'hour', 'dayofweek',
                        'quarter']
        
        for col in self.data.columns:
            if col not in correct_cols:
                raise ValueError(f"""the column '{col}' is in your data but
                                 not in the original data source.  Please
                                 double check that your data's columns are
                                 {correct_cols}""")
                                
    def _prep_data(self):
        """defines X & y, depending on outcome and scenario"""
        
        
        arrest_cols = ['city', 'subject_age', 'subject_race', 
                           'subject_sex', 'reason_for_stop',
                           'Observation of Suspected Contraband', 
                           'Informant Tip', 'Suspicious Movement',
                           'Witness Observation','Erratic/Suspicious Behavior',
                           'Other Official Information', 'hour', 'dayofweek',
                           'quarter']
        
        search_cols = ['city', 'subject_age', 'subject_race',
                           'subject_sex', 'reason_for_stop', 'hour',
                           'dayofweek', 'quarter']
        
        if self.scenario == 'pulled-over':
            if self.outcome == 'search':
                self.y = self.data['search_conducted']
                self.X = self.data[search_cols]
            elif self.outcome == 'arrest':
                self.y = self.data['arrest_made']
                self.X = self.data[search_cols]
            else:
                raise ValueError("""Your values for scenario and outcome were
                                 not valid.  Please double check them.""")
        elif self.scenario == 'being-searched':
            self.y = self.data['arrest_made']
            self.X = self.data[arrest_cols]
            
        elif self.scenario == 'search-completed':
            query = self.data.search_conducted == True
            self.data = self.data.loc[query, :]
            self.y = self.data['arrest_made']
            self.X = self.data[arrest_cols + ['contraband_found']]
        else:
            raise Exception(f"""The scenario you provided '{self.scenario}'
                            does not match any of the accept values.  Make sure
                            it is one of ['pulled-over', 'being-searched', 
                                          'search-completed']""")
                                          
    def _split_data(self, create_val=True):
        """Assigns training, testing, and validation sets"""
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self.X, self.y, test_size=0.1, random_state =
                             self.random_state, stratify=self.y)
        
        if create_val:
            self.X_train, self.X_val, self.y_train, self.y_val =\
                train_test_split(self.X_train, self.y_train, test_size=0.1,
                                 random_state = self.random_state, stratify=self.y_train)
                
    def _encode_data(self, encode_validation = True, encode_test = False,
                     encode_full = False):
        
        print("Encoding data for training.....")
        
        if encode_validation:
            self.initial_pipe = make_pipeline(ce.TargetEncoder(
                        cols=['city', 'reason_for_stop'], 
                        min_samples_leaf = self.min_samples_leaf
                        ), 
                        ce.OneHotEncoder(use_cat_names=True))
            self.X_train = self.initial_pipe.fit_transform(self.X_train,
                                                       self.y_train)
            self.X_val = self.initial_pipe.transform(self.X_val)
            
        if encode_test:
            self.X_train = self.initial_pipe.fit_transform(self.X_train, self.y_train)
            self.X_test  = self.initial_pipe.transform(self.X_test)
            
        if encode_full:
            self.X = self.initial_pipe.transform(self.X)
            
    def _train_model(self, how='early-stopping'):
        """Initializes model and trains it on data, using different data
        splits depending on argument given for 'how'
        
        Arguments:
        --------------------
        how: str, one of ['early-stopping', 'get-test-score', 'fit-full-model']
            'early-stopping': will using early stopping on validation set to
                              find the best number of boosting rounds
            'get-test-score': fits model on training set (including validation)
                              and uses this to get the test score
            'fit-full-model': fits model on the whole dataset, is used to
                              export the final model pipeline"""
        
        if how == 'early-stopping':
            print(f"""Fitting model for outcome: {self.outcome}, scenario:
                  {self.scenario}""")
                  
            self.mod = xgb.XGBClassifier(
                n_estimators      = self.boosting_rounds,
                learning_rate     = self.learning_rate,
                max_depth         = self.max_depth,
                objective         = 'binary:logistic',
                base_score        = self.y_train.mean(),
                scale_pos_weight  = 1 / self.y_train.mean(),
                use_label_encoder = False,
                tree_method       = self.fit_method)
                  
            self.mod.fit(self.X_train, self.y_train, 
                    eval_set=[(self.X_train,self.y_train), 
                              (self.X_val, self.y_val)],
                    early_stopping_rounds = 20, eval_metric= 'aucpr',
                    verbose=10)
            
            self.training_results = {
                    'n_estimators'   : self.mod.best_iteration,
                    'max_depth'      : self.max_depth,
                    'base_score'     : self.mod.base_score,
                    'learning_rate'  : self.learning_rate,
                    'train_score'    : self.mod.evals_result_['validation_0']\
                        ['aucpr'][self.mod.best_iteration]
                }
                
            self.mod.set_params(n_estimators = self.mod.best_iteration)
            
        elif how == 'get-test-score':
            self.mod.fit(self.X_train, self.y_train)
            
            self.training_results['test_score'] = average_precision_score(
                self.y_test, self.mod.predict_proba(self.X_test)[:, 1])
            
        elif how == 'fit-full-model':
            print("Fitting final model for deploymnent")
            self.mod.fit(self.X, self.y)
            
            
    def _find_best_parameters(self):
        """Uses Successive Halving with Randomized Search to find best model
        parameters"""
        
        grid_pipeline = make_pipeline(ce.TargetEncoder(
                        cols=['city', 'reason_for_stop'], 
                        min_samples_leaf = self.min_samples_leaf
                        ), 
                        ce.OneHotEncoder(use_cat_names=True), self.mod)
        
        kfold = StratifiedKFold(n_splits=3)
        param_grid = param_grid = {
            'xgbclassifier__scale_pos_weight': [1, 
                                (1  / self.y_train.mean()) * 0.5, 
                                 1  / self.y_train.mean(), 
                                 (1 / self.y_train.mean()) * 1.5, 
                                 (1 / self.y_train.mean()) * 2],
            'xgbclassifier__subsample': [1, 0.8, 0.6, 0.4],
            'xgbclassifier__gamma': [0, 1, 2, 5, 10, 15],
            'xgbclassifier__colsample_bytree': [1, 0.8, 0.6, 0.4]
            }
        
        grid = HalvingRandomSearchCV(grid_pipeline, param_grid, factor=1.5, cv=kfold,
                                     min_resources=400000 if self.scenario != 'search-completed' else 16000,
                                     n_jobs=-1,
                                     scoring=make_scorer(
                                         average_precision_score,
                                         needs_proba=True),
                                     random_state = self.random_state,
                                     refit=False, verbose=1)
        
        grid.fit(self.X_train, self.y_train)
        
        self.cv_results  = pd.DataFrame(grid.cv_results_)
        self.mod.set_params(subsample        = grid.best_params_\
                            ['xgbclassifier__subsample'],
                            scale_pos_weight = grid.best_params_\
                                ['xgbclassifier__scale_pos_weight'],
                            gamma        = grid.best_params_\
                                ['xgbclassifier__gamma'],
                            colsample_bytree = grid.best_params_\
                                ['xgbclassifier__colsample_bytree'])
        self.training_results['scale_pos_weight'] = grid.best_params_\
            ['xgbclassifier__scale_pos_weight']
        self.training_results['subsample'] = grid.best_params_\
            ['xgbclassifier__subsample']
        self.training_results['gamma'] = grid.best_params_\
            ['xgbclassifier__gamma']
        self.training_results['colsample_bytree'] = grid.best_params_\
            ['xgbclassifier__colsample_bytree']
        self.training_results['cv_best_score'] = grid.best_score_
            
            
            
    def _export_results(self):
        """Takes results from current analysis and exports them to a folder"""
        
        final_pipeline = make_pipeline(self.initial_pipe, self.mod)
        
        current_time   = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name       = f"""model_results-{self.outcome}-{self.scenario}-{current_time}"""
        
        path = os.path.join(self.output_dir, dir_name)
        
        os.mkdir(path)
        
        # write analysis results to a text file
        try:
            print(f"Exporting model results to directory: {self.output_dir}")
            with open(f"{path}/results.txt", 'w') as f:
                for key, value in self.training_results.items():
                    f.write(f"{key}: {value}\n")
                    
            # and export the serialized pipeline
            with open(f"{path}/mod_pipeline.pkl", 'wb') as model_results:
                pickle.dump(final_pipeline, model_results)
                
            self.cv_results.to_csv(f"{path}/cv_results.csv", index=False)
            print("Finished")
        except Exception as e:
            print(f"Export failed because: {e}")
            
    def evaluate_data(self):
        """Master function that will encapsulate all steps in the class and 
        run analysis from beginning to end"""
        
        # double check the inputs provided
        self._validate_inputs()
        # double check the data, making sure it matches master file
        self._validate_data()
        # store X and y for later use
        self._prep_data()
        # create training, validation, and test set
        self._split_data()
        # transform data using target encoding
        self._encode_data()
        # fit model on training data, log results on validation set
        self._train_model(how='early-stopping')
        # then, re-split data to combine validation and training sets
        self._split_data(create_val=False)
        # find best model parameters using successive halving
        self._find_best_parameters()
        # re-transform using target encoding
        self._encode_data(encode_validation=False, encode_test=True)
        # re-train model to get the test score
        self._train_model(how='get-test-score')
        # now, go ahead and fit model on entire dataset for export
        self._encode_data(encode_validation=False, encode_full=True)
        # and refit on all of dataset to get final version
        self._train_model(how='fit-full-model')
        # and finall export results
        self._export_results()
        
        
            
            
        
    
            
        
                            
    
            
            
    
