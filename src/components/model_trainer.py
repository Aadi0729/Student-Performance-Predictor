import os
import sys
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object, create_plot

@dataclass
class ModelTrainerConfig:
    # This class is used to store the configs, or any other files generated in this particular python file.
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self,model_train_config:ModelTrainerConfig = ModelTrainerConfig() ) -> None:
        self.model_trainer_config = model_train_config
        
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }
            
            params ={
                "Ridge": {
                "alpha": [0.1, 1, 10],
                "fit_intercept": [True, False],
                
                },
                    "Lasso": {
                "alpha": [0.1, 1, 10],
                "fit_intercept": [True, False],
                
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                    #"max_depth": [None, 5, 10],
                    #"min_samples_split": [2, 5, 10],
                    
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                     #'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256],
                    #"max_depth": [None, 5, 10],
                    #"min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256],
                    #"max_depth": [None, 5, 10],
                    #"min_samples_split": [2, 5, 10],
                },
                "Linear Regression":{ 
                    "fit_intercept": [True, False],
                    #"normalize": [True, False],
                    },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    #"max_depth": [None, 5, 10],
                    #"min_child_weight": [1, 3, 5],
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                    #"n_estimators": [50,100,250],
                    #"max_depth": [None, 5, 10],
                    #"reg_lambda": [0.1, 1, 10],
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256],
                }
                                
                }
            
            model_report: dict = evaluate_models(
                X_train = X_train,
                y_train =  y_train, 
                X_test  = X_test,
                y_test = y_test,
                models = models,
                param = params
                )
            print(model_report)
            
            model_report_df = pd.DataFrame(model_report, index=[0])
            model_report_df.to_csv("./assets/files/model_report.csv",index=False)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            

    
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            create_plot(y_test,predicted,type = 'scatter',model_name = best_model_name)
            create_plot(y_test,predicted, type = 'reg',model_name = best_model_name)
            
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)





# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }
#             params={
#                 "Decision Tree": {
#                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     # 'splitter':['best','random'],
#                     # 'max_features':['sqrt','log2'],
#                 },
#                 "Random Forest":{
#                     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
#                     # 'max_features':['sqrt','log2',None],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Gradient Boosting":{
#                     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#                     'learning_rate':[.1,.01,.05,.001],
#                     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#                     # 'criterion':['squared_error', 'friedman_mse'],
#                     # 'max_features':['auto','sqrt','log2'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Linear Regression":{},
#                 "XGBRegressor":{
#                     'learning_rate':[.1,.01,.05,.001],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "CatBoosting Regressor":{
#                     'depth': [6,8,10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                 },
#                 "AdaBoost Regressor":{
#                     'learning_rate':[.1,.01,0.5,.001],
#                     # 'loss':['linear','square','exponential'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 }
                
#             }

#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
#                                              models=models,param=params)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            



            
#         except Exception as e:
#             raise CustomException(e,sys)