"""
Stores hyperparameters associated with a single experiment.
"""
from prompts import * 


general_params = {

  "run_dir" : "runs/root_2_categories/",  # directory to store experiment runs
  "run_suffix_tag" : "overfit prompt",  # a way of easily identifying run. alter to name your run
  #"run_suffix_tag" : "plot-everything",  # a way of easily identifying run. alter to name your run
  "openai-key-path": "openai_api_key.txt",
  "run_info" : "binary classification",
  "save_run" : True, # keep on for now
  "verbose" : True

}


control_params = {

    "evaluate" : True,
    "inference" : True, 
   
    "reuse_run" : False,
    "reuse_run_dir" : None,

    "plot_metrics" : False,  # useful to toggle on plot_metrics and toggle off eval + inf. eval + inf usually done together. 
    "plot_dirs" : ['20221129-143038', '20221129-143351', '20221129-143624', '20221129-143914', '20221129-144250', '20221129-144405' ]  # array of run dirs whose eval metrics to plot. only specify timestamps
}



model_params = {
    "model":"text-davinci-003", 
    "max_tokens":200, 
    "top_p":1, 
    "temperature":0, 
    "n":1,
    
    "prompt" : PROMPT_HIGH,  # change this to your prompt

}


dataset_params = {

  "shuffle":False, # don't shuffle to keep consistent dataset between runs
  "num_samples":2, # None -> use all samples. Set low by default to avoid doing expensive forward pass
  "stratified":False,  # unused

  #"dataset_path" : "../../datasets/classify_datasets/cleaned_csc311_fall_2022_classify_prompt_v1.csv",
  "dataset_path" : "../datasets/311/csc311_2_categories.csv",  # path to dataset to classify
  #"dataset_path" : "../datasets/108/csc108_cleaned2.csv",  # path to dataset to classify
  #"dataset_info": "CSC311 Fall 2022. Randomly shuffled and 20 samples were picked.",
  "dataset_info": "CSC108 Fall 2022.",

  #"labels" : ['answerable', 'unanswerable', 'answerable with the right context'] 
  "labels" : ['answerable', 'other']  # dataset classification labels

}