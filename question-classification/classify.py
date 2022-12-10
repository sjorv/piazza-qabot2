"""
Main script used for running experiments. 

An experiment = generating and evaluating model predictions conditional on <dataset, prompt, gpt model parameters, ...>
"""

import openai
import re
import pandas
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pathlib import Path

from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score


from hyperparameters import *

# save current experiment in a new directory. Use timestamp to create a unique directory for each experiment.
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = f'{general_params["run_dir"]}{timestamp}_{general_params["run_suffix_tag"]}'



"""
Pick most common class in csv?
"""
def naive_baseline():
    #TODO
    pass


"""
"""
def prompt_gpt3(prompt, model=model_params['model'], max_tokens=model_params['max_tokens'], top_p=model_params['top_p'], temperature=model_params['temperature'], n=model_params['n']):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        n=n
    )
    
    return response


"""
Classify a single question in the CSV file. Helper function used by inference().
"""
def classify_question(row):
    question = row['question']
    prompt=model_params['prompt'].format(question)

    response = prompt_gpt3(prompt)
    
    text = response["choices"][0]["text"]

    row['predicted_category'] = text
    return row



"""
Takes a CSV of questions, adds a new column: "predicted_category" and fills it with model predictions.
"""
def inference():
    
    df = pandas.read_csv(dataset_params['dataset_path'])

    if dataset_params['shuffle']:
        df = df.sample(frac=1)

    df['predicted_category'] = ""
    if (dataset_params['num_samples']):
        df = df.head(dataset_params['num_samples'])
    df = df.apply(lambda row: classify_question(row), axis=1)

    out_path = RUN_DIR + '/' + 'inference.csv' 
    print(f'Saving Classified CSV to: {out_path}')
    df.to_csv(out_path, index=False)


"""

Performs evaluation on the csv file in the run_dir. Saves metrics to a new file: eval.txt in the run_dir.
Metrics tracked
    - accuracy
    - f1 micro + macro
    - confusion matrix

Adds special <unknown-category> token for model predictions that are not in the classification categories in the dataframe.
   - i.e. it's possible for gpt3 to output something other than the labels you tell it to output (i.e. answerable vs. other)

Precondition:
  1. data csv has columns named: target and predicted_category

"""
def evaluate():
    inference_path = RUN_DIR + '/' + 'inference.csv'

    df = pandas.read_csv(inference_path)

    targets = df['target'].str.lower().str.strip()
    predicted_categories = df['predicted_category'].str.lower().str.strip()
    
    predicted_categories[~predicted_categories.isin(dataset_params['labels'])] = '<unknown-category>'
    dataset_params['labels'].append('<unknown-category>')

    # compute and save cm if specified
    cm = confusion_matrix(targets, predicted_categories, labels=dataset_params['labels'])
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset_params['labels'])
    cmd.plot()

    if (general_params['save_run']):
        save_dir =  RUN_DIR + '/' + 'confusion_matrix.png'
        print(f'Saving to: {save_dir}')
        fig = plt.gcf()
        fig.set_size_inches(15, 15)

        fig.savefig(save_dir)

    # compute and save eval metrics if specified

    f1_macro = f1_score(targets, predicted_categories, labels=dataset_params['labels'], average='macro')
    f1_micro = f1_score(targets, predicted_categories, labels=dataset_params['labels'], average='micro')
    accuracy = accuracy_score(targets, predicted_categories)
    num_correct = accuracy_score(targets, predicted_categories, normalize=False)
    total = len(targets)

    print(f'accuracy: {accuracy} ({num_correct}/{total})')
    print(f'f1_macro: {f1_macro}')
    print(f'f1_micro: {f1_micro}')

    if (general_params['save_run']):
        eval_path = RUN_DIR + '/' + 'eval.txt'

        with open(eval_path, 'w') as f:
            f.write(f'Accuracy: {accuracy} ({num_correct}/{total})\n')
            f.write(f'f1_macro: {f1_macro}\n')
            f.write(f'f1_micro: {f1_micro}\n')

    # rename directory with val score?
    



"""
Plots metrics across different runs. Use this for plotting prompt vs. eval score across different runs.

Iterates through runs specified in control_params['plot_dirs'] and extracts their eval metrics stored in
eval.txt.

Precondition: eval.txt has the format: 
    Accuracy: <score>
    f1_macro: <score>
    f1_micro: <score>
"""
def plot_metrics():
    run_dirs = control_params['plot_dirs']
    accuracy, f1_macro, f1_micro, labels = [], [], [], []
    scores = [accuracy, f1_macro, f1_micro]

    for timestamp in run_dirs:
        for run in os.listdir(general_params['run_dir']):
            if timestamp in run:
                run_tag = None
                eval_file = general_params['run_dir']  + run + '/' + 'eval.txt'
                print(f'opening eval file: {eval_file}')
                with open(eval_file, 'r') as f:
                    for idx, line in enumerate(f):
                        scores[idx].append(float(line.replace('\n', '').split(' ')[-1]))
                        param_file = general_params['run_dir']  + run + '/' + 'hyperparameters.json'
                        print(f'opening hyperparameters file: {param_file}')
                        with open(param_file) as f2:
                         params = json.load(f2)
                         run_tag = params['general_params']['run_suffix_tag']
                labels.append(run_tag)

    
    # plot metrics

    plt.plot(labels, scores[0], label='accuracy', marker='o')
    plt.plot(labels, f1_macro, label='f1_macro', marker='o')
    plt.plot(labels, f1_micro, label='f1_micro', marker='o')
    plt.legend()
    plt.title('Prompt vs. Evaluation Score')
    plt.xlabel('Prompt')
    plt.ylabel('Evaluation Score')
    plt.xticks(rotation=15)

    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    

    save_plot_path = RUN_DIR + '/' + 'plot.png'
    fig.savefig(save_plot_path)



"""
Nicely format a dict in a text file
"""
def write_dict_to_file(d:dict, file):
    print(d)
    for k,v in d.items():
        print(k, v)
        file.write(f'    {k} : {v}\n')



"""
log parameters associated with experiment to hyperparameters.txt and hyperparameters.json
This makes it easy to see how hyperparams affect performance.
"""
def log_run_params():
    param_filename = RUN_DIR + '/' + 'hyperparameters'

    if (general_params['save_run']):
        print('Saving hyperparameters to {param_filename}')
        params = {
            "general_params" : general_params,
            "dataset_params" : dataset_params,
            "control_params" : control_params,
            "model_params" : model_params,
        }
        
        with open(param_filename + '.json', 'w') as f:
            json.dump(params, f)
            

        with open(param_filename + '.txt', 'w') as f:
            f.write(f'General Params:\n\n')
            write_dict_to_file(general_params, f)

            f.write(f'\n\nDataset Params:\n\n')
            write_dict_to_file(dataset_params, f)

            f.write(f'\n\nControl Params:\n\n')
            write_dict_to_file(control_params, f)

            f.write(f'\n\nModel Params:\n\n')
            write_dict_to_file(model_params, f)

    
    if (general_params['verbose']):
        print(f'General Params:\n\n')
        print(general_params)

        print(f'\n\nDataset Params:\n\n')
        print(dataset_params)

        print(f'\n\nControl Params:\n\n')
        print(control_params)


        print(f'\n\nModel Params:\n\n')
        print(model_params)

        

def main():
      
    openai.api_key_path = general_params['openai-key-path']

    # create run dir if not exists
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True) 

    if (general_params['save_run']):
        print(f'Run will be saved in: {RUN_DIR}')

    log_run_params()


    if (control_params['inference']):
        print(f'Performing inference: \n\n')
        inference()


    if (control_params['evaluate']):
        print(f'Performing evaluation: \n\n')
        evaluate()


    if (control_params['plot_metrics']):
        print(f'Plotting metrics: \n\n')
        plot_metrics()




if __name__ == "__main__":
    main()




"""
TODO: naive baseline -- choose common class

inference, eval go together while plot is separate

might want to plot across diff runs


other baselines:
 - naive -- pick the most common class 
 - rnn, lstm, gru, etc.


 accuracy overlaps with f1_micro
"""