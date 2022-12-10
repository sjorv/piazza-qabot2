# Piazza Question-Answering Bot

## Folder Structure

    .
    ├── question-classification/    submodule
        ├── runs/                   tracks state associated with each experiment
        ├── hyperparameters.py
        ├── prompts.py
        ├── classify.py   


## Notes

question-classification is a submodule used for categorizing questions. Right now, the goal is to simply distinguish between answerable vs. other questions. "answerable" questions are ones that we expect gpt to answer directly without external context.Ex. "How does gradient descent work?", "Can cross entropy be loss be used for regression?", etc. "other" questions are everything else. If we get this working, we could get a basic deployment out where gpt only responds to answerable questions and ignores all the other ones.

Note that we cannot simply rule out assignment/HW questions as "other" since some may be answerable. Ex. "Lab2 mentions we must use gradient descent to train our model. What does this mean?" I believe it is difficult for the model to correctly classify these types of questions. i.e. when it sees hw1, it automatically classifies the question as other. We probably need more examples in the prompt to classify this category with high accuracy.


## How the Framework Works / How to Use


The only file you need to worry about is `hyperparameters.py`. This contains all the parameters associated with an *experiment*. You can think of each experiment as a tuple of hyperparameters: (dataset, prompt, model_params). To run an experiment do: `python3 classify.py` which will apply the prompt to classify the questions in the dataset with the given model_params.


### hyperparameters.py

Everytime you do: `python3 classify.py`, the experiment will get saved to `run_dir`. This is so you can view how each experiment parameter affects performance (i.e. classification accuracy or f1). To distinguish between runs, a timestamp gets prepended to each run folder name. Your dataset should include a label called *target* which is the ground truth question category. The script will take this dataset, create a new column called *predicted_categories* and populate it with model predictions. `prompt.py` is a bit messy. It contains all my prompts. You can include your own prompt there and then include it in model_params['prompt'] in `hyperparameters.py`.

### Datasets

As of now, I'm using `311/csc311_2_categories` to engineer my prompts. Thus, CSC311 Fall 2022 is like my training set since I'm using samples from that course to engineer my prompts. I then validate on `108/csc108_cleaned2.csv`. We need more data to validate on. I'm thinking ~30 samples for various other courses (413, 384, 236, ...). i.e. 30 samples per course, not in total. 


### Current Performance

I've experimented with various prompting strategies which you can refer to in `runs/*/hyperparameters.txt`. Currently, my best prompt looks like: `runs/best_run/*` which achieves 18/20 training accuracy (311) and 32/38 validation accuracy (108). The # of training samples should be increased to ~40-50 and validation should be performed across ~30-40 samples across various courses so these numbers aren't final. Think of this like a game -- if you can beat my prompt in `runs/best_run/*`, with the given dataset, then let me know so we can use your prompt instead.








## Good Prompt Design
Our goal is to figure out how to prompt gpt so it can correctly classify between the different types of questions. Ideally, our prompts should generalize across different types of courses (i.e. ML/DL, CS1/CS2, ...). The idea is to prompt gpt with various (question, category) examples and gpt will learn to discriminate between each of the various question categories. Think of this like *cheap supervised learning*. Cheap since gpt should be able to draw decision boundaries between the categories from a small number of samples given in the prompt. Note that this is analogous to how humans discriminate between things. If I show a child a single, *good* picture of a cat, they will easily be able to distinguish between cats and other animals. However, the picture must be a representative/prototypical picture of a cat. You wouldn't want to give them some cat edge case picture (i.e. like a cat-leopard hybrid). Similar with gpt. The (question, category) samples you feed it should be representative. Ex. (How do we approach this question to the homework, other). 


## TODOs

- get more data to validate on so we can better measure prompt performance


