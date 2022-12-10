"""
gpt prompts used in experiments. Import the appropriate one into hyperparameters.py
"""

QUESTION_CATEGORIES = ["self-contained", "answerable-with-context", "un-answerable"]

"""
self-contained", "answerable-with-context", "un-answerable

answerable vs. unanswerable

************** confident vs. not-confident -- 0 shot  **************

Task Description: Classify the below question from an undergraduate course discussion board as either confident or not-confident depending on whether you feel confident
of your answer to the question.

Task Description: First, answer the below question. Then, classify the below question depending on whether you feel confident of your answer to the question. 

Task Description: Classify the below question from an undergraduate course discussion board as either confident or not-confident depending on whether you feel confident
of your answer to the question. If you feel uncertain of your answer, classify it as not-confident.


Question:{0}

Classification:


   - maybe reveals latent structure of questions
   - confident should correspond to knowledge encoded in parameters
   - try to answer assignment based q's i.e. q's w/ context
   - should feel confident answering questions whose knowledge are encoded in parameters. not confident answering assignment related Q's or time-based (perhaps)
   - ** presupposes we know gpt's answer to the question and what is encoded in its parameters. Maybe gpt actually doesn't have the knowledge to answer a self-contained question. **


   - could do <question, gpt-answer, label> where label=0|1 corresponds to not-confident|confident 
    - a self-contained question may be not confident if gpt doesn't know the answer to it
    - self-contained = no reference to course docs
    - conditional on <question,gpt-answer> is gpt confident about its answer?

    - maybe the model feels confident providing a bad answer to a question
    - have it rate the quality of its own question
    - in-context examples help show what you're talking about


    - tell gpt to classify a Q as answerable vs. unanswerable -- how does gpt define what these terms mean? what is an answerable vs. unanswerable q to gpt?


**************  confident vs. not-confident -- few shot  **************

Task Description: Classify the below question from an undergraduate course discussion board as either confident or not-confident depending on whether you feel confident
of the answer to the question.

Question:{0}

Classification:

Question:{0}

Classification:

Question:{0}

Classification:

Question:{0}

Classification:

Question:{0}

Classification:




**issue with self-contained**
  - conditional on 

"""

DAVINCI_CLASSIFY_QUESTIONS_0SHOT_v1 = """

Task Description: Classify the below question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question:{0}

Classification:"""

DAVINCI_CLASSIFY_QUESTIONS_0SHOT_v2 = """

Task Description: Classify the below question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor or one that depends on time.

Question:{0}

Classification:"""



DAVINCI_CLASSIFY_QUESTIONS_0SHOT_v1_1 = """

Task Description: First generate an answer to the below question in the form "Answer:<insert answer here>". Then, classify your answer as either "confident" or "not confident" depending on whether your feel confident with your answer to the question. Your classification should be in the form: "Classify:<insert classification here>
Question:{0}"""


DAVINCI_CLASSIFY_QUESTIONS_0SHOT_v1_2 = """

Task Description: First generate an answer to the below question in the form "Answer:<insert answer here>". Then, rate your answer on a scale of 1-5. Your rating should be in the form: "Rating:<insert rating here>"
Question:{0}"""







# davinci does not like trailing spaces

DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
Question:{0}
Classification:"""

DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_description2 = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Unsure of compute_accuracy
Hi, I couldn't complete the Lab 1 on time and submitted what I could so I'm going back and finishing it for my own understanding.

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question:A1 Q4a
What is the size of inp?

edit: nvm got it lol
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Major Lab
Hi, Sorry but I am not unable to understand How I got 0 for the "My_And" because I did not use "if, and" statements anywhere in my code. Plz explain
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context
 
Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
All other questions are "unanswerable". 

Question:{0}
Classification:"""




DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_description = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Unsure of compute_accuracy
Hi, I couldn't complete the Lab 1 on time and submitted what I could so I'm going back and finishing it for my own understanding.

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:A1 Q4a
What is the size of inp?

edit: nvm got it lol
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:Question about Justification of Results
Justification of Results mentions: "A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t." What does it refer to when it means method? Also my group and I are somewhat confused as to how much is expected here since it is worth 20 marks and we can't think of a great deal to write about here.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Major Lab
Hi, Sorry but I am not unable to understand How I got 0 for the "My_And" because I did not use "if, and" statements anywhere in my code. Plz explain
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context
 
Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:{0}
Classification:"""


DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_description_min_examples = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Unsure of compute_accuracy

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:Question about Justification of Results
Justification of Results mentions: "A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t." What does it refer to when it means method? Also my group and I are somewhat confused as to how much is expected here since it is worth 20 marks and we can't think of a great deal to write about here.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Major Lab
Hi, Sorry but I don't understand why I got 0 marks for the "My_And" function because I did not use "if, and" statements anywhere in my code. Plz explain.
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

 
Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:{0}
Classification:"""







DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_description_min_examples_a = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Unsure of compute_accuracy

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:Question about Justification of Results
Justification of Results mentions: "A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t." What does it refer to when it means method? Also my group and I are somewhat confused as to how much is expected here since it is worth 20 marks and we can't think of a great deal to write about here.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Major Lab
Hi, Sorry but I don't understand why I got 0 marks for the "My_And" function because I did not use "if, and" statements anywhere in my code. Plz explain.
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

 
Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question:{0}
Classification:"""




DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_description_min_examples_no_task_desc = """

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Answer: unanswerable

Question: Unsure of compute_accuracy

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Answer: unanswerable

Question:Question about Justification of Results
Justification of Results mentions: "A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t." What does it refer to when it means method? Also my group and I are somewhat confused as to how much is expected here since it is worth 20 marks and we can't think of a great deal to write about here.
Answer: unanswerable

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Answer: unanswerable

Question: Major Lab
Hi, Sorry but I don't understand why I got 0 marks for the "My_And" function because I did not use "if, and" statements anywhere in my code. Plz explain.
Answer: unanswerable

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Answer: answerable with the right context

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Answer: answerable

Question:{0}
Answer:"""



DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_no_task_description = """

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Answer: unanswerable

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Answer: unanswerable

Question: Unsure of compute_accuracy
Hi, I couldn't complete the Lab 1 on time and submitted what I could so I'm going back and finishing it for my own understanding.

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Answer: unanswerable

Question:A1 Q4a
What is the size of inp?

edit: nvm got it lol
Answer: unanswerable

Question:Question about Justification of Results
Justification of Results mentions: "A justification that your implemented method performed reasonably, given the difficulty of the problem—or a hypothesis for why it doesn’t." What does it refer to when it means method? Also my group and I are somewhat confused as to how much is expected here since it is worth 20 marks and we can't think of a great deal to write about here.
Answer: unanswerable

Question: A2 Release Date?
According to the syllabus, A2 is due the friday after we come back from reading week...Will A2 be released today? I would really like to work on it during reading week...
Answer: unanswerable

Question: Major Lab
Hi, Sorry but I am not unable to understand How I got 0 for the "My_And" because I did not use "if, and" statements anywhere in my code. Plz explain
Answer: unanswerable

Question: lab 7 last task should be comparing to part 3?
Is this a typo? We should comparing our part 4 result with the best result from part 3?
Answer: unanswerable


Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Answer: answerable with the right context

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Answer: answerable with the right context

Question: Cant seem to make solve_via_sgd work? need help
Answer: answerable with the right context
 
Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Answer: answerable

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Answer: answerable

Question:{0}
Answer:"""

















DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_all_simple_description = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: Unsure of compute_accuracy
Hi, I couldn't complete the Lab 1 on time and submitted what I could so I'm going back and finishing it for my own understanding.

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable


Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question:A1 Q4a
What is the size of inp?

edit: nvm got it lol
Classification: unanswerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context
 
Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 

Question:{0}
Classification:"""








DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_one_description = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Question: Unsure of compute_accuracy
Hi, I couldn't complete the Lab 1 on time and submitted what I could so I'm going back and finishing it for my own understanding.

I noticed the TA said I had coded this wrong,
So assumed we would use X_valid and t_valid instead of X_new/t_new but I keep getting this error.
I'm curious where I am going wrong.
Nvm just solved it.
Classification: unanswerable

Question:A1 Q4a
What is the size of inp?

edit: nvm got it lol
Classification: unanswerable

Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context
 
Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable


Question:{0}
Classification:"""

DAVINCI_CLASSIFY_QUESTIONS_FEWSHOT_one_description_alt = """

Task Description: Classify the question from an undergraduate course discussion board as either "answerable", "unanswerable" or "answerable with the right context". 
A question that is "answerable" is one that can be answered without needing any additional context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "answerable with the right context" is one that can be answered if it was conditioned on the right context such as a course assignment, lab, syllabus or a screenshot of a course snippet. 
A question that is "unanswerable" is one that can only be answered by an instructor, one that depends on time or one that has been resolved by the student.

Question: Florian's Office Hours. Are Florian's OH still happening right now?
Classification: unanswerable

Question: Test 2 grades. When can we expect to get our test 2 grades back?
Classification: unanswerable

Question: lab5 backward i get stuck on the function backward. I get a graph like this. I am not sure if I get my z2_bar in a wrong way.
Classification: answerable with the right context

Question: Pred
So for the pred function in part 2 of the lab, is this the correct way to do it? Can we assume that each value in y is between 0 and 1? 
Classification: answerable with the right context

Question: Cant seem to make solve_via_sgd work? need help
Classification: answerable with the right context
 
Question: I wonder if gradient descent is only used to find the optimal coefficients right? We cannot use it to find the optimal hyper-paramater? Thank you!
Classification: answerable

Question: I looked the post on piazza and relooked at lab 2. When we split data into training, validation, test sets, we want different categories(or different pattern) into different sets. We don't want it split randomly. Is my understanding correct?
Classification: answerable

Question:{0}
Classification:"""








ONLY_IN_CONTEXT_EXAMPLES = """

Question: When can we expect to get our test 2 grades back?
Classification: unanswerable

Question: How do we approach this question on the homework? 
Nvm just solved it.
Classification: unanswerable

Question: The assignment says 1+1=3. Is this a typo? Should it be 1+1=2?
Classification: unanswerable

Question: Will we lose marks for doing x on the assignment?
Classification: unanswerable

Question: The assignment question says "x". But in lecture we learnt "y", which contradicts what the assignment question says. Is this a typo?
Classification: unanswerable


Question:  What does x stand for in Part 3 of the lab?
Classification: answerable with the right context

Question: I am stuck on the function backward(). I get a graph like this. I am not sure if it is because I'm computing z2_bar correctly. Any ideas?
Classification: answerable with the right context

Question: I can't seem to make solve_via_sgd work? need help
Classification: answerable with the right context


Question:  How does gradient descent work?
Classification: answerable 

Question:  What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 


Question:{0}
Classification:"""


####################  BINARY PROMPTS  ########################



BINARY_FEWSHOT = """

Question: When can we expect to get our test 2 grades back?
Classification: other

Question: How do we approach this question on the homework? 
Nvm just solved it.
Classification: other

Question: Will we lose marks for doing x on the assignment?
Classification: other

Question:  How does gradient descent work?
Classification: answerable 

Question:  What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 


Question:{0}
Classification:"""


BINARY_FEWSHOT2 = """
Question: When can we expect to get our test 2 grades back?
Classification: other

Question:  How do we approach this question on the homework? 
Nvm just solved it.
Classification: other

Question: A4. Will we lose marks for doing x on the assignment?
Classification: other

Question: Lab3. So for the pred function in part 2 of the lab, is this the correct way to do it?
Classification: other

Question: lab5 backward. i get stuck on the function backward(). I get a graph like this. Any help?
Classification: other

Question: What is the derivate of f(x) with respect to x?
Classification: other

Question: How does gradient descent work?
Classification: answerable 

Question: What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 

Question:  Why does a neural net with a single hidden unit produce a linear decision boundary
Classification: answerable 

Question: Lab3. Does the function sklearn.metrics.accuracy_score() compute accuracy?
Classification: other

Question: A1 Q3h) What is the softmax function? How does it work?
Classification: answerable

Question: A2 Does it work to use an RNN to handle time-series data?
Classification: answerable

Question: {0}
Classification:"""












BINARY_FEWSHOT3 = """
Question: When can we expect to get our test 2 grades back?
Classification: other

Question: How do we approach this question on the homework? 
Nvm just solved it.
Classification: other

Question: A4. Will we lose marks for doing x on the assignment?
Classification: other

Question: Lab3. So for the pred function in part 2 of the lab, is this the correct way to do it?
Classification: other

Question: lab5 backward. i get stuck on the function backward(). I get a graph like this. Any help?
Classification: other

Question: I am stuck on the function backward(). Any help?
Classification: other

Question: How does gradient descent work?
Classification: answerable 

Question: What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 

Question: Why does a neural net with a single hidden unit produce a linear decision boundary
Classification: answerable 

Question: {0}
Classification:"""


PROMPT_HIGH = """
Question: When can we expect to get our test 2 grades back?  -- csc413
Classification: other

Question:  How do we approach this question on the homework? -- 
Nvm just solved it.
Classification: other

Question: A4. Will we lose marks for doing x on the assignment?
Classification: other

Question: Lab3. So for the pred function in part 2 of the lab, is this the correct way to do it?
Classification: other

Question: lab5 backward. i get stuck on the function backward(). I get a graph like this. Any help?
Classification: other

Question: I am stuck on the function backward(). Any help?
Classification: other

Question:  How does gradient descent work?
Classification: answerable 

Question:  What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 

Question:  Why does a neural net with a single hidden unit produce a linear decision boundary
Classification: answerable 

Question: {0}
Classification:"""


PROMPT_HIGH2 = """
Question: When do we get our midterm marks back?  
Classification: other

Question:  How do we approach this question on the homework? 
Nvm just solved it.
Classification: other

Question: A4. Will we lose marks for doing x on the assignment?
Classification: other

Question: Lab3. So for the pred function in part 2 of the lab, is this the correct way to do it?
Classification: other

Question: lab5 backward. i get stuck on the function backward(). I get a graph like this. Any help?
Classification: other

Question:  How does gradient descent work?
Classification: answerable 

Question:  What are some strategies for splitting the dataset into train, validation and test sets?
Classification: answerable 

Question:  Why does a neural net with a single hidden unit produce a linear decision boundary
Classification: answerable 

Question: {0}
Classification:"""


PAUL_PROMPT = """
The following questions were posted on a course discussion board for an undergraduate programming course.
If you are able to answer the question given the context, write "answerable". If there is not enough context to answer the question, write "other".
Q: Request for an Extension for Assignment 3
A: other
Q: Program running but no output?
A: answerable
Q: I am doing a Pytest. Why is my main.py not being called?
A: answerable
Q: Assignment 3
A: other
Q: Unable to access Python from textbook instructions (mac)
A: answerable
Q: T.B example CH3 P 94 clarification
A: other
Q: Equality Operator
A: answerable
Q: String Method - Strip()
A: answerable
Q: {0}
A:"""

