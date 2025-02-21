# THE BIG FIVE PERSONNALITY TEST
#### Video Demo: https://youtu.be/ual4KcvrxcE
#### Description:

# Introduction #
This project is a personnality test, the Big Five, that uses a machine learning algorithm (Kmeans) to classify you into different clusters of personnality. Basically, you take the test and the algorithm I trained will classify you into a specific cluster (or a spirit animal) based on the answers of 1 million around the world that answered the big five test (the dataset can be found on kaggle : https://www.kaggle.com/datasets/tunguz/big-five-personality-test). For the purpose of this project, I trained the ML model on 25 000 answers only.

The Big Five is the most reliable personnality test at the moment, and classify everyone into 5 different traits : Extraversion, Agreableness, Consciensciousness, Neuroticism, Openess. Many versions of the Big Five exists, but for the purpose of this project, I used the one with 50 questions. 

# How does it work #
The user is welcomed in the website and is invited to take the Big Five test. Their is 4 options on the navbar (homepage / WHat is the big Five / ML model used / Questions) where the user can learn more about the Big Five test, the model I trained, and questions if they want to contact me. When the users clicks on "take the test", he answers 50 questions (10 for each personnality test, randomly displayed). Finally, when he submits his answer, he will receive a feedback of his personnality trait, the percentage he got for each of them, personnalised advices and will be assigned a spirit animal, based of the cluster of personnality he ressembles the most, according to a machine learning model I previously trained.

# Pre-Requisite #
This project uses some python librairies for machine learning and data analysis, such as numpy, matplotlib, sklearn, joblib, seaborn, base64 and io, which are available in python 3.12 and above. IT alsos involves some HTML, javascript, CSS and flask code, which were already used in CS50.

To manage the package in a virtual environment, I used anaconda. You can use the same environment, using the file cs50.yml included in the folder of this project

## 1. Clone the Repository ##
Start by cloning the repository to your local machine:
```bash
git clone <repository-url>
cd <repository-name>
```

## 2. Create the Conda Environment ##
Use the provided `environment.yml` file to create the environment:
```bash
conda env create -f cs50.yml
```

## 3. Activate the Environment ##
Once the environment is created, activate it:
```bash
conda deactivate
```

## Notes ##

- If you don't have Conda installed, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it.
- For help with Conda commands, see the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/commands.html).


## Structure of the project
For this project, I used flask, python, CSS, HTML and a little bit of javascript. First of all, there is a style.css sheet and app.js on the folder 'static' for the codes in javascript and CSS across all the website.

## HTML files 
In 'base.html' in templates, I included a navbar that I implemented using jinja for so that it can be displayed on all the pages of the website. In 'index.html' there is the introduction to the website, where the user is welcomed and can decide to take the test, or read more aboutthe Big Five or learn about the machine model I used. In 'personnality.html', I included an explanation of the Big Five for the user. In 'ml.html', I explained that I trained a ML model to classify the use into a cluster of personnality.
In 'test.html', there is the 50 questions of the big five test on a 5 point scale (Strongly disagree, Disagree, Neutral, Agree, Strongly Agree). The user has to respond to all of them before submitting. And finally, in 'submit.html', the user will see the results of the test, with the % of personnality trait he got, a visualisation of his traits (using matplotlib), a picture of the spirit animal he got (the cluster he got assigned) and some insights on what he should know about the results he got for each trait. If the user has any question, he can click on "question" on the navbar to send me a request.

## Python files ##
Firstly, in model.ipynb (python using jupyter notebook), you can find the code for the ML model (Kmeans) that I trained, the pre-processing steps using MinMaxscaler and the steps to determine the number of cluster (using PCA and the elbow method). Using joblib, I saved this model to be used later as big_five_model.plk. Furthermore, I also uploaded scaler.plk, which is the tool I used to standardize the data.

Secondly, in app.py, you will find all the flask logic. More specifically, there is a result(), a function that takes the answers that the user made in test.html and calculates the average answer for each personnality trait and an histogram to visualise the results. Furthermore, these results are then added to the model I previously trained so the user is classified into a specific cluser (or spirit animal, which are : elephant, cheetah, dolphin, parrot, tiger, dog or a owl). Finally, depending of the percentage the user got for each trait, the result function will display specific advices tailored to the user's answer.








