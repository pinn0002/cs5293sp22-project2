Name: Divya Sai Pinnamaneni

Run Procedure:
>Install all the below specified libraries using the following command pipenv install modulename.

>Install pipenv using pip install pipenv

>Run the project using pipenv run python project2.py --N --ingredient where ingredient takes kind of ingredient that should be added to meal and this argument can be repeated any number of times in the command line argument and N is the number nearest neighbors to be displayed through prediction

>Run testfiles using pipenv run python -m pytest

Expected bugs

Libraries used in this project:

json,argparse,pandas,sklearn

Project Objective:

Goal of this project is to create a program which takes list of ingredients from user through command line and attempts to predict the nearest cuisine and similar meals.
Below steps need to be followed for successful creation of this project.

1. Train the provided food data i.e json file yummly.json in our case. we need to train the system with the input from yummly.json file.
2. create a argument which will allow User to provide the list of ingredients through command line arguments.
3. Using the trained model, predict the type of cuisne for the given ingredients passed by user and tell the user the type of cuisine.
4. Take N through command line argument which is to return total number of nearest neighbors for predicted value to be returned to the user. Here we are required to return the IDs of the neighbors.

Understand the provided data:
Yummly.json contains id, cuisine, ingredients. id is of type integer and cuisine is a kind of dish for the list of ingredients i.e. Mexican, American ..etc. ingredients is an array of list of items.

As our project requires two inputs, we need to add two arguments using argparse.

Two arguments:

--ingredient: It is of type string and this argument can be repeated any number of times and used append action to take all inputs of this argument.

--N: It takes integer value as input and used for getting N number of closest neighbors for the predicted value.

Use pd.read_json(file) to read the json file and convert it to dataFrame using pandas.

convert list of ingredients to a string to pass it to the vectorizer. For this use join function to combine list of ingredients to a string.

stringredient = [" ".join(doc).lower() for doc in data['ingredients']]

Assign string of ingredients to a dataframe with the column name stringIngredients.

data['stringIngredients'] = stringredient

create a TfidVectorizer with binary values and use the vectorizer to fit_transform of string of ingredients.

Use LabelEncoder to assign a label to each cuisine. By LabelEncoder we can get each label for every kind of cuisine.

LabelEncoder().fit_transform(data['cuisine'])

Use LogisticRegression to fit Ingredients and cuisines for traing the model with Ingredients and cuisines
    classifier = LogisticRegression(max_iter=100000).fit(Ingredients, cuisineml)

Get all input ingredients and transform vectorizer and convert to array.Using this vectorizer array to predict the input ingredients.

use inverse transform to obtain decoded value of encoded label. Initially we have LabelEncoder to label each ingredient. As we have only encoded values we should decode the values to get original cuisines. For decoding, we used Inverse_transorm unction to get original cuisine names.

Using cosine_similarity function, similarity between each input to the ingredients is found and assign this score the dataFrame. For obtaining closest distanced cuisine sort the score in decreasing order and get the first cusine and score which is of closest distance.
DataFrame would only contain predicted value row along with N nearest neighbors.

To obtain the output in Json format, convert nearest cuisine and closest distance, N nearest ids and scores to the required output which should be of form like below.

{

 "cuisine": "southern_us",

 "score": 0.43609088838272886,

 "closest": [

  {

   "id": 9944,

   "score": 0.3916287267961693

  },

  {

   "id": 13474,

   "score": 0.37219842908660455

  },

  {

   "id": 40583,

   "score": 0.35196471987094685

  },

  {

   "id": 27368,

   "score": 0.3181391481484255

  },

  {

   "id": 30881,

   "score": 0.3090660905794663

  }

 ]

}


