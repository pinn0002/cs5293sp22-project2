import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def main(args):
    input_ingredient = []
    for each in args.ingredient:
        input_ingredient.append(each)
    input_string = ""
    input_final = []

    for x, y in enumerate(input_ingredient):
        input_string = input_string + y + " "
    input_final.append(input_string)
    project2(args.N,input_final)

def project2(N,input_final):
    data = dataread()
    cuisine, vectors_array_test , Ingredients = LogisticModel(data,input_final)
    df , closestdistance = similarityScore(Ingredients , vectors_array_test, data ,N)

    # create array of dictionaries with id and similarity score for closest
    outputdict = df.apply(pd.Series.explode).to_dict(orient='records')
    output = {
        'cuisine': cuisine[0],
        'score': closestdistance,
        'closest': outputdict
    }
    print(json.dumps(output, indent=1))


def dataread():
    #open json and read and convert to a dataFrame
    with open('..\cs5293sp22-project2\yummly.json', 'r') as file:
        data = pd.read_json(file)
    return data

def LogisticModel(data, input_final):
    #convert list of ingredients to a string
    stringredient = [" ".join(doc).lower() for doc in data['ingredients']]
    #Assign string of ingredients to a dataframe with stringIngredients
    data['stringIngredients'] = stringredient

    #create a TfidVectorizer with binary values and use the vectoriizer to fit_transform of string of ingredients
    vectorizer = TfidfVectorizer(binary=True)
    Ingredients = vectorizer.fit_transform(stringredient)
    cuisineml1 = [doc for doc in data['cuisine']]
    # use LabelEncoder to assign a label to each cuisine
    cuisineml = LabelEncoder().fit_transform(data['cuisine'])

    # Use LogisticRegression to fit Ingredients and cuisines
    classifier = LogisticRegression(max_iter=100000).fit(Ingredients, cuisineml)

    vectors2 = vectorizer.transform(input_final)

    vectors_array_test = vectors2.toarray()
    #predict the input vectorizer of ingredients
    testpredictor = classifier.predict(vectors_array_test)
    # use inverse transform to obtain decoded value of encoded label
    cuisine = LabelEncoder().fit(cuisineml1).inverse_transform(testpredictor).tolist()

    return cuisine, vectors_array_test , Ingredients
    print(Ingredients , vectors_array_test, data ,N)

def similarityScore(Ingredients , vectors_array_test, data ,N):
    #Similary score between input to each ingredient is found
    similarityList = cosine_similarity(vectors_array_test[0:1],Ingredients)
    #Assign similary score to a dataFrame with column name score
    data["score"] = similarityList[0]
    #Obtain closest distance by sorting the list
    closestdistance = sorted(similarityList[0])[-1]
    #sort the dataFrame based on score column
    data = data.sort_values(by=['score'], ascending=False)
    #obtain closest value and it's nearest neighbors for N
    data = data.head(N+1)
    #drop the closest row
    data = data.iloc[1:, :]
    closestneighbors = []
    cosinescore = []
    for index, row in data.iterrows():
        closestneighbors.append(row['id'])
        cosinescore.append(row['score'])
    df = pd.DataFrame({"id": closestneighbors, "score": cosinescore})
    return df , closestdistance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient",type=str,required=True,help="input ingredients",action="append")
    parser.add_argument("--N", type=int, required=True, help="nearest neighbors")
    args = parser.parse_args()
    main(args)