import project2
import pytest
import pandas as pd
def test_dataread():
    data = project2.dataread()
    assert data is not None
    cuisine, vectors_array_test , Ingredients = project2.LogisticModel(data,['chicken pasta'])
    assert cuisine is not None
    assert vectors_array_test is not None
    df , closestdistance = project2.similarityScore(Ingredients , vectors_array_test, data ,5)
    assert df is not None
    assert closestdistance is not None

