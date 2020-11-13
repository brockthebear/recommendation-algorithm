import csv
import pandas as pd
from 

df = pd.read_csv("./data/movies.csv")
df["year"] = ""
df.to_csv("./data/result.csv", Index=False)

with 
