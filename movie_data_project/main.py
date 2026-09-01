import os
import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("TMDB_API_KEY")


movies = []

for i in range(1, 501):

    response = requests.get(
        f"https://api.themoviedb.org/3/movie/top_rated"
        f"?api_key={api_key}"
        f"&language=en-US"
        f"&page={i}"
    )

    print(i, response.status_code)

    data = response.json()

    if "results" not in data:
        print("API ERROR:")
        print(data)
        break

    temp_df = pd.DataFrame(data["results"])[
        [
            "id",
            "title",
            "overview",
            "release_date",
            "popularity",
            "vote_average",
            "vote_count",
        ]
    ]

    movies.append(temp_df)

df = pd.concat(movies, ignore_index=True)

print(df.head())
print(df.shape)

df.to_csv("movies.csv", index=False)