import requests
from PIL import Image
from io import BytesIO
import os
import re
from papzi.constants import BASE_DIR
from papzi.utils import load_label_map


# Configuration
TMDB_API_KEY = (
    "1ee728e3ef3eca766c5aab1b066e2e7a"  # Replace with your TMDb API key
)
ACTOR_NAME = (
    "benedict cumberbatch"  # Replace with the actor name you're interested in
)


class TMDbPosterDownloader:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer " + os.environ["TMDB_API_KEY"],
        }

    def get_actor_id(self, actor_name):
        url = f"{self.base_url}/search/person"
        params = {"query": actor_name}
        response = requests.get(url, params=params, headers=self.headers)
        data = response.json()
        print(data)
        if data["results"]:
            return data["results"][0]["id"]
        return None

    def get_movies_by_actor(self, actor_id):
        url = f"{self.base_url}/discover/movie"
        params = {"with_cast": actor_id, "sort_by": "popularity.desc"}
        response = requests.get(url, params=params, headers=self.headers)
        data = response.json()
        return data["results"][:10]

    def get_series_by_actor(self, actor_id):
        url = f"{self.base_url}/person/{actor_id}/tv_credits"
        params = {}
        response = requests.get(url, params=params, headers=self.headers)
        data = response.json()
        data = [
            i
            for i in data["cast"]
            if i["episode_count"] > 2
            and "show" not in i["original_name"].lower()
        ]
        return sorted(data, key=lambda x: x["popularity"], reverse=True)[:10]

    def download_image(self, url, path):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.save(path)

    def sanitize_filename(self, filename):
        return re.sub(r'[\\/:"*?<>|]+', "", filename)

    def download_posters(self, actor_names):
        output_dir = BASE_DIR / "posters"
        if not output_dir.exists():
            output_dir.mkdir()

        for actor_name in actor_names:
            actor_id = self.get_actor_id(actor_name)
            if not actor_id:
                print(f"No actor found with name {actor_name}")
                continue

            movies = self.get_movies_by_actor(actor_id)
            series = self.get_series_by_actor(actor_id)

            actor_dir = output_dir / self.sanitize_filename(actor_name)
            if not actor_dir.exists():
                actor_dir.mkdir()
            movie_dir = actor_dir / "movie"
            if not movie_dir.exists():
                movie_dir.mkdir()
            for movie in movies:
                poster_path = movie["poster_path"]
                if poster_path:
                    title = movie["title"]
                    year = (
                        movie["release_date"][:4]
                        if "release_date" in movie and movie["release_date"]
                        else "N/A"
                    )
                    rating = movie["vote_average"]
                    filename = f"{title}_{year}_{rating:.1f}.jpg"
                    sanitized_filename = self.sanitize_filename(filename)
                    poster_url = f"{self.image_base_url}{poster_path}"
                    self.download_image(
                        poster_url, movie_dir / sanitized_filename
                    )

            tv_dir = actor_dir / "tv"
            if not tv_dir.exists():
                tv_dir.mkdir()
            for serie in series:
                poster_path = serie["poster_path"]

                if poster_path:
                    title = serie["name"]
                    year = (
                        serie["first_air_date"][:4]
                        if "first_air_date" in serie
                        and serie["first_air_date"]
                        else "N/A"
                    )
                    rating = serie["vote_average"]
                    filename = f"{title}_{year}_{rating:.1f}.jpg"
                    sanitized_filename = self.sanitize_filename(filename)
                    poster_url = f"{self.image_base_url}{poster_path}"
                    self.download_image(
                        poster_url, tv_dir / sanitized_filename
                    )

            print(f"Downloaded posters for {actor_name}")


# Usage
label_map = load_label_map()
actor_names = list(label_map.values())  # Replace with your list of actor names
downloader = TMDbPosterDownloader()
downloader.download_posters(actor_names)
