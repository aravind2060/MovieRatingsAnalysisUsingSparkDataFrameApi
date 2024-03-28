import csv
import random
def generate_movie(movie_id):
 title = f"Movie {movie_id}"
 genre = random.choice(["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"])
 return [movie_id, title, genre]
def generate_rating(user_id, movie_id):
 rating = random.randint(1, 5)
 return [user_id, movie_id, rating]
num_movies = 100
num_users = 50
max_ratings_per_user = 20
movies_data = [generate_movie(movie_id) for movie_id in range(1, num_movies + 1)]
ratings_data = []
for user_id in range(1, num_users + 1):
 num_ratings = random.randint(1, max_ratings_per_user)
 rated_movies = random.sample(range(1, num_movies + 1), num_ratings)
 for movie_id in rated_movies:
  ratings_data.append(generate_rating(user_id, movie_id))
with open("../data/input/movies.csv", mode='w', newline='', encoding='utf-8') as file:
 writer = csv.writer(file)
 writer.writerow(["movieId", "title", "genre"])
 writer.writerows(movies_data)
with open("../data/input/ratings.csv", mode='w', newline='', encoding='utf-8') as file:
 writer = csv.writer(file)
 writer.writerow(["userId", "movieId", "rating"])
 writer.writerows(ratings_data)