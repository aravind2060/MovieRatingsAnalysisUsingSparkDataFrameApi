import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, desc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

def create_spark_session(master_node_url):
    spark = SparkSession.builder.appName('Movie Ratings Analysis').master(master_node_url).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def stop_spark_session(spark):
    spark.stop();


def define_schema():
    return StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genre", StringType(), True)
    ]), StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", IntegerType(), True)
    ])

def load_data(spark, file_path, schema):
    return spark.read.csv(file_path, header=True, schema=schema)

def preprocess_data(df):
    return df.na.drop()

def basic_analysis(movies_df, ratings_df, output_path):
    total_movies = movies_df.count()
    print(f"Total number of movies: {total_movies}")
    
    total_ratings = ratings_df.count()
    print(f"Total number of ratings: {total_ratings}")

    avg_ratings = ratings_df.groupBy('movieId').agg(avg('rating').alias('avg_rating'))
    top_movies = avg_ratings.join(movies_df, 'movieId').orderBy(desc('avg_rating'))
    
    top_movies.write.csv(f"{output_path}/top_movies", mode="overwrite", header=True)
    return top_movies

def main(master_node_url,movie_input_file_path, ratings_input_file_path, output_path):
    spark = create_spark_session(master_node_url)
    movie_schema, rating_schema = define_schema()

    movies_df = load_data(spark, movie_input_file_path, movie_schema)
    ratings_df = load_data(spark, ratings_input_file_path, rating_schema)

    movies_df = preprocess_data(movies_df)
    ratings_df = preprocess_data(ratings_df)

    top_movies = basic_analysis(movies_df, ratings_df, output_path)
    
    # while True:
    #     pass
    stop_spark_session(spark);
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: program.py <master_node_url> <movie_input_file_path> <ratings_input_file_path> <output_path>")
        sys.exit(1)

    master_node_url = sys.argv[1];
    movie_input_file_path = sys.argv[2]
    ratings_input_file_path = sys.argv[3]
    output_path = sys.argv[4]

    main(master_node_url,movie_input_file_path, ratings_input_file_path, output_path)
