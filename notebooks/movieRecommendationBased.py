import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql.functions import col
from pyspark.sql.functions import explode

def create_spark_session():
    spark = SparkSession.builder \
        .appName('Movie Recommendations with ALS') \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark, ratings_input_file_path, movies_input_file_path):
    # Load ratings data
    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", StringType(), True)
    ])
    
    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)
    ])

    ratings_df = spark.read.csv(ratings_input_file_path, header=True, schema=ratings_schema)
    movies_df = spark.read.csv(movies_input_file_path, header=True, schema=movies_schema)

    return ratings_df, movies_df

def train_als_model(ratings_df):
    # Train the ALS model
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(ratings_df)
    return model

def recommend_movies(spark, model, movies_df, user_id, num_recommendations):
    # Generate top N movie recommendations for a given user
    user_df = spark.createDataFrame([(user_id,)], ['userId'])
    recommendations = model.recommendForUserSubset(user_df, num_recommendations)

    # Explode the recommendations to flatten the structure
    recommendations = recommendations.select(
        col("userId"), explode(col("recommendations")).alias("recommendation")
    ).select(
        col("userId"), 
        col("recommendation.movieId").alias("movieId"),
        col("recommendation.rating").alias("rating")
    )

    # Join the recommendations with the movies DataFrame to get the movie titles
    recommendations = recommendations.alias('rec')\
        .join(movies_df.alias('movies'), col('rec.movieId') == col('movies.movieId'))\
        .select('rec.userId', 'movies.title', 'rec.rating')
   
    
    return recommendations

def main(movie_input_file_path, ratings_input_file_path, user_id, num_recommendations):
    
    spark = create_spark_session()
    ratings_df, movies_df = load_data(spark, ratings_input_file_path, movie_input_file_path)
    model = train_als_model(ratings_df)
    user_recommendations = recommend_movies(spark, model, movies_df, user_id, num_recommendations)

    # Show recommendations with movie names
    user_recommendations.show(truncate=False)
    
    # Define the path for the output CSV file
    output_csv_path = '../data/output/recommendations'
    
    # Write the recommendations to a CSV file
    user_recommendations.coalesce(1).write.csv(output_csv_path, mode="overwrite", header=True)
    
    spark.stop()




if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: program.py <movie_input_file_path> <ratings_input_file_path> <user_id> <num_recommendations> \n python //workspaces/MovieRatingsAnalysisUsingSparkDataFrameApi/notebooks/movieRatingsAnalysis.py  /workspaces/MovieRatingsAnalysisUsingSparkDataFrameApi/data/input/movies.csv /workspaces/MovieRatingsAnalysisUsingSparkDataFrameApi/data/input/ratings.csv 4 7")
        sys.exit(1)

    movie_input_file_path = sys.argv[1]
    ratings_input_file_path = sys.argv[2]
    user_id = int(sys.argv[3])
    num_recommendations = int(sys.argv[4])

    main(movie_input_file_path, ratings_input_file_path, user_id, num_recommendations)
