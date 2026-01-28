from pyspark.sql import SparkSession, functions as F

def main():
    spark = (
        SparkSession.builder
        .appName("LichessTimeControlByRating")
        .getOrCreate()
    )

    
    year = 2017 # years [2017-2023] manually
    months = [f"{m:02d}" for m in range(1, 13)]  # 12 months

    paths = [
        f"/data/doina/Lichess/lichess_db_standard_rated_{year}-{m}.parquet"
        for m in months
    ]

    # Read all months together
    df = spark.read.parquet(*paths)

    df = df.withColumn("has_eval", F.col("Moves").contains("eval"))

    # rating separation per player

    def rating_level(col):
        return (
            F.when(col < 1400, "Beginner")
             .when(col < 2000, "Intermediate")
             .otherwise("Expert")
        )

    df = (
        df
        .withColumn("WhiteLevel", rating_level(F.col("WhiteElo")))
        .withColumn("BlackLevel", rating_level(F.col("BlackElo")))
    )

    # TimeControl -> total seconds -> category

    parts = F.split(F.col("TimeControl"), r"\+")

    df = (
        df
        .withColumn("start_sec", parts.getItem(0).cast("int"))
        .withColumn("inc_sec",  parts.getItem(1).cast("int"))
        .withColumn("total_time_sec", F.col("start_sec") + 40 * F.col("inc_sec"))
        .withColumn(
            "TimeCategory",
            F.when(F.col("total_time_sec") <= 29,    "UltraBullet")
             .when(F.col("total_time_sec") <= 179,  "Bullet")
             .when(F.col("total_time_sec") <= 479,  "Blitz")
             .when(F.col("total_time_sec") <= 1499, "Rapid")
             .otherwise("Classic")
        )
        .drop("start_sec", "inc_sec")
    )

    
    white = df.select(
        F.lit("White").alias("Color"),
        F.col("WhiteElo").alias("Elo"),
        F.col("WhiteLevel").alias("RatingLevel"),
        F.col("TimeCategory")
    )

    black = df.select(
        F.lit("Black").alias("Color"),
        F.col("BlackElo").alias("Elo"),
        F.col("BlackLevel").alias("RatingLevel"),
        F.col("TimeCategory")
    )

    players = white.unionByName(black)

    # couting everything up
    counts = (
        players
        .groupBy("RatingLevel", "TimeCategory")
        .count()
    )

    totals = (
        players
        .groupBy("RatingLevel")
        .count()
        .withColumnRenamed("count", "total_count")
    )

    pct = (
        counts
        .join(totals, on="RatingLevel", how="inner")
        .withColumn(
            "percentage",
            F.col("count") / F.col("total_count") * 100.0
        )
    )


    #local filesystem:
    output_path = f"hdfs:///user/s3741117/timecontrol_by_rating_{year}_full"

    (
        pct
        .select("RatingLevel", "TimeCategory", "count", "total_count", "percentage")
        .coalesce(1)  #small data
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(output_path)
    )

    spark.stop()


if __name__ == "__main__":
    main()
