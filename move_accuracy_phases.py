# Disclaimer: Part of this code was David's code, I just added the game phases
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, posexplode, regexp_extract,
    when, lag, exp, lit, sha2, concat_ws, 
    floor, avg, count, sum, row_number, max
)
from pyspark.sql.window import Window


# ----------------------------
# Spark session
# ----------------------------
spark = (
    SparkSession.builder
    .appName("accuracy move phases")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

# ----------------------------
# Read data
# ----------------------------
chess_2017 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2018-*.parquet")
chess_2017 = chess_2017.filter(col("Moves").contains("%eval")).withColumn("year", lit(2018))

chess_2018 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2018-*.parquet")
chess_2018 = chess_2018.filter(col("Moves").contains("%eval")).withColumn("year", lit(2018))

chess_2019 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2019-*.parquet")
chess_2019 = chess_2019.filter(col("Moves").contains("%eval")).withColumn("year", lit(2019))

chess_2020 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2020-*.parquet")
chess_2020 = chess_2020.filter(col("Moves").contains("%eval")).withColumn("year", lit(2020))

chess_2021 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2021-*.parquet")
chess_2021 = chess_2021.filter(col("Moves").contains("%eval")).withColumn("year", lit(2021))

chess_2022 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2022-*.parquet")
chess_2022 = chess_2022.filter(col("Moves").contains("%eval")).withColumn("year", lit(2022))

chess_2023 = spark.read.parquet("/data/doina/Lichess/lichess_db_standard_rated_2023-*.parquet")
chess_2023 = chess_2023.filter(col("Moves").contains("%eval")).withColumn("year", lit(2023))

# Combine them all, eval moves already selected
chess = (
    chess_2017
    .union(chess_2018)
    .union(chess_2019)
    .union(chess_2020)
    .union(chess_2021)
    .union(chess_2022)
    .union(chess_2023)
)

# ----------------------------
# Create stable game_id
# ----------------------------
chess = chess.withColumn(
    "game_id",
    sha2(
        concat_ws("||", "Site", "UTCDate", "UTCTime"),
        256
    )
)

# ----------------------------
# Explode moves
# ----------------------------
moves = chess.select(
    "game_id",
    "Moves",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "year"
)

moves = moves.withColumn(
    "move_array",
    split(col("Moves"), r"\}\s*")
)

moves = moves.select(
    "*",
    posexplode("move_array").alias("move_index", "move_text")
)

# ----------------------------
# Extract eval and clock
# ----------------------------
moves = moves.withColumn(
    "eval_pawns",
    regexp_extract(
        "move_text", r"%eval\s+(-?\d+\.?\d*)", 1
    ).cast("double")
).withColumn(
    "centipawns",
    col("eval_pawns") * 100
).withColumn(
    "clock",
    regexp_extract(
        "move_text", r"%clk\s+(\d+:\d+:\d+)", 1
    )
)


# ----------------------------
# Player & Elo
# ----------------------------
moves = moves.withColumn(
    "player",
    when(col("move_index") % 2 == 0, lit("White"))
    .otherwise(lit("Black"))
)

moves = moves.withColumn(
    "player_elo",
    when(col("player") == "White", col("WhiteElo"))
    .otherwise(col("BlackElo"))
)


# ----------------------------
# Extract move notation
# ----------------------------
moves = moves.withColumn(
    "move_notation",
    regexp_extract(
        "move_text",
        r"^\s*\d+\.+\s*([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?)",
        1
    )
)

# New code starts here
# Get all moves that with either a capture or a king move
moves = moves.withColumn(
    "is_capture",
    col("move_notation").contains("x")
)

moves = moves.withColumn(
    "is_king_move",
    col("move_notation").startswith("K")
)

# Count cumulative captures and king moves per game per player
w_game_cumulative = (
    Window
    .partitionBy("game_id")
    .orderBy("move_index")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

# Get the total captures in the game so far, to determine the phase
moves = moves.withColumn(
    "total_captures",
    sum(when(col("is_capture"), 1).otherwise(0)).over(w_game_cumulative)
)

# Get the total king moves in the game so far, to determine the phase
w_player_cumulative = (
    Window
    .partitionBy("game_id", "player")
    .orderBy("move_index")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

moves = moves.withColumn(
    "player_king_moves",
    sum(when(col("is_king_move"), 1).otherwise(0)).over(w_player_cumulative)
)

# Check if either player has moved their king at least twice
w_game_king_check = (
    Window
    .partitionBy("game_id")
    .orderBy("move_index")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

moves = moves.withColumn(
    "max_king_moves_so_far",
    max(col("player_king_moves")).over(w_game_king_check)
)

# Determine the game stage (see report for explanation)
moves = moves.withColumn(
    "game_stage_raw",
    when(col("max_king_moves_so_far") >= 2, lit("end"))
    .when(col("total_captures") >= 5, lit("middle"))
    .otherwise(lit("beginning"))
)

# Rename the moves stages so we can take the max
moves = moves.withColumn(
    "stage_numeric",
    when(col("game_stage_raw") == "beginning", lit(1))
    .when(col("game_stage_raw") == "middle", lit(2))
    .when(col("game_stage_raw") == "end", lit(3))
)

# Take the maximum stage reached so far
moves = moves.withColumn(
    "max_stage_so_far",
    max(col("stage_numeric")).over(w_game_cumulative)
)

# Convert back to text, so all phases are correct
moves = moves.withColumn(
    "game_stage",
    when(col("max_stage_so_far") == 1, lit("beginning"))
    .when(col("max_stage_so_far") == 2, lit("middle"))
    .otherwise(lit("end"))
)

# End new code

# ----------------------------
# Clock to seconds
# ----------------------------
moves = moves.withColumn(
    "clock_seconds",
    split(col("clock"), ":")[0].cast("int") * 3600 +
    split(col("clock"), ":")[1].cast("int") * 60 +
    split(col("clock"), ":")[2].cast("int")
)

# ----------------------------
# Time spent per move (per game & player)
# ----------------------------
w_player = (
    Window
    .partitionBy("game_id", "player")
    .orderBy("move_index")
)

moves = moves.withColumn(
    "time_spent",
    lag("clock_seconds").over(w_player) - col("clock_seconds")
)

# ----------------------------
# Win probability
# ----------------------------
moves = moves.withColumn(
    "win_percent",
    50 + 50 * (
        2 / (1 + exp(-0.00368208 * col("centipawns"))) - 1
    )
)

# ----------------------------
# Accuracy (per game)
# ----------------------------
w_game = (
    Window
    .partitionBy("game_id")
    .orderBy("move_index")
)

moves = moves.withColumn(
    "win_percent_before",
    lag("win_percent").over(w_game)
)

moves = moves.withColumn(
    "accuracy",
    103.1668 * exp(
        -0.04354 * (
            col("win_percent_before") - col("win_percent")
        )
    ) - 3.1669
)

# Clamp accuracy to [0, 100]
moves = moves.withColumn(
    "accuracy",
    when(col("accuracy") > 100, 100)
    .when(col("accuracy") < 0, 0)
    .otherwise(col("accuracy"))
)

# ----------------------------
# Final dataset
# ----------------------------
final_moves = moves.select(
    "game_id",
    "move_index",
    "player",
    "player_elo",
    "centipawns",
    "win_percent_before",
    "win_percent",
    "accuracy",
    "time_spent",
    "TimeControl",
    "game_stage",
    "year"
)


binned_elo = (
    final_moves
    .filter(col("time_spent").isNotNull())
    .filter(col("time_spent").between(0, 60))
    .filter(col("player_elo").isNotNull())
    .withColumn("time_bin", floor(col("time_spent")))
    .withColumn(
        "elo_band",
        when(col("player_elo") < 1400, "beginner")
        .when(col("player_elo").between(1400, 1999), "intermediate")
        .otherwise("expert")
    )
    .groupBy("year", "elo_band", "time_bin", "game_stage")
    .agg(
        avg("accuracy").alias("avg_accuracy"),
        count("*").alias("n_moves"),           # number of moves in bin
        sum(lit(1)).alias("total_games")       # total games in this aggregation
    )
    .orderBy("year", "elo_band", "time_bin", "game_stage")
)

binned_elo.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("binned_accuracy_vs_time_elo_by_year")
