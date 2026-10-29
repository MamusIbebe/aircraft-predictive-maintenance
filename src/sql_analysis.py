"""Step 7: load the engine data into a DuckDB database and explore it with SQL."""
import duckdb

from data import check, load

# Put the training data into a database table called engine_flights.
train = check(load("data/raw/train_FD001.txt"), "train")
con = duckdb.connect("data/engines.duckdb")
con.execute("CREATE OR REPLACE TABLE engine_flights AS SELECT * FROM train")
print("Table engine_flights created.\n")

# Query 1 - GROUP BY: how many flights did each engine last?
q1 = """
SELECT unit                      AS engine,
       MAX(cycle)                AS flights_until_failure,
       ROUND(MAX(s4) - MIN(s4), 1) AS s4_temp_rise
FROM engine_flights
GROUP BY unit
ORDER BY flights_until_failure
LIMIT 5;
"""
print("Shortest-lived engines:")
print(con.execute(q1).df().to_string(index=False), "\n")

# Query 2 - CTE + window function + CASE: compare sensors by health stage.
q2 = """
WITH labeled AS (
    SELECT *,
           MAX(cycle) OVER (PARTITION BY unit) - cycle AS rul
    FROM engine_flights
)
SELECT CASE
           WHEN rul <= 20 THEN '1) 0-20 left (critical)'
           WHEN rul <= 50 THEN '2) 21-50 left (warning)'
           ELSE                '3) 50+ left (healthy)'
       END                 AS health_stage,
       COUNT(*)            AS flights,
       ROUND(AVG(s4), 1)   AS avg_s4_temp,
       ROUND(AVG(s11), 2)  AS avg_s11_pressure
FROM labeled
GROUP BY health_stage
ORDER BY health_stage;
"""
print("Sensor averages by health stage:")
print(con.execute(q2).df().to_string(index=False), "\n")

# Query 3 - rolling average in SQL.
q3 = """
SELECT unit, cycle, s4,
       ROUND(AVG(s4) OVER (PARTITION BY unit ORDER BY cycle
                           ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 2) AS s4_avg_last10
FROM engine_flights
WHERE unit = 1
ORDER BY cycle DESC
LIMIT 3;
"""
print("Engine 1, 10-flight rolling average of s4 (last 3 flights):")
print(con.execute(q3).df().to_string(index=False))
con.close()
