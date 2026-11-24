"""Spark SQL for the silver, gold and segmentation layers.

Each function returns a ``SELECT`` statement. Notebooks wrap it in
``CREATE OR REPLACE TABLE <name> AS ...`` (Delta on Databricks); tests run the same
statements on a local Spark session against temporary views. ``tn`` maps a logical table
name (e.g. ``"silver_trips"``) to the name to use in SQL.
"""

from __future__ import annotations

from collections.abc import Callable

from bikecast import config

TableNamer = Callable[[str], str]


def silver_trips(tn: TableNamer) -> str:
    """Typed, deduplicated, valid trips with start date and duration."""
    return f"""
WITH typed AS (
  SELECT
    ride_id,
    rideable_type,
    member_casual,
    TRIM(start_station_id)   AS start_station_id,
    start_station_name,
    to_timestamp(started_at) AS started_at,
    to_timestamp(ended_at)   AS ended_at
  FROM {tn("bronze_trips")}
),
dedup AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ride_id ORDER BY started_at) AS rn
  FROM typed
)
SELECT
  ride_id, rideable_type, member_casual, start_station_id, start_station_name,
  started_at, ended_at,
  to_date(started_at)                                             AS start_date,
  CAST(unix_timestamp(ended_at) - unix_timestamp(started_at) AS DOUBLE) / 60 AS duration_min
FROM dedup
WHERE rn = 1
  AND start_station_id IS NOT NULL AND start_station_id <> ''  -- dockless e-bikes: no station
  AND started_at BETWEEN TIMESTAMP'{config.START} 00:00:00' AND TIMESTAMP'{config.END} 23:59:59'
  AND (unix_timestamp(ended_at) - unix_timestamp(started_at))
      BETWEEN {config.MIN_DURATION_S} AND {config.MAX_DURATION_S}
"""


def silver_removal_counts(tn: TableNamer) -> str:
    """Why rows were removed between bronze and silver (mirrors the silver filters)."""
    return f"""
WITH typed AS (
  SELECT ride_id, TRIM(start_station_id) AS sid,
         to_timestamp(started_at) AS s, to_timestamp(ended_at) AS e
  FROM {tn("bronze_trips")}
),
dedup AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ride_id ORDER BY s) AS rn FROM typed
),
flags AS (
  SELECT rn,
         sid IS NULL OR sid = '' AS no_station,
         COALESCE(s BETWEEN TIMESTAMP'{config.START} 00:00:00'
                        AND TIMESTAMP'{config.END} 23:59:59', FALSE) AS in_period,
         COALESCE((unix_timestamp(e) - unix_timestamp(s))
                  BETWEEN {config.MIN_DURATION_S} AND {config.MAX_DURATION_S}, FALSE) AS good_dur
  FROM dedup
)
SELECT
  COUNT(*)                                                                    AS bronze_rows,
  SUM(CASE WHEN rn > 1 THEN 1 ELSE 0 END)                                     AS duplicates,
  SUM(CASE WHEN rn = 1 AND no_station THEN 1 ELSE 0 END)                      AS no_station,
  SUM(CASE WHEN rn = 1 AND NOT no_station AND NOT in_period THEN 1 ELSE 0 END) AS out_of_period,
  SUM(CASE WHEN rn = 1 AND NOT no_station AND in_period AND NOT good_dur
           THEN 1 ELSE 0 END)                                                 AS bad_duration,
  SUM(CASE WHEN rn = 1 AND NOT no_station AND in_period AND good_dur
           THEN 1 ELSE 0 END)                                                 AS silver_rows
FROM flags
"""


def dim_stations(tn: TableNamer) -> str:
    """One row per station: most frequent name, lifetime and GBFS coordinates/capacity."""
    return f"""
WITH names AS (
  SELECT start_station_id AS station_id, start_station_name AS station_name, COUNT(*) AS n,
         ROW_NUMBER() OVER (PARTITION BY start_station_id
                            ORDER BY COUNT(*) DESC, start_station_name) AS rk
  FROM {tn("silver_trips")}
  GROUP BY start_station_id, start_station_name
),
life AS (
  SELECT start_station_id AS station_id,
         MIN(start_date) AS first_trip_date, MAX(start_date) AS last_trip_date,
         COUNT(*) AS total_trips
  FROM {tn("silver_trips")}
  GROUP BY start_station_id
),
gbfs AS (
  SELECT TRIM(CAST(short_name AS STRING)) AS short_name,
         CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon, CAST(capacity AS INT) AS capacity,
         ROW_NUMBER() OVER (PARTITION BY TRIM(CAST(short_name AS STRING))
                            ORDER BY station_id) AS rk
  FROM {tn("gbfs_stations_raw")}
)
SELECT n.station_id, n.station_name, l.first_trip_date, l.last_trip_date, l.total_trips,
       g.lat, g.lon, g.capacity
FROM names n
JOIN life l ON l.station_id = n.station_id
LEFT JOIN gbfs g ON g.short_name = n.station_id AND g.rk = 1
WHERE n.rk = 1
"""


def dim_calendar(tn: TableNamer) -> str:
    """One row per day with day of week (1 = Sunday), weekend/holiday flags and weather."""
    return f"""
WITH days AS (
  SELECT explode(sequence(DATE'{config.START}', DATE'{config.END}', INTERVAL 1 DAY)) AS date
),
w AS (
  SELECT to_date(date) AS date,
         CAST(temperature_2m_mean AS DOUBLE) AS temp_mean,
         CAST(precipitation_sum AS DOUBLE)   AS precip_mm
  FROM {tn("weather_raw")}
),
h AS (
  SELECT DISTINCT to_date(date) AS date FROM {tn("holidays_raw")}
)
SELECT d.date,
       dayofweek(d.date)           AS dow,
       dayofweek(d.date) IN (1, 7) AS is_weekend,
       h.date IS NOT NULL          AS is_holiday,
       w.temp_mean, w.precip_mm
FROM days d
LEFT JOIN w ON w.date = d.date
LEFT JOIN h ON h.date = d.date
"""


def station_daily(tn: TableNamer) -> str:
    """Station x day grid (zero days filled) for stations alive the whole period."""
    return f"""
WITH eligible AS (
  SELECT station_id
  FROM {tn("dim_stations")}
  WHERE first_trip_date <= DATE'{config.ELIGIBLE_FIRST_TRIP_BY}'
    AND last_trip_date  >= DATE'{config.ELIGIBLE_LAST_TRIP_FROM}'
),
counts AS (
  SELECT start_station_id AS station_id, start_date AS date, COUNT(*) AS trips
  FROM {tn("silver_trips")}
  GROUP BY start_station_id, start_date
),
grid AS (
  SELECT e.station_id, c.date
  FROM eligible e CROSS JOIN {tn("dim_calendar")} c
)
SELECT g.station_id, g.date,
       CAST(COALESCE(ct.trips, 0) AS INT) AS trips,
       cal.dow, cal.is_weekend, cal.is_holiday, cal.temp_mean, cal.precip_mm
FROM grid g
LEFT JOIN counts ct ON ct.station_id = g.station_id AND ct.date = g.date
JOIN {tn("dim_calendar")} cal ON cal.date = g.date
"""


def gold_checks(tn: TableNamer) -> str:
    """Stations, days per station, total trips and share of zero days in station_daily."""
    return f"""
SELECT COUNT(DISTINCT station_id)                       AS stations,
       COUNT(*) / COUNT(DISTINCT station_id)            AS days_per_station,
       SUM(trips)                                       AS total_trips,
       AVG(CASE WHEN trips = 0 THEN 1.0 ELSE 0.0 END)   AS pct_zero_days
FROM {tn("station_daily")}
"""


def station_segments(tn: TableNamer) -> str:
    """ADI, CV² and segment per station on the training period only."""
    adi = "n_days / n_nonzero"
    cv2 = "POWER(sd_nonzero / mean_nonzero, 2)"
    return f"""
WITH train AS (
  SELECT * FROM {tn("station_daily")} WHERE date <= DATE'{config.TRAIN_END}'
),
stats AS (
  SELECT station_id,
         COUNT(*)                                   AS n_days,
         SUM(CASE WHEN trips > 0 THEN 1 ELSE 0 END) AS n_nonzero,
         AVG(trips)                                 AS mean_daily_trips,
         AVG(CASE WHEN trips > 0 THEN trips END)    AS mean_nonzero,
         STDDEV(CASE WHEN trips > 0 THEN trips END) AS sd_nonzero
  FROM train
  GROUP BY station_id
)
SELECT station_id, n_days, n_nonzero, mean_daily_trips,
       1 - n_nonzero / n_days AS pct_zero_days,
       {adi}                  AS adi,
       {cv2}                  AS cv2,
       CASE
         WHEN {adi} <  {config.ADI_CUT} AND {cv2} < {config.CV2_CUT} THEN 'smooth'
         WHEN {adi} <  {config.ADI_CUT}                               THEN 'erratic'
         WHEN {cv2} <  {config.CV2_CUT}                               THEN 'intermittent'
         ELSE 'lumpy'
       END AS segment
FROM stats
WHERE n_nonzero >= {config.MIN_NONZERO_DAYS}
"""
