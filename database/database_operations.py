from datetime import datetime
import psycopg2
import os
import pytz


def connect_to_database():
    """Connects to the database using Heroku config variables
    :return: database connection
    """
    db_connection = psycopg2.connect(os.environ['DATABASE_URL'])
    db_connection.autocommit = True
    return db_connection


def create_table() -> None:
    """Connects to the database and creates table
    if it doesn't exist
    """

    connection = connect_to_database()
    cur = connection.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS inferences (
        id serial PRIMARY KEY,
        inference_timestamp TIMESTAMP,
        inference_input json NOT NULL,
        prediction json NOT NULL
    );"""
    )

    print("Tables were created successfully")


def drop_table() -> None:
    """Drops existing table of inferences"""
    connection = connect_to_database()
    cur = connection.cursor()
    cur.execute(
        """
    DROP TABLE IF EXISTS
        inferences;"""
    )

    print("Tables were dropped successfully")


def get_inferences():
    """Selects and returns dataframe with
    10 most recent inferences
    :return: pandas Dataframe with recent inferences
    """
    connection = connect_to_database()
    curr = connection.cursor()
    curr.execute(
        """
    SELECT *
    FROM inferences
    ORDER BY inference_timestamp DESC 
    LIMIT 10;"""
    )
    rows = curr.fetchall()
    return rows


def insert_inference(json_input: dict, json_output: dict) -> None:
    """Inserts inference
    :param json_input: Input given to a model json
    :param json_output: Model output float
    """
    connection = connect_to_database()
    curr = connection.cursor()

    tz = pytz.timezone("Europe/Vilnius")
    dt = datetime.now(tz)

    curr.execute(
        f"INSERT INTO inferences(inference_timestamp, inference_input, prediction)"
        f"VALUES ('{dt}', '{json_input}', '{json_output}');"
    )
