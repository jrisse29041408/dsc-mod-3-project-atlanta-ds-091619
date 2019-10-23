import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sql_to_df(db, query):

    """
    Parameters
    -----------
    db: string
        sqlite database file path
    query: string
        sql query

    Returns
    -----------
    df: DataFrame
        dataframe created using the sql query
    """

    conn = sqlite3.connect(db)

    df = pd.read_sql_query(query, conn)

    conn.close()

    df.head()

    return df


# sql_to_df("Northwind_small.sqlite", "select * from Order;")

