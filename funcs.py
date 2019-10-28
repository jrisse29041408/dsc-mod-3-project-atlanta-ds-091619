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


def deg_free(data1, data2):
    n1 = len(data1)
    n2 = len(data2)

    s1 = data1.var(ddof=1)
    s2 = data2.var(ddof=1)

    df_num = (s1 + s2) ** 2
    df_denom = (s1) ** 2 / (n1 - 1) + (s2) ** 2 / (n2 - 1)

    return np.round(df_num / df_denom, decimals=3)


def visualize_t(t_stat, n_control, n_experimental):

    """
    Visualize the critical t values on a t distribution
    
    Parameters
    -----------
    t-stat: float
    n_control: int
    n_experiment: int
    
    Returns
    ----------
    None
    
    """
    # initialize a matplotlib "figure"
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()
    # generate points on the x axis between -4 and 4:
    xs = np.linspace(-4, 4, 500)

    # use stats.t.ppf to get critical value. For alpha = 0.05 and two tailed test
    crit = stats.t.ppf(1 - 0.025, (n_control + n_experimental - 2))

    # use stats.t.pdf to get values on the probability density function for the t-distribution

    ys = stats.t.pdf(xs, (n_control + n_experimental - 2), 0, 1)
    ax.plot(xs, ys, linewidth=3, color="darkred")

    ax.axvline(crit, color="black", linestyle="--", lw=5)
    ax.axvline(-crit, color="black", linestyle="--", lw=5)

    plt.show()
    return None


def ttest(a, n, b=None, one=True, independent=False):

    """
        Parameters
        -----------
        a : array
            experimental group 
        n : int
            sample size 
        b : array
            control group
        one : boolean default=True
            if False returns stats.ttest_1samp(a, n), if False may return ttest_rel(a, b) or ttest_ind(a, b)
        independent : boolean default=False
            if True returns stats.ttest_rel(a, b), if False return stats.ind(a, b)

        Returns
        --------
        Depending on parameters the function returns a ttest funciton that returns a t stat and a p-value (probability)
        
    """
    if one == True:
        return stats.ttest_1samp(a, n)
    elif one == False:
        if b == None:
            raise ValueError("ttest() calls for two arugments one was given.")
        if independent == True:
            return stats.ttest_ind(a, b)
        elif independent == False:
            return stats.ttest_rel(a, b)


def Cohen_d(group1, group2):
    """
    Calculates the effect size using the cohen's d test.
    
    Parameters
    ----------
    
    group1 : array
    
    group2 : array
    
    Returns
    -------
    
    Returns the effect size.
    """
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return abs(d)


def bootstrap(data, sample_size, n_samples=1):
    """
    Generates more sampled data utilizing 
    the bootstrap method.
    
    Parameters
    ----------
    data : list, array
    
    Returns
    -------
    a list of objects
        
    """
    from sklearn.utils import resample

    bootstrap = []
    for i in range(n_samples):
        boot = list(resample(data, n_samples=sample_size, replace=True))
        bootstrap.append(boot)

    return bootstrap
