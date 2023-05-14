import pandas as pd


def readCSV(filename):
    """
    Read the file called "filename", expected to be a csv file.

    Return the data as a pandas dataframe, removing the patient ID column
    """
    # return a list of lists
    df = pd.read_csv(filename)
    df.drop("id", axis=1, inplace=True)
    return df
