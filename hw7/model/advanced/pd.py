def advancedStats(df):
    """Advanced stats should leverage pandas to calculate
    some relevant statistics on the data."""

    # print skew and kurtosis for every column
    for i in range(len(df.columns)):
        if df.columns[i] == "diagnosis":
            # skip
            continue

        print("Column {} statistics:".format(i))
        col = df[df.columns[i]]
        print("\tSkewness:{}\tKurtosis:{}".format(col.skew(), col.kurtosis()))

    # Print out the describe
    print("\nDataframe statistics: {}".format(df.describe()))
