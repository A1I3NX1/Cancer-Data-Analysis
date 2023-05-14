import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")


def scatterMatrix(df, count=5):
    """Use seaborn to produce a pairplot of columns

    count: number of columns to scatter (larger will result in slower)
    """
    # convert to dataframe, limit number of columns shown for time reasons
    df = df[df.columns[: count + 1]]

    # pairplot
    sns.pairplot(df, hue="diagnosis")

    # show plot
    plt.show()


def correlationHeatmap(df):
    """Use seaborn to produce a heatmap of the columns' correlation"""
    # heatmap of correlations
    sns.heatmap(df.corr())

    # show plot
    plt.show()
