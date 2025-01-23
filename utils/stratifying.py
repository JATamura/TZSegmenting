import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# This is just an example which will need turning into a proper function based on  the format of the input dataframe/data.
# It currently just looks at the proportion of viable seeds and the total number of seeds, but more features could be added
# It also only splits into 3 bins, and it's not totally clear how many bins to use but 3 is really the minimum.
# This also includes some plots, which may want to be saved or modified and hopefully show that the relevant distributions are preserved in train/test

if __name__ == '__main__':

    # Example dataset
    data = pd.DataFrame({
        'viable': np.random.randint(1, 35, size=100),
        'nonviable': np.random.randint(1, 105, size=100),
        'empty': np.random.randint(1, 70, size=100),
    })

    # Compute total seeds and proportions
    data['total_seeds'] = data[['viable', 'nonviable', 'empty']].sum(axis=1)
    data['proportion_viable'] = data['viable'] / data['total_seeds']
    # data['proportion_nonviable'] = data['nonviable'] / data['total_seeds']
    # data['proportion_empty'] = data['empty'] / data['total_seeds']

    # Bin the proportions
    # Use pd.qcut for equal-frequency bins (handles skewness)
    # If this raises an error need to add a small jitter to break ties
    data['viable_bin'] = pd.qcut(data['proportion_viable'], q=3, labels=False)  # 3 quantile bins
    # data['nonviable_bin'] = pd.qcut(data['proportion_nonviable'], q=3, labels=False)
    # data['empty_bin'] = pd.qcut(data['proportion_empty'], q=3, labels=False)
    data['total_bin'] = pd.qcut(data['total_seeds'], q=3, labels=False)

    # Combine the bins into a single stratification label
    data['strat_label'] = (
            data['viable_bin'].astype(str) + "_" +
            # data['nonviable_bin'].astype(str) + "_" +
            # data['empty_bin'].astype(str) + "_" +
            data['total_bin'].astype(str)
    )

    # Stratify using the combined label
    train_idx, test_idx = train_test_split(
        data.index, test_size=0.2, random_state=42, stratify=data['strat_label']
    )

    assert len(set(train_idx.values)) == len(train_idx.values)
    assert len(set(test_idx.values)) == len(test_idx.values)

    # Create train and test sets
    train_data = data.loc[train_idx]
    test_data = data.loc[test_idx]

    train_data['group'] = 'train'
    test_data['group'] = 'test'
    all_data = pd.concat([train_data, test_data])


    sns.displot(all_data, x="viable", hue="group", kind="kde")
    plt.show()
    plt.close()

    sns.displot(all_data, x="proportion_viable", hue="group", kind="kde")
    plt.show()
    plt.close()

    sns.displot(all_data, x="proportion_viable",y='total_seeds', hue="group", kind="kde")
    plt.show()
    plt.close()

    # Validate the stratification
    print("Train Class Distribution:\n", train_data[['proportion_viable', 'total_seeds']].mean(axis=0))
    print("Test Class Distribution:\n", test_data[['proportion_viable', 'total_seeds']].mean(axis=0))
    print("Train Seed Distribution:\n", train_data['total_seeds'].describe())
    print("Test Seed Distribution:\n", test_data['total_seeds'].describe())