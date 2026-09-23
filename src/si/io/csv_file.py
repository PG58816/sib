import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, 
             sep:      str  = ',', 
             features: bool = False, 
             label:    bool = False) -> Dataset:
    
    """
    Reads a csv file and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        Value separator
    features: bool
        Whether the file has feature names (header)
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
    """

    df = pd.read_csv(filename, 
                     sep    = sep, 
                     header = 0 if features else None)

    if features and label:
        feature_names = df.columns[:-1].tolist()
        label_name    = df.columns[-1]
        X             = df.iloc[:, :-1].to_numpy()
        y             = df.iloc[:, -1].to_numpy()

    elif features and not label:
        feature_names = df.columns.tolist()
        label_name    = None
        X             = df.to_numpy()
        y             = None

    elif not features and label:
        feature_names = None
        label_name    = None
        X             = df.iloc[:, :-1].to_numpy()
        y             = df.iloc[:, -1].to_numpy()

    else:
        feature_names = None
        label_name    = None
        X             = df.to_numpy()
        y             = None

    return Dataset(X, 
                   y, 
                   features = feature_names, 
                   label    = label_name)


def write_csv(filename: str, 
              dataset:  Dataset, 
              sep:      str  = ',', 
              features: bool = False, 
              label:    bool = False) -> None:
    
    """
    Writes a Dataset object to a csv file.

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        Dataset object to write to the file
    sep: str
        Value separator
    features: bool
        Whether to write feature names (header)
    label: bool
        Whether to write y
    """

    df = pd.DataFrame(dataset.X, 
                      columns = dataset.features if features else None)

    if label and dataset.y is not None:
        df[dataset.label if features else df.shape[1]] = dataset.y

    df.to_csv(filename, 
              sep    = sep, 
              header = features, 
              index  = False)