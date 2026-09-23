import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, 
                   sep:      str  = ',', 
                   label:    bool = False) -> Dataset:
    
    """
    Reads a data file (no header) and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        Value separator
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
    """

    data = np.genfromtxt(filename, 
                         delimiter = sep,
                         ndmin     = 2)

    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(X, y)


def write_data_file(filename: str, 
                    dataset:  Dataset, 
                    sep:      str  = ',', 
                    label:    bool = False) -> None:
    
    """
    Writes a Dataset object to a data file.

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        Dataset object to write to the file
    sep: str
        Value separator
    label: bool
        Whether to write y
    """

    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, 
               data, 
               delimiter = sep)