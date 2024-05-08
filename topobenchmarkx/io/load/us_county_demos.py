import numpy as np
import pandas as pd
import torch
import torch_geometric


def load_us_county_demos(path, year=2012, y_col="Election"):
    r"""Load US County Demos dataset
    
    Parameters
    ----------
    path: str
        Path to the dataset.
    year: int
        Year to load the features.
    y_col: str
        Column to use as label.
    
    Returns
    -------
    torch_geometric.data.Data
        Data object of the graph for the US County Demos dataset.
    """
    
    edges_df = pd.read_csv(f"{path}/county_graph.csv")
    stat = pd.read_csv(f"{path}/county_stats_{year}.csv", encoding="ISO-8859-1")

    keep_cols = [
        "FIPS",
        "DEM",
        "GOP",
        "MedianIncome",
        "MigraRate",
        "BirthRate",
        "DeathRate",
        "BachelorRate",
        "UnemploymentRate",
    ]
    
    # Select columns, replace ',' with '.' and convert to numeric
    stat = stat.loc[:, keep_cols]
    stat["MedianIncome"] = stat["MedianIncome"].replace(',','.', regex=True)
    stat = stat.apply(pd.to_numeric, errors='coerce')
    
    # Step 2: Substitute NaN values with column mean
    for column in stat.columns:
        if column != "FIPS":
            mean_value = stat[column].mean()
            stat[column].fillna(mean_value, inplace=True)
    stat = stat[keep_cols].dropna()

    # Delete edges that are not present in stat df
    unique_fips = stat["FIPS"].unique()

    src_ = edges_df["SRC"].apply(lambda x: x in unique_fips)
    dst_ = edges_df["DST"].apply(lambda x: x in unique_fips)

    edges_df = edges_df[src_ & dst_]

    # Remove rows from stat df where edges_df['SRC'] or edges_df['DST'] are not present
    stat = stat[stat["FIPS"].isin(edges_df["SRC"]) & stat["FIPS"].isin(edges_df["DST"])]
    stat = stat.reset_index(drop=True)

    # Remove rows where SRC == DST
    edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Make edge_index undirected
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Convert edge_index back to pandas DataFrame
    edges_df = pd.DataFrame(edge_index.numpy().T, columns=["SRC", "DST"])

    del edge_index

    # Map stat['FIPS'].unique() to [0, ..., num_nodes]
    fips_map = {fips: i for i, fips in enumerate(stat["FIPS"].unique())}
    stat["FIPS"] = stat["FIPS"].map(fips_map)

    # Map edges_df['SRC'] and edges_df['DST'] to [0, ..., num_nodes]
    edges_df["SRC"] = edges_df["SRC"].map(fips_map)
    edges_df["DST"] = edges_df["DST"].map(fips_map)

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Remove isolated nodes (Note: this function maps the nodes to [0, ..., num_nodes] automatically)
    edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(edge_index)

    # Conver mask to index
    index = np.arange(mask.size(0))[mask]
    stat = stat.iloc[index]
    stat = stat.reset_index(drop=True)

    # Get new values for FIPS from current index
    # To understand why please print stat.iloc[[516, 517, 518, 519, 520]] for 2012 year
    # Basically the FIPS values has been shifted
    stat["FIPS"] = stat.reset_index()["index"]

    # Create Election variable
    stat["Election"] = (stat["DEM"] - stat["GOP"]) / (stat["DEM"] + stat["GOP"])

    # Drop DEM and GOP columns and FIPS
    stat = stat.drop(columns=["DEM", "GOP", "FIPS"])

    # Prediction col

    x_col = list(set(stat.columns).difference(set([y_col])))

    # stat["MedianIncome"] = (
    #     stat["MedianIncome"]
    #     .apply(lambda x: x.replace(",", ""))
    #     .to_numpy()
    #     .astype(float)
    # )

    x = torch.tensor(stat[x_col].to_numpy(), dtype=torch.float32)
    y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

    return data
