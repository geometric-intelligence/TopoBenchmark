import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_scatter


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ["bn", "ln", "None"]
        if Normalization == "bn":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == "ln":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == "Identity"):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels  # first normalization
        flops += (
            num_samples * self.in_channels * self.hidden_channels
        )  # first linear layer
        flops += num_samples * self.hidden_channels  # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels  # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += (
            num_samples * self.out_channels * self.hidden_channels
        )  # last linear layer

        return flops


class PlainMLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super().__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class EquivSetConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mlp1_layers=1,
        mlp2_layers=1,
        mlp3_layers=1,
        aggr="add",
        alpha=0.5,
        dropout=0.0,
        normalization="None",
        input_norm=False,
    ):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(
                in_features,
                out_features,
                out_features,
                mlp1_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(
                in_features + out_features,
                out_features,
                out_features,
                mlp2_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(
                out_features,
                out_features,
                out_features,
                mlp3_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(
            Xve, edges, dim=-2, reduce=self.aggr
        )  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(
            Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N
        )  # [N, C]

        X = Xv

        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X


class JumpLinkConv(nn.Module):
    def __init__(self, in_features, out_features, mlp_layers=2, aggr="add", alpha=0.5):
        super().__init__()
        self.W = MLP(
            in_features,
            out_features,
            out_features,
            mlp_layers,
            dropout=0.0,
            Normalization="None",
            InputNorm=False,
        )

        self.aggr = aggr
        self.alpha = alpha

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0, beta=1.0):
        N = X.shape[-2]

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(
            Xve, edges, dim=-2, reduce=self.aggr
        )  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xv = torch_scatter.scatter(
            Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N
        )  # [N, C]

        X = Xv

        Xi = (1 - self.alpha) * X + self.alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X


class MeanDegConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_features=None,
        mlp1_layers=1,
        mlp2_layers=1,
        mlp3_layers=2,
    ):
        super().__init__()
        if init_features is None:
            init_features = out_features
        self.W1 = MLP(
            in_features,
            out_features,
            out_features,
            mlp1_layers,
            dropout=0.0,
            Normalization="None",
            InputNorm=False,
        )
        self.W2 = MLP(
            in_features + out_features + 1,
            out_features,
            out_features,
            mlp2_layers,
            dropout=0.0,
            Normalization="None",
            InputNorm=False,
        )
        self.W3 = MLP(
            in_features + out_features + init_features + 1,
            out_features,
            out_features,
            mlp3_layers,
            dropout=0.0,
            Normalization="None",
            InputNorm=False,
        )

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X[..., vertex, :])  # [nnz, C]
        Xe = torch_scatter.scatter(
            Xve, edges, dim=-2, reduce="mean"
        )  # [E, C], reduce is 'mean' here as default

        deg_e = torch_scatter.scatter(
            torch.ones(Xve.shape[0], device=Xve.device), edges, dim=-2, reduce="sum"
        )
        Xe = torch.cat([Xe, torch.log(deg_e)[..., None]], -1)

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(
            Xev, vertex, dim=-2, reduce="mean", dim_size=N
        )  # [N, C]

        deg_v = torch_scatter.scatter(
            torch.ones(Xev.shape[0], device=Xev.device), vertex, dim=-2, reduce="sum"
        )
        X = self.W3(torch.cat([Xv, X, X0, torch.log(deg_v)[..., None]], -1))

        return X


class EDGNN(nn.Module):
    def __init__(
        self,
        num_features,
        input_dropout=0.2,
        dropout=0.2,
        activation="relu",
        MLP_num_layers=2,
        MLP2_num_layers=-1,
        MLP3_num_layers=-1,
        All_num_layers=2,
        edconv_type="EquivSet",
        restart_alpha=0.5,
        aggregate="add",
        normalization="None",
        AllSet_input_norm=False,
    ):
        """EDGNN

        Args:
            num_features (int): number of input features
            input_dropout (float, optional): dropout rate for input features. Defaults to 0.2.
            dropout (float, optional): dropout rate for hidden layers. Defaults to 0.2.
            activation (str, optional): activation function. Defaults to 'relu'.
            MLP_num_layers (int, optional): number of layers in MLP. Defaults to 2.
            MLP2_num_layers (int, optional): number of layers in the second MLP. Defaults to -1.
            MLP3_num_layers (int, optional): number of layers in the third MLP. Defaults to -1.
            All_num_layers (int, optional): number of layers in the EDConv. Defaults to 2.
            edconv_type (str, optional): type of EDConv. Defaults to 'EquivSet'.
            restart_alpha (float, optional): restart alpha. Defaults to 0.5.
            aggregate (str, optional): aggregation method. Defaults to 'add'.
            normalization (str, optional): normalization method. Defaults to 'None'.
            AllSet_input_norm (bool, optional): whether to normalize input features. Defaults to False.

        """
        super().__init__()
        act = {"Id": nn.Identity(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = act[activation]
        self.input_drop = nn.Dropout(input_dropout)  # 0.6 is chosen as default
        self.dropout = nn.Dropout(dropout)  # 0.2 is chosen for GCNII

        self.in_channels = num_features
        self.hidden_channels = self.in_channels

        self.mlp1_layers = MLP_num_layers
        self.mlp2_layers = MLP_num_layers if MLP2_num_layers < 0 else MLP2_num_layers
        self.mlp3_layers = MLP_num_layers if MLP3_num_layers < 0 else MLP3_num_layers
        self.nlayer = All_num_layers
        self.edconv_type = edconv_type

        if edconv_type == "EquivSet":
            self.conv = EquivSetConv(
                self.in_channels,
                self.in_channels,
                mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers,
                alpha=restart_alpha,
                aggr=aggregate,
                dropout=dropout,
                normalization=normalization,
                input_norm=AllSet_input_norm,
            )
        elif edconv_type == "JumpLink":
            self.conv = JumpLinkConv(
                self.in_channels,
                self.in_channels,
                mlp_layers=self.mlp1_layers,
                alpha=restart_alpha,
                aggr=aggregate,
            )
        elif edconv_type == "MeanDeg":
            self.conv = MeanDegConv(
                self.in_channels,
                self.in_channels,
                init_features=self.in_channels,
                mlp1_layers=self.mlp1_layers,
                mlp2_layers=self.mlp2_layers,
                mlp3_layers=self.mlp3_layers,
            )
        else:
            raise ValueError(f"Unsupported EDConv type: {edconv_type}")

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        if edge_index.layout == torch.sparse_coo:
            edge_index, _ = torch_geometric.utils.to_edge_index(edge_index)
        V, E = edge_index[0], edge_index[1]
        x0 = x
        for _ in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        return x, None
