import math
from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# TODO: Convolutional architecture: possibly with Trees similar to Conv DLGN.
# Check regularization if it is separate for each neuron in the FullPolyLayer or if it is shared across the layer.
# Better connection learning scheme: Can iteratively select new connections by dismissing the ones that have the bottom _ weights every N epochs. Should there be a kind of moving average recorded for such decision?
# Checkout the connection learning scheme of PolYLut paper
# Use the regularization formula of the PolYLut paper
# Pruning strategy? Checkout ConvLUT and PolYLut papers for pruning strategies
# Discretizing to Boolean functions using the Hadamard matrix of the Warp-LUT paper

def _normalize_percent(value: float) -> float:
    """Accept either 0..1 or 0..100 percentages and normalize to a fraction."""
    if value <= 0:
        return 0.0
    if value <= 1.0:
        return float(value)
    return float(value) / 100.0


class SparsePolyNeuron(nn.Module):
    """
    A neuron with sparse random connectivity and learnable monomial coefficients.

    Each neuron selects a random subset of previous-layer inputs of size
    k = round(input_percent * in_features). The selected inputs are weighted by
    learned input weights w_j, then all monomials over the weighted inputs are
    built. The neuron output is a sum over coefficient-weighted monomials
    followed by a sigmoid activation.

    Example for selected inputs [x0, x1, x2]:
        [1, x0, x1, x2, x0*x1, x0*x2, x1*x2, x0*x1*x2]
    and the neuron computes:
        y = sigmoid(sum_i c_i * monomial_i)
    """

    def __init__(
        self,
        in_features: int,
        input_percent: float,
        activation: Optional[str] = "sigmoid",
        max_selected_features: int = 8,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.input_percent = _normalize_percent(input_percent)
        self.max_selected_features = max(1, int(max_selected_features))
        self.num_selected = max(1, int(round(self.input_percent * self.in_features)))
        self.num_selected = min(self.num_selected, self.in_features, self.max_selected_features)
        self.activation_name = activation or "sigmoid"

        self.input_weights = nn.Parameter(torch.empty(self.in_features))
        self.monomial_coeffs = nn.Parameter(torch.empty(2 ** self.num_selected))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("selected_indices", torch.empty(self.num_selected, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.input_weights, mean=0.0, std=0.5)
        nn.init.normal_(self.monomial_coeffs, mean=0.0, std=0.8)
        self.bias.data.zero_()
        self.selected_indices.copy_(torch.randperm(self.in_features)[: self.num_selected])

    @staticmethod
    def _all_monomials(weighted_inputs: torch.Tensor) -> torch.Tensor:
        """
        Build all monomials over selected weighted inputs.
        If weighted_inputs has shape (B, k), the result has shape (B, 2^k)
        and entries are [1, x1, x2, x1*x2, x3, x1*x3, ...].
        """
        batch_size = weighted_inputs.shape[0]
        terms = torch.ones(batch_size, 1, device=weighted_inputs.device, dtype=weighted_inputs.dtype)
        for feature in weighted_inputs.t():
            terms = torch.cat([terms, terms * feature.unsqueeze(1)], dim=1)
        return terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x to be shape [batch, in_features], got {tuple(x.shape)}")

        selected_idx = self.selected_indices.to(x.device)
        selected_x = x[:, selected_idx]
        selected_w = self.input_weights[selected_idx]
        weighted_inputs = selected_x * selected_w.view(1, -1)

        monomial_terms = self._all_monomials(weighted_inputs)
        z = torch.sum(monomial_terms * self.monomial_coeffs.view(1, -1), dim=1) + self.bias

        if self.activation_name == "sigmoid":
            return torch.sigmoid(z)
        if self.activation_name == "relu":
            return torch.relu(z)
        if self.activation_name == "identity":
            return z
        if self.activation_name == "tanh":
            return torch.tanh(z)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def l1_regularization(self, lambda_value: float) -> torch.Tensor:
        return lambda_value * torch.abs(self.input_weights).sum()

    def top_dominant_inputs(self, k: int = 3) -> torch.Tensor:
        return torch.topk(self.input_weights.abs(), k=min(k, self.in_features)).indices


class FullPolyLayer(nn.Module):
    """A polynomial layer whose operations are performed in one batched matrix form."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_percent: float,
        activation: str = "sigmoid",
        max_selected_features: int = 8,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.input_percent = _normalize_percent(input_percent)
        self.max_selected_features = max(1, int(max_selected_features))
        self.num_selected = max(1, int(round(self.input_percent * self.in_features)))
        self.num_selected = min(self.num_selected, self.in_features, self.max_selected_features)
        self.activation_name = activation or "sigmoid"

        self.input_weights = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.monomial_coeffs = nn.Parameter(torch.empty(self.out_features, 2 ** self.num_selected))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.register_buffer("selected_indices", torch.empty(self.out_features, self.num_selected, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.input_weights, mean=0.0, std=0.5)
        nn.init.normal_(self.monomial_coeffs, mean=0.0, std=0.8)
        self.bias.data.zero_()
        for i in range(self.out_features):
            self.selected_indices[i].copy_(torch.randperm(self.in_features)[: self.num_selected])

    @staticmethod
    def _all_monomials_matrix(weighted_inputs: torch.Tensor) -> torch.Tensor:
        """Build monomials for all neurons in the layer together."""
        batch_size, num_neurons, k = weighted_inputs.shape
        terms = torch.ones(batch_size, num_neurons, 1, device=weighted_inputs.device, dtype=weighted_inputs.dtype)
        for idx in range(k):
            terms = torch.cat([terms, terms * weighted_inputs[:, :, idx : idx + 1]], dim=2)
        return terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x to be shape [batch, in_features], got {tuple(x.shape)}")

        selected_idx = self.selected_indices.to(x.device)
        selected_x = x.unsqueeze(1).expand(-1, self.out_features, -1).gather(
            2, selected_idx.unsqueeze(0).expand(x.size(0), -1, -1)
        )
        selected_w = self.input_weights.gather(1, selected_idx)
        weighted_inputs = selected_x * selected_w.unsqueeze(0)

        monomial_terms = self._all_monomials_matrix(weighted_inputs)
        z = torch.sum(monomial_terms * self.monomial_coeffs.view(1, self.out_features, -1), dim=2) + self.bias.view(1, -1)

        if self.activation_name == "sigmoid":
            return torch.sigmoid(z)
        if self.activation_name == "relu":
            return torch.relu(z)
        if self.activation_name == "identity":
            return z
        if self.activation_name == "tanh":
            return torch.tanh(z)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def l1_regularization(self, lambda_value: float) -> torch.Tensor:
        return lambda_value * torch.abs(self.input_weights).sum()

    def top_dominant_inputs(self, top_k: int = 3) -> torch.Tensor:
        return torch.topk(self.input_weights.abs(), k=min(top_k, self.in_features), dim=1)


class FullPolyConv2d(nn.Module):
    """
    A sparse convolutional kernel built as a tree of polynomial neurons.

    The receptive field is a square patch with shape (kh, kw) and the same shared
    connectivity pattern is applied across all sliding windows in the image.

    Each kernel has:
      - branch_count: number of branches at each node in the tree
      - depth: number of levels in the tree
      - num_leaves = branch_count ** depth
      - connectivity: [out_channels, num_leaves, num_inputs_per_leaf, 3]
        where entry [channel, x, y] defines the receptive-field input for a leaf.
      - input_weights: [out_channels, num_leaves, num_inputs_per_leaf]
      - monomial_coeffs: [out_channels, num_leaves, 2 ** num_inputs_per_leaf]
      - tree_weights: [out_channels, num_leaves] to combine leaf outputs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        input_percent: float = 0.25,
        branch_count: int = 4,
        depth: int = 2,
        group_size: int = 1,
        group_mode: str = "ordered",
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size), int(kernel_size)) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.input_percent = _normalize_percent(input_percent)
        self.branch_count = int(branch_count)
        self.depth = int(depth)
        self.group_size = max(1, int(group_size))
        self.group_mode = group_mode
        self.activation_name = activation or "sigmoid"

        if self.branch_count <= 1:
            raise ValueError("branch_count must be >= 2")
        if self.depth <= 0:
            raise ValueError("depth must be >= 1")

        self.kernel_h, self.kernel_w = self.kernel_size
        self.num_leaves = self.branch_count ** self.depth
        self.num_window_positions = self.kernel_h * self.kernel_w
        self.num_inputs_per_leaf = max(1, int(round(self.input_percent * self.in_channels * self.num_window_positions)))
        self.num_inputs_per_leaf = min(self.num_inputs_per_leaf, self.in_channels * self.num_window_positions)

        self.num_internal_nodes = 0
        remaining = self.num_leaves
        while remaining > 1:
            self.num_internal_nodes += remaining // self.branch_count
            remaining = remaining // self.branch_count

        self.input_weights = nn.Parameter(torch.empty(self.out_channels, self.num_leaves, self.num_inputs_per_leaf))
        self.monomial_coeffs = nn.Parameter(torch.empty(self.out_channels, self.num_leaves, 2 ** self.num_inputs_per_leaf))
        self.tree_weights = nn.Parameter(torch.empty(self.out_channels, self.num_internal_nodes, 2 ** self.branch_count))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

        self.register_buffer(
            "connectivity",
            torch.empty(self.out_channels, self.num_leaves, self.num_inputs_per_leaf, 3, dtype=torch.long),
        )
        self.register_buffer(
            "channel_group_assignments",
            torch.empty(self.out_channels, dtype=torch.long),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.input_weights, mean=0.0, std=0.5)
        nn.init.normal_(self.monomial_coeffs, mean=0.0, std=0.8)
        nn.init.normal_(self.tree_weights, mean=0.0, std=0.1)
        self.bias.data.zero_()

        num_groups = max(1, int(math.ceil(self.in_channels / self.group_size)))
        if self.group_mode == "random":
            group_ids = torch.randint(0, num_groups, (self.out_channels,), dtype=torch.long)
            self.channel_group_assignments.copy_(group_ids)
        else:
            self.channel_group_assignments.copy_(torch.arange(self.out_channels, dtype=torch.long) % num_groups)

        for k in range(self.out_channels):
            group_id = int(self.channel_group_assignments[k].item())
            group_start = group_id * self.group_size
            group_end = min(self.in_channels, group_start + self.group_size)
            allowed_channel_ids = list(range(group_start, group_end))
            if len(allowed_channel_ids) == 0:
                allowed_channel_ids = list(range(self.in_channels))

            for leaf in range(self.num_leaves):
                for inp in range(self.num_inputs_per_leaf):
                    channel = allowed_channel_ids[(leaf * inp + k) % len(allowed_channel_ids)]
                    x = (leaf + k + inp) % self.kernel_w
                    y = (leaf + k + inp + channel) % self.kernel_h
                    self.connectivity[k, leaf, inp, 0] = channel
                    self.connectivity[k, leaf, inp, 1] = x
                    self.connectivity[k, leaf, inp, 2] = y

    @staticmethod
    def _all_monomials(weighted_inputs: torch.Tensor) -> torch.Tensor:
        """Shape [N, K, L, 2^I]."""
        terms = torch.ones(*weighted_inputs.shape[:-1], 1, device=weighted_inputs.device, dtype=weighted_inputs.dtype)
        for idx in range(weighted_inputs.shape[-1]):
            terms = torch.cat([terms, terms * weighted_inputs[..., idx : idx + 1]], dim=-1)
        return terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be shape [batch, channels, H, W], got {tuple(x.shape)}")

        batch_size, in_channels, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {in_channels}")

        padded = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        out_h = (height + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_w) // self.stride + 1

        windows = padded.unfold(2, self.kernel_h, self.stride).unfold(3, self.kernel_w, self.stride)
        windows = windows.permute(0, 2, 3, 1, 4, 5)
        positions = out_h * out_w
        flat_windows = windows.reshape(batch_size, positions, in_channels * self.kernel_h * self.kernel_w)

        connectivity = self.connectivity.to(x.device)
        flat_idx = (
            connectivity[..., 0].to(x.device) * (self.kernel_h * self.kernel_w)
            + connectivity[..., 1].to(x.device) * self.kernel_w
            + connectivity[..., 2].to(x.device)
        ).long()

        sampled_parts = []
        for out_channel in range(self.out_channels):
            channel_gather = []
            for leaf in range(self.num_leaves):
                idx = flat_idx[out_channel, leaf]
                values = flat_windows[:, :, idx].to(x.device)
                channel_gather.append(values)
            sampled_parts.append(torch.stack(channel_gather, dim=2))

        sampled = torch.stack(sampled_parts, dim=0).permute(1, 0, 2, 3, 4)
        # [batch, out_channels, positions, num_leaves, num_inputs_per_leaf]
        selected_weights = self.input_weights.to(x.device).view(1, self.out_channels, 1, self.num_leaves, self.num_inputs_per_leaf)
        weighted_inputs = sampled * selected_weights
        monomial_terms = self._all_monomials(weighted_inputs)

        leaf_scores = torch.sum(
            monomial_terms * self.monomial_coeffs.view(1, self.out_channels, 1, self.num_leaves, -1),
            dim=-1,
        ) + self.bias.view(1, self.out_channels, 1, 1)
        leaf_outputs = self._apply_activation(leaf_scores)

        tree_values = leaf_outputs
        coeff_offset = 0
        while tree_values.shape[-1] > 1:
            num_groups = tree_values.shape[-1] // self.branch_count
            coeff_block = self.tree_weights[:, coeff_offset: coeff_offset + num_groups, :]
            grouped = tree_values.reshape(batch_size, self.out_channels, out_h * out_w, num_groups, self.branch_count)
            grouped_monomials = self._all_monomials(grouped)
            coeff = coeff_block.unsqueeze(0).unsqueeze(2)
            tree_values = torch.sum(grouped_monomials * coeff, dim=-1)
            coeff_offset += num_groups

        out = self._apply_activation(tree_values.squeeze(-1)).reshape(batch_size, self.out_channels, out_h, out_w)
        return out

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "relu":
            return torch.relu(x)
        if self.activation_name == "identity":
            return x
        if self.activation_name == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def l1_regularization(self, lambda_value: float) -> torch.Tensor:
        return lambda_value * (torch.abs(self.input_weights).sum() + torch.abs(self.tree_weights).sum())


class SparsePolyLayer(nn.Module):
    """Stack of sparse polynomial neurons."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_percent: float,
        activation: str = "sigmoid",
        max_selected_features: int = 8,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.input_percent = _normalize_percent(input_percent)
        self.neurons = nn.ModuleList(
            [
                SparsePolyNeuron(
                    self.in_features,
                    self.input_percent,
                    activation=activation,
                    max_selected_features=max_selected_features,
                )
                for _ in range(self.out_features)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([neuron(x) for neuron in self.neurons], dim=1)

    def regularization_loss(self, lambda_value: float) -> torch.Tensor:
        return sum(neuron.l1_regularization(lambda_value) for neuron in self.neurons)


class SparsePolynomialNetwork(nn.Module):
    """A simple MLP whose hidden units are sparse polynomial neurons."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int = 1,
        input_percent: float = 0.3,
        hidden_activation: str = "sigmoid",
        output_activation: Optional[str] = "sigmoid",
        l1_lambda: float = 1e-4,
        max_selected_features: int = 10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.output_dim = int(output_dim)
        self.input_percent = _normalize_percent(input_percent)
        self.l1_lambda = float(l1_lambda)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.max_selected_features = max(1, int(max_selected_features))

        self.layers = nn.ModuleList()
        prev = self.input_dim
        for hidden_size in self.hidden_layers:
            self.layers.append(
                SparsePolyLayer(
                    prev,
                    hidden_size,
                    self.input_percent,
                    activation=self.hidden_activation,
                    max_selected_features=self.max_selected_features,
                )
            )
            prev = hidden_size

        self.output_head = nn.Linear(prev, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        logits = self.output_head(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.output_activation == "identity":
            return logits
        if self.output_activation == "tanh":
            return torch.tanh(logits)
        raise ValueError(f"Unsupported output activation: {self.output_activation}")

    def regularization_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.regularization_loss(self.l1_lambda)
        return total

    def prune_top_k(self, top_k: int = 3) -> None:
        for layer in self.layers:
            for neuron in layer.neurons:
                keep = neuron.top_dominant_inputs(k=top_k)
                with torch.no_grad():
                    mask = torch.zeros_like(neuron.input_weights, dtype=torch.bool)
                    mask[keep] = True
                    neuron.input_weights.data[~mask] = 0.0


class FullPolyNetwork(nn.Module):
    """A network built from FullPolyLayer blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int = 1,
        input_percent: float = 0.3,
        hidden_activation: str = "sigmoid",
        output_activation: Optional[str] = "sigmoid",
        l1_lambda: float = 1e-4,
        max_selected_features: int = 8,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.output_dim = int(output_dim)
        self.input_percent = _normalize_percent(input_percent)
        self.l1_lambda = float(l1_lambda)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.max_selected_features = max(1, int(max_selected_features))

        self.layers = nn.ModuleList()
        prev = self.input_dim
        for hidden_size in self.hidden_layers:
            self.layers.append(
                FullPolyLayer(
                    prev,
                    hidden_size,
                    self.input_percent,
                    activation=self.hidden_activation,
                    max_selected_features=self.max_selected_features,
                )
            )
            prev = hidden_size

        self.output_head = nn.Linear(prev, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        logits = self.output_head(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.output_activation == "identity":
            return logits
        if self.output_activation == "tanh":
            return torch.tanh(logits)
        raise ValueError(f"Unsupported output activation: {self.output_activation}")

    def regularization_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.l1_regularization(self.l1_lambda)
        return total


class ExampleTrainer:
    """Train the sparse polynomial network on MNIST."""

    @staticmethod
    def make_loaders(batch_size: int = 64, train_limit: Optional[int] = None, test_limit: Optional[int] = None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        if train_limit is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_limit)))
        if test_limit is not None:
            test_dataset = torch.utils.data.Subset(test_dataset, list(range(test_limit)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader

    @staticmethod
    def run_demo() -> None:
        train_loader, test_loader = ExampleTrainer.make_loaders(batch_size=128, train_limit=20000, test_limit=2000)

        model = FullPolyNetwork(
            input_dim=28 * 28,
            hidden_layers=[128, 128, 128],
            output_dim=10,
            input_percent=0.2,
            hidden_activation="sigmoid",
            output_activation="softmax",
            l1_lambda=1e-5,
            #max_selected_features=6,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.NLLLoss()

        for epoch in range(30):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.view(xb.size(0), -1)
                optimizer.zero_grad()
                probs = model(xb)
                loss = criterion(torch.log(probs.clamp_min(1e-8)), yb) + model.regularization_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.view(xb.size(0), -1)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            print(f"epoch={epoch:02d}  avg_loss={train_loss / max(1, len(train_loader.dataset)):.4f}  test_acc={correct / total}")

        # model.prune_top_k(top_k=6)
        # print("\nDominant weights after pruning:")
        # for layer_idx, layer in enumerate(model.layers):
        #     print(f"Layer {layer_idx}:")
        #     for neuron_idx, neuron in enumerate(layer.neurons):
        #         top = neuron.top_dominant_inputs(k=6).tolist()
        #         print(f"  neuron {neuron_idx}: dominant={top}")


if __name__ == "__main__":
    ExampleTrainer.run_demo()
