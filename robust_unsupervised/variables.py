from .prelude import *


class Variable(nn.Module):
    def __init__(self, G: networks.Generator, data: torch.Tensor):
        super().__init__()
        self.G = G
        self.data = data

    # ------------------------------------
    
    @staticmethod
    def sample_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    @staticmethod
    def sample_random_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    def to_input_tensor(self):
        raise NotImplementedError

    # ------------------------------------

    def parameters(self):
        return [self.data]

    def to_image(self):
        return self.render_image(self.to_input_tensor())

    def render_image(self, ws: torch.Tensor): # todo 
        """
        ws shape: [batch_size, num_layers, 512]
        """
        return (self.G.synthesis(ws, noise_mode="const", force_fp32=True) + 1.0) / 2.0

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def clone(self):
        data = self.data.detach().clone().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def interpolate(self, other: "Variable", alpha: float = 0.5):
        assert self.G == other.G
        return self.__class__(self.G, self.data.lerp(other.data, alpha))

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def unbind(self):
        """
        Splits this (batched) variable into a a list of variables with batch size 1. 
        """
        return [
            self.__class__(
                self.G,
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]
    

class WVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping.w_avg.reshape(1, G.w_dim).repeat(batch_size, 1)

        return WVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping(
            torch.randn(batch_size, G.z_dim).cuda(),
            None,
            skip_w_avg_update=True,
        )[:, 0]

        return WVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data.unsqueeze(1).repeat(1, self.G.num_ws, 1)

    @torch.no_grad()
    def truncate(self, truncation: float=1.0):
        assert 0.0 <= truncation <= 1.0
        self.data.lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
        return self


class WpVariable(Variable):
    def __init__(self, G, data: torch.Tensor):
        super().__init__(G, data)

    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.to_input_tensor(WVariable.sample_from(G, batch_size))

        return WpVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            G.mapping(
                (torch.randn(batch_size * G.mapping.num_ws, G.z_dim).cuda()),
                None,
                skip_w_avg_update=True,
            )
            .mean(dim=1)
            .reshape(batch_size, G.mapping.num_ws, G.w_dim)
        )

        return WpVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data

    def mix(self, other: "WpVariable", num_layers: float):
        return WpVariable(
            self.G,
            torch.cat(
                (self.data[:, :num_layers, :], other.data[:, num_layers:, :]), dim=1
            ),
        )

    @staticmethod
    def from_W(W: WVariable):
        return WpVariable(
            W.G, nn.parameter.Parameter(W.to_input_tensor())
        )

    @torch.no_grad()
    def truncate(self, truncation=1.0, *, layer_start = 0, layer_end: Optional[int] = None):
        assert 0.0 <= truncation <= 1.0
        mu = self.G.mapping.w_avg
        target = mu.reshape(1, 1, 512).repeat(1, self.G.mapping.num_ws, 1)
        self.data[:, layer_start:layer_end].lerp_(target[:, layer_start:layer_end], 1.0 - truncation)
        return self


class WppVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.sample_from(G, batch_size).to_input_tensor().repeat(1, 512, 1)

        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            WVariable.sample_random_from(G, batch_size)
            .to_input_tensor()
            .repeat(1, 512, 1)
        )

        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def from_w(W: WVariable):
        data = W.data.detach().repeat(1, 512 * W.G.num_ws, 1)

        return WppVariable(W.G, nn.parameter.Parameter(data))

    @staticmethod
    def from_Wp(Wp: WpVariable):
        # Detach to start Phase 3 with a clean slate
        data = Wp.data.detach().clone()
        # Ensure it is a Parameter so NGD can optimize it
        return WppVariable(Wp.G, nn.Parameter(data))

    def to_input_tensor(self):
        # Reshape the flattened W++ parameters back to the StyleGAN synthesis format
        # [batch, 512 * num_ws, 512] -> [batch, num_ws, 512] via averaging or slicing
        # For LFMA, we typically treat the data as a delta over the Wp base.
        if self.data.shape[1] == self.G.num_ws:
            return self.data
        return self.data.reshape(-1, self.G.num_ws, 512)
        
    @torch.no_grad()
    def project(self, basis: torch.Tensor):
        """
        Projects the current W++ variable onto a provided orthogonal basis.
        basis: [rank, 512] matrix generated from Phase 2.
        """
        # Flattened projection: W_proj = B * (B^T * W)
        # This keeps the latent vector within the high-variance subspace found in Phase 2
        batch, num_ws, dim = self.data.shape
        flat_data = self.data.reshape(-1, dim)
        
        # Linear algebra projection
        projected = flat_data @ basis.T @ basis
        self.data.copy_(projected.reshape(batch, num_ws, dim))
        return self

