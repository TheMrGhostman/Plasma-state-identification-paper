import torch
import torch.nn as nn

class ResNetBlock1d_stride(nn.Module):
	def __init__(self, in_channels, out_channels=64, kernel_size=5, layers=2, stride = 2, activation=nn.ReLU):
		super(ResNetBlock1d_stride, self).__init__()
		self.activation = activation()
		self.skip = nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1,
			stride=stride
		) if in_channels!=out_channels else lambda x: x
		block = [
			nn.Conv1d(
				in_channels=in_channels, 
				out_channels=out_channels,
				kernel_size=kernel_size,
				stride=stride, 
				padding=int(kernel_size//2)
			),
			nn.BatchNorm1d(num_features=out_channels)
			]
		for i in range(layers-1):
			block = [
				*block,
				nn.Conv1d(
					in_channels=out_channels, 
					out_channels=out_channels,
					kernel_size=kernel_size,
					stride=1, 
					padding=int(kernel_size//2)
				),
				nn.BatchNorm1d(num_features=out_channels),
			]
			if i < layers-1:
				block.append(activation())
			in_channels=out_channels

		self.block = nn.Sequential(*block)

	def forward(self, x):
		x = self.activation(self.block(x) + self.skip(x))
		return x
	
class GRU_CLF(nn.Module):
    def __init__(self, in_channels, out_channels=64, gru_layers=2, dropout=0, hidden_dim=32, nclasses=4, activation=nn.ReLU()):
        super(GRU_CLF, self).__init__()
        self.grus = nn.GRU(in_channels, out_channels, num_layers=gru_layers, dropout=dropout)
        self.dense = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            activation,
            nn.Linear(hidden_dim, nclasses)
        )

    def forward(self, x):
        h = self.grus(x.permute(2,0,1))
        h = h[0].permute(1,2,0)[:,:,-1]
        y = self.dense(h)
        return y