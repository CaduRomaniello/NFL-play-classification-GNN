import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, node_features, config):
        super(GCN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(config.RANDOM_SEED)
        
        # Store config parameters
        self.hidden_channels = config.GCN.HIDDEN_CHANNELS
        self.hidden_layers = config.GCN.HIDDEN_LAYERS
        self.dropout = config.GCN.DROPOUT
        
        # Validate hidden_layers
        if self.hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        
        # Create layers dynamically
        self.convs = torch.nn.ModuleList()
        
        # First layer: input -> hidden
        self.convs.append(GCNConv(node_features, self.hidden_channels))
        
        # Hidden layers: hidden -> hidden
        for _ in range(self.hidden_layers - 1):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        
        # Output layer
        self.classifier = torch.nn.Linear(self.hidden_channels, 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better training stability"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch):
        # Apply all conv layers with ReLU activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            
            # Optional: Add dropout between layers (not just at the end)
            if i < len(self.convs) - 1:  # Don't apply to last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (graph-level representation)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # Final dropout and classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'hidden_channels': self.hidden_channels,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'total_parameters': self.get_num_parameters()
        }