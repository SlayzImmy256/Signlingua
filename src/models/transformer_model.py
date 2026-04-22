"""Transformer-based model for sign language recognition"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MediaPipeTransformer(nn.Module):
    """Transformer model for MediaPipe landmarks"""
    
    def __init__(self,
                 input_dim: int = 258,  # 33*4 + 21*3 + 21*3 (pose + hands, no face)
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.3,
                 num_classes: int = 100,
                 max_seq_length: int = 64):
        """
        Args:
            input_dim: Dimension of input landmarks
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_classes: Number of output classes
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional padding mask of shape (batch_size, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for padding
        if mask is not None:
            # Convert padding mask to attention mask
            # True values are masked (padding), False values are not masked
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_mask = attn_mask.expand(-1, -1, x.size(1), -1)  # (batch, 1, seq_len, seq_len)
            attn_mask = attn_mask.squeeze(1)  # (batch, seq_len, seq_len)
        else:
            attn_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (batch, seq_len, d_model)
        
        # Global average pooling over time dimension
        if mask is not None:
            # Mask out padding before pooling
            mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # (batch, d_model)
        else:
            x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get attention weights for visualization"""
        # This is a simplified version - full implementation would require
        # modifying the transformer encoder to return attention weights
        with torch.no_grad():
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            # Note: Standard PyTorch transformer doesn't return attention weights
            # You would need to implement a custom version for this
        return None


class LSTMModel(nn.Module):
    """LSTM baseline model for comparison"""
    
    def __init__(self,
                 input_dim: int = 258,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 100,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: Dimension of input landmarks
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Classification head
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional padding mask
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim * 2)
        else:
            hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(hidden)  # (batch, num_classes)
        
        return logits


def create_model(model_name: str = "transformer", **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of model ('transformer' or 'lstm')
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    if model_name.lower() == "transformer":
        return MediaPipeTransformer(**kwargs)
    elif model_name.lower() == "lstm":
        return LSTMModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
