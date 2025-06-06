import torch
import torch.nn as nn
from typing import Dict, Union
from torch_geometric.nn import GATConv
import torch.nn.functional as F

# ============================
# contact map prediction model
# ============================

def extract_features(
    model,
    inp: torch.Tensor,
    has_cls: bool = True,
    has_eos: bool = True,
    need_head_weights: Union[bool, Dict] = False,
):
    """
    Extract attention features from the model.

    Args:
        model: The transformer model.
        inp: Input tensor.
        has_cls: Whether the input has a CLS token.
        has_eos: Whether the input has an EOS token.
        need_head_weights: Whether head weights are needed.

    Returns:
        Attention features.
    """
    out_start_idx = 1 if has_cls else 0
    out_end_idx = -1 if has_eos else None
    result = model(inp, output_attentions=True, output_hidden_states=True)

    attentions = torch.stack(result.attentions, 1)
    batch, layer, head, seqlen, _ = attentions.size()
    attentions = attentions.reshape(batch, layer * head, seqlen, seqlen)

    output_attentions = attentions.detach()
    output_attentions.requires_grad = False

    return output_attentions[:, :, out_start_idx:out_end_idx, out_start_idx:out_end_idx]


def _process_output(out_dict):
    """
    Process the output dictionary from the model.

    Args:
        out_dict: Dictionary containing model outputs.

    Returns:
        Processed output dictionary.
    """
    out_dict['logits'] = out_dict['logits'].permute(0, 2, 3, 1).contiguous()

    if 'logits' in out_dict:
        out_dict['dist_logits'] = out_dict['logits']
        del out_dict['logits']
    for k in [k for k in out_dict if k.endswith('_logits')]:
        targetname = k[:-len('_logits')]
        out_dict[f'p_{targetname}'] = torch.sigmoid(out_dict[k])
    return out_dict


class ResidualBlock(nn.Module):
    """
    Residual block with convolutional layers and batch normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualProjectionDistogramModel(nn.Module):
    """
    Model to predict contact maps using residual projection architecture.
    """

    def __init__(self, num_heads=20 * 33, output_dim=1, channels=[512, 256, 128, 64]):
        super().__init__()
        layers = []
        in_channels = num_heads

        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels, output_dim, kernel_size=1)

    def forward(self, attentions_asym):
        out = self.feature_extractor(attentions_asym)
        out = self.final_conv(out)
        return {'logits': self.symmetrize(out)}

    @staticmethod
    def symmetrize(x, scale=1.0):
        return scale * (x + x.transpose(-1, -2))


def load_contact(base_model, model_path, sequence, tokenizer, device, threshold=0.65, 
                num_heads=20*33, channels=[512, 256, 128, 64, 32, 16]):
    """
    Load a contact prediction model and predict contacts for a protein sequence.
    
    Args:
        base_model: ESM model for feature extraction
        model_path: Path to the saved model weights
        sequence: Protein sequence
        tokenizer: ESM tokenizer
        device: Device to run the model on
        threshold: Threshold for contact prediction
        num_heads: Number of attention heads in the model
        channels: List of channel dimensions for residual blocks
        
    Returns:
        Predicted contact map as a binary matrix
    """
    # Create and load the model
    model = ResidualProjectionDistogramModel(
        num_heads=num_heads,
        output_dim=1,
        channels=channels
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Predict contacts
    inputs = tokenizer(sequence, return_tensors='pt', padding="max_length", 
                     truncation=True, max_length=len(sequence) + 2).to(device)
    
    with torch.no_grad():
        # Extract attention features
        attentions = extract_features(base_model, inputs['input_ids'], need_head_weights=True)
        
        # Predict contacts
        out_dict = model(attentions)
        outputs = _process_output(out_dict)
    
    contacts = (outputs['p_dist'] > threshold).float()[0, :, :, 0]
    contacts = torch.nan_to_num(contacts, nan=0)
    contacts = torch.clamp(contacts, 0, 1)
    
    return contacts.cpu().detach().numpy()

# ============================
# GAT model
# ============================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GAT(nn.Module):
    def __init__(self, config, 
                 esm_input_dim, 
                 physio_input_dim, 
                 hidden_channels, 
                 out_channels, 
                 heads, 
                 heads_intermediate, 
                 heads_final, 
                 dropratio):
        super(GAT, self).__init__()
        self.config = config
        
        # Define MLPs for ESM2 embeddings and physicochemical properties
        self.mlp_aa = MLP(input_dim=esm_input_dim, hidden_dim=2 * config.fusion_dim, output_dim=config.fusion_dim)
        self.mlp_pc = MLP(input_dim=physio_input_dim, hidden_dim=2 * config.fusion_dim, output_dim=config.fusion_dim)
        
        # Define GAT Convolution layers
        self.conv1 = GATConv(config.fusion_dim, hidden_channels, heads=heads, dropout=dropratio)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads_intermediate, dropout=dropratio)
        self.conv3 = GATConv(hidden_channels * heads_intermediate, hidden_channels, heads=heads_intermediate, dropout=dropratio)
        self.conv4 = GATConv(hidden_channels * heads_intermediate, out_channels, heads=heads_final, concat=False, dropout=dropratio)

    def forward(self, x, edge_index):
        """
        Forward pass without augmentation.
        
        Args:
            x (torch.Tensor): Node features after possible augmentation.
            edge_index (torch.Tensor): Edge indices.
        
        Returns:
            torch.Tensor: Output logits.
        """
        # Split the combined features back into ESM and physicochemical features if needed
        
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = self.conv4(x, edge_index)
        return x  # For cross-entropy loss