# trainer.py
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from dataset import ContrastiveDataset, split_dataset
from model import ContrastiveModel
from loss import ContrastiveLoss
from utils import evaluate_model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Split the dataset if files don't exist
    train_file = os.path.join(os.path.dirname(args.data_file), "train_split.csv")
    val_file = os.path.join(os.path.dirname(args.data_file), "val_split.csv")
    test_file = os.path.join(os.path.dirname(args.data_file), "test_split.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        print("Splitting dataset...")
        train_file, val_file, test_file = split_dataset(
            args.data_file, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
    else:
        print("Using existing dataset splits...")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ContrastiveDataset(
        csv_file=train_file, 
        all_molecules_csv=args.data_file,
        protein_feature_dir=args.protein_feature_dir,
        molecule_feature_dir=args.molecule_feature_dir
    )
    
    val_dataset = ContrastiveDataset(
        csv_file=val_file,
        all_molecules_csv=args.data_file,
        protein_feature_dir=args.protein_feature_dir,
        molecule_feature_dir=args.molecule_feature_dir
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = ContrastiveModel(
        protein_dim=args.protein_dim, 
        molecule_dim=args.molecule_dim, 
        projection_dim=args.projection_dim
    )
    model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ContrastiveLoss(temperature=args.temperature)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Initialize early stopping
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    early_stopping_path = os.path.join(args.save_dir, f"contrastive_model_best_{timestamp}.pth")
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        delta=args.early_stopping_delta,
        path=early_stopping_path
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for up to {args.epochs} epochs (with early stopping)...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as t:
            for batch in t:
                protein_features = batch['protein_feature'].to(device)
                molecule_features = batch['molecule_feature'].to(device)
                labels = batch['label'].to(device) if 'label' in batch else None
                
                optimizer.zero_grad()
                protein_proj, molecule_proj = model(protein_features, molecule_features)
                loss = criterion(protein_proj, molecule_proj, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                protein_features = batch['protein_feature'].to(device)
                molecule_features = batch['molecule_feature'].to(device)
                labels = batch['label'].to(device) if 'label' in batch else None
                
                protein_proj, molecule_proj = model(protein_features, molecule_features)
                loss = criterion(protein_proj, molecule_proj, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        scheduler.step()
    
    # Load the best model
    model.load_state_dict(torch.load(early_stopping_path))
    
    # Save final model (in case we want a different name)
    final_model_path = os.path.join(args.save_dir, f"contrastive_model_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_curve_path = os.path.join(args.save_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")
    
    # Evaluate on test set if requested
    if args.evaluate:
        print("Evaluating on test set...")
        test_dataset = ContrastiveDataset(
            csv_file=test_file,
            all_molecules_csv=args.data_file,
            protein_feature_dir=args.protein_feature_dir,
            molecule_feature_dir=args.molecule_feature_dir
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        
        results = evaluate_model(model, test_loader, device=device)
        
        # Save evaluation results
        if not os.path.exists("results"):
            os.makedirs("results")
        
        results_path = os.path.join("results", f"test_results_{timestamp}.npz")
        np.savez(
            results_path, 
            similarities=results['similarities'], 
            labels=results['labels']
        )
        print(f"Test results saved to {results_path}")
    
    print("Training completed!")
    return model, final_model_path

def main():
    parser = argparse.ArgumentParser(description="Contrastive Learning for Protein-Molecule Interaction")
    
    # Dataset arguments
    parser.add_argument("--data_file", type=str, default="./data/augmented_data.csv",
                        help="Path to the original CSV file")
    parser.add_argument("--protein_feature_dir", type=str, default="data/protein_data",
                        help="Directory containing protein features")
    parser.add_argument("--molecule_feature_dir", type=str, default="data/molecule_data",
                        help="Directory containing molecule features")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of test data")
    
    # Model arguments
    parser.add_argument("--protein_dim", type=int, default=256,
                        help="Dimension of protein features")
    parser.add_argument("--molecule_dim", type=int, default=768,
                        help="Dimension of molecule features")
    parser.add_argument("--projection_dim", type=int, default=128,
                        help="Dimension of projection space")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter for contrastive loss")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--early_stopping_delta", type=float, default=0.0001,
                        help="Minimum change in validation loss to qualify as improvement")
    
    # Misc arguments
    parser.add_argument("--save_dir", type=str, default="model_weights",
                        help="Directory to save models")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on test set after training")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Train model
    train(args)

if __name__ == "__main__":
    main()