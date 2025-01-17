import os
from datetime import datetime

class Config:
    def __init__(self, args):
        # Create unique experiment ID
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Base directory settings with default fallback
        self.base_dir = args.base_dir if hasattr(args, 'base_dir') else "experiments"
        self.experiment_dir = os.path.join(self.base_dir, f"{args.model_type}_{self.experiment_id}")
        
        # Create experiment directories
        self.create_directories()
        
        # Data related paths with default values
        self.splits_dir = os.path.join(self.experiment_dir, "splits")
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        
        # Model configuration
        self.model_type = args.model_type
        self.cnn_out_channels = getattr(args, 'cnn_out_channels', 64)  # Default value
        self.hidden_dim = getattr(args, 'hidden_dim', 128)  # Default value
        self.num_layers = getattr(args, 'num_layers', 2)  # Default value
        
        # Training flags with defaults
        self.train = getattr(args, 'train', True)
        self.fine_tune = getattr(args, 'fine_tune', False)
        self.test = getattr(args, 'test', False)
        self.resume = getattr(args, 'resume', False)
        self.save_current = getattr(args, 'save_current', True)
        
        # Training parameters with defaults
        self.prune = getattr(args, 'prune', False)
        self.epochs = getattr(args, 'epochs', 50)
        self.k_folds = getattr(args, 'k_folds', 5)
        self.fine_tune_epochs = getattr(args, 'fine_tune_epochs', 10)
        
        # Optimizer configuration with defaults
        self.optimizer = getattr(args, 'optimizer', 'adam')
        self.momentum = getattr(args, 'momentum', 0.9)
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.lr = getattr(args, 'lr', 0.001)
        
        # Data and file paths with validation
        self.data_dir = self._validate_path(args.data_dir)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.sample_size = getattr(args, 'sample_size', -1)
        self.fine_tune_data_dir = self._validate_path(getattr(args, 'fine_tune_data_dir', ''))
        self.test_data_dir = self._validate_path(getattr(args, 'test_data_dir', ''))
        self.explain_data_dir = self._validate_path(getattr(args, 'explain_data_dir', ''))
        self.checkpoint_path = self._validate_path(getattr(args, 'checkpoint_path', ''))
        self.best_checkpoint_path = self._validate_path(getattr(args, 'best_checkpoint_path', ''))
        self.test_checkpoint_path = self._validate_path(getattr(args, 'test_checkpoint_path', ''))
        self.fine_tuned_model_checkpoint_path = self._validate_path(getattr(args, 'fine_tuned_model_checkpoint_path', ''))
        
        # Hyperparameter search and explain flags
        self.hyperparameter_search = getattr(args, 'hyperparameter_search', False)
        self.explain = getattr(args, 'explain', False)
        self.explain_checkpoint_path = self._validate_path(getattr(args, 'explain_checkpoint_path', ''))
        self.use_exist = getattr(args, 'use_exist', False)
        
        # Save configuration
        self.save_config(args)
    
    def create_directories(self):
        """创建实验所需的所有目录"""
        dirs = [
            self.experiment_dir,
            os.path.join(self.experiment_dir, "splits"),
            os.path.join(self.experiment_dir, "models"),
            os.path.join(self.experiment_dir, "logs"),
            os.path.join(self.experiment_dir, "results"),
            os.path.join(self.experiment_dir, "plots")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def save_config(self, args):
        """保存实验配置到文件"""
        config_path = os.path.join(self.experiment_dir, "config.txt")
        with open(config_path, 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
    
    def _validate_path(self, path):
        """Validate and return a path, returning None if path is empty or invalid"""
        if not path:
            return None
        path = os.path.expanduser(path)  # Expand user directory if present
        return path if os.path.exists(os.path.dirname(path) or '.') else None