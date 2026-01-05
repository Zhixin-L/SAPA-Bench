"""
Configuration loader module for evaluation scripts.
Provides utilities for loading and accessing configuration settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader with support for environment variable overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.json file. If None, searches for config.json
                        in current directory and parent directories.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config_path = config_path
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _find_config_file(self) -> str:
        """Find config.json file in current or parent directories."""
        current = Path.cwd()
        for path in [current] + list(current.parents):
            config_file = path / "config.json"
            if config_file.exists():
                return str(config_file)
        raise FileNotFoundError("config.json not found. Please create it or specify the path.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Override API key if set in environment
        if 'OPENAI_API_KEY' in os.environ:
            self.config['api']['api_key'] = os.environ['OPENAI_API_KEY']
        
        if 'OPENAI_BASE_URL' in os.environ:
            self.config['api']['base_url'] = os.environ['OPENAI_BASE_URL']
        
        # Override paths if set in environment
        if 'GROUND_TRUTH_PATH' in os.environ:
            self.config['paths']['ground_truth'] = os.environ['GROUND_TRUTH_PATH']
        
        if 'OUTPUT_DIR' in os.environ:
            self.config['paths']['output_dir'] = os.environ['OUTPUT_DIR']
    
    def get_path(self, key: str) -> str:
        """Get a path from configuration."""
        return self.config['paths'][key]
    
    def get_eval_config(self, eval_type: str) -> Dict[str, Any]:
        """Get evaluation configuration for a specific evaluation type."""
        return self.config['evaluation'][eval_type]
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config['api']
    
    def get_mapping(self, mapping_type: str) -> Dict[str, str]:
        """Get option mapping configuration."""
        return self.config['mappings'][mapping_type]
    
    def resolve_path(self, *path_parts: str) -> str:
        """
        Resolve a path relative to the config file location.
        
        Args:
            *path_parts: Path components to join.
        
        Returns:
            Absolute path string.
        """
        base_dir = Path(self.config_path).parent
        return str(base_dir / Path(*path_parts))
    
    def get_full_path(self, config_key: str, *sub_paths: str) -> str:
        """
        Get full path by combining base path from config with sub-paths.
        
        Args:
            config_key: Key in paths configuration.
            *sub_paths: Additional path components.
        
        Returns:
            Absolute path string.
        """
        base_path = self.get_path(config_key)
        return self.resolve_path(base_path, *sub_paths)


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get or create global configuration loader instance.
    
    Args:
        config_path: Optional path to config file.
    
    Returns:
        ConfigLoader instance.
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

