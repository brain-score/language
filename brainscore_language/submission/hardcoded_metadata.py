"""
Temporary hardcoded metadata generator for language domain.

This module provides a dummy metadata generator that creates placeholder
metadata.yml files. Once language metadata generation is implemented,
replace the import in the workflow with the real generator.
"""

import yaml
import os
from typing import Optional


def generate_hardcoded_metadata(plugin_dir: str, plugin_type: str) -> Optional[str]:
    """
    Generate a hardcoded metadata.yml file for a plugin.
    
    Args:
        plugin_dir: Path to the plugin directory (e.g., "brainscore_language/models/test_model")
        plugin_type: Type of plugin ("models" or "benchmarks")
    
    Returns:
        Path to the generated metadata file, or None if generation failed
    """
    plugin_name = os.path.basename(plugin_dir)
    
    if plugin_type == "models":
        metadata = {
            "models": {
                plugin_name: {
                    "architecture": "DCNN",
                    "model_family": plugin_name,
                    "total_parameter_count": 1234567,
                    "trainable_parameter_count": 1234567,
                    "total_layers": 55,
                    "trainable_layers": 40,
                    "model_size_mb": 1202,
                    "training_dataset": None,
                    "task_specialization": None,
                    "brainscore_link": f"https://github.com/brain-score/language/tree/master/{plugin_dir}",
                    "huggingface_link": None,
                    "extra_notes": "Temporary hardcoded metadata - will be replaced with actual generation",
                    "runnable": True
                }
            }
        }
    elif plugin_type == "benchmarks":
        metadata = {
            "benchmarks": {
                plugin_name: {
                    "stimulus_set": None,
                    "data_metadata": None,
                    "metric_information": None,
                    "brainscore_link": f"https://github.com/brain-score/language/tree/master/{plugin_dir}",
                    "extra_notes": "Temporary hardcoded metadata - will be replaced with actual generation"
                }
            }
        }
    else:
        print(f"Unknown plugin type: {plugin_type}")
        return None
    
    # Ensure directory exists
    os.makedirs(plugin_dir, exist_ok=True)
    
    # Write metadata file
    metadata_path = os.path.join(plugin_dir, "metadata.yml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created hardcoded metadata.yml at: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: hardcoded_metadata.py <plugin_dir> <plugin_type>")
        sys.exit(1)
    
    plugin_dir = sys.argv[1]
    plugin_type = sys.argv[2]
    result = generate_hardcoded_metadata(plugin_dir, plugin_type)
    if not result:
        sys.exit(1)
