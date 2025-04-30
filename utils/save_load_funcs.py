import os
import numpy as np
import jax.numpy as jnp

def save_data(folder_path, filenames, *data):
    """
    Save JAX arrays and lists to the specified folder.
    Creates subfolders if they do not exist.

    Args:
        folder_path (str): Path to the folder where data will be saved.
        filenames (list of str): List of filenames (without extension) corresponding to data items.
        *data: Variable number of data items (JAX arrays or Python lists).
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Check if the number of filenames matches the number of data items
    if len(filenames) != len(data):
        raise ValueError("Number of filenames must match the number of data items.")
    
    # Iterate through the data and filenames to save each item
    for name, item in zip(filenames, data):
        file_path = os.path.join(folder_path, f"{name}.npy")
        if isinstance(item, jnp.ndarray):
            # Convert JAX array to NumPy array and save
            np.save(file_path, np.array(item))
        elif isinstance(item, list):
            # Convert list to NumPy array and save
            np.save(file_path, np.array(item))
        else:
            raise ValueError(f"Unsupported data type for {name}: {type(item)}")
        # print(f"Saved {name} to {file_path}")

