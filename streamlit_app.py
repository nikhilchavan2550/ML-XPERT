import streamlit as st
import os
import sys

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Ensure the src directory is in sys.path
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the main app
from app import main

# Run the main application
if __name__ == "__main__":
    main()
