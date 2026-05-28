import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Sab CSV files load karo aur combine karo
data_folder = 'data'

files = os.listdir(data_folder)
csv_files = [f for f in files if f.endswith('.csv')]
print("Files found:", csv_files)