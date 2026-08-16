import pandas as pd
import numpy as np

def load_data(file_path="Mobile.csv"):
    df = pd.read_csv(file_path)
    
    # ดึงรายชื่อ Class ทั้งหมดออกมาอัตโนมัติ
    classes = sorted(df['price_range'].unique().tolist())
    print("Detected classes:", classes)
    
    label_map = {c: i for i, c in enumerate(classes)}
    labels = df['price_range'].map(label_map).values
    
    return df, labels, classes