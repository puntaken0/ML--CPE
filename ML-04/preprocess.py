import pandas as pd
import numpy as np

def preprocess_mobile_data(df):
    df_clean = df.copy()
    
    # 1. ตัดหน่วยข้อความออกจากตัวเลข
    numeric_cols = [
        'Battery_power_mAh', 'Front_camera', 'Internal_memeory_gb', 
        'Mobile_depth', 'Mobile_weight', 'Primary_camera', 
        'px_height', 'Pixel_width', 'Ram_mb', 'Screen_height', 'Screen_weight'
    ]
    for col in numeric_cols:
        df_clean[col] = df_clean[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # 2. แปลง Yes/No เป็นตัวเลข 1 และ 0
    binary_cols = ['Bluetooh', 'Dual_sim', '4G', '3G', 'touch_screen', 'wifi']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
    # 3. ตัดคอลัมน์ Target ออกเพื่อสร้าง Feature Matrix X
    X = df_clean.drop(columns=['price_range']).values.astype(np.float32)
    return X