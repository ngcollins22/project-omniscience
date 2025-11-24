import pandas as pd
import re
import os
from pathlib import Path


def col_to_idx(col):
    col = col.upper()
    idx = 0
    for ch in col:
        if not ('A' <= ch <= 'Z'):
            break
        idx = idx*26 + (ord(ch) - ord('A') + 1)
    return idx-1


def _sanitize_headers(seq):
    out = []
    for i, c in enumerate(seq):
        if pd.isna(c):
            out.append(f"col{i+1}")
        else:
            out.append(str(c))
    return out

def read_constellations(xlsx, sheet, cell_range) -> pd.DataFrame:
    df_full = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl", header=None)
    m = re.match(r'^\s*([A-Za-z]+)(\d+)\s*:\s*([A-Za-z]+)(\d+)\s*$', cell_range) # this line was AI generated idk how to debug this regex
    if not m:
        raise ValueError(f"Range must be like A1:G200, got: {cell_range}")
    c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    r1_i, r2_i = r1-1, r2-1
    c1_i, c2_i = col_to_idx(c1), col_to_idx(c2)
    df_slice = df_full.iloc[r1_i:r2_i+1, c1_i:c2_i+1].copy()
    # First row in the slice is the header
    df_slice.columns = _sanitize_headers(df_slice.iloc[0].tolist())
    df_slice = df_slice.iloc[1:, :]
    df_slice = df_slice.reset_index(drop=True)
    return df_slice

def clean_path(path_str: str) -> Path:
    """
    Clean up a filesystem path string and return a pathlib.Path object.

    - Strips surrounding quotes (single or double).
    - Expands ~ (home directory).
    - Expands environment variables like %USERPROFILE% or $HOME.
    - Normalizes separators for the current OS.
    """
    # Strip whitespace and surrounding quotes
    s = path_str.strip().strip('"').strip("'")
    
    # Expand environment vars and ~
    s = os.path.expandvars(os.path.expanduser(s))
    
    # Create a Path and normalize
    return Path(s)



    
