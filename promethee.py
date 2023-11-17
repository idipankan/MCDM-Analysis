import pandas as pd
import numpy as np

def getPROMETHEEsoln_batch(df, alt_col, weights, beneficial_attrs, enable_json=False, batch_size=1000):
    alternatives = df.iloc[:, 0]
    df = df.set_index(df.columns[0])
    raw_data = df.to_numpy()
    m, n = raw_data.shape
    
    # Normalize the dataset
    max_vals = np.max(raw_data, axis=0)
    min_vals = np.min(raw_data, axis=0)
    norm_data = np.where(np.isin(np.arange(n), beneficial_attrs),
                         (raw_data - min_vals) / (max_vals - min_vals),
                         (max_vals - raw_data) / (max_vals - min_vals))
    
    num_alternatives = len(alternatives)
    
    # Create a memory-mapped array for the preference_matrix
    filename = 'preference_matrix.mmap'
    if os.path.exists(filename):
        os.unlink(filename)
    preference_matrix = np.memmap(filename, dtype='float64', mode='w+', shape=(num_alternatives, num_alternatives))
    
    # Batch processing
    for i_start in range(0, num_alternatives, batch_size):
        i_end = min(i_start + batch_size, num_alternatives)
        for j_start in range(0, num_alternatives, batch_size):
            j_end = min(j_start + batch_size, num_alternatives)
            
            diff_mod = np.maximum(norm_data[i_start:i_end, None, :] - norm_data[None, j_start:j_end, :], 0)
            preference_matrix[i_start:i_end, j_start:j_end] = np.tensordot(diff_mod, weights, axes=([2], [0]))
    
    np.fill_diagonal(preference_matrix, 0)
    
    # Calculate the net flow matrix
    positive_flow = np.sum(preference_matrix, axis=1) / (num_alternatives - 1)
    negative_flow = np.sum(preference_matrix, axis=0) / (num_alternatives - 1)
    net_flow = positive_flow - negative_flow
    
    # Rank the alternatives based on net flow
    ranks = pd.DataFrame({f'{alt_col}': alternatives, 'Net Flow': net_flow}).sort_values('Net Flow', ascending=False)
    ranks['Rank'] = range(1, len(ranks) + 1)
    if enable_json:
        return ranks[[f'{alt_col}', 'Rank']].to_json(orient='records')
    
    return ranks[[f'{alt_col}', 'Rank']]
