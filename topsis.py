import pandas as pd, numpy as np

def getTOPSISsoln(df,alt_col,weights,benefit_attributes,verbose=False,enable_json=False):
  if verbose:
    df = df.astype({f'{alt_col}':'str'})
    altcol_array = np.array(df[f'{alt_col}'].tolist())
    df = df.set_index(f'{alt_col}')
    m = len(df) #rows
    n = len(df.columns) #columns
    raw_data = df.to_numpy()
    print("Step 1: Basic Dataframe Head \n")
    print(df.head().to_string(index=True))
    print("\n")

    # Normalize the matrix by TOPSIS normalization
    divisors = np.empty(n)
    for j in range(n):
        column = raw_data[:,j]
        divisors[j] = np.sqrt(column @ column)

    raw_data /= divisors
    columns = ["X%d" % j for j in range(n)]

    raw_data *= weights
    print("Step 2: Post-normalization, weight multiplied dataframe preview \n")
    print(pd.DataFrame(data=raw_data, index=altcol_array, columns=columns).head().to_string(index=True))
    print("\n")

    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    for j in range(n):
        column = raw_data[:,j]
        max_val = np.max(column)
        min_val = np.min(column)
        
        # Take max for beneficial ideal best and min for beneficial ideal worst, and vice-versa
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    print("Step 3: Ideal best, ideal worst (V+, V-): \n")
    print(pd.DataFrame(data=[a_pos, a_neg], index=["V+", "V-"], columns=columns).head().to_string(index=True))
    print("\n")

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    print("Step 4: Order by Performance Scoring \n")
    print(pd.DataFrame(data=zip(altcol_array,sp, sn, cs), columns=[f"{alt_col}","S+", "S-", "perf_score"]).sort_values(by=['perf_score'],ascending=False).head().to_string(index=True))
    print("\n")

    results = pd.DataFrame(data=zip(altcol_array,sp, sn, cs), columns=[f"{alt_col}","S+", "S-", "perf_score"]).sort_values(by=['perf_score'],ascending=False)
    results['Rank'] = range(1, len(results) + 1)

    if enable_json:
      return results[[f'{alt_col}','Rank']].to_json(orient='records')
    return results[[f'{alt_col}','Rank']]
  
  else:
    df = df.astype({f'{alt_col}':'str'})
    altcol_array = np.array(df[f'{alt_col}'].tolist())
    df = df.set_index(f'{alt_col}')
    m = len(df) #rows
    n = len(df.columns) #columns
    raw_data = df.to_numpy()

    # Normalize the matrix by TOPSIS normalization
    divisors = np.empty(n)
    for j in range(n):
        column = raw_data[:,j]
        divisors[j] = np.sqrt(column @ column)

    raw_data /= divisors
    columns = ["X%d" % j for j in range(n)]

    raw_data *= weights
    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    for j in range(n):
        column = raw_data[:,j]
        max_val = np.max(column)
        min_val = np.min(column)
        
        # Take max for beneficial ideal best and min for beneficial ideal worst, and vice-versa
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    results = pd.DataFrame(data=zip(altcol_array,sp, sn, cs), columns=[f"{alt_col}","S+", "S-", "perf_score"]).sort_values(by=['perf_score'],ascending=False)
    results['Rank'] = range(1, len(results) + 1)
    
    if enable_json:
      return results[[f'{alt_col}','Rank']].to_json(orient='records')
    return results[[f'{alt_col}','Rank']]
