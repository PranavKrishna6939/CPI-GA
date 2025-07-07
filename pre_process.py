import pandas as pd
import numpy as np

def preprocess_cpindex_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['Description'])
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
    df = df.sort_values('Date')
    
    unique_descriptions = df['Description'].unique()
    comprehensive_data = []
    time_periods = df[['Year', 'Month', 'Date']].drop_duplicates().sort_values('Date')
    
    for _, period in time_periods.iterrows():
        year = period['Year']
        month = period['Month']
        period_data = df[(df['Year'] == year) & (df['Month'] == month)]
        row_data = {'Year': year, 'Month': month}
        
        for desc in unique_descriptions:
            desc_data = period_data[period_data['Description'] == desc]
            
            if not desc_data.empty:
                data_row = desc_data.iloc[0]
                row_data[f"{desc}_Rural"] = data_row['Rural']
                row_data[f"{desc}_Urban"] = data_row['Urban'] 
                row_data[f"{desc}_Combined"] = data_row['Combined']
            else:
                row_data[f"{desc}_Rural"] = np.nan
                row_data[f"{desc}_Urban"] = np.nan
                row_data[f"{desc}_Combined"] = np.nan
        
        comprehensive_data.append(row_data)
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.columns = comprehensive_df.columns.str.replace(' ', '_')
    
    return comprehensive_df

def save_preprocessed_data(comprehensive_df, base_filename='raw_cpi_data'):
    output_file = f'{base_filename}.csv'
    comprehensive_df.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to: {output_file}")

if __name__ == "__main__":
    file_path = 'CPIndex_Jan14-To-May25.csv'
    
    print("Reading and preprocessing data:")
    comprehensive_df = preprocess_cpindex_data(file_path)
    
    print(f"DataFrame shape: {comprehensive_df.shape}")
    save_preprocessed_data(comprehensive_df)
    
    print("Column names:")
    for i, col in enumerate(comprehensive_df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("Preprocessing complete!")