import pandas as pd

def remove_na(df: pd.DataFrame,
              save: bool = True,
              filename: str = "EThOS_CSV_202210.csv") -> pd.DataFrame:
    null_df = df[df["IR URL"].isnull()]
    for index, row in null_df.iterrows():
        new_row = []
        # Title
        new_row.append(row["Title"].split(",")[0])
        # DOI
        new_row.append(row["Title"].split(",")[1])
        # Author
        first_name = row["Title"].split(",")[2]
        second_name = row["DOI"][:-1]
        new_row.append(first_name + second_name)
        # Remaining columns
        colnames = [col for col in df.columns
                    if col not in ["Title", "DOI", "IR URL"]]
        new_row += row[colnames].tolist()
        df.loc[index] = new_row
    df = df.fillna(' ')
    if save:
        df.to_csv(f"cleaned_{filename}")
    return df
