import csv

def convert_to_csv(input_path, output_path):
    """
    Convert fixed-width formatted file to proper CSV format
    """
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        
        csv_writer.writerow([
            "eKYCTransactionID",
            "DistCode",
            "CreatedBy",
            "InsertedOn",
            "LastUpdatedOn",
            "eKYCStatus",
            "AppID",
            "ConsumerID"
        ])
        
        next(infile)
        next(infile)
        
        for line in infile:
            if not line.strip():
                continue
            
            details = line.split(" ")
            slist = list(filter(None, details))
            if len(slist) < 10:
                # memory limitation
                fields = [0,0,0,0,0,0,0,0]
                csv_writer.writerow(fields)
                break

            fields = [
                slist[0],
                slist[1],
                slist[2],
                slist[3]+ " " + slist[4],
                slist[5]+ " " + slist[6],
                slist[7],
                slist[8],
                slist[9],
            ]
            
            csv_writer.writerow(fields)

convert_to_csv('./failure.csv', './cleaned-failure.csv')

# To verify the conversion:
def verify_conversion(filepath):
    """
    Print first few lines of the converted CSV
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    return df

verify_conversion("./cleaned-failure.csv")