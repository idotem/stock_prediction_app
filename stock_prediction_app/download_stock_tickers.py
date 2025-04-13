import ftplib
import csv

# Step 1: Download file from NASDAQ FTP server
def download_nasdaq_symbols():
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")

    # Download NASDAQ-listed.txt file
    filename = "nasdaqlisted.txt"
    with open(filename, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)

    ftp.quit()
    return filename


def download_other_exchanges_symbols():
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")

    # Download other-listed.txt file
    filename = "otherlisted.txt"
    with open(filename, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)

    ftp.quit()
    return filename


# Step 2: Parse file and extract stock symbols
def parse_symbols(file_path):
    symbols = []
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader)  # Skip header row
        for row in reader:
            if len(row) > 0 and "File Creation Time" not in row[0]:  # Filter valid rows
                symbols.append(row[0])  # Add symbol (first column)
    return symbols


