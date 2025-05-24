import os
import re
import subprocess

import pypandoc
from inscriptis import get_text
from sec_edgar_downloader import Downloader

pypandoc.download_pandoc()

dl = Downloader('Meto', 'madaa_fakaa@yahoo.com')


def convert_html_to_txt(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    input_path = os.path.abspath(input_path)  # Ensure absolute path
    output_txt = os.path.abspath(output_path)

    # Ensure the input file exists, create it if not
    if not os.path.isfile(input_path):
        print(f"⚠️ File not found at {input_path}.")

    print(f"Running inscriptis command...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            output = get_text(html_content)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✅ Conversion successful! File saved at: {output_txt}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def download_10k_from_ticker(ticker):
    filing_type = "10-K"

    print(f"Downloading {filing_type} filings for {ticker}...")
    # Download the filing (automatically goes to sec-edgar-filings/ticker/filing-type/full-submission.txt
    # but it is not actually .txt, it is some kind of html, that's why we convert it to clean txt!
    dl.get(filing_type, ticker, limit=1)

    # Locate the latest downloaded filing
    filing_dir = f"sec-edgar-filings/{ticker}/{filing_type}"
    latest_filing = sorted(os.listdir(filing_dir))[-1]  # Get most recent folder
    txt_file = os.path.join(filing_dir, latest_filing, "full-submission.txt")  # Expected file

    print(f"Processing downloaded file...")
    # Read the raw SEC file
    with open(txt_file, "r", encoding="utf-8") as f:
        raw_content = f.read()

    print(f"Extracting HTML content from SEC file...")
    # **Extract only the real HTML part**
    html_match = re.search(r"(<html.*?</html>)", raw_content, re.DOTALL | re.IGNORECASE)
    if html_match:
        html_content = html_match.group(1)
    else:
        print("Warning: No proper HTML found! Using full file as fallback.")
        html_content = raw_content  # Fall back to full text if no HTML is found

    print(f"Cleaning HTML content...")
    # Save to a cleaned html file
    html_file = f"data/10k-html/{ticker}.html"
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(html_content)

    input_file = f"data/10k-html/{ticker}.html"
    output_file = f"data/next_docs_to_index/{ticker}.txt"

    convert_html_to_txt(input_file, output_file)
