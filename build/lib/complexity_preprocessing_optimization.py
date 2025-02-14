import pandas as pd
import numpy as np
from logger import setup_logger

logger = setup_logger("logger", "logs.log")

def extract_required_value(cell, metric_type):

    """
    Extract the required value based on the metric type.
    - ApEn and SampEn: Take the last value.
    - PermEn: Take the first value.
    """
    try:
    # Handle space-separated values in brackets (e.g., "[2.6 1.2 0.7]")
        if isinstance(cell, str) and "[" in cell and "]" in cell:
            values = [float(x) for x in cell.strip("[]").split()]
            return values[-1] if len(values) > 0 else np.nan  # Always return the last value

    # Handle string representations of arrays (e.g., "array([2.6, 1.2, 0.7])")
        elif isinstance(cell, str) and "array" in cell:
            start = cell.find("[")
            end = cell.find("]")
            if start != -1 and end != -1:
                values = [float(x) for x in cell[start + 1:end].split(",")]
                return values[-1] if len(values) > 0 else np.nan  # Always return the last value

    # If already numeric, return as-is
        return float(cell)
    except (ValueError, AttributeError):
     return np.nan  # Return NaN if extraction fails

def preprocess_entropy_metrics(df):

    """
    Process the DataFrame columns to extract the required values
    for ApEn, SampEn, and PermEn.
    """
    try:

        for col in df.columns:
            if "ApEn" in col:
                df[col] = df[col].apply(lambda x: extract_required_value(x, "ApEn"))
            elif "SampEn" in col:
                df[col] = df[col].apply(lambda x: extract_required_value(x, "SampEn"))
            elif "PermEn" in col:
                df[col] = df[col].apply(lambda x: extract_required_value(x, "PermEn"))
    except Exception as e:
        logger.error(f"Error processing dataframe columns {e}")
        df = pd.DataFrame([])
    return df

def main():
    # Load your CSV file
    df = pd.read_csv("data/complex.csv")

    # Preprocess the entropy metrics
    df = preprocess_entropy_metrics(df)

    # Save the cleaned DataFrame to a new CSV file (optional)
    df.to_csv("data/complexity_csv_file.csv", index=False)


if __name__ == "__main__":
    main()