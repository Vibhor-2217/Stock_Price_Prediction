from data.loaders.save_processed import process_and_save

process_and_save(
    input_path="data/raw/AAPL.csv",
    output_path="data/processed/AAPL_processed.csv"
)
