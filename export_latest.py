"""Quick script: export the newest pipeline checkpoint (by stage order) to data/latest_output.csv."""
from pipeline import Pipeline
from utils import latest_checkpoint_in_order, load_checkpoint, save_dataframe_csv

order = Pipeline().list_stages()
latest = latest_checkpoint_in_order(order)
if latest:
    print(f"Exporting checkpoint: {latest} (last completed stage in pipeline order)")
    df = load_checkpoint(latest)
    save_dataframe_csv(df, "latest_output")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
else:
    print("No checkpoints found.")
