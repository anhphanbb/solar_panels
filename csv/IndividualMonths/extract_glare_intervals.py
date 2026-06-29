# %%
# Read the CSV file and extract glare intervals into a new CSV.
#
# Input:
#   Feb2026.csv
#
# Output:
#   glare_intervals.csv
#
# Logic:
#   For each row where Glare == 1:
#       Save:
#           Orbit, Glare Start 1, Glare End 1
#
#       If Glare Start 2 and Glare End 2 exist:
#           Save another row:
#           Orbit, Glare Start 2, Glare End 2

import pandas as pd

# -----------------------------
# Input / Output
# -----------------------------
input_csv = "April2026.csv"
output_csv = "glare_intervals_April2026.csv"

# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(input_csv)

# -----------------------------
# Collect output rows
# -----------------------------
output_rows = []

for _, row in df.iterrows():

    # Only keep rows where Glare == 1
    if row.get("Glare", 0) != 1:
        continue

    orbit = row.get("Orbit")
    # -------------------------
    # Glare interval 1
    # -------------------------
    start1 = row.get("Glare Start 1")
    end1   = row.get("Glare End 1")

    if pd.notna(start1) and pd.notna(end1):
        output_rows.append({
            "Orbit #": orbit,
            "glare_initial": start1,
            "glare_final": end1
        })

    # -------------------------
    # Glare interval 2
    # -------------------------
    start2 = row.get("Glare Start 2")
    end2   = row.get("Glare End 2")
    

    if pd.notna(start2) and pd.notna(end2):
        output_rows.append({
            "Orbit #": orbit,
            "glare_initial": start2,
            "glare_final": end2
        })
     
# -----------------------------
# Save output CSV
# -----------------------------
output_df = pd.DataFrame(output_rows)

output_df.to_csv(output_csv, index=False)

print(f"Saved {len(output_df)} glare intervals to:")
print(output_csv)