import os
import re
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

# Glare indices refer to the files AFTER trimming.
# Both start and end indices are inclusive.
csv_file_path = 'csv/IndividualMonths/glare_intervals_Dec2023.csv'

parent_directory = r'E:\soc\l0c\2023\12'
output_directory = r'E:\soc\l0c\2023\12\no_glare'

TRIM_START = 10
TRIM_END = 10


def extract_glare_intervals(data):
    glare_intervals = {}

    for _, row in data.iterrows():
        if (
            pd.notna(row['Orbit #'])
            and pd.notna(row['glare_initial'])
            and pd.notna(row['glare_final'])
        ):
            orbit = int(row['Orbit #'])
            start = int(row['glare_initial'])
            end = int(row['glare_final'])

            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid glare interval for orbit {orbit}: {start}..{end}"
                )

            glare_intervals.setdefault(orbit, []).append((start, end))

    return glare_intervals


def remove_glare_and_save(nc_file_path, glare_intervals):
    match = re.search(r'_(\d{5})_', os.path.basename(nc_file_path))
    if not match:
        print(f"Skipping file (orbit not found in name): {nc_file_path}")
        return

    orbit_number = int(match.group(1))

    with Dataset(nc_file_path, 'r') as nc:
        total_frames = len(nc.dimensions['time'])

        if total_frames <= TRIM_START + TRIM_END:
            print(
                f"Skipping {os.path.basename(nc_file_path)}: "
                f"only {total_frames} frames; none would remain after trimming."
            )
            return

        # Original-file indices remaining after trimming.
        trimmed_indices = np.arange(
            TRIM_START, total_frames - TRIM_END
        )

        # Glare intervals are indexed relative to this shortened sequence.
        keep_mask = np.ones(len(trimmed_indices), dtype=bool)

        for start, end in glare_intervals.get(orbit_number, []):
            # Clip intervals to the shortened file's bounds.
            start = max(0, start)
            stop = min(len(trimmed_indices), end + 1)

            if start < stop:
                keep_mask[start:stop] = False

        # Map surviving shortened-file indices back to original-file indices.
        keep_indices = trimmed_indices[keep_mask]
        glare_removed = int(np.count_nonzero(~keep_mask))

        if len(keep_indices) == 0:
            print(
                f"Skipping {os.path.basename(nc_file_path)}: "
                "no frames remain after glare removal."
            )
            return

        output_file_path = os.path.join(
            output_directory, os.path.basename(nc_file_path)
        )

        with Dataset(output_file_path, 'w', format='NETCDF4') as new_nc:
            new_nc.setncatts({
                attr: nc.getncattr(attr)
                for attr in nc.ncattrs()
            })

            output_sizes = {}
            for dim_name, dim in nc.dimensions.items():
                size = (
                    len(keep_indices)
                    if dim_name == 'time'
                    else len(dim)
                )
                output_sizes[dim_name] = size
                new_nc.createDimension(
                    dim_name, None if dim.isunlimited() else size
                )

            for var_name, var in nc.variables.items():
                original_chunks = var.chunking()
                safe_chunks = None

                if isinstance(original_chunks, (list, tuple)):
                    safe_chunks = tuple(
                        max(1, min(output_sizes[dim], chunk))
                        for dim, chunk in zip(
                            var.dimensions, original_chunks
                        )
                    )

                filters = var.filters() or {}
                create_options = {
                    'zlib': filters.get('zlib', False),
                    'chunksizes': safe_chunks,
                }

                if filters.get('zlib', False):
                    create_options.update(
                        complevel=filters.get('complevel', 4),
                        shuffle=filters.get('shuffle', True),
                    )

                # _FillValue must be supplied when creating the variable.
                if '_FillValue' in var.ncattrs():
                    create_options['fill_value'] = var.getncattr(
                        '_FillValue'
                    )

                new_var = new_nc.createVariable(
                    var_name,
                    var.datatype,
                    var.dimensions,
                    **create_options
                )

                new_var.setncatts({
                    attr: var.getncattr(attr)
                    for attr in var.ncattrs()
                    if attr != '_FillValue'
                })

                # Copy stored values without unpacking/repacking.
                var.set_auto_maskandscale(False)
                new_var.set_auto_maskandscale(False)
                var.set_auto_chartostring(False)
                new_var.set_auto_chartostring(False)

                if 'time' in var.dimensions:
                    # Select along the time axis, wherever it occurs.
                    selection = [slice(None)] * var.ndim
                    selection[var.dimensions.index('time')] = keep_indices
                    new_var[...] = var[tuple(selection)]
                else:
                    new_var[...] = var[...]

    print(
        f"Processed: {output_file_path}\n"
        f"  Original: {total_frames}; "
        f"trimmed: {TRIM_START + TRIM_END}; "
        f"glare removed: {glare_removed}; "
        f"remaining: {len(keep_indices)}"
    )


def process_file(nc_file_path, glare_intervals):
    remove_glare_and_save(nc_file_path, glare_intervals)


if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)

    data = pd.read_csv(csv_file_path)
    glare_intervals = extract_glare_intervals(data)

    nc_files = [
        os.path.join(parent_directory, filename)
        for filename in os.listdir(parent_directory)
        if filename.endswith(".nc")
    ]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_file, nc_file, glare_intervals
            ): nc_file
            for nc_file in nc_files
        }

        for future in as_completed(futures):
            nc_file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {nc_file}: {e}")