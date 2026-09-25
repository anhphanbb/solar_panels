# -*- coding: utf-8 -*-
"""
Apply manual corrections to MLSPB for selected orbit/frame ranges.

Frame ranges in MANUAL_EDITS are inclusive at both ends. By default, the
script writes modified copies and leaves the original NetCDF files unchanged.

@author: anhph
"""

from pathlib import Path
import re
import shutil

import netCDF4 as nc
import numpy as np


# -----------------------------------------------------------------------------
# Paths and settings
# -----------------------------------------------------------------------------

# Search this folder and all of its subfolders for the requested orbit files.
PARENT_DIRECTORY = Path(r"E:\soc\l0d\2026\02")

# Used only when EDIT_IN_PLACE is False.
OUTPUT_DIRECTORY = PARENT_DIRECTORY / "manual_mlspb_edits"

# False: create modified copies in OUTPUT_DIRECTORY (recommended).
# True: modify the original files and optionally make backup copies first.
EDIT_IN_PLACE = False
CREATE_BACKUP_WHEN_EDITING_IN_PLACE = True

# If False, an existing output file causes the script to stop for that orbit.
OVERWRITE_OUTPUT = True

MLSPB_VARIABLE_NAME = "MLSPB"

# Each tuple is: (first_frame, last_frame, value).
# first_frame and last_frame are both included.
MANUAL_EDITS = {
    
    8155: [
        (800, 1650, 0),
    ],
    9100: [
        (800, 1643, 0),
    ],
    10140: [
        (800, 1402, 0),
    ],
    
}


def is_inside(path, directory):
    """Return True when path is inside directory."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def find_orbit_file(parent_directory, orbit_number):
    """Find exactly one NetCDF file containing the five-digit orbit number."""
    orbit_str = f"{orbit_number:05d}"
    pattern = re.compile(
        rf"^awe_l.*_{re.escape(orbit_str)}_.*\.nc$",
        flags=re.IGNORECASE,
    )

    matches = []
    for path in parent_directory.rglob("*.nc"):
        # Do not select files produced by an earlier run of this script.
        if is_inside(path, OUTPUT_DIRECTORY):
            continue
        if pattern.match(path.name):
            matches.append(path)

    matches.sort()

    if not matches:
        raise FileNotFoundError(
            f"No NetCDF file found for orbit {orbit_str} under "
            f"{parent_directory}"
        )

    if len(matches) > 1:
        match_list = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Found more than one file for orbit {orbit_str}. "
            f"Please narrow PARENT_DIRECTORY:\n{match_list}"
        )

    return matches[0]


def validate_edits(variable, orbit_number, edits):
    """Validate MLSPB dimensions and all requested frame ranges."""
    if variable.ndim < 2:
        raise ValueError(
            f"Orbit {orbit_number:05d}: {MLSPB_VARIABLE_NAME} has shape "
            f"{variable.shape}; expected time plus at least one box dimension."
        )

    number_of_frames = variable.shape[0]

    for start_frame, end_frame, value in edits:
        if value not in (0, 1):
            raise ValueError(
                f"Orbit {orbit_number:05d}: MLSPB value must be 0 or 1, "
                f"not {value}."
            )
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(
                f"Orbit {orbit_number:05d}: invalid inclusive range "
                f"{start_frame}-{end_frame}."
            )
        if end_frame >= number_of_frames:
            raise IndexError(
                f"Orbit {orbit_number:05d}: requested frame {end_frame}, but "
                f"{MLSPB_VARIABLE_NAME} contains frames 0-{number_of_frames - 1}."
            )


def edit_mlspb_file(file_path, orbit_number, edits):
    """Apply edits to one NetCDF file and verify each edited range."""
    with nc.Dataset(file_path, mode="r+") as dataset:
        if MLSPB_VARIABLE_NAME not in dataset.variables:
            raise KeyError(
                f"Orbit {orbit_number:05d}: variable "
                f"'{MLSPB_VARIABLE_NAME}' was not found in {file_path.name}."
            )

        mlspb = dataset.variables[MLSPB_VARIABLE_NAME]
        validate_edits(mlspb, orbit_number, edits)

        for start_frame, end_frame, value in edits:
            # The ellipsis selects every MLSPB box in each requested frame.
            mlspb[start_frame : end_frame + 1, ...] = value

        dataset.sync()

        # Read back the edited ranges before closing the file.
        for start_frame, end_frame, value in edits:
            edited_data = mlspb[start_frame : end_frame + 1, ...]
            if not np.all(np.asarray(edited_data) == value):
                raise RuntimeError(
                    f"Orbit {orbit_number:05d}: verification failed for "
                    f"frames {start_frame}-{end_frame}."
                )


def prepare_target_file(source_path):
    """Return the file to edit, copying or backing it up as configured."""
    if EDIT_IN_PLACE:
        if CREATE_BACKUP_WHEN_EDITING_IN_PLACE:
            backup_path = source_path.with_name(source_path.name + ".before_mlspb_edit.bak")
            if not backup_path.exists():
                shutil.copy2(source_path, backup_path)
                print(f"  Backup: {backup_path}")
            else:
                print(f"  Backup already exists: {backup_path}")
        return source_path

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    target_path = OUTPUT_DIRECTORY / source_path.name

    if target_path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Output already exists: {target_path}\n"
            "Set OVERWRITE_OUTPUT = True to replace it."
        )

    shutil.copy2(source_path, target_path)
    return target_path


def main():
    if not PARENT_DIRECTORY.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {PARENT_DIRECTORY}")

    print(f"Searching under: {PARENT_DIRECTORY}")
    print(f"Edit in place: {EDIT_IN_PLACE}")

    completed = []

    for orbit_number, edits in MANUAL_EDITS.items():
        orbit_str = f"{orbit_number:05d}"
        print(f"\nOrbit {orbit_str}")

        source_path = find_orbit_file(PARENT_DIRECTORY, orbit_number)
        print(f"  Source: {source_path}")

        target_path = prepare_target_file(source_path)
        edit_mlspb_file(target_path, orbit_number, edits)

        for start_frame, end_frame, value in edits:
            print(
                f"  Set all MLSPB boxes in frames "
                f"{start_frame}-{end_frame} to {value}"
            )
        print(f"  Saved: {target_path}")
        completed.append((orbit_str, target_path))

    print("\nCompleted all requested MLSPB edits:")
    for orbit_str, path in completed:
        print(f"  Orbit {orbit_str}: {path}")


if __name__ == "__main__":
    main()
