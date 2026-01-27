#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

def copy_dm_files(src_root: str, dst_root: str):
    if not os.path.isdir(src_root):
        print(f"Error: source directory '{src_root}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(dst_root, exist_ok=True)

    for entry in os.listdir(src_root):
        src_sub = os.path.join(src_root, entry)
        if not os.path.isdir(src_sub):
            continue

        dst_sub = os.path.join(dst_root, entry)
        os.makedirs(dst_sub, exist_ok=True)

        copied = False
        for fname in os.listdir(src_sub):
            if fname.lower().endswith('.dm'):
                src_file = os.path.join(src_sub, fname)
                dst_file = os.path.join(dst_sub, fname)
                shutil.copy2(src_file, dst_file)
                copied = True
        if copied:
            print(f"Copied .DM files from '{src_sub}' → '{dst_sub}'")
        else:
            print(f"No .DM files found in '{src_sub}'")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Copy .DM files from each subfolder of a source dir into matching subfolders of a new destination dir"
    )
    p.add_argument('source', help="Path to the source run directory")
    p.add_argument('destination', help="Path to the (new) destination directory")
    args = p.parse_args()

    copy_dm_files(args.source, args.destination)
