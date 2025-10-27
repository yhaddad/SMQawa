import os
import pickle
from hist import Hist

def merge_histograms_in_directory(directory, output_file):
    """
    Merge all .pkl files in a directory that contain dicts of Hist objects.
    The histograms are added together by key (bin-by-bin addition).
    """
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    files.sort()
    print(f"Found {len(files)} .pkl file(s) in '{directory}'.")

    if not files:
        print("No pickle files to merge. Exiting.")
        return

    merged_hists = {}

    for filename in files:
        path = os.path.join(directory, filename)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            print(f"Skipping '{filename}' — not a dict of histograms.")
            continue

        for key, hist in data.items():
            if key not in merged_hists:
                merged_hists[key] = hist.copy()
            else:
                merged_hists[key] += hist  # Hist objects support +=
        print(f"Merged histograms from '{filename}'.")

    # Save combined histograms
    with open(output_file, "wb") as f:
        pickle.dump(merged_hists, f)

    print(f"\n Successfully merged {len(files)} files into '{output_file}'.")
    print(f"Total histograms combined: {len(merged_hists)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge dicts of Hist objects from .pkl files.")
    parser.add_argument("directory", help="Directory containing .pkl histogram files")
    parser.add_argument("output", help="Output .pkl file name (e.g. merged.pkl)")
    args = parser.parse_args()

    merge_histograms_in_directory(args.directory, args.output)
