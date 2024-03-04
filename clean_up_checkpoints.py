import os
import re

def get_highest_evaluated_files(root_dir):
    highest_files = {}
    for archive_size in os.listdir(root_dir):
        size_dir = os.path.join(root_dir, archive_size)
        if os.path.isdir(size_dir):
            for run_num in os.listdir(size_dir):
                run_dir = os.path.join(size_dir, run_num)
                if os.path.isdir(run_dir):
                    for file in os.listdir(run_dir):
                        match = re.match(r'archive(_genome)?(\d+).dat', file)
                        if match:
                            num_evaluations = int(match.group(2))
                            key = (archive_size, run_num, match.group(1) or '')
                            if key not in highest_files or highest_files[key][1] < num_evaluations:
                                highest_files[key] = (file, num_evaluations)
    return highest_files

def cleanup_directories(root_dir, highest_files):
    for archive_size in os.listdir(root_dir):
        size_dir = os.path.join(root_dir, archive_size)
        if os.path.isdir(size_dir):
            for run_num in os.listdir(size_dir):
                run_dir = os.path.join(size_dir, run_num)
                if os.path.isdir(run_dir):
                    for file in os.listdir(run_dir):
                        if file != 'log.dat':  # Skip the log.dat file
                            key = (archive_size, run_num, '_genome' if 'genome' in file else '')
                            if key in highest_files and highest_files[key][0] != file:
                                os.remove(os.path.join(run_dir, file))
                                print(f"Removed {os.path.join(run_dir, file)}")

def main():
    root_dir = 'mapElitesOutput'  # Update this path if needed
    highest_files = get_highest_evaluated_files(root_dir)
    cleanup_directories(root_dir, highest_files)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
