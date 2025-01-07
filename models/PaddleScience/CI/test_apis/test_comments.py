import os


def bracket_check_file(file_path, left_bracket, right_bracket):
    """Check bracket matching in a file."""
    # Forward check
    s1 = []
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        for linenum, line in enumerate(lines, start=1):
            for chnum, ch in enumerate(line, start=1):
                if ch == left_bracket:
                    s1.append((ch, linenum, chnum))
                elif ch == right_bracket:
                    if not s1:
                        raise ValueError(f"Error at file {file_path}:{linenum}:{chnum}")
                    elif s1[-1][0] != left_bracket:
                        raise ValueError(
                            f"Error at file {file_path}:{linenum}:{chnum} which is pair to "
                            f"{file_path}:{s1[-1][1]}:{s1[-1][2]} for bracket: {left_bracket}, {right_bracket}"
                        )
                    else:
                        s1.pop()

    # Reverse check
    s1 = []
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        lines = lines[::-1]
        for linenum, line in enumerate(lines, start=1):
            line = line[::-1]
            for chnum, ch in enumerate(line, start=1):
                if ch == right_bracket:
                    s1.append((ch, linenum, chnum))
                elif ch == left_bracket:
                    if not s1:
                        raise ValueError(f"Error at file {file_path}:{linenum}:{chnum}")
                    elif s1[-1][0] != right_bracket:
                        raise ValueError(
                            f"Error at file {file_path}:{linenum}:{chnum} which is pair to "
                            f"{file_path}:{s1[-1][1]}:{s1[-1][2]} for bracket: {left_bracket}, {right_bracket}"
                        )
                    else:
                        s1.pop()


def solve(path: str, whitelist: list = []):
    """Recursively check bracket matching in Python files."""

    print(f"Solving file {path}")
    if os.path.isdir(path):
        pathes = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        for subpath in pathes:
            solve(os.path.join(path, subpath), whitelist)
        files = os.listdir(path)
        for file in files:
            if file.endswith(".py") and file not in whitelist:
                solve(os.path.join(path, file), whitelist)
    else:
        if not path.endswith(".py"):
            return
        if os.path.basename(path) in whitelist:
            return
        try:
            bracket_check_file(path, "(", ")")
            bracket_check_file(path, "[", "]")
            bracket_check_file(path, "{", "}")
            print(f"File {path} successfully checked.")
        except ValueError as e:
            print(str(e))


def get_all_files(folder_path):
    """Retrieve all files recursively from a folder."""
    files_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)

    return files_list


if __name__ == "__main__":
    current_dir = os.path.abspath(os.curdir)
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    print(f"Grandparent directory: {grandparent_dir}")
    folder_path = os.path.join(grandparent_dir, "jointContribution")  # Folder path
    files_list = get_all_files(folder_path)  # Get all files in the folder
    whitelist = ["atmospheric_dataset.py"]  # Example whitelist
    solve(grandparent_dir, whitelist)
    
