from pathlib import Path


def make_dir(dir_path: str):
    """
    Create the directory
    :param dir_path:             path to the directory
    :return:                null
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def read_file(filename: str):
    """
    Returns a string of file contents, after changing
    new lines to spaces

    :param filename:        path to the file to be read
    :return:                file contents
    """
    file = open(filename, 'r', encoding='utf8', errors='ignore')
    content = file.read()
    return ' '.join(content.split('\n'))


def write_list_to_file(filename: str, content: list):
    """
    Writes each element of the list on a new line in the given file

    :param filename:        path to file to which list is to be written
    :param content:         list to be written
    :return:                null
    """
    parent = get_parent_dir(filename)
    if not Path(parent).exists():
        Path(parent).mkdir(parents=True, exist_ok=True)

    file = open(filename, 'w')
    file.write('\n'.join(content))


def get_parent_dir(filename: str):
    """
    Get the path of the parent directory of the given file

    :param filename:        path to the file
    :return:                path to the parent directory of the file
    """
    return '/'.join(filename.split('/')[:-1])


def read_file_to_list(filename: str):
    """
    Returns a list, where each element is a line in the given file

    :param filename:        path to the file to be read
    :return:                list of lines in the file
    """
    file = open(filename, 'r')
    content = file.read()
    return content.split('\n')
