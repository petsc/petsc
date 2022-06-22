""" Helper to make links in HTML pages relative to a root """

import os
import fileinput

from ext.html5_petsc import PETSC_DOC_OUT_ROOT_PLACEHOLDER
from build_classic_docs import classic_docs_subdirs


def make_links_relative(root, placeholder=PETSC_DOC_OUT_ROOT_PLACEHOLDER):
    """ For .html files in root, replace placeholder with a relative path back up to root

        Exclude a specific set of subdirectories.
    """
    excludes = ["_static", "_sources", "_images"]
    excludes.extend(classic_docs_subdirs("pre"))
    excludes.extend(classic_docs_subdirs("post"))
    root_level = root.count(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in excludes]
        level = dirpath.count(os.path.sep) - root_level
        relpath = os.path.sep.join([".."] * level)
        for filename in filenames:
            if filename.endswith(".html"):
                filename_from_root = os.path.join(dirpath, filename)
                with fileinput.FileInput(filename_from_root, inplace=True) as file:
                    for line in file:
                        print(line.replace(placeholder, relpath), end='')  # prints to file
