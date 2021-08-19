""" Add a version header line to HTML pages """

import os
import fileinput

def add_version_header(root, version_string):
    """ For .html files in root, add a header using version string """
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for filename in filenames:
            if filename.endswith(".html"):
                filename_from_root = os.path.join(dirpath, filename)
                with fileinput.FileInput(filename_from_root, inplace=True) as current_file:
                    for line in current_file:
                        print(line, end='')  # prints to file
                        if line.lstrip().lower().startswith('<body'):
                            print('<div id="version" align="right"><b>PETSc version %s</b></div>' % version_string)
