#!/usr/bin/env python
""" A brittle approach to making "Edit this Page" links work on man pages """

import os
import re
import fileinput

EDIT_URL_PATTERN = re.compile(
    r'^<p><a.*href="(.*)">Edit on GitLab</a></p>')  # very brittle


def _get_edit_url(filename):
    """ Get the edit URL for a manual page HTML page """
    with open(filename, "r") as f:
        for line in f:
            m = re.match(EDIT_URL_PATTERN, line)
            if m:
                return m.group(1)
    return None


def _check_edit_link(filename):
    """ Return true if the file has an edit link to be updated """
    found = False
    with open(filename, "r") as f:
        for line in f:
            if line.lstrip().startswith("<div") and "editthispage" in line:
                return True
        return False


def _update_edit_link(filename, url):
    """ Update the edit this page URL """
    with fileinput.FileInput(filename, inplace=True) as f:
        replace_link_line = False
        done = False
        for line in f:
            if done:
                print(line, end="")  # prints to file
            else:
                if line.lstrip().startswith("<div") and "editthispage" in line:
                    replace_link_line = True
                if replace_link_line and line.lstrip().startswith("<a"):
                    print("<a href=%s>" % url)  # prints to file
                    done = True
                else:
                    print(line, end="")  # prints to file


def fix_man_page_edit_links(root):
    base = os.path.join(root, "docs", "manualpages")
    for root, dirs, filenames in os.walk(base):
        for filename in filenames:
            if filename.endswith(".html"):
                filename_full = os.path.join(root, filename)
                url = _get_edit_url(filename_full)
                if url is not None and _check_edit_link(filename_full):
                    _update_edit_link(filename_full, url)
