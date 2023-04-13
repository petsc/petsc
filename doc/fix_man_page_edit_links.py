#!/usr/bin/env python
""" A brittle approach to making "Edit this Page" links work on man pages """

import os
import re
import fileinput

EDIT_URL_PATTERN = re.compile(
    r'^<p><a.*href="(.*)">Edit on GitLab</a></p>')  # very brittle

SHOW_SOURCE_PATTERN = re.compile(
    r'(?s)<div class="toc-item">\s*<div class="tocsection sourcelink">.*<a href=.*>.*<i class="fa-solid fa-file-lines"></i> Show Source\s*</a>\s*</div>\s*</div>')


def _get_edit_url(filename):
    """ Get the edit URL for the source code for a manual page that was added by lib/petsc/bin/maint/wwwindex.py"""
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
    """ Change the URL for editthispage that Sphinx generates to the URL for GitLab repository location of the file
        Remove the Edit on GitLab line added by lib/petsc/bin/maint/wwwindex.py since it is now a duplicate"""
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
                elif not 'Edit on GitLab' in line:
                    print(line, end="")  # prints to file

def fix_man_page_edit_links(root):
    base = os.path.join(root, "manualpages")
    for root, dirs, filenames in os.walk(base):
        for filename in filenames:
            if filename.endswith(".html"):
                filename_full = os.path.join(root, filename)
                url = _get_edit_url(filename_full)
                if url is not None and _check_edit_link(filename_full):
                    _update_edit_link(filename_full, url)
            # remove Show Source line
            with open(filename_full) as f:
                str = f.read()
                newstr = re.sub(SHOW_SOURCE_PATTERN,'',str)
            with open(filename_full,'w') as f:
                f.write(newstr)
