#!/usr/bin/env python

import sys
import re
import codecs
import os
import subprocess

encoding = 'utf8'

def comment_finder(text):
    pattern = re.compile( r'//.*?$|/\*.*?\*/', re.DOTALL | re.MULTILINE)
    result = pattern.findall(text)
    return result

def print_command(filename):
    codefile = codecs.open(filename, 'r', encoding)
    lines = codefile.read()
    codefile.close()
    list_of_comments = comment_finder(lines)
    for comment in list_of_comments:
        if comment[0:2] == "//":
                comment_to_write = comment[2:]
        else:
            comment_to_write = comment[2:-2]
        if comment_to_write.endswith("\r"):
            comment_to_write = comment_to_write[0:-1]
        if len(comment_to_write) != 0:
            print(comment_to_write)

if __name__ == "__main__":
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).splitlines()
    files = subprocess.check_output(["git", "ls-files", "--full-name", "../../.."]).splitlines()
    for name in files:
        decoded = os.path.join(root[0].decode("utf-8"), name.decode("utf-8"))
        if decoded.endswith(('.c', '.h', '.cxx')):
            print_command(decoded)
