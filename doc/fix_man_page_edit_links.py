#!/usr/bin/env python3
""" A brittle approach to making "Edit this Page" links work on man pages """

import os
import re
import fileinput

EDIT_URL_PATTERN = re.compile(r'<p><a.*href="(.*)">Edit on GitLab</a></p>')  # very brittle
SHOW_SOURCE_PATTERN = re.compile(r'(?s)<div class="toc-item">\s*<div class="tocsection sourcelink">.*<a href=.*>.*<i class="fa-solid fa-file-lines"></i> Show Source\s*</a>\s*</div>\s*</div>')

def fix_man_page_edit_link(root,filename):
   with open(os.path.join(root,filename)) as f:
     str = f.read()
   m = re.findall(EDIT_URL_PATTERN, str)
   if not m:
     # print("Cannot find Edit on Gitlab string "+os.path.join(root,filename))
     return
   url = m[0]
   str = re.sub(SHOW_SOURCE_PATTERN,'',str)
   replace_link_line = False
   done = False
   with open(os.path.join(root,filename),'w') as f:
     for line in str.split('\n'):
       if done:
         f.write(line+'\n')
       else:
         if line.lstrip().startswith("<div") and "editthispage" in line:
           replace_link_line = True
         if replace_link_line and line.lstrip().startswith("<a"):
           f.write("<a href=%s>\n" % url)
           done = True
         elif not 'Edit on GitLab' in line:
           f.write(line+'\n')

def fix_man_page_edit_links(root):
    base = os.path.join(root, "manualpages")
    for root, dirs, filenames in os.walk(base):
        for filename in filenames:
            if filename.endswith(".html"):
              fix_man_page_edit_link(root,filename)
