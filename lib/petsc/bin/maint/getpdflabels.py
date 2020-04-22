#!/usr/bin/env python
#!/bin/env python
#
# change python to whatever is needed on your system to invoke python
#
#  Generates the PDF label given to each latex label in the users manual
#
#  Calling sequence
#      | getpdflabels.py
#
from __future__ import print_function
import os
import sys
from sys import *
import re

def main():
#
#  get the entire list of labels, chapters, sections and subsections
#
    lines = sys.stdin.readlines()
#
    regchapter    = re.compile('(chapter{)([^}]*)(})')
    regsection    = re.compile('(section{)([^}]*)(})')
    regsubsection = re.compile('(subsection{)([^}]*)(})')
    reglabel      = re.compile(r'(label{)([^}]*)(})')

    chapter    = 1
    section    = 0
    subsection = 0
    label      = 'chapter.1'
    title      = 'Introduction'

    n = len(lines)

    for i in range(0,n):
      fl = regchapter.search(lines[i])
      if fl:
          chapter = chapter + 1
          section    = 0
          subsection = 0
          label = 'chapter.'+str(chapter)
          title = 'Chapter '+str(chapter)+' '+fl.group(2)
      fl = regsection.search(lines[i])
      if fl:
          section = section + 1
#          print 'section',chapter,section
          subsection = 0
          label = 'section.'+str(chapter)+'.'+str(section)
          title = 'Section '+str(chapter)+'.'+str(section)+' '+fl.group(2)
      fl = regsubsection.search(lines[i])
      if fl:
          subsection = subsection + 1
#          print 'subsection',chapter,section,subsection
          label = 'subsection.'+str(chapter)+'.'+str(section)+'.'+str(subsection)
          title = 'Section '+str(chapter)+'.'+str(section)+'.'+str(subsection)+' '+fl.group(2)
      fl = reglabel.search(lines[i])
      if fl:
#	  print fl.group(2),label,title
          print('man:+'+fl.group(2)+'++'+title+'++++man+../../manual.pdf#'+label)


#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
    main()

