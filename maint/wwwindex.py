#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: wwwindex.py,v 1.1 1999/01/19 16:33:06 balay Exp balay $ 
#
# Reads in all the generated manual pages, and Creates the index
# for the manualpages, ordering the indices into sections based
# on the 'Level of Difficulty'
#
#  Usage:
#    wwwindex.py PETSC_DIR
#
import os
import posixpath
import regsub
from exceptions import *
from sys import *
from string import *

# Now use the level info, and print a html formatted index
# table
def printindex(outfilename,levels,tables):
      try:
            fd = open(outfilename,'w')
      except:
            print 'Error writing to file',outfilename
            exit()
      # Add the HTML Header info here. None?
      fd.write('<TABLE>')
      for i in range(len(levels)):
            level = levels[i]
            fd.write('</TR><TD>')
            fd.write('<B>'+ upper(level[0]) + level[1:] + '</B>')
            fd.write('</TD></TR>')
            for filename in tables[i]:
                  path,name     = posixpath.split(filename)
                  func_name,ext = posixpath.splitext(name)
                  rel_dir       = split(path,'/')[-1]
                  mesg          = '<TD WIDTH=250><A HREF="' + rel_dir + '/' + name + '">' + \
                                  func_name + '</A></TD>'
                  fd.write(mesg)
                  if tables[i].index(filename) % 3 == 2 : fd.write('<TR>')
      fd.write('</TABLE>')
      # Add HTML tail info here
      fd.write('<BR><A HREF="manualpages.html"><IMG SRC="up.xbm">Table of Contents</A>')
      fd.close()

# Add the BOLD HTML format to Level field, and write the file
def writeupdatedfile(filename,buf):
      outbuf = regsub.sub('\nLevel:','\n<B>Level:</B>',buf)
      try:
            fd = open(filename[:-1],'w')
      except:
            print 'Error! Cannot write to file:',filename[:-1]
            exit()            
      fd.write(outbuf)
      fd.close()
      
# Read in the filename contents, and search for the formatted
# String 'Level:' and return the level info.
# Also adds the BOLD HTML format to Level field
def extractlevel(filename):
      try:
            fd = open(filename,'r')
      except:
            print 'Error! Cannot open file:',filename
            exit()
      buf    = fd.read()
      fd.close()
      writeupdatedfile(filename,buf)

      lines = split(buf,'\n')
      for i in range(len(lines)):
            line = lines[i]
            if strip(line) == 'Level:':
                  # The next line has the level info
                  level = strip(lines[i+1])
                  return level
      print 'Error! No level info in',filename

def makeboldlevel(filename):
      try:
            fd = open(filename,'r')
      except:
            print 'Error! Cannot open file:',filename
            exit()
      lines = split(fd.read(),'\n')
      for i in range(len(lines)):
            line = lines[i]
            if strip(line) == 'Level:':
                  # The next line has the level info
                  level = strip(lines[i+1])
                  return level
      print 'Error! No level info in',filename
  
      
      
# Go through each manpage file, present in dirname,
# and create and return a table for it, wrt levels specified.
def createtable(dirname,levels):
      fd = os.popen('ls '+ dirname + '/*.html')
      buf = fd.read()
      if buf == '':
            print 'Error! Empty directory:',dirname
            return None

      table = []
      for level in levels: table.append([])
      for filename in split(strip(buf),'\n'):
            level = extractlevel(filename)
            if not level: continue
            if level in levels:
                  table[levels.index(level)].append(filename)
            else:
                  print 'Error! Unknown level \''+ level + '\' in', filename
      return table
      
# Gets the list of man* dirs present in the doc dir.
# Each dir will have an index created for it.
def getallmandirs(buf):
      mandirs = []
      for filename in split(strip(buf),'\n'):
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs

# Extracts PETSC_DIR from the command line and
# starts genrating index for all the manpages.
def main():
      arg_len = len(argv)
      
      if arg_len < 2: 
            print 'Error! Insufficient arguments.'
            print 'Usage:', argv[0], 'PETSC_DIR'
            exit()

      PETSC_DIR = argv[1]
      fd        = os.popen('ls -d '+ PETSC_DIR + '/docs/manualpages/man*')
      buf       = fd.read()
      mandirs   = getallmandirs(buf)

      levels =['beginner','intermediate','advanced','developer']
      for dirname in mandirs:
            table       = createtable(dirname,levels)
            if not table: continue
            outfilename = dirname + '.html'
            printindex(outfilename,levels,table)


# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
    
