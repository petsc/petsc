#!/usr/bin/env python1.5
#!/bin/env python1.5
# $Id: concepts.py,v 1.1 2000/09/22 00:43:20 balay Exp balay $ 
# 
# reads in docs/tex/exampleconcepts,manconcepts, and create
# the file help.html
# 
#
#  Usage:
#    concepts.py
#
import os
import glob
import posixpath
from exceptions import *
from sys import *
from string import *

# dict[prim_key][sec_key][link_title] = filename

def comptxt(a,b):
      a = lower(a)
      b = lower(b)
      return cmp(a,b)
      

# Scan and extract format information from each line
def updatedata(dict,line):
      # The first filed is the name of the file which will be used for link
      filename     = split(line," ")[0]
      concept_list = join(split(line," ")[1:]," ")
      concept_list = strip(concept_list)

      #check for a man page - html suffix
      if split(filename,'.')[-1] == 'html':
            link_title = split(split(filename,'/')[-1],'.')[0]
      else:
            link_title = filename
      
      # ';' is a field separator
      keys = split(concept_list,";")
      for key in keys:
            # '^' is a subsection separator
            prim_sub_keys = split(key,"^")
            if len(prim_sub_keys) == 1:
                  prim_key = prim_sub_keys[0]
                  sub_key  = "PetscNoKey"
            elif len(prim_sub_keys) == 2:
                  prim_key = prim_sub_keys[0]
                  sub_key  = prim_sub_keys[1]
            else:
                  prim_key = prim_sub_keys[0]
                  sub_key  = prim_sub_keys[1]
                  print "Error! more than 2 levels if keys specified "  + filename
                  print line
            prim_key = strip(prim_key)
            sub_key  = strip(sub_key)
            if prim_key == '':
                  continue
            if not dict.has_key(prim_key):
                  dict[prim_key] = {}  #use dict to store sub_key
            if not dict[prim_key].has_key(sub_key):
                  dict[prim_key][sub_key] = {}
            dict[prim_key][sub_key][link_title] = filename

# print the dict in html format
def printdata(fd,dict):

      # Put some  HTML Header 
      fd.write("<HTML>")
      fd.write("<TITLE>Concepts_File</TITLE>")
      fd.write("<BODY>")
      
      # Put the Table Header
      fd.write("<H1><center> PETSc Help Index</center></H1>")
    
      prim_keys = dict.keys()
      prim_keys.sort(comptxt)

      for prim_key in prim_keys:
            fd.write("<TABLE>")
            fd.write("<TD WIDTH=4 ><BR></TD>")
            fd.write("<TD WIDTH=1000 ><B><H3>")
            fd.write(prim_key)
            fd.write("<H3></B></TD>")
            fd.write("</TR>")
            fd.write("</TABLE>")

            sub_keys = dict[prim_key].keys()
            sub_keys.sort(comptxt)

            for sub_key in sub_keys:
                  if not sub_key == 'PetscNoKey':
                        fd.write("<TABLE>")
                        fd.write("<TD WIDTH=60 ><BR></TD>")
                        fd.write("<TD WIDTH=1000><FONT COLOR=\"#CC3333\"><B>")
                        fd.write(sub_key)
                        fd.write("</B></FONT></TD>")
                        fd.write("</TR>")
                        fd.write("</TABLE>")

                  link_names = dict[prim_key][sub_key].keys()
                  link_names.sort(comptxt)
                  for link_name in link_names:
                        filename = dict[prim_key][sub_key][link_name]
                        temp = "<A HREF=\"" + "../../" + filename + "\">" + link_name + "</A>"
                        fd.write("<TABLE>")
                        fd.write("<TD WIDTH=192 ><BR></TD>")
                        fd.write("<TD WIDTH=300 >")
                        fd.write(temp)
                        fd.write("</TD>")
                        fd.write("</TR>")
                        fd.write("</TABLE>")

      # HTML Tail
      fd.write("</BODY>")
      fd.write("</HTML>")

# Extracts PETSC_DIR from the command line and
# starts genrating index for all the manpages.
def main():

      PETSC_DIR = '/home/bsmith/petsc'
      dict = {}

      # open and read in the input files
      fd1 = open( PETSC_DIR + '/docs/tex/exampleconcepts','r')
      fd2 = open( PETSC_DIR + '/docs/tex/manconcepts','r')

      for line in fd1.readlines():
            updatedata(dict,strip(line))
      for line in fd2.readlines():
            updatedata(dict,strip(line))

      fd1.close()
      fd2.close()

      fd = open( PETSC_DIR + '/docs/manualpages/help.html','w')
      printdata(fd,dict)
      fd.close()
      
# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__': 
      main()
    
