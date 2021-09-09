#! /usr/bin/env python
#!/bin/env python
#
# reads in docs/exampleconcepts and creates the file help.html
#
#
#  Usage:
#    helpindex.py
#
from __future__ import print_function
import os
import glob
import posixpath
from sys import *

# dict[prim_key][sec_key][link_title] = filename

def gethelpstring(PETSC_DIR,filename):
      filename = PETSC_DIR + '/' + filename
      fd = open(filename,'r')

      for line in fd.readlines():
            tmp = line.find('help[]')
            if not tmp == -1:
                  tmp1 = line.find('"')
                  tmp2 = line.find('.')
                  if tmp1 == -1 or tmp2 == -1:
                        return "PetscNoHelp"
                  buf = line.split('"')[1].strip()
                  buf = buf.split('.')[0].strip()
                  fd.close()
                  return buf
      return "PetscNoHelp"



# Scan and extract format information from each line
def updatedata(PETSC_DIR,dict,line):
      # The first filed is the name of the file which will be used for link
      filename     = line.split()[0]
      concept_list = ' '.join(line.split()[1:])
      concept_list = concept_list.strip()

      #check for a man page - html suffix
      if os.path.splitext(filename)[-1] == '.html':
            link_title = os.path.splitext(os.path.basename(filename))[0]
      else:
            # should be an example file
            help_str = gethelpstring(PETSC_DIR,filename)
            if not help_str == "PetscNoHelp":
                  link_title = help_str
            else:
                  link_title = (filename.replace('src/', '')
                    .replace('examples/', '')
                    .replace('tutorials/', ''))

      # ';' is a field separator
      keys = concept_list.split(";")
      for key in keys:
            # '^' is a subsection separator
            prim_sub_keys = key.split("^")
            if len(prim_sub_keys) == 1:
                  prim_key = prim_sub_keys[0]
                  sub_key  = "PetscNoKey"
            elif len(prim_sub_keys) == 2:
                  prim_key = prim_sub_keys[0]
                  sub_key  = prim_sub_keys[1]
            else:
                  prim_key = prim_sub_keys[0]
                  sub_key  = prim_sub_keys[1]
                  print("Error! more than 2 levels if keys specified "  + filename)
                  print(line)
            prim_key = prim_key.strip()
            sub_key  = sub_key.strip()
            if prim_key == '':
                  continue
            if prim_key not in dict:
                  dict[prim_key] = {}  #use dict to store sub_key
            if sub_key not in dict[prim_key]:
                  dict[prim_key][sub_key] = {}
            dict[prim_key][sub_key][link_title] = filename

# print the dict in html format
def printdata(LOC,fd,dict):

      # Put some  HTML Header
      fd.write("<HTML>\n")
      fd.write("<TITLE>Concepts_File</TITLE>\n")
      fd.write("<!-- Created by helpindex.py -->\n");
      fd.write("<BODY>\n")

      # Put the Table Header
      fd.write("<H1><center> PETSc Help Index</center></H1>\n")

      prim_keys = sorted(dict.keys(), key=str.lower)

      alphabet_index = {}
      for key in prim_keys:
            alphabet_index[key[0].upper()] = 'junk'
      alphabet_keys = sorted(alphabet_index.keys())

      a_key_tmp = ''
      for prim_key in prim_keys:
            # First check and print the alphabet index
            a_key = prim_key[0].upper()
            if not a_key == a_key_tmp:
                  a_key_tmp = a_key
                  # Print the HTML tag for this section
                  fd.write('<A NAME="' + a_key + '"></A>\n' )
                  # Print the HTML index at the beginning of each section
                  fd.write('<H3> <CENTER> | ')
                  for key_tmp in alphabet_keys:
                        if a_key == key_tmp:
                              fd.write( '<FONT COLOR="#CC3333">' + key_tmp.upper() + '</FONT> | \n' )
                        else:
                              fd.write('<A HREF="help.html#' + key_tmp + '"> ' + \
                                       key_tmp.upper() + ' </A> | \n')
                  fd.write('</CENTER></H3> \n')

            # Precheck the sub_keys so that if it has 'PetscNoKey', then start printing
            # the filename (data) in the same column
            sub_keys = sorted(dict[prim_key].keys(), key=str.lower)
            # Now move 'PetscNoKey' to the beginning of this list
            if not sub_keys.count('PetscNoKey') == 0:
                  sub_keys.remove('PetscNoKey')
                  sub_keys.insert(0,'PetscNoKey')

            if  sub_keys[0] == 'PetscNoKey':
                  link_names = sorted(dict[prim_key]['PetscNoKey'].keys(), key=str.lower)
                  link_name  = link_names[0]
                  filename = dict[prim_key]['PetscNoKey'][link_name]
                  del dict[prim_key]['PetscNoKey'][link_name]

                  temp = "<A HREF=\"" + "../../" + filename + ".html\">" + link_name + "</A>"
                  fd.write("<TABLE>\n")
                  fd.write("<TD WIDTH=4 ><BR></TD>")
                  fd.write("<TD WIDTH=260 ><B><FONT SIZE=4>")
                  # If prim_key exists in the concepts directory,
                  # create a link to it.
                  concept_filename = prim_key.lower().replace(' ', '_')
                  concept_filename = "concepts/" + concept_filename + ".html"

                  #if os.access(concept_filename,os.F_OK):
                  fd_tmp = os.popen('ls '+ LOC + '/docs/manualpages/'+ concept_filename)
                  buf = fd_tmp.read()
                  if not buf == '':
                        fd.write("<A HREF=\"")
                        fd.write(concept_filename)
                        fd.write("\">")
                        fd.write(prim_key)
                        fd.write("</A>")
                  else:
                        fd.write(prim_key)
                  fd.write("</FONT></B></TD>")
                  fd.write("<TD WIDTH=500>")
                  fd.write(temp)
                  fd.write("</TD>")

                  fd.write("</TR>\n")
                  fd.write("</TABLE>\n")
            else:
                  fd.write("<TABLE>")
                  fd.write("<TD WIDTH=4 ><BR></TD>")
                  fd.write("<TD WIDTH=300 ><B><FONT SIZE=4>")
                  # If prim_key exists in the concepts directory,
                  # create a link to it.
                  concept_filename = prim_key.lower().replace(' ', '_')
                  concept_filename = "concepts/" + concept_filename + ".html"

                  #if os.access(concept_filename,os.F_OK):
                  fd_tmp = os.popen('ls '+ LOC + '/docs/manualpages/' + concept_filename)
                  buf = fd_tmp.read()
                  if not buf == '':
                        fd.write("<A HREF=\"")
                        fd.write(concept_filename)
                        fd.write("\">")
                        fd.write(prim_key)
                        fd.write("</A>")
                  else:
                        fd.write(prim_key)
                  fd.write("</FONT></B></TD>")
                  fd.write("</TR>\n")
                  fd.write("</TABLE>\n")



            for sub_key in sub_keys:
                  link_names = sorted(dict[prim_key][sub_key].keys(), key=str.lower)

                  if not sub_key == 'PetscNoKey':
                        # Extract the first element from link_names
                        link_name = link_names[0]
                        link_names = link_names[1:]
                        filename = dict[prim_key][sub_key][link_name]
                        temp = "<A HREF=\"" + "../../" + filename + ".html\">" + link_name + "</A>"
                        fd.write("<TABLE>")
                        fd.write("<TD WIDTH=60 ><BR></TD>")
                        fd.write("<TD WIDTH=205><FONT COLOR=\"#CC3333\"><B>")
                        fd.write(sub_key)
                        fd.write("</B></FONT></TD>")
                        fd.write("<TD WIDTH=500 >")
                        fd.write(temp)
                        fd.write("</TD>")
                        fd.write("</TR>\n")
                        fd.write("</TABLE>\n")

                  for link_name in link_names:
                        filename = dict[prim_key][sub_key][link_name]
                        temp = "<A HREF=\"" + "../../" + filename + ".html\">" + link_name + "</A>"
                        fd.write("<TABLE>")
                        fd.write("<TD WIDTH=270><BR></TD>")
                        fd.write("<TD WIDTH=500>")
                        fd.write(temp)
                        fd.write("</TD>")
                        fd.write("</TR>\n")
                        fd.write("</TABLE>\n")

      # HTML Tail
      fd.write("</BODY>")
      fd.write("</HTML>")

# Extracts PETSC_DIR from the command line and
# starts genrating index for all the manpages.
def main():

      arg_len = len(argv)

      if arg_len < 3:
            print('Error! Insufficient arguments.')
            print('Usage:', argv[0], 'PETSC_DIR','LOC')
            exit()

      PETSC_DIR = argv[1]
      LOC       = argv[2]

      dict = {}

      # open and read in the input files
      fd1 = open( LOC + '/docs/exampleconcepts','r')

      for line in fd1.readlines():
            updatedata(PETSC_DIR,dict,line.strip())

      fd1.close()
      #make sure there is no problem re-writing to this file
      os.system('/bin/rm -f ' + LOC + '/docs/manualpages/help.html')
      fd = open( LOC + '/docs/manualpages/help.html','w')

      printdata(LOC,fd,dict)
      fd.close()

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
      main()

