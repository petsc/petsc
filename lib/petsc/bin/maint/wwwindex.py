#!/usr/bin/env python
#!/bin/env python
#
# Reads in all the generated manual pages, and Creates the index
# for the manualpages, ordering the indices into sections based
# on the 'Level of Difficulty'
#
#  Usage:
#    wwwindex.py PETSC_DIR LOC
#
from __future__ import print_function
import os
import glob
import posixpath
from sys import *
import subprocess

# This routine reorders the entries int he list in such a way, so that
# When they are printed in a row order, the entries are sorted by columns
# In this subroutine row,col,nrow,ncol correspond to the new layout
# of a given data-value
def maketranspose(data,ncol):
      nrow = (len(data)+ncol-1)//ncol
      newdata = []
      # use the complete nrow by ncol matrix
      for i in range(nrow*ncol):
            newdata.append('')
      for i in range(len(data)):
            col           = i//nrow
            row           = i%nrow
            newi          = row*ncol+col
            newdata[newi] = data[i]
      return newdata

# Now use the level info, and print a html formatted index
# table. Can also provide a header file, whose contents are
# first copied over.
def printindex(outfilename,headfilename,levels,titles,tables):
      # Read in the header file
      headbuf = ''
      if posixpath.exists(headfilename) :
            try:
                  fd = open(headfilename,'r')
            except:
                  print('Error reading file',headfilename)
                  exit()
            headbuf = fd.read()
            headbuf = headbuf.replace('PETSC_DIR','../../../')
            fd.close()
      else:
            print('Header file \'' + headfilename + '\' does not exist')

      # Now open the output file.
      try:
            fd = open(outfilename,'w')
      except:
            print('Error writing to file',outfilename)
            exit()

      # Add the HTML Header info here.
      fd.write(headbuf)
      # Add some HTML separators
      fd.write('\n<P>\n')
      fd.write('<TABLE>\n')
      for i in range(len(levels)):
            level = levels[i]
            title = titles[i]

            if len(tables[i]) == 0:
                  # If no functions in 'None' category, then don't print
                  # this category.
                  if level == 'none':
                        continue
                  else:
                        # If no functions in any other category, then print
                        # the header saying no functions in this cagetory.
                        fd.write('<TR><TD WIDTH=250 COLSPAN="3">')
                        fd.write('<B>' + 'No ' + level +' routines' + '</B>')
                        fd.write('</TD></TR>\n')
                        continue

            fd.write('<TR><TD WIDTH=250 COLSPAN="3">')
            #fd.write('<B>' + upper(title[0])+title[1:] + '</B>')
            fd.write('<B>' + title + '</B>')
            fd.write('</TD></TR>\n')
            # Now make the entries in the table column oriented
            tables[i] = maketranspose(tables[i],3)
            for filename in tables[i]:
                  path,name     = posixpath.split(filename)
                  func_name,ext = posixpath.splitext(name)
                  mesg          = ' <TD WIDTH=250><A HREF="'+ './' + name + '">' + \
                                  func_name + '</A></TD>\n'
                  fd.write(mesg)
                  if tables[i].index(filename) % 3 == 2 : fd.write('<TR>\n')
      fd.write('</TABLE>\n')
      # Add HTML tail info here
      fd.write('<BR><A HREF="../../../docs/manualpages/index.html">Table of Contents</A>\n')
      fd.close()

# This routine takes in as input a dictionary, which contains the
# alhabetical index to all the man page functions, and prints them all in
# a single index page
def printsingleindex(outfilename,alphabet_dict):
      # Now open the output file.
      try:
            fd = open(outfilename,'w')
      except:
            print('Error writing to file',outfilename)
            exit()

      alphabet_index = list(alphabet_dict.keys())
      alphabet_index.sort()

      # Now print each section, beginning with a title
      for key in alphabet_index:

            # Print the HTML tag for this section
            fd.write('<A NAME="' + key + '"></A>\n' )

            # Print the HTML index at the beginning of each section
            fd.write('<H3> <CENTER> | ')
            for key_tmp in alphabet_index:
                  if key == key_tmp:
                        fd.write( '<FONT COLOR="#CC3333">' + key_tmp.upper() + '</FONT> | \n' )
                  else:
                        fd.write('<A HREF="singleindex.html#' + key_tmp + '"> ' + \
                                 key_tmp.upper() + ' </A> | \n')
            fd.write('</CENTER></H3> \n')

            # Now write the table entries
            fd.write('<TABLE>\n')
            fd.write('<TR><TD WIDTH=250 COLSPAN="3">')
            fd.write('</TD></TR>\n')
            function_dict  = alphabet_dict[key]
            function_index = list(function_dict.keys())
            function_index.sort()
            function_index = maketranspose(function_index,3)
            for name in function_index:
                  if name:
                        path_name = function_dict[name]
                  else:
                        path_name = ''
                  mesg = '<TD WIDTH=250><A HREF="'+ './' + path_name + '">' + \
                         name + '</A></TD>\n'
                  fd.write(mesg)
                  if function_index.index(name) %3 == 2: fd.write('<TR>\n')

            fd.write('</TABLE>')

      fd.close()
      return


# Read in the filename contents, and search for the formatted
# String 'Level:' and return the level info.
# Also adds the BOLD HTML format to Level field
def modifylevel(filename,secname):
      import re
      try:
            fd = open(filename,'r')
      except:
            print('Error! Cannot open file:',filename)
            exit()
      buf    = fd.read()
      fd.close()

      re_name = re.compile('<P><B><FONT COLOR="#CC3333">Location:</FONT></B><A HREF=".*">(.*)<')
      m = re_name.search(buf)
      if m:
        loc =m.group(1)
        if loc:
          re_loc = re.compile('<BODY .*>')
          git_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip()
          git_ref_release = subprocess.check_output(['git', 'rev-parse', 'origin/release']).rstrip()
          edit_branch = 'release' if git_ref == git_ref_release else 'main'
          replacementtext = '<BODY BGCOLOR="FFFFFF">\n<div id="edit" align=right><a href="https://gitlab.com/petsc/petsc/-/edit/'+edit_branch+'/'+loc+'"><small>Fix/Edit manual page</small></a></div>'
          buf = re_loc.sub(replacementtext,buf)
      else:
        print('Error! No location in file:', filename)

      re_level = re.compile(r'(Level:)\s+(\w+)')
      m = re_level.search(buf)
      level = 'none'
      if m:
            level = m.group(2)
      else:
            print('Error! No level info in file:', filename)

      # Now takeout the level info, and move it to the end,
      # and also add the bold format.
      tmpbuf = re_level.sub('',buf)
      re_loc = re.compile('(<FONT COLOR="#CC3333">Location:</FONT>)')
      tmpbuf = re_loc.sub('</B><H3><FONT COLOR="#CC3333">Level</FONT></H3>' + level + r'<BR>\n<H3><FONT COLOR="#CC3333">Location</FONT></H3>\n',tmpbuf)

      # Modify .c#,.h#,.cu#,.cxx# to .c.html#,.h.html#,.cu.html#,.cxx.html#
      re_loc = re.compile('.c#')
      tmpbuf = re_loc.sub('.c.html#',tmpbuf)
      re_loc = re.compile('.h#')
      tmpbuf = re_loc.sub('.h.html#',tmpbuf)
      re_loc = re.compile('.cu#')
      tmpbuf = re_loc.sub('.cu.html#',tmpbuf)
      re_loc = re.compile('.cxx#')
      tmpbuf = re_loc.sub('.cxx.html#',tmpbuf)

      re_loc = re.compile('</BODY></HTML>')
      outbuf = re_loc.sub('<BR><A HREF="./index.html">Index of all ' + secname + ' routines</A>\n<BR><A HREF="../../../docs/manualpages/index.html">Table of Contents for all manual pages</A>\n<BR><A HREF="../singleindex.html">Index of all manual pages</A>\n</BODY></HTML>',tmpbuf)

      re_loc = re.compile(r' (http://[A-Za-z09_\(\)\./]*)[ \n]')
      outbuf = re_loc.sub(' <a href="\\1">\\1 </a> ',outbuf)

      # write the modified manpage
      try:
            #fd = open(filename[:-1],'w')
            fd = open(filename,'w')
      except:
            print('Error! Cannot write to file:',filename)
            exit()
      fd.write(outbuf)
      fd.close()
      return level

# Go through each manpage file, present in dirname,
# and create and return a table for it, wrt levels specified.
def createtable(dirname,levels,secname):
      htmlfiles = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith('.html')]
      htmlfiles.sort()
      if htmlfiles == []:
            print('Error! Empty directory:',dirname)
            return None

      table = []
      for level in levels: table.append([])

      for filename in htmlfiles:
            level = modifylevel(filename,secname)
            #if not level: continue
            if level.lower() in levels:
                  table[levels.index(level.lower())].append(filename)
            else:
                  print('Error! Unknown level \''+ level + '\' in', filename)
      return table

# This routine is called for each man dir. Each time, it
# adds the list of manpages, to the given list, and returns
# the union list.

def addtolist(dirname,singlelist):
      htmlfiles = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith('.html')]
      htmlfiles.sort()
      if htmlfiles == []:
            print('Error! Empty directory:',dirname)
            return None

      for filename in htmlfiles:
            singlelist.append(filename)

      return singlelist

# This routine creates a dictionary, with entries such that each
# key is the alphabet, and the vaue corresponds to this key is a dictionary
# of FunctionName/PathToFile Pair.

def createdict(singlelist):

      newdict = {}
      for filename in singlelist:
            path,name     = posixpath.split(filename)
            # grab the short path Mat from /wired/path/Mat
            junk,path     = posixpath.split(path)
            index_char    = name[0:1].lower()
            # remove the .name suffix from name
            func_name,ext = posixpath.splitext(name)
            if index_char not in newdict:
                  newdict[index_char] = {}
            newdict[index_char][func_name] = path + '/' + name

      return newdict


# Gets the list of man* dirs present in the doc dir.
# Each dir will have an index created for it.
def getallmandirs(dirs):
      mandirs = []
      for filename in dirs:
            path,name = posixpath.split(filename)
            if name == 'RCS' or name == 'sec' or name == "concepts" or name  == "SCCS" : continue
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs


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
      #fd        = os.popen('/bin/ls -d '+ PETSC_DIR + '/docs/manualpages/*')
      #buf       = fd.read()
      #dirs      = split(strip(buf),'\n')
      dirs      = glob.glob(LOC + '/docs/manualpages/*')
      mandirs   = getallmandirs(dirs)

      levels = ['beginner','intermediate','advanced','developer','deprecated','none']
      titles = ['Beginner - Basic usage',
                'Intermediate - Setting options for algorithms and data structures',
                'Advanced - Setting more advanced options and customization',
                'Developer - Interfaces intended primarily for library developers, not for typical applications programmers',
                'Deprecated - Functionality scheduled for removal in future versions',
                'None: Not yet cataloged']

      singlelist = []
      for dirname in mandirs:
            outfilename  = dirname + '/index.html'
            dname,secname  = posixpath.split(dirname)
            headfilename = PETSC_DIR + '/src/docs/manualpages-sec/header_' + secname
            table        = createtable(dirname,levels,secname)
            if not table: continue
            singlelist   = addtolist(dirname,singlelist)
            printindex(outfilename,headfilename,levels,titles,table)

      alphabet_dict = createdict(singlelist)
      outfilename   = LOC + '/docs/manualpages' + '/singleindex.html'
      printsingleindex (outfilename,alphabet_dict)

# The classes in this file can also
# be used in other python-programs by using 'import'
if __name__ ==  '__main__':
      main()

