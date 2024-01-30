#!/usr/bin/env python
#!/bin/env python

""" Reads in all the generated manual pages, and creates the index
for the manualpages, ordering the indices into sections based
on the 'Level of Difficulty'.
"""

import os
import sys
import re
import glob
import posixpath
import subprocess

numberErrors = 0
HLIST_COLUMNS = 3

# Read an optional header file, whose contents are first copied over
# Use the level info, and print a formatted index table of all the manual pages
#
def printindex(outfilename, headfilename, levels, titles, tables):
      global numberErrors
      # Read in the header file
      headbuf = ''
      if posixpath.exists(headfilename) :
            with open(headfilename, "r") as fd:
                headbuf = fd.read()
                headbuf = headbuf.replace('PETSC_DIR', '../../../')
      else:
            print('Error! SUBMANSEC header file "%s" does not exist' % headfilename)
            print('Likley you introduced a new set of manual pages but did not add the header file that describes them')
            numberErrors = numberErrors + 1

      with open(outfilename, "w") as fd:
          # Since it uses three columns we must remove right sidebar so all columns are displayed completely
          # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/page-toc.html
          fd.write(':html_theme.sidebar_secondary.remove: true\n')
          fd.write(headbuf)
          fd.write('\n')
          all_names = []
          for i, level in enumerate(levels):
                title = titles[i]
                if not tables[i]:
                      if level != 'none' and level != 'deprecated':
                          fd.write('\n## No %s routines\n' % level)
                      continue

                fd.write('\n## %s\n' % title)
                fd.write('```{hlist}\n')
                fd.write("---\n")
                fd.write("columns: %d\n" % HLIST_COLUMNS)
                fd.write("---\n")

                for filename in tables[i]:
                      path,name     = posixpath.split(filename)
                      func_name,ext = posixpath.splitext(name)
                      fd.write('- [](%s)\n' % name)
                      all_names.append(name)
                fd.write('```\n\n\n')

          fd.write('\n## Single list of manual pages\n')
          fd.write('```{hlist}\n')
          fd.write("---\n")
          fd.write("columns: %d\n" % HLIST_COLUMNS)
          fd.write("---\n")
          for name in sorted(all_names):
              fd.write('- [](%s)\n' % name)
          fd.write('```\n\n\n')


# This routine takes in as input a dictionary, which contains the
# alhabetical index to all the man page functions, and prints them all in
# a single index page
def printsingleindex(outfilename, alphabet_dict):
      global numberErrors
      with open(outfilename, "w") as fd:
          fd.write("# Single Index of all PETSc Manual Pages\n\n")
          fd.write(" Also see the [Manual page table of contents, by section](/manualpages/index.rst).\n\n")
          for key in sorted(alphabet_dict.keys()):
                fd.write("## %s\n\n" % key.upper())
                fd.write("```{hlist}\n")
                fd.write("---\n")
                fd.write("columns: %d\n" % HLIST_COLUMNS)
                fd.write("---\n")
                function_dict = alphabet_dict[key]
                for name in sorted(function_dict.keys()):
                      if name:
                            path_name = function_dict[name]
                      else:
                            path_name = ''
                      fd.write("- [%s](%s)\n" % (name, path_name))
                fd.write("```\n")


# Read in the filename contents, and search for the formatted
# String 'Level:' and return the level info.
# Also adds the BOLD HTML format to Level field
def modifylevel(filename,secname,edit_branch):
      global numberErrors
      with open(filename, "r") as fd:
          buf = fd.read()

      re_name = re.compile('\*\*Location:\*\*(.*)')  # As defined in myst.def
      m = re_name.search(buf)
      if m:
        loc_html = m.group(1)
        if loc_html:
          pattern = re.compile(r"<A.*>(.*)</A>")
          loc = re.match(pattern, loc_html)
          if loc:
              source_path = loc.group(1)
              buf += "\n\n---\n[Edit on GitLab](https://gitlab.com/petsc/petsc/-/edit/%s/%s)\n\n" % (edit_branch, source_path)
          else:
              print("Warning. Could not find source path in %s" % filename)
      else:
        print('Error! No location in file:', filename)
        numberErrors = numberErrors + 1

      re_level = re.compile(r'(Level:)\s+(\w+)')
      m = re_level.search(buf)
      level = 'none'
      if m:
            level = m.group(2)
      else:
            print('Error! No level info in file:', filename)
            numberErrors = numberErrors + 1

      # Reformat level and location
      tmpbuf = re_level.sub('',buf)
      re_loc = re.compile('(\*\*Location:\*\*)')
      tmpbuf = re_loc.sub('\n## Level\n' + level + '\n\n## Location\n',tmpbuf)

      # Modify .c#,.h#,.cu#,.cxx# to .c.html#,.h.html#,.cu.html#,.cxx.html#
      tmpbuf = re.sub('.c#', '.c.html#', tmpbuf)
      tmpbuf = re.sub('.h#', '.h.html#', tmpbuf)
      tmpbuf = re.sub('.cu#', '.cu.html#', tmpbuf)
      tmpbuf = re.sub('.cxx#', '.cxx.html#', tmpbuf)

      # Add footer links
      outbuf = tmpbuf + '\n[Index of all %s routines](index.md)  \n' % secname + '[Table of Contents for all manual pages](/manualpages/index.rst)  \n' + '[Index of all manual pages](/manualpages/singleindex.md)  \n'

      # write the modified manpage
      with open(filename, "w") as fd:
          fd.write(':orphan:\n'+outbuf)

      return level

# Go through each manpage file, present in dirname,
# and create and return a table for it, wrt levels specified.
def createtable(dirname,levels,secname,editbranch):
      global numberErrors
      listdir =  os.listdir(dirname)
      mdfiles = [os.path.join(dirname,f) for f in listdir if f.endswith('.md')]
      mdfiles.sort()
      if mdfiles == []:
            return None

      table = []
      for level in levels: table.append([])

      for filename in mdfiles:
            level = modifylevel(filename,secname,editbranch)
            if level.lower() in levels:
                  table[levels.index(level.lower())].append(filename)
            else:
                  print('Error! Unknown level \''+ level + '\' in', filename)
                  numberErrors = numberErrors + 1
      return table

# This routine is called for each man dir. Each time, it
# adds the list of manpages, to the given list, and returns
# the union list.

def addtolist(dirname,singlelist):
      global numberErrors
      mdfiles = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith('.md')]
      mdfiles.sort()
      if mdfiles == []:
            print('Error! Empty directory:',dirname)
            numberErrors = numberErrors + 1
            return None

      singlelist.extend(mdfiles)

      return singlelist

# This routine creates a dictionary, with entries such that each
# key is the alphabet, and the vaue corresponds to this key is a dictionary
# of FunctionName/PathToFile Pair.
def createdict(singlelist):
      global numberErrors
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


def getallmandirs(dirs):
      """ Gets the list of man* dirs present in the doc dir. Each dir will have an index created for it. """
      global numberErrors
      mandirs = []
      for filename in dirs:
            path,name = posixpath.split(filename)
            if name == 'RCS' or name == 'sec' or name == "concepts" or name  == "SCCS" : continue
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs


def main(PETSC_DIR):
      global numberErrors
      HEADERDIR = 'doc/manualpages/MANSECHeaders'
      dirs      = glob.glob(os.path.join(PETSC_DIR,'doc','manualpages','*'))
      mandirs   = getallmandirs(dirs)

      levels = ['beginner','intermediate','advanced','developer','deprecated','none']
      titles = ['Beginner - Basic usage',
                'Intermediate - Setting options for algorithms and data structures',
                'Advanced - Setting more advanced options and customization',
                'Developer - Interfaces rarely needed by applications programmers',
                'Deprecated - Functionality scheduled for removal in the future',
                'None: Not yet cataloged']

      singlelist = []
      git_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip()
      try:
        git_ref_release = subprocess.check_output(['git', 'rev-parse', 'origin/release']).rstrip()
        edit_branch = 'release' if git_ref == git_ref_release else 'main'
      except subprocess.CalledProcessError:
        print("WARNING: checking branch for man page edit links failed")
        numberErrors = numberErrors + 1
        edit_branch = 'main'

      for dirname in mandirs:
            outfilename  = dirname + '/index.md'
            dname,secname  = posixpath.split(dirname)
            headfilename = PETSC_DIR + '/' + HEADERDIR + '/' + secname
            table        = createtable(dirname,levels,secname,edit_branch)
            if not table: continue
            singlelist   = addtolist(dirname,singlelist)
            printindex(outfilename,headfilename,levels,titles,table)

      alphabet_dict = createdict(singlelist)
      outfilename   = os.path.join(PETSC_DIR,'doc','manualpages','singleindex.md')
      printsingleindex (outfilename,alphabet_dict)
      if numberErrors:
        raise RuntimeError('Stopping document build since errors were detected in generating manual page indices')

if __name__ == '__main__':
      main(os.path.abspath(os.environ['PETSC_DIR']))
