#!/usr/bin/env python
#!/bin/env python

""" Reads in all the generated manual pages, and creates the index
for the manualpages, ordering the indices into sections based
on the 'Level of Difficulty'.

 Usage:
   wwwindex.py PETSC_DIR LOC
"""

import os
import sys
import re
import glob
import posixpath
import subprocess

HLIST_COLUMNS = 3

# Now use the level info, and print a html formatted index
# table. Can also provide a header file, whose contents are
# first copied over.
def printindex(outfilename, headfilename, levels, titles, tables):
      # Read in the header file
      headbuf = ''
      if posixpath.exists(headfilename) :
            with open(headfilename, "r") as fd:
                headbuf = fd.read()
                headbuf = headbuf.replace('PETSC_DIR', '../../../')
      else:
            print('Header file "%s" does not exist' % headfilename)

      with open(outfilename, "w") as fd:
          fd.write(headbuf)
          fd.write('[Manual Pages Table of Contents](/docs/manualpages/index.md)\n')
          all_names = []
          fd.write('\n## Manual Pages by Level\n')
          for i, level in enumerate(levels):
                title = titles[i]
                if not tables[i]:
                      if level != 'none':
                          fd.write('\n### No %s routines\n' % level)
                      continue

                fd.write('\n### %s\n' % title)
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
          fd.write('```{toctree}\n')
          fd.write("---\n")
          fd.write("maxdepth: 1\n")
          fd.write("---\n")
          for name in sorted(all_names):
              fd.write('%s\n' % name)
          fd.write('```\n\n\n')


# This routine takes in as input a dictionary, which contains the
# alhabetical index to all the man page functions, and prints them all in
# a single index page
def printsingleindex(outfilename, alphabet_dict):
      with open(outfilename, "w") as fd:
          fd.write("# Single Index of all PETSc Manual Pages\n\n")
          fd.write(" Also see the [Manual page table of contents, by section](/docs/manualpages/index.md).\n\n")
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
def modifylevel(filename,secname):
      with open(filename, "r") as fd:
          buf = fd.read()

      re_name = re.compile('\*\*Location:\*\*(.*)')  # As defined in myst.def
      m = re_name.search(buf)
      if m:
        loc_html = m.group(1)
        if loc_html:
          git_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip()
          try:
              git_ref_release = subprocess.check_output(['git', 'rev-parse', 'origin/release']).rstrip()
              edit_branch = 'release' if git_ref == git_ref_release else 'main'
          except subprocess.CalledProcessError:
              print("WARNING: checking branch for man page edit links failed")
              edit_branch = 'main'
          pattern = re.compile(r"<A.*>(.*)</A>")
          loc = re.match(pattern, loc_html)
          if loc:
              source_path = loc.group(1)
              buf += "\n\n---\n[Edit on GitLab](https://gitlab.com/petsc/petsc/-/edit/%s/%s)\n\n" % (edit_branch, source_path)
          else:
              print("Warning. Could not find source path in %s" % filename)
      else:
        print('Error! No location in file:', filename)

      re_level = re.compile(r'(Level:)\s+(\w+)')
      m = re_level.search(buf)
      level = 'none'
      if m:
            level = m.group(2)
      else:
            print('Error! No level info in file:', filename)

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
      outbuf = tmpbuf + '\n[Index of all %s routines](index.md)  \n' % secname + '[Table of Contents for all manual pages](/docs/manualpages/index.md)  \n' + '[Index of all manual pages](/docs/manualpages/singleindex.md)  \n'

      # write the modified manpage
      with open(filename, "w") as fd:
          fd.write(outbuf)

      return level

# Go through each manpage file, present in dirname,
# and create and return a table for it, wrt levels specified.
def createtable(dirname,levels,secname):
      mdfiles = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith('.md')]
      mdfiles.sort()
      if mdfiles == []:
            print('Error! Empty directory:',dirname)
            return None

      table = []
      for level in levels: table.append([])

      for filename in mdfiles:
            level = modifylevel(filename,secname)
            if level.lower() in levels:
                  table[levels.index(level.lower())].append(filename)
            else:
                  print('Error! Unknown level \''+ level + '\' in', filename)
      return table

# This routine is called for each man dir. Each time, it
# adds the list of manpages, to the given list, and returns
# the union list.

def addtolist(dirname,singlelist):
      mdfiles = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith('.md')]
      mdfiles.sort()
      if mdfiles == []:
            print('Error! Empty directory:',dirname)
            return None

      for filename in mdfiles:
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


def getallmandirs(dirs):
      """ Gets the list of man* dirs present in the doc dir. Each dir will have an index created for it. """
      mandirs = []
      for filename in dirs:
            path,name = posixpath.split(filename)
            if name == 'RCS' or name == 'sec' or name == "concepts" or name  == "SCCS" : continue
            if posixpath.isdir(filename):
                  mandirs.append(filename)
      return mandirs


def main():
      arg_len = len(sys.argv)

      if arg_len < 3:
            print('Error! Insufficient arguments.')
            print('Usage:', sys.argv[0], 'PETSC_DIR','LOC')
            exit()

      PETSC_DIR = sys.argv[1]
      LOC       = sys.argv[2]
      HEADERDIR = (sys.argv[3] if arg_len > 3 else 'doc/classic/manualpages-sec')
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
            outfilename  = dirname + '/index.md'
            dname,secname  = posixpath.split(dirname)
            headfilename = PETSC_DIR + '/' + HEADERDIR + '/header_' + secname
            table        = createtable(dirname,levels,secname)
            if not table: continue
            singlelist   = addtolist(dirname,singlelist)
            printindex(outfilename,headfilename,levels,titles,table)

      alphabet_dict = createdict(singlelist)
      outfilename   = LOC + '/docs/manualpages/singleindex.md'
      printsingleindex (outfilename,alphabet_dict)


if __name__ == '__main__':
      main()
