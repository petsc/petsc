#!/usr/bin/env python
""" Runs c2html and mapnames on a single source file and cleans up all text"""

import os
import re
import subprocess
import pathlib
import sys

def main(petsc_dir,loc,git_sha,c2html,mapnames,rel_dir,file):
  with open(os.path.join(rel_dir,file), "r") as fd:
    txt = fd.read()

  # TODO change text processing parts to Python
  cmd = 'sed -E "s/PETSC[A-Z]*_DLLEXPORT//g"  | '+ \
         c2html + ' -n | \
         awk \'{ sub(/<pre width="80">/,"<pre width=\"80\">\\n"); print }\' | \
         grep -E -v "(PetscValid|#if !defined\(__|#define __|#undef __|EXTERN_C )" | ' + \
         mapnames + ' -map htmlmap.tmp -inhtml'
  txt = subprocess.check_output(cmd, text=True, input=txt, shell = True)

  # make the links to manual pages relative
  rel_dot = '../'
  for c in rel_dir:
    if c == '/':
      rel_dot = rel_dot + '../'
  txt = txt.replace('HTML_ROOT/',rel_dot)

  # make the links to include files relative
  ntxt = ''
  for line in txt.split('\n'):
    if 'include' in line:
      ins = re.search('#include [ ]*&lt;',line)
      if ins:
        includename = line[ins.end():re.search('&gt;[a-zA-Z0-9/<>#*"=. ]*',line).start()]
        ln = re.search('<a name="line[0-9]*">[ 0-9]*: </a>',line)
        linenumber = line[ln.start():ln.end()]
        if os.path.isfile(includename):
          line = linenumber+'#include <A href="'+includename+'.html">&lt;'+includename+'&gt;</A>'
        elif os.path.isfile(os.path.join('include',includename)):
          line = linenumber+'#include <A href="'+os.path.relpath(os.path.join(rel_dot,'include',includename))+'.html">&lt;'+includename+'&gt;</A>'
        elif os.path.isfile(os.path.join(includename)):
          line = linenumber+'#include <A href="'+os.path.relpath(os.path.join(rel_dot,includename))+'.html">&lt;'+includename+'&gt;</A>'
    ntxt = ntxt + line + '\n'

  with open(os.path.join(loc,rel_dir,file+'.html'), "w") as fdw:
    fdw.write('<center><a href="https://gitlab.com/petsc/petsc/-/blob/'+git_sha+'/'+rel_dir+'/'+file+'">Actual source code: '+file+'</a></center><br>\n')
    fdw.write(ntxt)

if __name__ == "__main__":
  main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],os.path.dirname(sys.argv[6]),os.path.basename(sys.argv[6]))
