#!/usr/bin/env python3
""" Runs c2html and mapnames on a single source file and cleans up all text"""

import os
import re
import subprocess
import pathlib
import sys
import uuid

def _run_c2html(rel_dir,file,c2html):
  with open(os.path.join(rel_dir,file), "r") as fd:
    txt = fd.read()

  txt = re.sub(r'PETSC[A-Z]*_DLLEXPORT', '', txt)
  txt = subprocess.check_output([c2html, '-n'], text=True, input=txt)
  txt = ''.join(
    line.replace('<pre width="80">', '<pre width="80">\n', 1) + '\n'
    for line in txt.splitlines()
  )
  excluded = ('PetscValid', '#define __', '#undef __', 'EXTERN_C ')
  txt = ''.join(
    line + '\n'
    for line in txt.splitlines()
    if '#if !defined(__' not in line and not any(pattern in line for pattern in excluded)
  )
  return txt

def _run_mapnames(petsc_dir,mapnames,txt):
  return subprocess.check_output(
    [mapnames, '-map', os.path.join(petsc_dir, 'htmlmap.tmp'), '-inhtml'],
    text=True,
    input=txt,
  )

def _write_html(petsc_dir,loc,git_sha,rel_dir,file,txt):
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

def _map_batch(petsc_dir,mapnames,texts):
  mapped = [None] * len(texts)
  batched = []
  for i,txt in enumerate(texts):
    # mapnames drops an unterminated final identifier, so preserve EOF for such inputs.
    if txt and not txt.endswith('\n'):
      mapped[i] = _run_mapnames(petsc_dir,mapnames,txt)
    else:
      batched.append(i)
  if batched:
    while True:
      separator = '\n><!-- PETSC_C2HTML_BATCH_BOUNDARY_' + uuid.uuid4().hex + ' -->\n'
      if all(separator not in texts[i] for i in batched): break
    output = _run_mapnames(petsc_dir,mapnames,separator.join(texts[i] for i in batched)).split(separator)
    if len(output) != len(batched):
      raise RuntimeError('mapnames changed the C2HTML batch separator')
    for i,txt in zip(batched,output):
      mapped[i] = txt
  return mapped

def main(petsc_dir,loc,git_sha,c2html,mapnames,rel_dir,file):
  txt = _run_c2html(rel_dir,file,c2html)
  txt = _run_mapnames(petsc_dir,mapnames,txt)
  _write_html(petsc_dir,loc,git_sha,rel_dir,file,txt)

def main_batch(petsc_dir,loc,git_sha,c2html,mapnames,sourcefiles):
  files = [(os.path.dirname(sourcefile),os.path.basename(sourcefile)) for sourcefile in sourcefiles]
  texts = [_run_c2html(rel_dir,file,c2html) for rel_dir,file in files]
  for (rel_dir,file),txt in zip(files,_map_batch(petsc_dir,mapnames,texts)):
    _write_html(petsc_dir,loc,git_sha,rel_dir,file,txt)
  return len(files)

if __name__ == "__main__":
  main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],os.path.dirname(sys.argv[6]),os.path.basename(sys.argv[6]))
