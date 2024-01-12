#!/usr/bin/env python
""" Builds the html files for all the source"""

import os
import re
import subprocess
import pathlib
from itertools import chain

def compute_make_np(i):
  '''Number of cores to run make c2html on'''
  f16 = .80
  f32 = .65
  f64 = .50
  f99 = .30
  if (i<=2):    return 2
  elif (i<=4):  return i
  elif (i<=16): return int(4+(i-4)*f16)
  elif (i<=32): return int(4+12*f16+(i-16)*f32)
  elif (i<=64): return int(4+12*f16+16*f32+(i-32)*f64)
  else:         return int(4+12*f16+16*f32+32*f64+(i-64)*f99)

def main(petsc_dir,loc,c2html,mapnames):
  os.chdir(petsc_dir)

  # reformat file that maps manual pages to directory location and add MPI manual pages
  with open('htmlmap.tmp', "w") as fdw, open(os.path.join(petsc_dir,'doc','manualpages','htmlmap'), "r") as fd:
    fdw.write(fd.read().replace('man+manualpages/','man+HTML_ROOT/manualpages/'))
  with open('htmlmap.tmp', "a") as fdw, open(os.path.join(petsc_dir,'doc','manualpages','mpi.www.index'), "r") as fd:
    fdw.write(fd.read())

  # walk directories generating list of all source code that needs processing and creating index.html for each directory
  SKIPDIRS = set('public html benchmarks output arch doc binding config petsc-doc lib bin .git systems share mpiuni kernels khash valgrind interfaces data linter'.split())
  SUFFIXES = set('.F90 .F .c .cxx .cpp .h .cu'.split())
  allfiles = []
  for root, dirs, files in chain.from_iterable(os.walk(path) for path in [petsc_dir]):
    dirs[:] = [d for d in dirs if not any([s for s in SKIPDIRS if d.startswith(s)])]
    root = root[len(petsc_dir)+1:]
    if not root: continue
    if not os.path.isdir(os.path.join(loc,root)): os.makedirs(os.path.join(loc,root))
    allfiles.extend([os.path.join(loc,root,f+'.html') for f in files if any([s for s in SUFFIXES if f.endswith(s)])])

    # create index.html file for each directory
    with open(os.path.join(loc,root,'index.html'),'w') as fdw:
      if root.startswith('src'):

        # get MANSEC from the makefile and copy the MANSEC basic information into the index
        # TODO: the text is actually .md so needs processing
        if os.path.isfile(os.path.join(root,'makefile')):
          with open(os.path.join(root,'makefile')) as mklines:
            mansecl = [line for line in mklines if line.startswith('MANSEC')]
            if mansecl:
              mansec = re.sub('MANSEC[ ]*=[ ]*','',mansecl[0]).strip('\n').strip()
              with open(os.path.join('doc','manualpages','MANSECHeaders',mansec)) as fd:
                fdw.write(fd.read())

      fdw.write('\n<p>\n')

      # TODO: use HTML lists for the list below

      # list examples
      if root.find('/tests') > -1 or root.find('tutorials') > -1:
        fdw.write('\n<p>\nExamples\n<p>')
        for f in files:
          if any([s for s in SUFFIXES if f.endswith(s)]):
            with open(os.path.join(root,f)) as fd:
              line = fd.readline()
              l = line.find('char help[] = ')
              if l > -1:
                s = line.find('\\n')
                line = line[l + 15:s]
              else: line = ''
              fdw.write('<a href="' + f + '.html">' + f + ': ' + line + '</a><br>\n')

      # list source code
      else:
        if any([f for f in files if any([s for s in SUFFIXES if f.endswith(s)])]):
          fdw.write('\n<p>\nSource files\n<p>')
        for f in files:
          if any([s for s in SUFFIXES if f.endswith(s)]):
            fdw.write('<a href=\"' + f + '.html\">' + f + '</a><br>\n')

      # list subdirectories
      if dirs:
        fdw.write('\n<p>\nDirectories\n<p>')
      for d in dirs:
        fdw.write('<a href="' + os.path.join(d,'index.html') + '">' + d + '</a><br>\n')

  # create makefile that will run c2html on all source files in parallel
  git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).rstrip()
  with open(os.path.join(petsc_dir,'c2html.mk'),'w') as fd:
    fd.write('files = ')
    fd.write(' '.join(allfiles))
    fd.write('\n')
    fd.write('\n')
    fd.write(os.path.join(loc,'%.html')+' : %\n')
    fd.write('	@' + os.path.join(petsc_dir,'doc','build_c2html_file.py') + ' ' + petsc_dir + ' ' + loc + ' '+ git_sha + ' ' + c2html + ' ' + mapnames + ' $< $@\n')
    fd.write('\n')
    fd.write('all: $(files)\n')

  import multiprocessing
  command = ['make', '-j', str(compute_make_np(multiprocessing.cpu_count())), '-f', 'c2html.mk', 'all']
  subprocess.run(command, cwd=petsc_dir, check=True)

  os.unlink('c2html.mk')
  os.unlink('htmlmap.tmp')

if __name__ == "__main__":
  main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
