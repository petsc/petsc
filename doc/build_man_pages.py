#!/usr/bin/env python
""" Loops through all the source using doctext to generate the manual pages"""

import os
import re
import subprocess
import pathlib

def findlmansec(file):
    mansec = None
    submansec = None
    with open(file) as mklines:
      #print(file)
      submansecl = [line for line in mklines if (line.find('SUBMANSEC') > -1 and line.find('BFORT') == -1)]
      if submansecl:
        submansec = re.sub('[ ]*/\* [ ]*SUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip('*/').strip()
        if submansec == submansecl[0].strip('\n'):
          submansec = re.sub('SUBMANSEC[ ]*=[ ]*','',submansecl[0]).strip('\n').strip()
        #print(':SUBMANSEC:'+submansec)
        return submansec
    with open(file) as mklines:
      mansecl = [line for line in mklines if line.startswith('MANSEC')]
      if mansecl:
        mansec = re.sub('MANSEC[ ]*=[ ]*','',mansecl[0]).strip('\n').strip()
        #print(':MANSEC:'+mansec)
        return mansec
    return None

def processdir(petsc_dir, dir, doctext):
  '''Runs doctext on each source file in the directory'''
  #print('Processing '+dir)
  #print('build_man_pages: Using doctext '+doctext)
  loc = os.path.join(petsc_dir,'doc')
  doctext_path = os.path.join(petsc_dir,'doc','manualpages','doctext')
  lmansec = None
  if os.path.isfile(os.path.join(dir,'makefile')):
    lmansec = findlmansec(os.path.join(dir,'makefile'))

  numberErrors = 0
  for file in os.listdir(dir):
    llmansec = lmansec
    if os.path.isfile(os.path.join(dir,file)) and pathlib.Path(file).suffix in ['.c', '.cxx', '.h', '.cu', '.cpp', '.hpp']:
      #print('Processing '+file)
      if not llmansec:
        llmansec = findlmansec(os.path.join(dir,file))
        if not llmansec: continue
      if not os.path.isdir(os.path.join(loc,'manualpages',llmansec)): os.mkdir(os.path.join(loc,'manualpages',llmansec))

      command = [doctext,
                 '-myst',
                 '-mpath',    os.path.join(loc,'manualpages',llmansec),
                 '-heading',  'PETSc',
                 '-defn',     os.path.join(loc,'manualpages','doctext','myst.def'),
                 '-indexdir', '../'+llmansec,
                 '-index',    os.path.join(loc,'manualpages','manualpages.cit'),
                 '-locdir',   dir[len(petsc_dir)+1:]+'/',
                 '-Wargdesc', os.path.join(loc,'manualpages','doctext','doctextcommon.txt'),
                 file]
      #print(command)
      sp = subprocess.run(command, cwd=dir, capture_output=True, encoding='UTF-8', check=True)
      if sp.stdout and sp.stdout.find('WARNING') > -1:
        print(sp.stdout)
        numberErrors = numberErrors + 1
      if sp.stderr and sp.stderr.find('WARNING') > -1:
        print(sp.stderr)
        numberErrors = numberErrors + 1
  return numberErrors


def processkhash(T, t, KeyType, ValType, text):
  '''Replaces T, t, KeyType, and ValType in text (from include/petsc/private/hashset.txt) with a set of supported values'''
  import re
  return re.sub('<ValType>',ValType,re.sub('<KeyType>',KeyType,re.sub('<t>',t,re.sub('<T>',T,text))))

def main(petsc_dir, doctext):
  # generate source code for manual pages for PETSc khash functions
  text = ''
  for f in ['hashset.txt', 'hashmap.txt']:
    with open(os.path.join(petsc_dir,'include','petsc','private',f)) as mklines:
      text = mklines.read()
      with open(os.path.join(petsc_dir,'include','petsc','private',f+'.h'),mode='w') as khash:
        khash.write(processkhash('I','i','PetscInt','',text))
        khash.write(processkhash('IJ','ij','struct {PetscInt i, j;}','',text))
        khash.write(processkhash('I','i','PetscInt','PetscInt',text))
        khash.write(processkhash('IJ','ij','struct {PetscInt i, j;}','PetscInt',text))
        khash.write(processkhash('IJ','ij','struct {PetscInt i, j;}','PetscScalar',text))
        khash.write(processkhash('IV','iv','PetscInt','PetscScalar',text))
        khash.write(processkhash('Obj','obj','PetscInt64','PetscObject',text))

  # generate the .md files for the manual pages from all the PETSc source code
  try:
    os.unlink(os.path.join(petsc_dir,'doc','manualpages','manualpages.cit'))
  except:
    pass
  numberErrors = 0
  for dirpath, dirnames, filenames in os.walk(os.path.join(petsc_dir),topdown=True):
    dirnames[:] = [d for d in dirnames if d not in ['tests', 'tutorials', 'doc', 'output', 'ftn-custom', 'f90-custom', 'ftn-auto', 'f90-mod', 'binding', 'binding', 'config', 'lib', '.git', 'share', 'systems'] and not d.startswith('arch')]
    numberErrors = numberErrors + processdir(petsc_dir,dirpath,doctext)
  if numberErrors:
    raise RuntimeError('Stopping document build since errors were detected in generating manual pages')

  # generate list of all manual pages
  with open(os.path.join(petsc_dir,'doc','manualpages','htmlmap'),mode='w') as map:
    with open(os.path.join(petsc_dir,'doc','manualpages','manualpages.cit')) as cit:
      map.write(re.sub('man\+../','man+manualpages/',cit.read()))
    with open(os.path.join(petsc_dir,'doc','manualpages','mpi.www.index')) as mpi:
      map.write(mpi.read())

if __name__ == "__main__":
  # TODO Accept doctext from command line
  main(os.path.abspath(os.environ['PETSC_DIR']))
