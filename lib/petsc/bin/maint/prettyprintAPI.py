#!/usr/bin/env python3
#
# change python to whatever is needed on your system to invoke python
#
#  Processes PETSc's include/petsc*.h files to pretty print the PETSc API
#
#  Calling sequence:
#      prettyprintAPI (must be run in the PETSc directory)
##
import os
import re
import sys
import pathlib

def prettyprintAPI():
  sys.path.insert(0, os.path.abspath(os.path.join('config','utils')))
  import getAPI

  classes, enums, senums, typedefs, structs, funcs, files, mansecs, submansecs = getAPI.getAPI()

  for c in classes:
    print(classes[c])

  for e in enums:
    print(enums[e])

  for t in typedefs:
    print(typedefs[t])

  for s in structs:
    print(structs[s])

  for f in funcs:
    print(funcs[f])

  exit(0)

  with open(os.path.join('doc','objects.md'),"w") as fd:
    fd.write(':html_theme.sidebar_secondary.remove: true\n')
    fd.write('::::{tab-set}\n\n')

    fd.write(':::{tab-item} PETSc objects\n')
    for i in sorted(list(classes.keys())):
      fd.write('- '+i+'\n')
      for j in classes[i].keys():
        fd.write('  - '+j+str(classes[i][j])+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Typedefs to basic types\n')
    for i in sorted(list(typedefs.keys())):
      if i in ['VecScatter', 'VecScatterType']: continue
      if typedefs[i].name:
        fd.write('- '+i+' = '+typedefs[i].value+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Structs\n')
    for i in sorted(list(structs.keys())):
      fd.write('- '+i+' ' + str(structs[i].opaque)+'\n')
      for j in structs[i]:
        fd.write('  - '+j+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Enums\n')
    for i in sorted(list(enums.keys())):
      if i in ['PetscEnum']: continue
      fd.write(':::{dropdown} '+i+'\n')
      for j in enums[i]:
         v = j.strip()
         if v.find('=') > -1 : v = v[0:v.find('=')].strip()
         fd.write('  - '+v+'\n')
      fd.write(':::\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} String enums\n')
    for i in sorted(list(senums.keys())):
      fd.write('- '+i+'\n')
      for j in senums[i]:
         fd.write('  - '+j+'\n')
    fd.write(':::\n')

#
if __name__ ==  '__main__':
  prettyprintAPI()

