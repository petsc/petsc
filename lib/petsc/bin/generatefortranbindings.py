#!/usr/bin/env python3
#
#    Generates Fortran function stubs and module interface definitions for PETSc and SLEPc
#
#    Note 1:
#      const char *title[] gets mapped to character(*) title and the string gets copied into the space provided by the caller
#
#    This tool looks for the values MANSEC and [BFORT]SUBMANSEC (where BFORTSUBMANSEC has priority over SUBMANSEC)
#    defined in the makefile
#
#    The generated interface files are stored in $PETSC_ARCH/ftn/MANSEC/petsc[BFORT]SUBMANSEC.*
#    These are then included by the src/MANSEC/ftn-mod/MANSECmod.F90 files to create the Fortran module files
#
#    The generated C stub files are stored in $PETSC_ARCH/ftn/MANSEC/**/ where ** is the directory under MANSEC of the original source
#
#    Stubs/interfaces generated from include can only involve sys files
#
#    SUBMANSEC (but not BFORTSUBMANSEC) is also used (in the documentation generating part of PETSc) to determine what
#    directory in doc/manualpages/ the manual pages are deposited.
#
#    An example of when the BFORTSUBMANSEC may be different than SUBMANSEC is src/dm/label/impls/ephemeral/plex where we would like
#    the documentation to be stored under DMLabel but the function interfaces need to go into the DMPLEX Fortran module
#    (not the DM Fortran module) since they depend on DMPlexTransform.
#
from __future__ import print_function
import os
import pathlib
import shutil
import sys
import subprocess
from subprocess import check_output
sys.path.insert(0,os.path.realpath(os.path.dirname(__file__)))
import getAPI

CToFortranTypes = {'int':'integer4', 'ptrdiff_t':'PetscInt64', 'float':'PetscFortranFloat', 'int32_t':'integer4',
                   'double':'PetscFortranDouble', 'short':None, 'size_t':'PetscSizeT', 'rocblas_status':None, 'PetscBT':None,
                   'PetscEnum':None, 'PetscDLHandle':None}

Letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','w','x','y']
verbose = False

def verbosePrint(text):
  '''Prints the text if run with verbose option'''
  if verbose: print(text)

def generateFortranInterface(pkgname, petscarch, classes, enums, structs, senums, funname, fun):
  '''Generates the interface definition for a function'''
  '''This is used both by class functions and standalone functions'''
  # check for functions for which we cannot build interfaces
  if fun.opaque:
    return
  if fun.name.find('_') > -1: return
  for k in fun.arguments:
    ktypename = k.typename
    if ktypename in CToFortranTypes and not CToFortranTypes[ktypename]:
      fun.opaque = True
      return
    if ktypename.find('_') and not ktypename.startswith('MPI') > -1:
      fun.opaque = True
      return
    if ktypename.endswith('Func'):
      # these function typedef are soon to be eliminated and so can this check
      fun.opaque = True
      return
    if (ktypename == 'void' and not k.isfunction) or ktypename == 'PeCtx':
      return

  mansec = fun.mansec
  if not mansec: mansec = fun.submansec
  file = fun.includefile + '90'
  if not file.startswith(pkgname): file = pkgname + file
  with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(mansec),file),"a") as fd:
    if funname in ['PetscObjectQuery', 'PetscObjectCompose']:
      # for macro polymorphism the objects are passed directly as obj%d
      fun.arguments[0].typename = 'PetscFortranAddr'
      fun.arguments[2].typename = 'PetscFortranAddr'
      funname = funname + 'Raw'
    fd.write('  interface ' + funname + '\n')
    fi = fun
    func = ''
    dims = ['']
    # if ((funname).startswith('MatDenseGetArray') or (funname).startswith('MatDenseRestoreArray')) and fi[-1].endswith('[]'): dims = ['1d','2d']
    for dim in dims:
      fd.write('  subroutine ' + funname + func + dim + '(')
      simportset = set()
      simport = ''
      cnt = 0
      for k in fi.arguments:
        if k.stringlen: continue
        ktypename = k.typename
        if cnt: fd.write(',')
        fd.write(Letters[cnt])
        if not ktypename in simportset:
          if ktypename in classes or ktypename == 'VecScatter':
            if simport: simport = simport + ','
            simport = simport + 't' + ktypename
          if ktypename in enums:
            if simport: simport = simport + ','
            simport = simport + 'e' + ktypename
          if ktypename in structs and not structs[ktypename].opaque:
            if simport: simport = simport + ','
            simport = simport + 's' + ktypename
        simportset.add(ktypename)
        cnt = cnt + 1
      if cnt: fd.write(',')
      fd.write(' z)\n')
      fd.write('  use, intrinsic :: ISO_C_binding\n')
      if simport: fd.write('  import ' + simport + '\n')

      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        ktypename = k.typename
        if ktypename in CToFortranTypes:
          ktypename =CToFortranTypes[ktypename]
        if ktypename == 'char':
          if getattr(k, 'char_type', None) == 'single':
            fd.write('  character :: ' + Letters[cnt] + '\n')
          else:
            fd.write('  character(*) :: ' + Letters[cnt] + '\n')
        elif ktypename in senums:
          fd.write('  character(*) :: ' + Letters[cnt] + '\n')
        elif k.array and k.stars:
          if not dim or dim == '1d': fd.write('  ' + ktypename + ', pointer :: ' +  Letters[cnt]  + '(:)\n')
          else: fd.write('  ' + ktypename + ', pointer :: ' +  Letters[cnt]  + '(:,:)\n')
        elif k.array:
          fd.write('  ' + ktypename + ' :: ' +  Letters[cnt]  + '(*)\n')
        elif k.isfunction:
          fd.write('  ' + 'external ' + Letters[cnt]  + '\n')
        else:
          fd.write('  ' + ktypename + ' :: ' + Letters[cnt] + '\n')
        cnt = cnt + 1
      fd.write('  PetscErrorCode z\n')
      fd.write('  end subroutine\n')
    fd.write('  end interface\n')
    fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
    fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + funname + func + dim  + '\n')
    fd.write('#endif\n')

    if fun.name in ['PetscObjectQuery', 'PetscObjectCompose']:
      # under change above
      fun.arguments[0].typename = 'PetscObject'
      fun.arguments[2].typename = 'PetscObject'

def generateCStub(pkgname,petscarch,manualstubsfound,senums,classes,structs,funname,fun):
  '''Generates the C stub that is callable from Fortran for a function'''
  #
  #
  #  PETSc returns strings in two ways either
  #     - with a pointer to an array: char *[]
  #     - or by copying the string into a given array with a given length: char [], size_t len
  #
  #  in both cases the Fortran API passes in a character(*) with enough room to hold the result.
  #  It does not pass in the string length as a separate argument
  #
  #  The following Argument class values used to indicate which case is being generated
  #     - stringlen    - True indicates the argument is the length of the previous argument which is a character string
  #     - const        - indicates the string argument is an input, not an output
  #     - stars == 1   - indicates the string is (in C) returned by a pointer to a string array
  #
  if fun.penss: return

  skipbody = False
  if fun.opaque or fun.opaquestub: skipbody = True
  for k in fun.arguments:
    if k.array and k.stars and not k.typename == 'char': skipbody = True
    if k.stars and k.typename == 'MPI_Fint': skipbody = True
    if k.stars == 2 and k.typename == 'void': skipbody = True
    if k.isfunction: skipbody = True
  if skipbody and fun.name.lower() in manualstubsfound: return

  for k in fun.arguments:
    # no automatic stub if function returns an array, except if it is a string
    if not skipbody and k.array and k.stars and not k.typename == 'char': return
    if k.stars and k.typename == 'MPI_Fint': return   # TODO add support for returning MPI_Fint
    if k.stars == 2 and k.typename == 'void': return

    # no manual stub if dealing with multidimensional arrays, voids, etc
    if skipbody:
      if k.stars > 1: return
      if k.typename == 'void': return
      if k.typename == 'char' and not k.array: return
      return

  if not skipbody:
    if fun.file.endswith('.h'):
      dir = os.path.join(petscarch,'ftn',fun.mansec.lower(),'stubs')
      filename = fun.file.replace('.h','.c')
    else:
      dir = os.path.join(petscarch,fun.dir.replace('src/','ftn/'))
      filename = fun.file
  else:
    dir = os.path.join(fun.dir,'ftn-custom')
    filename = 'z' + fun.file
  if not os.path.isdir(dir): os.makedirs(dir)

  with open(os.path.join(dir,filename.replace('.c','f.c')),'a') as fd:
    fd.write('#include "petscsys.h"\n')
    fd.write('#include "petscfix.h"\n')
    fd.write('#include "petsc/private/ftnimpl.h"\n')
    fd.write('#include <' + pkgname + fun.mansec + '.h>\n')
    fd.write('#include <' + pkgname + fun.includefile.replace(pkgname,'') + '>\n')

    suffix = ''
    # not used because generating the Fortran modules takes too long
    #for k in fun.arguments:
    #  if k.optional:
    #    suffix = 'raw'
    if funname in ['PetscObjectQuery', 'PetscObjectCompose']:
      suffix = 'raw'

    fd.write('#if defined(PETSC_HAVE_FORTRAN_CAPS)\n')
    fd.write('  #define ' + (funname + suffix).lower() + '_ ' + (funname + suffix).upper() + '\n')
    fd.write('#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)\n')
    fd.write('  #define ' + (funname + suffix).lower() + '_ ' + (funname + suffix).lower() + '\n')
    fd.write('#endif\n')

    # output function declaration prototype
    fd.write(pkgname.upper() + '_EXTERN void ' + (funname + suffix).lower() + '_(')
    cnt = 0
    for k in fun.arguments:
      if k.stringlen: continue
      ktypename = k.typename
      if cnt: fd.write(', ')
      if k.const and not ((k.typename == 'char' or k.typename in senums) and k.array):
        # see note one at the top of the file why const is not added for this case
        fd.write('const ')
      if k.typename in senums:
        fd.write('char *')
      elif k.typename == 'char' and not k.array and not k.stars and getattr(k, 'char_type', None) == 'single':
        fd.write('char *')
      else:
        if k.stars == 1 and k.array and not ktypename == 'char':
          fd.write('F90Array1d *')
        else:
          fd.write(ktypename)
          fd.write(' ')
      if k.typename in structs.keys() and structs[k.typename].opaque:
        fd.write('*')
      if not (k.typename == 'char' or k.typename in senums or k.array or k.typename == 'PeCtx'):
        fd.write('*')
      fd.write(Letters[cnt])
      if (k.typename == 'char' and getattr(k, 'char_type', None) != 'single') or (not k.stars and k.array): fd.write('[]')
      cnt = cnt + 1
    if cnt: fd.write(', ')
    fd.write('PetscErrorCode *ierr')
    # add the lengths of the string arguments put in by the Fortran compiler
    cnt = 0
    for k in fun.arguments:
      if k.stringlen: continue
      if k.typename in senums or k.typename == 'char':
        fd.write(', PETSC_FORTRAN_CHARLEN_T l_'  + Letters[cnt])
      cnt = cnt + 1
    fd.write(')\n{\n')

    if skipbody:
      fd.write('  PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_SUP, PETSC_ERROR_INITIAL, "Add Fortran stub body here!");\n')
      fd.write('  *ierr = PETSC_ERR_SUP;\n')
      fd.write('  // You may need the code fragments below\n');
      fd.write('  // *ierr = F90Array1dCreate(C array, MPIU_XXX, 1, length, Fortran array\n');
      fd.write('  // *ierr = F90Array1dAccess(Fortran array , MPIU_XXX, (void **)&C array\n');
      fd.write('  // *ierr = F90Array1dDestroy(Fortran array, MPIU_XXX\n');
    else:
      # functions that destroy objects should return immediately if null, -2, -3
      if fun.arguments and fun.arguments[0].typename in classes and fun.name.endswith('Destroy'):
        fd.write('  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(a);\n')

      # handle arguments that may return a null object
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.stars and k.typename  in classes:
          fd.write('  PetscBool null_' + Letters[cnt] + ' = !*(void**) ' + Letters[cnt] + ' ? PETSC_TRUE : PETSC_FALSE;\n')
        cnt = cnt + 1

      # prevent an existing object from being overwritten by a new create
      if fun.arguments and fun.arguments[-1].typename in classes and fun.name.endswith('Create'):
        fd.write('  PETSC_FORTRAN_OBJECT_CREATE(' + Letters[len(fun.arguments)-1] + ');\n')

      # handle string argument fixes
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.typename == 'char':
          if getattr(k, 'char_type', None) == 'single': pass
          elif not k.stars:
            if k.const:
              fd.write('  char* c_' + Letters[cnt] + ';\n')
              fd.write('  FIXCHAR(' + Letters[cnt] + ', l_' + Letters[cnt] + ', c_' + Letters[cnt] + ');\n')
          elif k.stars:
            fd.write('  char* c_' + Letters[cnt] + ' = PETSC_NULLPTR;\n')
        elif k.typename in senums:
          if not k.stars:
            fd.write('  char* c_' + Letters[cnt] + ';\n')
            fd.write('  FIXCHAR(' + Letters[cnt] + ', l_' + Letters[cnt] + ', c_' + Letters[cnt] + ');\n')
          elif k.stars:
            fd.write('  char* c_' + Letters[cnt] + ' = PETSC_NULLPTR;\n')
        cnt = cnt + 1

      # handle viewer argument fixes
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.typename == 'PetscViewer' and not k.stars and not k.array:
          fd.write('  PetscViewer v_' + Letters[cnt] + ' = PetscPatchDefaultViewers(' + Letters[cnt] + ');\n')
        cnt = cnt + 1

      # handle any arguments that may be null
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.stars or k.array:
          if k.typename in classes:
            fd.write('  CHKFORTRANNULLOBJECT(' + Letters[cnt] + ');\n')
          elif k.typename == 'PetscInt':
            fd.write('  CHKFORTRANNULLINTEGER(' + Letters[cnt] + ');\n')
          elif k.typename == 'PetscReal':
            fd.write('  CHKFORTRANNULLREAL(' + Letters[cnt] + ');\n')
          elif k.typename == 'PetscScalar':
            fd.write('  CHKFORTRANNULLSCALAR(' + Letters[cnt] + ');\n')
          elif k.typename == 'PetscBool':
            fd.write('  CHKFORTRANNULLBOOL(' + Letters[cnt] + ');\n')
        cnt = cnt + 1

      # call C function
      fd.write('  *ierr = ' + funname + '(')
      cnt = 0
      for k in fun.arguments:
        if cnt: fd.write(', ')
        if (k.typename in senums or k.typename == 'char') and k.stars:
          fd.write('(const char **)&')
        if k.typename == 'char':
          if getattr(k, 'char_type', None) == 'single':
            fd.write('*')
          elif k.const:
            fd.write('c_')
        elif k.typename in senums:
          fd.write('c_')
        elif k.typename == 'MPI_Fint':
          fd.write('MPI_Comm_f2c(*(')
        elif not k.stars and not k.array and not k.stringlen and not k.typename == 'PetscViewer' and not k.typename == 'PeCtx':
          fd.write('*')
#        if k.typename == 'void' and k.stars == 2:
#          fd.write('&')
        if k.stringlen:
          fd.write('l_' + Letters[cnt - 1])
          continue
        if k.typename == 'PetscViewer' and not k.stars and not k.array:
          fd.write('v_')
        if k.typename in structs.keys() and structs[k.typename].opaque:
          fd.write('*')
        fd.write(Letters[cnt])
        if k.typename == 'PetscBool' and not k.stars and not k.array:
          # handle bool argument fixes (-1 needs to be corrected to 1 for Intel compilers)
          fd.write(' ? PETSC_TRUE : PETSC_FALSE')
        if k.typename == 'MPI_Fint':
          fd.write('))')
        cnt = cnt + 1
      fd.write(');\n')
      fd.write('  if (*ierr) return;\n');

      # cleanup any string arguments fixes
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.typename == 'char' or k.typename in senums:
          if k.typename == 'char' and getattr(k, 'char_type', None) == 'single': pass
          elif not k.stars and (k.const or k.typename in senums):
            fd.write('  FREECHAR(' + Letters[cnt] + ', c_' + Letters[cnt] + ');\n')
          else:
            if k.stars:
              fd.write('  *ierr = PetscStrncpy((char *)' + Letters[cnt] + ', c_' + Letters[cnt] + ', l_' + Letters[cnt] + ');\n')
              fd.write('  if (*ierr) return;\n');
            fd.write('  FIXRETURNCHAR(PETSC_TRUE, ' + Letters[cnt] + ', l_' + Letters[cnt] + ');\n')
        cnt = cnt + 1

      # handle arguments that may return a null PETSc object
      cnt = 0
      for k in fun.arguments:
        if k.stringlen: continue
        if k.stars and k.typename in classes:
          fd.write('  if (! null_' + Letters[cnt] + ' && !*(void**) ' + Letters[cnt] + ') *(void **) ' + Letters[cnt] + ' = (void *)-2;\n')
        cnt = cnt + 1

    fd.write('}\n')
    if not skipbody:
      shutil.copy(os.path.join(fun.dir,'makefile'), os.path.join(dir,'makefile'))
    else:
      with open(os.path.join(fun.dir,'makefile')) as fin:
        with open(os.path.join(dir,'makefile'),'w') as fout:
          fout.write(fin.read().replace('petscdir.mk','../petscdir.mk'))
        output = check_output('git add ' + os.path.join(dir,'makefile'), shell=True).decode('utf-8')
      print('Fix the manual stub for ' + fun.name + ' in ' + os.path.join(dir,filename.replace('.c','f.c')))
      output = check_output('git add ' + os.path.join(dir,filename.replace('.c','f.c')), shell=True).decode('utf-8')

def generateFortranStub(senums, funname, fun, fd, opts):
  '''For functions with optional arguments generate the Fortran stub that calls the C stub'''
  for k in fun.arguments:
    # no C stub if function returns an array, except if it is a string
    # TODO: generate fillible stub for functions that return arrays
    if k.array and k.stars and not k.typename == 'char': return
    if k.stars and k.typename == 'MPI_Fint': return   # TODO add support for returning MPI_Fint
    if k.stars == 2 and k.typename == 'void': return
  for fi in opts:
    fd.write('  subroutine ' + funname + ''.join(fi) + '(')
    cnt = 0
    for k in fun.arguments:
      if k.stringlen: continue
      if cnt: fd.write(',')
      fd.write(Letters[cnt])
      cnt = cnt + 1
    fd.write(',z)\n')
    cnt = 0
    for k in fun.arguments:
      if k.stringlen: continue
      ktypename = k.typename
      if fi[cnt] == 'O':
        fd.write('  PetscNull :: ' + Letters[cnt] + '\n')
      elif ktypename in senums or ktypename == 'char':
        fd.write('  character(*) :: ' + Letters[cnt] + '\n')
      elif k.array and k.stars:
        fd.write('  ' + ktypename + ', pointer :: ' +  Letters[cnt] + '(:)\n')
      elif k.array:
        fd.write('  ' + ktypename + ' :: '+ Letters[cnt] + '(*)\n')
      else:
        fd.write('  '+ ktypename + ' :: ' + Letters[cnt] + '\n')
      cnt = cnt + 1
    fd.write('  PetscErrorCode z\n')
    fd.write('  call ' + funname + 'Raw(')
    cnt = 0
    for k in fun.arguments:
      if k.stringlen: continue
      ktypename = k.typename
      if cnt: fd.write(',')
      if fi[cnt] == 'a':
        fd.write(Letters[cnt])
      else:
        typename = ktypename.upper().replace('PETSC','')
        if typename == 'INT': typename = 'INTEGER'
        fd.write('PETSC_NULL_' + typename)
        if k.array and k.stars: fd.write('_POINTER')
        elif k.array: fd.write('_ARRAY')
      cnt = cnt + 1
    fd.write(',z)\n')
    fd.write('  end subroutine\n')

##########  main

def main(petscdir,slepcdir,petscarch):
  '''Generates all the Fortran include and C stub files needed for the Fortran API'''
  import pickle
  del sys.path[0]

  pkgname = 'slepc' if slepcdir else 'petsc'

  if not slepcdir:
    classes, enums, senums, typedefs, structs, funcs, files, mansecs, submansecs = getAPI.getAPI(petscdir,'petsc')

    with open(os.path.join(petscdir,petscarch,'lib','petsc','conf','classes.data'),'wb') as file:
      pickle.dump(classes,file)
      pickle.dump(enums,file)
      pickle.dump(senums,file)
      pickle.dump(structs,file)
      pickle.dump(typedefs,file)

    petscobjectfunctions = classes['PetscObject'].functions
    classesext = classes.copy()
    structsext = structs.copy()
    enumsext   = enums.copy()
    senumsext  = senums.copy()
  else:
    with open(os.path.join(petscdir,'' if petscarch.startswith('installed-') else petscarch,'lib','petsc','conf','classes.data'),'rb') as file:
      petscclasses  = pickle.load(file)
      petscenums    = pickle.load(file)
      petscsenums   = pickle.load(file)
      petscstructs  = pickle.load(file)
      petsctypedefs = pickle.load(file)

    petscobjectfunctions = petscclasses['PetscObject'].functions
    classes, enums, senums, typedefs, structs, funcs, files, mansecs, submansecs = getAPI.getAPI(slepcdir,'slepc')
    classesext = classes.copy(); classesext.update(petscclasses)
    structsext = structs.copy(); structsext.update(petscstructs)
    enumsext = enums.copy(); enumsext.update(petscenums)
    senumsext = senums.copy(); senumsext.update(petscsenums)

##########  $PETSC_ARCH/include/petsc/finclude/*.h

  dir     = os.path.join(petscarch,'include', pkgname, 'finclude')
  skipinc = [pkgname + 'version.h']
  if os.path.isdir(dir): shutil.rmtree(dir)
  os.makedirs(dir)

  for i in files.keys():
    if i.endswith('types.h') or i in skipinc: continue
    with open(os.path.join(dir, i),'w') as fd:
      dname = pkgname.upper() + i.upper()[0:-2] + 'DEF_H'
      fd.write('#if !defined(' + dname + ')\n#define ' + dname + '\n\n')
      fb = os.path.join('include', pkgname, 'finclude',i.replace('.h','base.h'))
      if os.path.isfile(fb):
        fd.write('#include "' + os.path.join(pkgname,'finclude',i.replace('.h','base.h')) + '"\n')
      for j in files[i].included:
        if j in skipinc: continue
        j = j.replace('types.h','.h')
        if i == j: continue
        fd.write('#include "' + os.path.join(('petsc' if j.startswith('petsc') else 'slepc'),'finclude',j) + '"\n')
      fd.write('\n')

  for i in enums.keys():
    if i in ['PetscBool', 'PetscEnum']: continue
    with open(os.path.join(dir, enums[i].includefile),"a") as fd:
      fd.write('#define ' + i + ' type(e' + i + ')\n')

  for i in typedefs.keys():
    if not typedefs[i].name: continue
    if i in ['PetscBool', 'PetscSizeT']: continue
    value = typedefs[i].value
    if value in CToFortranTypes:
      if not CToFortranTypes[value]: continue
      value = CToFortranTypes[value]
    with open(os.path.join(dir, typedefs[i].includefile),"a") as fd:
      fd.write('#define ' + i + ' ' + value + '\n')

  for i in structs.keys():
    with open(os.path.join(dir, structs[i].includefile),"a") as fd:
      if not structs[i].opaque:
        fd.write('#define ' +  i  + ' type(s' + i + ')\n')
      else:
        fd.write('#define ' + i + ' PetscFortranAddr\n')

  for i in files.keys():
    if i.endswith('types.h') or i in skipinc: continue
    with open(os.path.join(dir, i),'a') as fd:
      fd.write('\n')

  for i in senums.keys():
    with open(os.path.join(dir, senums[i].includefile),"a") as fd:
      fd.write('#define ' + i + ' CHARACTER(80)\n')

  for i in files.keys():
    if i.endswith('types.h') or i in skipinc: continue
    with open(os.path.join(dir, i),'a') as fd:
      fd.write('\n')

  for i in classes.keys():
    if i in ['PetscIntStack']: continue
    with open(os.path.join(dir, classes[i].includefile),"a") as fd:
      fd.write('#define ' + i + ' type(t' + i + ')\n')

  if not slepcdir:
    # special polymorphic routines handled with macros
    with open(os.path.join(petscarch,'include', 'petsc', 'finclude', 'petscsys.h'),"a") as fd:
      fd.write('#define PetscObjectCompose(a,b,c,z) PetscObjectComposeRaw(a%v,b,c%v,z)\n')
      fd.write('#define PetscObjectQuery(a,b,c,z) PetscObjectQueryRaw(a%v,b,c%v,z)\n')

  for i in files.keys():
    if i.endswith('types.h') or i in skipinc: continue
    with open(os.path.join(dir, i),'a') as fd:
      fd.write('\n#endif\n')

###########  $PETSC_ARCH/ftn/MANSEC/*.h

  dir = os.path.join(petscarch,'ftn')
  if os.path.isdir(dir): shutil.rmtree(dir)
  os.makedirs(dir)

  for i in mansecs.keys():
    if not slepcdir:
      dir = os.path.join(petscarch,'ftn', i)
      if os.path.isdir(dir): shutil.rmtree(dir)
      os.makedirs(dir)
    elif i!='sys':
      dir = os.path.join(petscarch,'ftn', getAPI.mansecpath(i))
      if os.path.isdir(dir): shutil.rmtree(dir)
      os.makedirs(dir)

  for i in classes.keys():
    if i in ['PetscIntStack', 'PetscTabulation']: continue
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(classes[i].mansec),classes[i].includefile),"a") as fd:
      if not classes[i].petscobject:
        fd.write('  type t' + i + '\n')
        fd.write('    PetscFortranAddr:: v = PETSC_FORTRAN_TYPE_INITIALIZE\n')
        fd.write('  end type t' + i + '\n')
      else:
        fd.write('  type, extends(tPetscObject) ::  t' + i + '\n')
        fd.write('  end type t' + i + '\n')
      v = (pkgname.upper() + '_NULL_' + i.upper().replace(pkgname.upper(),'').strip('_')).replace('_NULL_NULL','_NULL')
      fd.write('  ' + i + ', parameter :: ' + v + ' = t' + i + '(0)\n')
      fd.write('  ' + i + ', target :: ' + v + '_ARRAY(1) = [t' + i + '(0)]\n')
      fd.write('  ' + i + ', pointer :: ' + v + '_POINTER(:) => ' + v + '_ARRAY\n')
      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + v + '\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + v + '_ARRAY\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + v + '_POINTER\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in enums.keys():
    if i in ['PetscBool', 'PetscEnum']: continue
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(enums[i].mansec),enums[i].includefile),"a") as fd:
      fd.write('  type e' + i + '\n')
      fd.write('    PetscEnum:: v = PETSC_FORTRAN_TYPE_INITIALIZE\n')
      fd.write('  end type e' + i + '\n\n')
      v = (pkgname.upper() + '_NULL_' + i.upper().replace(pkgname.upper(),'').replace('NULL','')).strip('_').replace('_NULL_NULL','_NULL')
      fd.write('  ' + i + ', parameter :: ' + v + ' = e' + i + '(-50)\n')
      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + v + '\n')
      fd.write('#endif\n')
      cnt = 0
      givenvalue = 0
      for j in enums[i].values:
        if j.find('=') > -1:
          if givenvalue == -1:
            print('Some enum values for ' + i + ' are set but others are not set')
          v = j.replace(' = ',' = e' + i + '(') + ')'
          givenvalue = 1
        else:
          if givenvalue == 1:
            print('Some enum values for ' + i + ' are set but others are not set')
          v = j + ' = e' + i + '(' + str(cnt) + ')'
          givenvalue = -1
        fd.write('    ' + i + ', parameter :: ' + v + '\n')
        cnt = cnt + 1
      fd.write('\n')

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      for j in enums[i].values:
         if j.count('='): v = j[0:j.find('=')]
         else: v = j
         fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + v + '\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in senums.keys():
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(senums[i].mansec),senums[i].includefile),"a") as fd:
      for j in senums[i].values:
        fd.write('  CHARACTER(LEN=*), PARAMETER :: ' + j + ' = \'' + senums[i].values[j].replace('"','') + '\'\n')
      fd.write('\n')

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      for j in senums[i].values:
        fd.write('!DEC$ ATTRIBUTES DLLEXPORT::' + j + '\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in structs.keys():
    if structs[i].opaque: continue
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(structs[i].mansec),structs[i].includefile),"a") as fd:
      fd.write('  type s' + i + '\n')
      for j in structs[i].records:
        fd.write('    ' + j.type.replace('[','(').replace(']',')') + '\n')
      fd.write('  end type s' + i + '\n')
      fd.write('\n')

###########  $PETSC_ARCH/ftn/MANSEC/*.h90

  for i in classes.keys():
    # generate interface definitions for all objects' methods
    if i in ['PetscIntStack']: continue
    for j in classes[i].functions: # loop over functions in class
      generateFortranInterface(pkgname,petscarch,classesext,enumsext,structs,senumsext,j,classes[i].functions[j])

    if i in ['PetscObject', 'PetscTabulation','SlepcConvMon']: continue
    file = classes[i].includefile + '90'
    if not file.startswith(pkgname): file = pkgname + file
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(classes[i].mansec),file),"a") as fd:
      fd.write('  interface operator(.ne.)\n')
      fd.write('    module procedure ' + i + 'notequals\n')
      fd.write('  end interface operator (.ne.)\n')
      fd.write('  interface operator(.eq.)\n')
      fd.write('    module procedure ' + i + 'equals\n')
      fd.write('  end interface operator (.eq.)\n\n')

    # generate interface definitions for PetscObject methods for each PetscObject subclass (KSP etc)
    if not classes[i].petscobject: continue
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(classes[i].mansec),file),"a") as fd:
      ii = i.replace('Petsc','')
      fd.write('  interface PetscObjectCast\n')
      fd.write('    module procedure PetscObjectCast' + ii + '\n')
      fd.write('  end interface\n')
      fd.write('  interface PetscBarrier\n')
      fd.write('  module procedure PetscBarrier' + ii + '\n')
      fd.write('  end interface\n')

      for funname in petscobjectfunctions:
        if funname in ['PetscObjectCompose', 'PetscObjectQuery']: continue
        fi = petscobjectfunctions[funname]

        # the subclassing only works for PetscObjectXXX(PetscObject xxx,...) class methods
        if not fi.arguments or not fi.arguments[0].typename == 'PetscObject': continue

        # cannot print Fortran interface definition if any arguments are void * or void **
        opaque = False
        for k in fi.arguments:
          if k.typename == 'void' or k.typename == 'PeCtx': opaque = True
        if opaque: continue

        if funname.startswith('PetscObjectSAWs') or funname == 'PetscObjectViewSAWs':
          # if we always generate the fortran stubs simply mark these functions as opaque when SAWs is not available
          fd.write('#if defined(PETSC_HAVE_SAWS)\n')
        fd.write('  interface ' + funname + '\n')
        fd.write('    module procedure ' + funname  + ii + '\n')
        fd.write('  end interface\n')
        if funname.startswith('PetscObjectSAWs') or funname == 'PetscObjectViewSAWs':
          fd.write('#endif\n')

  # generate interface definitions for all standalone functions
  for j in funcs.keys():
    generateFortranInterface(pkgname,petscarch,classesext,enumsext,structs,senumsext,funcs[j].name,funcs[j])

  # generate .eq. and .neq. for enums
  for i in enums.keys():
    if i in ['PetscEnum', 'PetscBool3']: continue
    file = enums[i].includefile + '90'
    if not file.startswith(pkgname): file = pkgname + file
    with open(os.path.join(petscarch,'ftn',getAPI.mansecpath(enums[i].mansec),file),"a") as fd:
      fd.write('  interface operator(.ne.)\n')
      fd.write('    module procedure ' + i + 'notequals\n')
      fd.write('  end interface operator (.ne.)\n')
      fd.write('  interface operator(.eq.)\n')
      fd.write('    module procedure ' + i + 'equals\n')
      fd.write('  end interface operator (.eq.)\n\n')

##########  $PETSC_ARCH/ftn/MANSEC/*.hf90

  for i in classes.keys():
    if i in ['PetscObject', 'PetscIntStack', 'PetscTabulation','SlepcConvMon']: continue
    with open(os.path.join(petscarch,'ftn', getAPI.mansecpath(classes[i].mansec),classes[i].includefile + 'f90'),"a") as fd:

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: ' + i + 'notequals\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: ' + i + 'equals\n')
      fd.write('#endif\n\n')
      fd.write('  function ' + i + 'notequals(A,B)\n')
      fd.write('    logical ' + i + 'notequals\n')
      fd.write('    type(t' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'notequals = (A%v .ne. B%v)\n')
      fd.write('  end function\n')
      fd.write('  function ' + i + 'equals(A,B)\n')
      fd.write('    logical ' + i + 'equals\n')
      fd.write('    type(t' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'equals = (A%v .eq. B%v)\n')
      fd.write('  end function\n')

       # generate Fortran subroutines for PetscObject methods for each PetscObject subclass (KSP etc)
      if classes[i].petscobject:
        ii = i.replace('Petsc', '')
        fd.write('  function PetscObjectCast' + ii + '(a)\n')
        fd.write('    ' + i + ' a\n')
        fd.write('    PetscObject PetscObjectCast' + ii + '\n')
        fd.write('    PetscObjectCast' + ii + '%v = a%v\n')
        fd.write('  end function \n')
        fd.write('  subroutine PetscBarrier' + ii + '(a,z)\n')
        fd.write('    ' + i + ' a\n')
        fd.write('    PetscErrorCode z\n')
        fd.write('    call PetscBarrier(PetscObjectCast(a),z)\n')
        fd.write('  end subroutine \n')

        for funname in petscobjectfunctions:
          if funname in ['PetscObjectCompose', 'PetscObjectQuery']: continue
          fi = petscobjectfunctions[funname]
          if not fi.arguments or not fi.arguments[0].typename == 'PetscObject': continue

          # cannot generate Fortran functions if any argument is void or PeCtx
          opaque = False
          for k in fi.arguments:
            if k.typename == 'void' or k.typename == 'PeCtx': opaque = True
          if opaque: continue

          # write the PetscObject class function for the specific object that calls the base function
          if funname.startswith('PetscObjectSAWs') or funname == 'PetscObjectViewSAWs':
            fd.write('#if defined(PETSC_HAVE_SAWS)\n')
          fd.write('  subroutine ' + funname + ii + '(')
          cnt = 0
          for k in fi.arguments:
            if k.stringlen: continue
            if cnt: fd.write(', ')
            fd.write(Letters[cnt])
            cnt = cnt + 1
          fd.write(' , z)\n')
          fd.write('  ' + i + ' a\n')
          cnt = 1
          for k in fi.arguments[1:]:
            if k.stringlen: continue
            ktypename = k.typename
            if ktypename in CToFortranTypes:
              ktypename = CToFortranTypes[ktypename]
            if ktypename in senumsext:
              fd.write('  character(*) :: ' + Letters[cnt] + '\n')
            elif ktypename == 'char':
              if k.char_type != 'single':
                fd.write('  character(*) :: ' + Letters[cnt] + '\n')
              elif k.char_type == 'single':
                fd.write('  character(len=1) :: ' + Letters[cnt] + '\n')
            elif k.array and k.stars:
              fd.write('  ' + ktypename + ', pointer :: ' +  Letters[cnt]  + '(:)\n')
            elif k.array:
              fd.write('  ' + ktypename + ' :: ' +  Letters[cnt]  + '(*)\n')
            else:
              fd.write('  ' + ktypename + ' :: ' + Letters[cnt] + '\n')
            cnt = cnt + 1
          fd.write('  PetscErrorCode z\n')
          fd.write('  call ' + funname  + '(PetscObjectCast(')
          cnt = 0
          for k in fi.arguments:
            if k.stringlen: continue
            if cnt: fd.write(', ')
            fd.write(Letters[cnt])
            if cnt == 0: fd.write(')')
            cnt = cnt + 1
          fd.write(', z)\n')
          fd.write('  end subroutine \n')
          if funname.startswith('PetscObjectSAWs') or funname == 'PetscObjectViewSAWs':
            fd.write('#endif\n')

  # generate .eq. and .neq. for enums
  for i in enums.keys():
    if i in ['PetscEnum', 'PetscBool3']: continue
    with open(os.path.join(petscarch,'ftn',getAPI.mansecpath(enums[i].mansec),enums[i].includefile + 'f90'),"a") as fd:

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: ' + i + 'notequals\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: ' + i + 'equals\n')
      fd.write('#endif\n\n')
      fd.write('  function ' + i + 'notequals(A,B)\n')
      fd.write('    logical ' + i + 'notequals\n')
      fd.write('    type(e' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'notequals = (A%v .ne. B%v)\n')
      fd.write('  end function\n')
      fd.write('  function ' + i + 'equals(A,B)\n')
      fd.write('    logical ' + i + 'equals\n')
      fd.write('    type(e' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'equals = (A%v .eq. B%v)\n')
      fd.write('  end function\n')

##########  $PETSC_ARCH/ftn/MANSEC/**/*f.c

  import re
  reg = re.compile(r'[-a-zA-Z0-9/._]*: [ ]*#define [ ]*([a-z0-9]*)_ [ ]*[a-z0-9]*')
  output = check_output('find src -type f -path "*/ftn-custom/*.c" | xargs grep "[ ]*#define [a-z0-9]*_ [ ]*[a-z0-9]*$"', shell=True).decode('utf-8')
  manualstubsfound = set()
  for f in output.split('\n'):
    manualstubsfound.add(reg.sub(r'\1',f))

  # convert function arguments from MPI_Comm to MPI_Fint
  for i in funcs:
    for j in funcs[i].arguments:
      j.typename = j.typename.replace('MPI_Comm','MPI_Fint')

  for i in classes:
    for j in classes[i].functions:
      for k in classes[i].functions[j].arguments:
        k.typename = k.typename.replace('MPI_Comm','MPI_Fint')

  for i in classes.keys():
    if i in ['PetscIntStack']: continue
    for j in classes[i].functions: # loop over functions in class
      generateCStub(pkgname,petscarch,manualstubsfound,senumsext,classes,structsext,j,classes[i].functions[j])

  for j in funcs.keys():
    if funcs[j].name in ['SlepcDebugViewMatrix']: continue
    generateCStub(pkgname,petscarch,manualstubsfound,senumsext,classes,structsext,funcs[j].name,funcs[j])

##########  $PETSC_ARCH/ftn/MANSEC/petscall.*

  # petscall.* contains all the include files associated with C petscMANSEC.h
  # these are used by src/MANSEC/ftn-mod/petscMANSECmod.F to generate the module for C petscMANSEC.h
  # src/MANSEC/ftn-mod/petscMANSECmod.F may also define additional modules that use petscMANSEC
  for i in mansecs.keys():
    d = os.path.join(petscarch,'ftn', getAPI.mansecpath(i))
    dd = os.path.join('../','ftn', getAPI.mansecpath(i))
    args = [os.path.join(d,i) for i in os.listdir(d) if i.endswith('.h')]
    for j in args:
      if not os.path.getsize(j): os.path.remove(j)
    with open(os.path.join(d,pkgname + 'all.h'),'w') as fd, open(os.path.join(d,pkgname + 'all.h90'),'w') as fd90, open(os.path.join(d,pkgname + 'all.hf90'),'w') as fdf90:
      if not i.startswith(pkgname): f = pkgname + i + '.h'
      else: f = i + '.h'
      includes = set()
      for j in files[f].included:
        if j in skipinc: continue
        j = j.replace('types.h','.h')
        includes.add(j)
        fd.write('#include <' + os.path.join(('petsc' if j.startswith('petsc') else 'slepc'),'finclude',j) + '>\n')
        if os.path.isfile(os.path.join(d,j)) :
          fd.write('#include <' + os.path.join(dd,j) + '>\n')
        if os.path.isfile(os.path.join(d,j + '90')):
          fd90.write('#include <' + os.path.join(dd,j + '90') + '>\n')
        if os.path.isfile(os.path.join(d,j + 'f90')):
          fdf90.write('#include <' + os.path.join(dd,j + 'f90') + '>\n')
      if f not in includes:
        fd.write('#include <' + os.path.join(pkgname,'finclude',f) + '>\n')
        fd.write('#include <' + os.path.join(dd,f) + '>\n')
        if os.path.isfile(os.path.join(d,f + '90')):
          fd90.write('#include <' + os.path.join(dd,f + '90') + '>\n')
        if os.path.isfile(os.path.join(d,f + 'f90')):
          fdf90.write('#include <' + os.path.join(dd,f + 'f90') + '>\n')

#
if __name__ ==  '__main__':
  import sys
  import argparse

  parser = argparse.ArgumentParser(description='generate PETSc/SLEPc FORTRAN stubs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--petsc-dir', metavar='path', required=True, help='PETSc root directory')
  parser.add_argument('--slepc-dir', metavar='path', required=False, help='SLEPc root directory when generating SLEPc bindings')
  parser.add_argument('--petsc-arch', metavar='string', required=True, help='PETSc arch name')
  args = parser.parse_args()

  ret = main(args.petsc_dir, args.slepc_dir, args.petsc_arch)
  sys.exit(ret)
