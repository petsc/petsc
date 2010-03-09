#define PETSC_DLL
/*
      Code for manipulating files.
*/
#include "petscsys.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_DIRECT_H)
#include <direct.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscGetWorkingDirectory"
/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Not Collective

   Input Parameters:
.  len  - maximum length of path

   Output Parameter:
.  path - use to hold the result value. The string should be long enough
          to hold the path.

   Level: developer

   Concepts: working directory

@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetWorkingDirectory(char path[],size_t len)
{
#if defined(PETSC_HAVE_GETCWD)
  PetscFunctionBegin;
  getcwd(path,len);
  PetscFunctionReturn(0);
#elif defined(PETSC_HAVE__GETCWD)
  PetscFunctionBegin;
  _getcwd(path,len);
  PetscFunctionReturn(0);
#elif defined(PETSC_HAVE_GETWD)
  PetscFunctionBegin;
  getwd(path);
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_ERR_SUP_SYS, "Could not find getcwd() or getwd()");
#endif
}

