/*
      Code for manipulating files.
*/
#include "petsc.h"
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
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscGetHomeDirectory"
/*@C
   PetscGetHomeDirectory - Returns home directory name.

   Not Collective

   Input Parameter:
.  maxlen - maximum lengh allowed

   Output Parameter:
.  dir - contains the home directory. Must be long enough to hold the name.

   Level: developer

   Note:
   If PETSc cannot determine the home directory it makes dir a null string

   On Windows machines the enviornmental variable HOME specifies the home directory.

   Concepts: home directory
@*/
int PetscGetHomeDirectory(char dir[],int maxlen)
{
  int  ierr;
  char *d1 = 0;
#if defined(PETSC_HAVE_GETPWUID)
  struct passwd *pw = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_GETPWUID)
  pw = getpwuid(getuid());
  if (pw)  {
    d1 = pw->pw_dir;
  }
#else
  d1 = getenv("HOME");
#endif
  if (d1) {
    ierr = PetscStrncpy(dir,d1,maxlen);CHKERRQ(ierr);
  } else if (maxlen > 0) {
    dir[0] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscFixFilename"
/*@C
    PetscFixFilename - Fixes a file name so that it is correct for both Unix and 
    Windows by using the correct / or \ to seperate directories.

   Not Collective

   Input Parameter:
.  filein - name of file to be fixed

   Output Parameter:
.  fileout - the fixed name. Should long enough to hold the filename.

   Level: advanced

   Notes:
   Call PetscFixFilename() just before calling fopen().
@*/
int PetscFixFilename(const char filein[],char fileout[])
{
  int    i,ierr;
  size_t n;

  PetscFunctionBegin;
  if (!filein || !fileout) PetscFunctionReturn(0);

  ierr = PetscStrlen(filein,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
#if defined(PARCH_win32)
    if (filein[i] == '/') fileout[i] = '\\';
#else
    if (filein[i] == '\\') fileout[i] = '/';
#endif
    else fileout[i] = filein[i];
  }
  fileout[n] = 0;

  PetscFunctionReturn(0);
}
