/*$Id: ghome.c,v 1.39 2001/03/23 23:20:30 balay Exp $*/
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
   On Windows NT machine the enviornmental variable HOME specifies the home directory.

   Concepts: home directory
@*/
int PetscGetHomeDirectory(char dir[],int maxlen)
{
  int ierr;
#if defined(PARCH_win32) || defined(PARCH_win32_gnu)
  char *d1 = getenv("HOME");
#else
  struct passwd *pw = 0;
#endif

  PetscFunctionBegin;
#if defined(PARCH_win32) || defined(PARCH_win32_gnu)
  if (!d1) d1 ="c:";
  ierr = PetscStrncpy(dir,d1,maxlen);CHKERRQ(ierr);
#elif !defined(PETSC_MISSING_GETPWUID)
  pw = getpwuid(getuid());
  if (!pw)  {dir[0] = 0; PetscFunctionReturn(0);}
  ierr = PetscStrncpy(dir,pw->pw_dir,maxlen);CHKERRQ(ierr);
#else 
  dir[0] = 0;
#endif
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
  int i,n,ierr;

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
