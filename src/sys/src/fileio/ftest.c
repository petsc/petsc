/*$Id: ftest.c,v 1.23 1999/10/24 14:01:25 bsmith Exp bsmith $*/

#include "petsc.h"
#include "sys.h"
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
#include "pinclude/petscfix.h"

#if defined (PARCH_win32)
#undef __FUNC__  
#define __FUNC__ "PetscTestFile"
/*+
  PetscTestFile - Test for a file existing with a specified mode.

  Input Parameters:
+ fname - name of file
- mode  - mode.  One of 'r', 'w', 'x'

  Output Parameter:
  flg - PETSC_TRUE if file exists with given mode, PETSC_FALSE otherwise.

+*/
int PetscTestFile( const char fname[], char mode,PetscTruth *flg)
{
  int m;
  
  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);
  
  if (mode == 'r') m = 4;
  if (mode == 'w') m = 2;
  if(!_access(fname,4))  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#else 
#undef __FUNC__  
#define __FUNC__ "PetscTestFile"
int PetscTestFile( const char fname[], char mode,PetscTruth *flg)
{
  struct stat statbuf;
  int         err,stmode, rbit, wbit, ebit;
  uid_t       uid;
  gid_t       gid;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);

  /* Get the (effective) user and group of the caller */
  uid = geteuid();
  gid = getegid();

#if !defined (PARCH_ascired)  
  err = stat( fname, &statbuf );
#else
  err = stat( (char*)fname, &statbuf );
#endif
  if (err != 0) PetscFunctionReturn(0);

  /* At least the file exists ... */
  stmode = statbuf.st_mode;
  /*
     Except for systems that have this broken stat macros (rare), this
     is the correct way to check for a (not) regular file */
  if (!S_ISREG(stmode)) PetscFunctionReturn(0);

  /* Test for accessible. */
  if (statbuf.st_uid == uid) {
    rbit = S_IRUSR;
    wbit = S_IWUSR;
    ebit = S_IXUSR;
  } else if (statbuf.st_gid == gid) {
    rbit = S_IRGRP;
    wbit = S_IWGRP;
    ebit = S_IXGRP;
  } else {
    rbit = S_IROTH;
    wbit = S_IWOTH;
    ebit = S_IXOTH;
  }
  if (mode == 'r') {
    if ((stmode & rbit))   *flg = PETSC_TRUE;
  } else if (mode == 'w') {
    if ((stmode & wbit))   *flg = PETSC_TRUE;
  } else if (mode == 'x') {
    if ((stmode & ebit))   *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#endif
