/*$Id: ftest.c,v 1.39 2001/04/04 21:18:39 bsmith Exp $*/

#include "petsc.h"
#include "petscsys.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
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
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

#if defined (PETSC_HAVE_U_ACCESS) || defined(PETSC_HAVE_ACCESS)
#if !defined(R_OK)
#define R_OK 04
#endif
#if !defined(W_OK)
#define W_OK 02
#endif
#if !defined(X_OK)
#define X_OK 01
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscTestFile"
/*@C
  PetscTestFile - Test for a file existing with a specified mode.

  Input Parameters:
+ fname - name of file
- mode  - mode.  One of r, w, or x

  Output Parameter:
.  flg - PETSC_TRUE if file exists with given mode, PETSC_FALSE otherwise.

  Level: intermediate

@*/
int PetscTestFile(const char fname[],char mode,PetscTruth *flg)
{
  int m;
  
  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);
  
  if (mode == 'r') m = R_OK;
  else if (mode == 'w') m = W_OK;
  else if (mode == 'x') m = X_OK;
  else SETERRQ(1,"Mode must be one of r, w, or x");
#if defined(PETSC_HAVE_U_ACCESS)
  if (m == X_OK) SETERRQ1(PETSC_ERR_SUP,"Unable to check execute permission for file %s",fname);
  if(!_access(fname,m)) *flg = PETSC_TRUE;
#else
  if(!access(fname,m))  *flg = PETSC_TRUE;
#endif
  PetscFunctionReturn(0);
}
#else 
#undef __FUNCT__  
#define __FUNCT__ "PetscTestFile"
int PetscTestFile(const char fname[],char mode,PetscTruth *flg)
{
  struct stat statbuf;
  int         err,stmode,rbit,wbit,ebit;
  uid_t       uid;
  gid_t       gid;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);

  /* Get the (effective) user and group of the caller */
  uid = geteuid();
  gid = getegid();

#if defined(PETSC_HAVE_STAT_NO_CONST)
  err = stat((char*)fname,&statbuf);
#else
  err = stat(fname,&statbuf);
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
