#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ftest.c,v 1.13 1998/01/06 20:09:29 bsmith Exp curfman $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscTestFile"
/*+
  PetscTestFile - Test for a file existing with a specified mode.

  Input Parameters:
+ fname - name of file
- mode  - mode.  One of 'r', 'w', 'x'

  Output Parameter:
  flag - PETSC_TRUE if file exists with given mode, PETSC_FALSE otherwise.
+*/
#if defined (PARCH_nt)
int PetscTestFile( char *fname, char mode,PetscTruth *flag)
{
  int m;
  
  PetscFunctionBegin;
  *flag = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);
  
  if (mode == 'r') m = 4;
  if (mode == 'w') m = 2;
  if(!_access(fname,4))  *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#else 
int PetscTestFile( char *fname, char mode,PetscTruth *flag)
{
  struct stat statbuf;
  int         err,stmode, rbit, wbit, ebit;
  uid_t       uid;
  gid_t       gid;

  PetscFunctionBegin;
  *flag = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);

  /* Get the (effective) user and group of the caller */
  uid = geteuid();
  gid = getegid();
  
  err = stat( fname, &statbuf );
  if (err != 0) PetscFunctionReturn(0);

  /* At least the file exists ... */
  stmode = statbuf.st_mode;
  /* Except for systems that have this broken stat macros (rare), this
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
    if ((stmode & rbit))   *flag = PETSC_TRUE;
  } else if (mode == 'w') {
    if ((stmode & wbit))   *flag = PETSC_TRUE;
  } else if (mode == 'x') {
    if ((stmode & ebit))   *flag = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#endif
