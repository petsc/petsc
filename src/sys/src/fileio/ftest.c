#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ftest.c,v 1.11 1997/09/05 18:39:47 gropp Exp bsmith $";
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
. fname - name of file
. mode  - mode.  One of 'r', 'w', 'e'
. uid,gid - user and group id to use

  Returns:
  1 if file exists with given mode, 0 otherwise.
+*/
#if defined (PARCH_nt)
int PetscTestFile( char *fname, char mode)
{
  int m;
  if (!fname) PetscFunctionReturn(0);
  
  if (mode == 'r') m = 4;
  if (mode == 'w') m = 2;
  if(!_access(fname,4))   PetscFunctionReturn(1);
  PetscFunctionReturn(0);
}
#else 
int PetscTestFile( char *fname, char mode,uid_t uid, gid_t gid )
{
  int         err;
  struct stat statbuf;
  int         stmode, rbit, wbit, ebit;
  
  PetscFunctionBegin;
  if (!fname) PetscFunctionReturn(0);
  
  /* Check to see if the environment variable is a valid regular FILE */
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
  }
  else if (statbuf.st_gid == gid) {
    rbit = S_IRGRP;
    wbit = S_IWGRP;
    ebit = S_IXGRP;
  }
  else {
    rbit = S_IROTH;
    wbit = S_IWOTH;
    ebit = S_IXOTH;
  }
  if (mode == 'r') {
    if ((stmode & rbit))   PetscFunctionReturn(1);
  }
  else if (mode == 'w') {
    if ((stmode & wbit))   PetscFunctionReturn(1);
  }
  else if (mode == 'e') {
    if ((stmode & ebit))   PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

#endif
