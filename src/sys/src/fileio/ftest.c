#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ftest.c,v 1.8 1997/02/27 00:44:16 balay Exp balay $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscTestFile" /* ADIC Ignore */
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
  if (!fname) return 0;
  
  if (mode == 'r') m = 4;
  if (mode == 'w') m = 2;
  if(!_access(fname,4)) return 1;
  return 0;
}
#else 
int PetscTestFile( char *fname, char mode,uid_t uid, gid_t gid )
{
  int         err;
  struct stat statbuf;
  int         stmode, rbit, wbit, ebit;
  
  if (!fname) return 0;
  
  /* Check to see if the environment variable is a valid regular FILE */
  err = stat( fname, &statbuf );
  if (err != 0) return 0;

  /* At least the file exists ... */
  stmode = statbuf.st_mode;
#if defined(PARCH_rs6000) || defined(PARCH_hpux)
#define S_IFREG _S_IFREG
#endif
#if defined(PARCH_alpha) || defined(PARCH_paragon)
  if (!S_ISREG(stmode)) return 0;
#else
  if (!(S_IFREG & stmode)) return 0;
#endif
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
    if ((stmode & rbit)) return 1;
  }
  else if (mode == 'w') {
    if ((stmode & wbit)) return 1;
  }
  else if (mode == 'e') {
    if ((stmode & ebit)) return 1;
  }
  return 0;
}

#endif
