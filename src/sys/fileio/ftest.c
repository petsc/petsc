
#include <petscsys.h>
#include <errno.h>
#if defined(PETSC_HAVE_PWD_H)
  #include <pwd.h>
#endif
#include <ctype.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
  #include <unistd.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
  #include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_IO_H)
  #include <io.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
  #include <sys/systeminfo.h>
#endif

#if defined(PETSC_HAVE__ACCESS) || defined(PETSC_HAVE_ACCESS)

static PetscErrorCode PetscTestOwnership(const char fname[], char mode, uid_t fuid, gid_t fgid, int fmode, PetscBool *flg)
{
  int m = R_OK;

  PetscFunctionBegin;
  if (mode == 'r') m = R_OK;
  else if (mode == 'w') m = W_OK;
  else if (mode == 'x') m = X_OK;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mode must be one of r, w, or x");
  #if defined(PETSC_HAVE_ACCESS)
  if (!access(fname, m)) {
    PetscCall(PetscInfo(NULL, "System call access() succeeded on file %s\n", fname));
    *flg = PETSC_TRUE;
  } else {
    PetscCall(PetscInfo(NULL, "System call access() failed on file %s\n", fname));
    *flg = PETSC_FALSE;
  }
  #else
  PetscCheck(m != X_OK, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unable to check execute permission for file %s", fname);
  if (!_access(fname, m)) *flg = PETSC_TRUE;
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#else /* PETSC_HAVE_ACCESS or PETSC_HAVE__ACCESS */

static PetscErrorCode PetscTestOwnership(const char fname[], char mode, uid_t fuid, gid_t fgid, int fmode, PetscBool *flg)
{
  uid_t  uid;
  gid_t *gid = NULL;
  int    numGroups;
  int    rbit = S_IROTH;
  int    wbit = S_IWOTH;
  int    ebit = S_IXOTH;
  #if !defined(PETSC_MISSING_GETGROUPS)
  int    err;
  #endif

  PetscFunctionBegin;
  /* Get the number of supplementary group IDs */
  #if !defined(PETSC_MISSING_GETGROUPS)
  numGroups = getgroups(0, gid);
  PetscCheck(numGroups >= 0, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to count supplementary group IDs");
  PetscCall(PetscMalloc1(numGroups + 1, &gid));
  #else
  numGroups = 0;
  #endif

  /* Get the (effective) user and group of the caller */
  uid    = geteuid();
  gid[0] = getegid();

  /* Get supplementary group IDs */
  #if !defined(PETSC_MISSING_GETGROUPS)
  err = getgroups(numGroups, gid + 1);
  PetscCheck(err >= 0, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to obtain supplementary group IDs");
  #endif

  /* Test for accessibility */
  if (fuid == uid) {
    rbit = S_IRUSR;
    wbit = S_IWUSR;
    ebit = S_IXUSR;
  } else {
    int g;

    for (g = 0; g <= numGroups; g++) {
      if (fgid == gid[g]) {
        rbit = S_IRGRP;
        wbit = S_IWGRP;
        ebit = S_IXGRP;
        break;
      }
    }
  }
  PetscCall(PetscFree(gid));

  if (mode == 'r') {
    if (fmode & rbit) *flg = PETSC_TRUE;
  } else if (mode == 'w') {
    if (fmode & wbit) *flg = PETSC_TRUE;
  } else if (mode == 'x') {
    if (fmode & ebit) *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_ACCESS */

static PetscErrorCode PetscGetFileStat(const char fname[], uid_t *fileUid, gid_t *fileGid, int *fileMode, PetscBool *exists)
{
  struct stat statbuf;
  int         ierr;

  PetscFunctionBegin;
  *fileMode = 0;
  *exists   = PETSC_FALSE;
#if defined(PETSC_HAVE_STAT_NO_CONST)
  ierr = stat((char *)fname, &statbuf);
#else
  ierr = stat(fname, &statbuf);
#endif
  if (ierr) {
#if defined(EOVERFLOW)
    PetscCheck(errno != EOVERFLOW, PETSC_COMM_SELF, PETSC_ERR_SYS, "EOVERFLOW in stat(), configure PETSc --with-large-file-io=1 to support files larger than 2GiB");
#endif
    PetscCall(PetscInfo(NULL, "System call stat() failed on file %s\n", fname));
    *exists = PETSC_FALSE;
  } else {
    PetscCall(PetscInfo(NULL, "System call stat() succeeded on file %s\n", fname));
    *exists   = PETSC_TRUE;
    *fileUid  = statbuf.st_uid;
    *fileGid  = statbuf.st_gid;
    *fileMode = statbuf.st_mode;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscTestFile - checks for the existence of a file

   Not Collective

   Input Parameters:
+  fname - the filename
-  mode - either 'r', 'w', 'x' or '\0'

   Output Parameter:
.  flg - the file exists and satisfies the mode

   Level: intermediate

   Note:
   If mode is '\0', no permissions checks are performed

.seealso: `PetscTestDirectory()`, `PetscLs()`
@*/
PetscErrorCode PetscTestFile(const char fname[], char mode, PetscBool *flg)
{
  uid_t     fuid;
  gid_t     fgid;
  int       fmode;
  PetscBool exists;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscGetFileStat(fname, &fuid, &fgid, &fmode, &exists));
  if (!exists) PetscFunctionReturn(PETSC_SUCCESS);
  /* Except for systems that have this broken stat macros (rare), this is the correct way to check for a regular file */
  if (!S_ISREG(fmode)) PetscFunctionReturn(PETSC_SUCCESS);
  /* return if asked to check for existence only */
  if (mode == '\0') {
    *flg = exists;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscTestOwnership(fname, mode, fuid, fgid, fmode, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscTestDirectory - checks for the existence of a directory

   Not Collective

   Input Parameters:
+  dirname - the directory name
-  mode - either 'r', 'w', or 'x'

   Output Parameter:
.  flg - the directory exists and satisfies the mode

   Level: intermediate

.seealso: `PetscTestFile()`, `PetscLs()`, `PetscRMTree()`
@*/
PetscErrorCode PetscTestDirectory(const char dirname[], char mode, PetscBool *flg)
{
  uid_t     fuid;
  gid_t     fgid;
  int       fmode;
  PetscBool exists;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!dirname) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscGetFileStat(dirname, &fuid, &fgid, &fmode, &exists));
  if (!exists) PetscFunctionReturn(PETSC_SUCCESS);
  /* Except for systems that have this broken stat macros (rare), this
     is the correct way to check for a directory */
  if (!S_ISDIR(fmode)) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscTestOwnership(dirname, mode, fuid, fgid, fmode, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscLs - produce a listing of the files in a directory

   Collective

   Input Parameters:
+  comm - the MPI communicator
.  dirname - the directory name
-  tlen - the length of the buffer `found`

   Output Parameters:
+  found - listing of files
-  flg - the directory exists

   Level: intermediate

.seealso: `PetscTestFile()`, `PetscRMTree()`, `PetscTestDirectory()`
@*/
PetscErrorCode PetscLs(MPI_Comm comm, const char dirname[], char found[], size_t tlen, PetscBool *flg)
{
  size_t len;
  char  *f, program[PETSC_MAX_PATH_LEN];
  FILE  *fp;

  PetscFunctionBegin;
  PetscCall(PetscStrncpy(program, "ls ", sizeof(program)));
  PetscCall(PetscStrlcat(program, dirname, sizeof(program)));
#if defined(PETSC_HAVE_POPEN)
  PetscCall(PetscPOpen(comm, NULL, program, "r", &fp));
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Cannot run external programs on this machine");
#endif
  f = fgets(found, tlen, fp);
  if (f) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  while (f) {
    PetscCall(PetscStrlen(found, &len));
    f = fgets(found + len, tlen - len, fp);
  }
  if (*flg) PetscCall(PetscInfo(NULL, "ls on %s gives \n%s\n", dirname, found));
#if defined(PETSC_HAVE_POPEN)
  PetscCall(PetscPClose(comm, fp));
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Cannot run external programs on this machine");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
