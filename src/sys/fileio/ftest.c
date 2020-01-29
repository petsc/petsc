
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

static PetscErrorCode PetscTestOwnership(const char fname[], char mode, uid_t fuid, gid_t fgid, int fmode, PetscBool  *flg)
{
  int            m = R_OK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == 'r') m = R_OK;
  else if (mode == 'w') m = W_OK;
  else if (mode == 'x') m = X_OK;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Mode must be one of r, w, or x");
#if defined(PETSC_HAVE_ACCESS)
  if (!access(fname, m)) {
    ierr = PetscInfo1(NULL,"System call access() succeeded on file %s\n",fname);CHKERRQ(ierr);
    *flg = PETSC_TRUE;
  } else {
    ierr = PetscInfo1(NULL,"System call access() failed on file %s\n",fname);CHKERRQ(ierr);
    *flg = PETSC_FALSE;
  }
#else
  if (m == X_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Unable to check execute permission for file %s", fname);
  if (!_access(fname, m)) *flg = PETSC_TRUE;
#endif
  PetscFunctionReturn(0);
}

#else  /* PETSC_HAVE_ACCESS or PETSC_HAVE__ACCESS */

static PetscErrorCode PetscTestOwnership(const char fname[], char mode, uid_t fuid, gid_t fgid, int fmode, PetscBool  *flg)
{
  uid_t          uid;
  gid_t          *gid = NULL;
  int            numGroups;
  int            rbit = S_IROTH;
  int            wbit = S_IWOTH;
  int            ebit = S_IXOTH;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Get the number of supplementary group IDs */
#if !defined(PETSC_MISSING_GETGROUPS)
  numGroups = getgroups(0, gid); if (numGroups < 0) SETERRQ(PETSC_COMM_SELF,numGroups, "Unable to count supplementary group IDs");
  ierr = PetscMalloc1(numGroups+1, &gid);CHKERRQ(ierr);
#else
  numGroups = 0;
#endif

  /* Get the (effective) user and group of the caller */
  uid    = geteuid();
  gid[0] = getegid();

  /* Get supplementary group IDs */
#if !defined(PETSC_MISSING_GETGROUPS)
  ierr = getgroups(numGroups, gid+1); if (ierr < 0) SETERRQ(PETSC_COMM_SELF,ierr, "Unable to obtain supplementary group IDs");
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
  ierr = PetscFree(gid);CHKERRQ(ierr);

  if (mode == 'r') {
    if (fmode & rbit) *flg = PETSC_TRUE;
  } else if (mode == 'w') {
    if (fmode & wbit) *flg = PETSC_TRUE;
  } else if (mode == 'x') {
    if (fmode & ebit) *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_ACCESS */

static PetscErrorCode PetscGetFileStat(const char fname[], uid_t *fileUid, gid_t *fileGid, int *fileMode,PetscBool  *exists)
{
  struct stat    statbuf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *fileMode = 0;
  *exists = PETSC_FALSE;
#if defined(PETSC_HAVE_STAT_NO_CONST)
  ierr = stat((char*) fname, &statbuf);
#else
  ierr = stat(fname, &statbuf);
#endif
  if (ierr) {
#if defined(EOVERFLOW)
    if (errno == EOVERFLOW) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"EOVERFLOW in stat(), configure PETSc --with-large-file-io=1 to support files larger than 2GiB");
#endif
    ierr    = PetscInfo1(NULL,"System call stat() failed on file %s\n",fname);CHKERRQ(ierr);
    *exists = PETSC_FALSE;
  } else {
    ierr      = PetscInfo1(NULL,"System call stat() succeeded on file %s\n",fname);CHKERRQ(ierr);
    *exists   = PETSC_TRUE;
    *fileUid  = statbuf.st_uid;
    *fileGid  = statbuf.st_gid;
    *fileMode = statbuf.st_mode;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscTestFile - checks for the existence of a file

   Not Collective

   Input Parameter:
+  fname - the filename
-  mode - either 'r', 'w', 'x' or '\0'

   Output Parameter:
.  flg - the file exists and satisfies the mode

   Level: intermediate

   Notes: if mode is '\0', no permissions checks are performed

.seealso: PetscTestDirectory(), PetscLs()
@*/
PetscErrorCode  PetscTestFile(const char fname[], char mode, PetscBool  *flg)
{
  uid_t          fuid;
  gid_t          fgid;
  int            fmode;
  PetscErrorCode ierr;
  PetscBool      exists;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!fname) PetscFunctionReturn(0);

  ierr = PetscGetFileStat(fname, &fuid, &fgid, &fmode, &exists);CHKERRQ(ierr);
  if (!exists) PetscFunctionReturn(0);
  /* Except for systems that have this broken stat macros (rare), this is the correct way to check for a regular file */
  if (!S_ISREG(fmode)) PetscFunctionReturn(0);
  /* return if asked to check for existence only */
  if (mode == '\0') { *flg = exists; PetscFunctionReturn(0); }
  ierr = PetscTestOwnership(fname, mode, fuid, fgid, fmode, flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscTestDirectory - checks for the existence of a directory

   Not Collective

   Input Parameter:
+  dirname - the directory name
-  mode - either 'r', 'w', or 'x'

   Output Parameter:
.  flg - the directory exists and satisfies the mode

   Level: intermediate

.seealso: PetscTestFile(), PetscLs()
@*/
PetscErrorCode  PetscTestDirectory(const char dirname[],char mode,PetscBool  *flg)
{
  uid_t          fuid;
  gid_t          fgid;
  int            fmode;
  PetscErrorCode ierr;
  PetscBool      exists;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  if (!dirname) PetscFunctionReturn(0);

  ierr = PetscGetFileStat(dirname, &fuid, &fgid, &fmode,&exists);CHKERRQ(ierr);
  if (!exists) PetscFunctionReturn(0);
  /* Except for systems that have this broken stat macros (rare), this
     is the correct way to check for a directory */
  if (!S_ISDIR(fmode)) PetscFunctionReturn(0);

  ierr = PetscTestOwnership(dirname, mode, fuid, fgid, fmode, flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscLs - produce a listing of the files in a directory

   Collective

   Input Parameter:
+  comm - the MPI communicator
.  dirname - the directory name
-  tlen - the length of the buffer found[]

   Output Parameter:
+  found - listing of files
-  flg - the directory exists

   Level: intermediate

.seealso: PetscTestFile(), PetscLs()
@*/
PetscErrorCode  PetscLs(MPI_Comm comm,const char dirname[],char found[],size_t tlen,PetscBool  *flg)
{
  PetscErrorCode ierr;
  size_t         len;
  char           *f,program[PETSC_MAX_PATH_LEN];
  FILE           *fp;

  PetscFunctionBegin;
  ierr = PetscStrcpy(program,"ls ");CHKERRQ(ierr);
  ierr = PetscStrcat(program,dirname);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(comm,NULL,program,"r",&fp);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
  f = fgets(found,tlen,fp);
  if (f) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  while (f) {
    ierr = PetscStrlen(found,&len);CHKERRQ(ierr);
    f    = fgets(found+len,tlen-len,fp);
  }
  if (*flg) {ierr = PetscInfo2(NULL,"ls on %s gives \n%s\n",dirname,found);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPClose(comm,fp);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
  PetscFunctionReturn(0);
}
