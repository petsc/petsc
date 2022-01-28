#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for realpath() */
#include <petscsys.h>
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
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

/*@C
   PetscGetRealPath - Get the path without symbolic links etc. and in absolute form.

   Not Collective

   Input Parameter:
.  path - path to resolve

   Output Parameter:
.  rpath - resolved path

   Level: developer

   Notes:
   rpath is assumed to be of length PETSC_MAX_PATH_LEN.

   Systems that use the automounter often generate absolute paths
   of the form "/tmp_mnt....".  However, the automounter will fail to
   mount this path if it is not already mounted, so we remove this from
   the head of the line.  This may cause problems if, for some reason,
   /tmp_mnt is valid and not the result of the automounter.

.seealso: PetscGetFullPath()
@*/
PetscErrorCode  PetscGetRealPath(const char path[],char rpath[])
{
  PetscErrorCode ierr;
  char           tmp3[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
#if !defined(PETSC_HAVE_REALPATH) && defined(PETSC_HAVE_READLINK)
  char           tmp1[PETSC_MAX_PATH_LEN],tmp4[PETSC_MAX_PATH_LEN],*tmp2;
  size_t         N,len,len1,len2;
  int            n,m;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_REALPATH)
  PetscAssertFalse(!realpath(path,rpath),PETSC_COMM_SELF,PETSC_ERR_LIB,"realpath()");
#elif defined(PETSC_HAVE_READLINK)
  /* Algorithm: we move through the path, replacing links with the real paths.   */
  ierr = PetscStrcpy(rpath,path);CHKERRQ(ierr);
  ierr = PetscStrlen(rpath,&N);CHKERRQ(ierr);
  while (N) {
    ierr    = PetscStrncpy(tmp1,rpath,N);CHKERRQ(ierr);
    tmp1[N] = 0;
    n       = readlink(tmp1,tmp3,PETSC_MAX_PATH_LEN);
    if (n > 0) {
      tmp3[n] = 0; /* readlink does not automatically add 0 to string end */
      if (tmp3[0] != '/') {
        ierr    = PetscStrchr(tmp1,'/',&tmp2);CHKERRQ(ierr);
        ierr    = PetscStrlen(tmp1,&len1);CHKERRQ(ierr);
        ierr    = PetscStrlen(tmp2,&len2);CHKERRQ(ierr);
        m       = len1 - len2;
        ierr    = PetscStrncpy(tmp4,tmp1,m);CHKERRQ(ierr);
        tmp4[m] = 0;
        ierr    = PetscStrlen(tmp4,&len);CHKERRQ(ierr);
        ierr    = PetscStrlcat(tmp4,"/",PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
        ierr    = PetscStrlcat(tmp4,tmp3,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
        ierr    = PetscGetRealPath(tmp4,rpath);CHKERRQ(ierr);
        ierr    = PetscStrlcat(rpath,path+N,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      } else {
        ierr = PetscGetRealPath(tmp3,tmp1);CHKERRQ(ierr);
        ierr = PetscStrncpy(rpath,tmp1,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
        ierr = PetscStrlcat(rpath,path+N,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
    ierr = PetscStrchr(tmp1,'/',&tmp2);CHKERRQ(ierr);
    if (tmp2) {
      ierr = PetscStrlen(tmp1,&len1);CHKERRQ(ierr);
      ierr = PetscStrlen(tmp2,&len2);CHKERRQ(ierr);
      N    = len1 - len2;
    } else {
      ierr = PetscStrlen(tmp1,&N);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrncpy(rpath,path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
#else /* Just punt */
  ierr = PetscStrcpy(rpath,path);CHKERRQ(ierr);
#endif

  /* remove garbage some automounters put at the beginning of the path */
  ierr = PetscStrncmp("/tmp_mnt/",rpath,9,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcpy(tmp3,rpath + 8);CHKERRQ(ierr);
    ierr = PetscStrcpy(rpath,tmp3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
