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
   rpath is assumed to be of length `PETSC_MAX_PATH_LEN`.

   Systems that use the automounter often generate absolute paths
   of the form "/tmp_mnt....".  However, the automounter will fail to
   mount this path if it is not already mounted, so we remove this from
   the head of the line.  This may cause problems if, for some reason,
   /tmp_mnt is valid and not the result of the automounter.

.seealso: `PetscGetFullPath()`
@*/
PetscErrorCode PetscGetRealPath(const char path[], char rpath[])
{
  char      tmp3[PETSC_MAX_PATH_LEN];
  PetscBool flg;
#if !defined(PETSC_HAVE_REALPATH) && defined(PETSC_HAVE_READLINK)
  char   tmp1[PETSC_MAX_PATH_LEN], tmp4[PETSC_MAX_PATH_LEN], *tmp2;
  size_t N, len, len1, len2;
  int    n, m;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_REALPATH)
  PetscCheck(realpath(path, rpath), PETSC_COMM_SELF, PETSC_ERR_LIB, "realpath()");
#elif defined(PETSC_HAVE_READLINK)
  /* Algorithm: we move through the path, replacing links with the real paths.   */
  PetscCall(PetscStrcpy(rpath, path));
  PetscCall(PetscStrlen(rpath, &N));
  while (N) {
    PetscCall(PetscStrncpy(tmp1, rpath, N));
    tmp1[N] = 0;
    n       = readlink(tmp1, tmp3, PETSC_MAX_PATH_LEN);
    if (n > 0) {
      tmp3[n] = 0; /* readlink does not automatically add 0 to string end */
      if (tmp3[0] != '/') {
        PetscCall(PetscStrchr(tmp1, '/', &tmp2));
        PetscCall(PetscStrlen(tmp1, &len1));
        PetscCall(PetscStrlen(tmp2, &len2));
        m = len1 - len2;
        PetscCall(PetscStrncpy(tmp4, tmp1, m));
        tmp4[m] = 0;
        PetscCall(PetscStrlen(tmp4, &len));
        PetscCall(PetscStrlcat(tmp4, "/", PETSC_MAX_PATH_LEN));
        PetscCall(PetscStrlcat(tmp4, tmp3, PETSC_MAX_PATH_LEN));
        PetscCall(PetscGetRealPath(tmp4, rpath));
        PetscCall(PetscStrlcat(rpath, path + N, PETSC_MAX_PATH_LEN));
      } else {
        PetscCall(PetscGetRealPath(tmp3, tmp1));
        PetscCall(PetscStrncpy(rpath, tmp1, PETSC_MAX_PATH_LEN));
        PetscCall(PetscStrlcat(rpath, path + N, PETSC_MAX_PATH_LEN));
      }
      PetscFunctionReturn(0);
    }
    PetscCall(PetscStrchr(tmp1, '/', &tmp2));
    if (tmp2) {
      PetscCall(PetscStrlen(tmp1, &len1));
      PetscCall(PetscStrlen(tmp2, &len2));
      N = len1 - len2;
    } else {
      PetscCall(PetscStrlen(tmp1, &N));
    }
  }
  PetscCall(PetscStrncpy(rpath, path, PETSC_MAX_PATH_LEN));
#else /* Just punt */
  PetscCall(PetscStrcpy(rpath, path));
#endif

  /* remove garbage some automounters put at the beginning of the path */
  PetscCall(PetscStrncmp("/tmp_mnt/", rpath, 9, &flg));
  if (flg) {
    PetscCall(PetscStrcpy(tmp3, rpath + 8));
    PetscCall(PetscStrcpy(rpath, tmp3));
  }
  PetscFunctionReturn(0);
}
