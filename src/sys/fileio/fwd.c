
/*
      Code for manipulating files.
*/
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
#if defined(PETSC_HAVE_DIRECT_H)
#include <direct.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Not Collective

   Input Parameters:
.  len  - maximum length of path

   Output Parameter:
.  path - use to hold the result value. The string should be long enough
          to hold the path.

   Level: developer

@*/
PetscErrorCode  PetscGetWorkingDirectory(char path[],size_t len)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_GETCWD)
  PetscCheckFalse(!getcwd(path,len),PETSC_COMM_SELF,PETSC_ERR_LIB,"getcwd()");
#elif defined(PETSC_HAVE__GETCWD)
  _getcwd(path,len);
#elif defined(PETSC_HAVE_GETWD)
  getwd(path);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS, "Could not find getcwd() or getwd()");
#endif
  PetscFunctionReturn(0);
}

