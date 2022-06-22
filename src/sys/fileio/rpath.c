
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
   PetscGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Not Collective

   Input parameters:
+  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
-  flen     - size of path

   Level: developer

.seealso: `PetscGetFullPath()`
@*/
PetscErrorCode  PetscGetRelativePath(const char fullpath[],char path[],size_t flen)
{
  char           *p;

  PetscFunctionBegin;
  /* Find string after last / or entire string if no / */
  PetscCall(PetscStrrchr(fullpath,'/',&p));
  PetscCall(PetscStrncpy(path,p,flen));
  PetscFunctionReturn(0);
}
