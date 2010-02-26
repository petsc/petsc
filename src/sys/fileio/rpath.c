#define PETSC_DLL

#include "petscsys.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscGetRelativePath"
/*@C
   PetscGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Not Collective

   Input parameters:
+  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
-  flen     - size of path

   Level: developer

   Concepts: relative path
   Concepts: path^relative

.seealso: PetscGetFullPath()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetRelativePath(const char fullpath[],char path[],size_t flen)
{
  char           *p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Find string after last / or entire string if no / */
  ierr = PetscStrrchr(fullpath,'/',&p);CHKERRQ(ierr);
  ierr = PetscStrncpy(path,p,flen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
