#define PETSC_DLL
/*
      Code for manipulating files.
*/
#include "petscsys.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_WINDOWS_H)
#include <windows.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif


#if defined(PETSC_HAVE_GET_USER_NAME)
#undef __FUNCT__  
#define __FUNCT__ "PetscGetUserName"
PetscErrorCode PETSC_DLLEXPORT PetscGetUserName(char name[],size_t nlen)
{
  PetscFunctionBegin;
  GetUserName((LPTSTR)name,(LPDWORD)(&nlen));
  PetscFunctionReturn(0);
}

#elif defined(PETSC_HAVE_PWD_H)
#undef __FUNCT__  
#define __FUNCT__ "PetscGetUserName"
/*@C
    PetscGetUserName - Returns the name of the user.

    Not Collective

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains user name.  Must be long enough to hold the name

    Level: developer

    Concepts: user name

.seealso: PetscGetHostName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetUserName(char name[],size_t nlen)
{
  struct passwd *pw=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_GETPWUID)
  pw = getpwuid(getuid());
#endif
  if (!pw) {ierr = PetscStrncpy(name,"Unknown",nlen);CHKERRQ(ierr);}
  else     {ierr = PetscStrncpy(name,pw->pw_name,nlen);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "PetscGetUserName"
PetscErrorCode PETSC_DLLEXPORT PetscGetUserName(char *name,size_t nlen)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(name,"Unknown",nlen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* !PETSC_HAVE_PWD_H */

