/*$Id: fuser.c,v 1.31 2001/03/23 23:20:45 balay Exp $*/
/*
      Code for manipulating files.
*/
#include "petscconfig.h"
#include "petsc.h"
#include "petscsys.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#include "petscfix.h"


#if defined(PETSC_HAVE_GET_USER_NAME)
#undef __FUNCT__  
#define __FUNCT__ "PetscGetUserName"
int PetscGetUserName(char name[],int nlen)
{
  PetscFunctionBegin;
  GetUserName((LPTSTR)name,(LPDWORD)(&nlen));
  PetscFunctionReturn(0);
}

#elif defined(HAVE_PWD_H)
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
int PetscGetUserName(char name[],int nlen)
{
  struct passwd *pw;
  int           ierr;

  PetscFunctionBegin;
  pw = getpwuid(getuid());
  if (!pw) {ierr = PetscStrncpy(name,"Unknown",nlen);CHKERRQ(ierr);}
  else     {ierr = PetscStrncpy(name,pw->pw_name,nlen);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "PetscGetUserName"
int PetscGetUserName(char *name,int nlen)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(name,"Unknown",nlen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* !HAVE_PWD_H */

