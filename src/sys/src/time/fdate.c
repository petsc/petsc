/*$Id: fdate.c,v 1.33 2000/01/11 20:59:38 bsmith Exp bsmith $*/

#include "petsc.h"
#include "sys.h"
#include "petscfix.h"
#include "pinclude/ptime.h"
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
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
#include <fcntl.h>
#include <time.h>  
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

static char starttime[64];
   
/*
  This function is called once during the initialize stage.
  It stashes the timestamp, and uses it when needed. This is so that 
  error handlers may report the date without generating possible
  additional system errors during the call to get the date.

*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscGetDate"
/*@C
    PetscGetDate - Gets the current date.

   Not collective

  Input Parameter:
.  len - length of string to hold date

  Output Parameter:
.  date - the date

  Level: beginner

  Notes:
    This is Y2K compliant.

    This function DOES make a system call and thus SHOULD NOT be called
    from an error handler. Use PetscGetInitialDate() instead.

.seealso: PetscGetInitialDate()

@*/
int PetscGetDate(char date[],int len)
{
  char           *str=0;
#if defined (PARCH_win32)
  time_t         aclock;
  int            ierr;
#else
  struct timeval tp;
  int            ierr;
#endif

  PetscFunctionBegin;
#if defined (PARCH_win32)
  time(&aclock);
  ierr = PetscStrncpy(date,asctime(localtime(&aclock)),len);CHKERRQ(ierr);
#else
  gettimeofday(&tp,(struct timezone *)0);
  ierr = PetscStrncpy(date,asctime(localtime((time_t*)&tp.tv_sec)),len);CHKERRQ(ierr);
#endif
  /* now strip out the new-line chars at the end of the string */
  ierr = PetscStrstr(date,"\n",&str);CHKERRQ(ierr);
  if (str) str[0] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscSetInitialDate"
int PetscSetInitialDate(void)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscGetDate(starttime,64);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscGetInitialDate"
/*@C
    PetscGetInitialDate - Gets the date the program was started 
      on.

   Not collective

  Input Parameter:
.  len - length of string to hold date

  Output Parameter:
.  date - the date

  Level: beginner

  Notes:
    This is Y2K compliant.

    This function does not make a system call and thus may be called
    from an error handler.

.seealso: PetscGetDate()

@*/
int PetscGetInitialDate(char date[],int len)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(date,starttime,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

