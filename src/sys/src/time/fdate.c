/* $Id: fdate.c,v 1.41 2001/03/23 23:20:44 balay Exp $*/

#include "petsc.h"
#include "petscsys.h"
#include "petscfix.h"
#if defined(HAVE_SYS_TIME_H)
#include <sys/types.h>
#include <sys/time.h>
#endif
#include <time.h>
#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
EXTERN_C_BEGIN
EXTERN int gettimeofday(struct timeval *,struct timezone *);
EXTERN_C_END
#endif

static char starttime[64];
   
/*
  This function is called once during the initialize stage.
  It stashes the timestamp, and uses it when needed. This is so that 
  error handlers may report the date without generating possible
  additional system errors during the call to get the date.

*/
#undef __FUNCT__  
#define __FUNCT__ "PetscGetDate"
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

#undef __FUNCT__  
#define __FUNCT__ "PetscSetInitialDate"
int PetscSetInitialDate(void)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscGetDate(starttime,64);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGetInitialDate"
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

