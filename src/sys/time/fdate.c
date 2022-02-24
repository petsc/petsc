
#include <petscsys.h>
#if defined(PETSC_HAVE_SYS_TIME_H)
#include <sys/time.h>
#endif
#include <time.h>
#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
PETSC_EXTERN int gettimeofday(struct timeval*,struct timezone*);
#endif

/*
  This function is called once during the initialize stage.
  It stashes the timestamp, and uses it when needed. This is so that
  error handlers may report the date without generating possible
  additional system errors during the call to get the date.

*/
/*@C
    PetscGetDate - Gets the current date.

   Not collective

  Input Parameter:
.  len - length of string to hold date

  Output Parameter:
.  date - the date

  Level: beginner

    This function DOES make a system call and thus SHOULD NOT be called
    from an error handler.

@*/
PetscErrorCode  PetscGetDate(char date[],size_t len)
{
  char           *str=NULL;
#if defined(PETSC_HAVE_TIME)
  time_t         aclock;
#else
  struct timeval tp;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_TIME)
  time(&aclock);
  CHKERRQ(PetscStrncpy(date,asctime(localtime(&aclock)),len));
#else
  gettimeofday(&tp,(struct timezone*)0);
  CHKERRQ(PetscStrncpy(date,asctime(localtime((time_t*)&tp.tv_sec)),len));
#endif
  /* now strip out the new-line chars at the end of the string */
  CHKERRQ(PetscStrstr(date,"\n",&str));
  if (str) str[0] = 0;
  PetscFunctionReturn(0);
}
