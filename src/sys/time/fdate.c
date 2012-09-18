
#include <petscsys.h>
#if defined(PETSC_HAVE_SYS_TIME_H)
#include <sys/types.h>
#include <sys/time.h>
#endif
#include <time.h>
#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
EXTERN_C_BEGIN
extern int gettimeofday(struct timeval *,struct timezone *);
EXTERN_C_END
#endif

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

    This function DOES make a system call and thus SHOULD NOT be called
    from an error handler.

@*/
PetscErrorCode  PetscGetDate(char date[],size_t len)
{
  char           *str=PETSC_NULL;
#if defined(PETSC_HAVE_TIME)
  time_t         aclock;
#else
  struct timeval tp;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_TIME)
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

