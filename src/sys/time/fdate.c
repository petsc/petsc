#include <petscsys.h>
#if defined(PETSC_HAVE_SYS_TIME_H)
  #include <sys/time.h>
#endif
#include <time.h>

/*@C
  PetscGetDate - Gets the current date.

  Not Collective

  Input Parameter:
. len - length of string to hold date

  Output Parameter:
. date - the date

  Level: beginner

  Note:
  This function makes a system call and thus SHOULD NOT be called from an error handler.

  Developer Notes:
  This function is called once during `PetscInitialize()`.
  It stashes the timestamp, and uses it when needed. This is so that
  error handlers may report the date without generating possible
  additional system errors during the call to get the date.

.seealso: `PetscGetHostName()`
@*/
PetscErrorCode PetscGetDate(char date[], size_t len)
{
  char *str = NULL;
#if defined(PETSC_HAVE_TIME)
  time_t aclock;
#else
  struct timeval tp;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_TIME)
  time(&aclock);
  PetscCall(PetscStrncpy(date, asctime(localtime(&aclock)), len));
#else
  gettimeofday(&tp, NULL);
  PetscCall(PetscStrncpy(date, asctime(localtime((time_t *)&tp.tv_sec)), len));
#endif
  /* now strip out the new-line chars at the end of the string */
  PetscCall(PetscStrstr(date, "\n", &str));
  if (str) str[0] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
