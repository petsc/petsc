/*
      Code for manipulating files.
*/
#include <petscsys.h>
#if defined(PETSC_HAVE_WINDOWS_H)
  #include <windows.h>
#endif

/*@C
  PetscGetUserName - Get the login name of the user running the program on the current MPI process

  Not Collective

  Input Parameter:
. nlen - length of the `name` buffer

  Output Parameter:
. name - on output, holds the user name (null-terminated)

  Level: developer

.seealso: `PetscGetHostName()`, `PetscGetProgramName()`
@*/
#if defined(PETSC_HAVE_GET_USER_NAME)
PetscErrorCode PetscGetUserName(char name[], size_t nlen)
{
  PetscFunctionBegin;
  GetUserName((LPTSTR)name, (LPDWORD)(&nlen));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#else

PetscErrorCode PetscGetUserName(char name[], size_t nlen)
{
  const char *user;

  PetscFunctionBegin;
  user = getenv("USER");
  if (!user) user = "Unknown";
  PetscCall(PetscStrncpy(name, user, nlen));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
