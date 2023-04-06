
/*
      Code for manipulating files.
*/
#include <petscsys.h>

/*@C
   PetscGetHomeDirectory - Returns home directory name.

   Not Collective

   Input Parameter:
.  maxlen - maximum length allowed

   Output Parameter:
.  dir - contains the home directory. Must be long enough to hold the name.

   Level: developer

   Notes:
   If PETSc cannot determine the home directory it makes `dir` an empty string

   On Microsoft Windows machines the environmental variable `HOME` specifies the home directory.

.seealso: `PetscGetTmp()`, `PetscSharedTmp()`, `PetscGetWorkingDirectory()`
@*/
PetscErrorCode PetscGetHomeDirectory(char dir[], size_t maxlen)
{
  const char *d1;

  PetscFunctionBegin;
  d1 = getenv("HOME");
  if (d1) {
    PetscCall(PetscStrncpy(dir, d1, maxlen));
  } else if (maxlen > 0) dir[0] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscFixFilename - Fixes a file name so that it is correct for both Unix and
    Microsoft Windows by using the correct / or \ to separate directories.

   Not Collective

   Input Parameter:
.  filein - name of file to be fixed

   Output Parameter:
.  fileout - the fixed name. Should long enough to hold the filename.

   Level: advanced

   Note:
   Call `PetscFixFilename()` just before calling `fopen()`.
@*/
PetscErrorCode PetscFixFilename(const char filein[], char fileout[])
{
  size_t i, n;

  PetscFunctionBegin;
  if (!filein || !fileout) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscStrlen(filein, &n));
  for (i = 0; i < n; i++) {
    if (filein[i] == PETSC_REPLACE_DIR_SEPARATOR) fileout[i] = PETSC_DIR_SEPARATOR;
    else fileout[i] = filein[i];
  }
  fileout[n] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
