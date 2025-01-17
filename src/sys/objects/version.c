#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/

/*@C
  PetscGetVersion - Gets the PETSc version information in a string.

  Not Collective; No Fortran Support

  Input Parameter:
. len - length of the string

  Output Parameter:
. version - version string

  Level: developer

  Note:
  For doing runtime checking of supported versions we recommend using `PetscGetVersionNumber()` instead of this routine.

.seealso: `PetscGetProgramName()`, `PetscGetVersionNumber()`
@*/
PetscErrorCode PetscGetVersion(char version[], size_t len)
{
  PetscFunctionBegin;
#if (PETSC_VERSION_RELEASE == 1)
  PetscCall(PetscSNPrintf(version, len, "PETSc Release Version %d.%d.%d, %s", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, PETSC_VERSION_DATE));
#else
  PetscCall(PetscSNPrintf(version, len, "PETSc Development Git Revision: %s Git Date: %s", PETSC_VERSION_GIT, PETSC_VERSION_DATE_GIT));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscGetVersionNumber - Gets the PETSc version information from the library

  Not Collective

  Output Parameters:
+ major    - the major version (optional, pass `NULL` if not requested)
. minor    - the minor version (optional, pass `NULL` if not requested)
. subminor - the subminor version (patch number)  (optional, pass `NULL` if not requested)
- release  - indicates the library is from a release, not random git repository  (optional, pass `NULL` if not requested)

  Level: developer

  Notes:
  The C macros `PETSC_VERSION_MAJOR`, `PETSC_VERSION_MINOR`, `PETSC_VERSION_SUBMINOR`, `PETSC_VERSION_RELEASE` provide the information at
  compile time. This can be used to confirm that the shared library being loaded at runtime has the appropriate version updates.

  This function can be called before `PetscInitialize()`

.seealso: `PetscGetProgramName()`, `PetscGetVersion()`, `PetscInitialize()`
@*/
PetscErrorCode PetscGetVersionNumber(PetscInt *major, PetscInt *minor, PetscInt *subminor, PetscInt *release)
{
  if (major) *major = PETSC_VERSION_MAJOR;
  if (minor) *minor = PETSC_VERSION_MINOR;
  if (subminor) *subminor = PETSC_VERSION_SUBMINOR;
  if (release) *release = PETSC_VERSION_RELEASE;
  return PETSC_SUCCESS;
}
#if defined(PETSC_HAVE_BLI_THREAD_SET_NUM_THREADS)
EXTERN_C_BEGIN
void bli_thread_set_num_threads(int);
EXTERN_C_END
#elif defined(PETSC_HAVE_MKL_SET_NUM_THREADS)
  #include <mkl.h>
#elif defined(PETSC_HAVE_OPENBLAS_SET_NUM_THREADS)
EXTERN_C_BEGIN
void openblas_set_num_threads(int);
EXTERN_C_END
#endif
PetscInt PetscNumBLASThreads = 1;

/*@
  PetscBLASSetNumThreads - set the number of threads for calls to BLAS to use

  Input Parameter:
. nt - the number of threads

  Options Database Key:
. -blas_num_threads <nt> - set the number of threads when PETSc is initialized

  Level: intermediate

  Notes:
  The environmental variables `BLIS_NUM_THREADS`, `MKL_NUM_THREADS`, or `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`
  may also affect the number of threads used depending on the BLAS libraries being used. A call to this function
  overwrites those values.

  With the BLIS BLAS implementation one can use `BLIS_THREAD_IMPL=pthread` or `BLIS_THREAD_IMPL=openmp` to determine how
  BLIS implements the parallelism.

.seealso: `PetscInitialize()`, `PetscBLASGetNumThreads()`
@*/
PetscErrorCode PetscBLASSetNumThreads(PetscInt nt)
{
  PetscFunctionBegin;
  PetscNumBLASThreads = nt;
#if defined(PETSC_HAVE_BLI_THREAD_SET_NUM_THREADS)
  bli_thread_set_num_threads(nt);
  PetscCall(PetscInfo(NULL, "Setting number of threads used for BLIS provided BLAS %" PetscInt_FMT "\n", PetscNumBLASThreads));
#elif defined(PETSC_HAVE_MKL_SET_NUM_THREADS)
  mkl_set_num_threads((int)nt);
  PetscCall(PetscInfo(NULL, "Setting number of threads used for MKL provided BLAS %" PetscInt_FMT "\n", PetscNumBLASThreads));
#elif defined(PETSC_HAVE_OPENBLAS_SET_NUM_THREADS)
  openblas_set_num_threads((int)nt);
  PetscCall(PetscInfo(NULL, "Setting number of threads used for OpenBLAS provided BLAS %" PetscInt_FMT "\n", PetscNumBLASThreads));
#else
  PetscCall(PetscInfo(NULL, "Cannot set number of threads used for BLAS %" PetscInt_FMT ", will be ignored\n", PetscNumBLASThreads));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBLASGetNumThreads - get the number of threads for calls to BLAS to use

  Output Parameter:
. nt - the number of threads

  Level: intermediate

.seealso: `PetscInitialize()`, `PetscBLASSetNumThreads()`
@*/
PetscErrorCode PetscBLASGetNumThreads(PetscInt *nt)
{
  PetscFunctionBegin;
  PetscAssertPointer(nt, 1);
  *nt = PetscNumBLASThreads;
  PetscFunctionReturn(PETSC_SUCCESS);
}
