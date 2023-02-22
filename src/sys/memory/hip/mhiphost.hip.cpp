#include <petscsys.h>        /*I   "petscsys.h"   I*/
#include <petscdevice_hip.h> /* Needed to provide PetscCallHIP() */

PETSC_EXTERN PetscErrorCode PetscHIPHostMalloc(size_t a, PetscBool clear, int lineno, const char function[], const char filename[], void **result)
{
  PetscCallHIP(hipHostMalloc(result, a));
  return PETSC_SUCCESS;
}

PETSC_EXTERN PetscErrorCode PetscHIPHostFree(void *aa, int lineno, const char function[], const char filename[])
{
  PetscCallHIP(hipHostFree(aa));
  return PETSC_SUCCESS;
}

PETSC_EXTERN PetscErrorCode PetscHIPHostRealloc(size_t a, int lineno, const char function[], const char filename[], void **result)
{
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "HIP has no Realloc()");
}

static PetscErrorCode (*PetscMallocOld)(size_t, PetscBool, int, const char[], const char[], void **);
static PetscErrorCode (*PetscReallocOld)(size_t, int, const char[], const char[], void **);
static PetscErrorCode (*PetscFreeOld)(void *, int, const char[], const char[]);

/*@C
   PetscMallocSetHIPHost - Set `PetscMalloc()` to use `HIPHostMalloc()`
     Switch the current malloc and free routines to the HIP malloc and free routines

   Not Collective

   Level: developer

   Note:
     This provides a way to use the HIP malloc and free routines temporarily. One
     can switch back to the previous choice by calling `PetscMallocResetHIPHost()`.

.seealso: `PetscMallocSetCUDAHost()`, `PetscMallocResetHIPHost()`
@*/
PETSC_EXTERN PetscErrorCode PetscMallocSetHIPHost(void)
{
  PetscFunctionBegin;
  /* Save the previous choice */
  PetscMallocOld  = PetscTrMalloc;
  PetscReallocOld = PetscTrRealloc;
  PetscFreeOld    = PetscTrFree;
  PetscTrMalloc   = PetscHIPHostMalloc;
  PetscTrRealloc  = PetscHIPHostRealloc;
  PetscTrFree     = PetscHIPHostFree;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMallocResetHIPHost - Reset the changes made by `PetscMallocSetHIPHost()`

   Not Collective

   Level: developer

.seealso: `PetscMallocSetHIPHost()`
@*/
PETSC_EXTERN PetscErrorCode PetscMallocResetHIPHost(void)
{
  PetscFunctionBegin;
  PetscTrMalloc  = PetscMallocOld;
  PetscTrRealloc = PetscReallocOld;
  PetscTrFree    = PetscFreeOld;
  PetscFunctionReturn(PETSC_SUCCESS);
}
