#include <petscsys.h>             /*I   "petscsys.h"   I*/
#include <petschipblas.h>          /* Needed to provide CHKERRHIP() */

PETSC_EXTERN PetscErrorCode PetscHIPHostMalloc(size_t a,PetscBool clear,int lineno,const char function[],const char filename[],void **result)
{
  hipError_t ierr;
  ierr = hipHostMalloc(result,a);CHKERRHIP(ierr);
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscHIPHostFree(void *aa,int lineno,const char function[],const char filename[])
{
  hipError_t ierr;
  ierr = hipHostFree(aa);CHKERRHIP(ierr);
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscHIPHostRealloc(size_t a,int lineno,const char function[],const char filename[],void **result)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"HIP has no Realloc()");
}

static PetscErrorCode (*PetscMallocOld)(size_t,PetscBool,int,const char[],const char[],void**);
static PetscErrorCode (*PetscReallocOld)(size_t,int,const char[],const char[],void**);
static PetscErrorCode (*PetscFreeOld)(void*,int,const char[],const char[]);

/*@C
   PetscMallocSetHIPHost - Set PetscMalloc to use HIPHostMalloc
     Switch the current malloc and free routines to the HIP malloc and free routines

   Not Collective

   Level: developer

   Notes:
     This provides a way to use the HIP malloc and free routines temporarily. One
     can switch back to the previous choice by calling PetscMallocResetHIPHost().

.seealso: PetscMallocResetHIPHost()
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
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocResetHIPHost - Reset the changes made by PetscMallocSetHIPHost

   Not Collective

   Level: developer

.seealso: PetscMallocSetHIPHost()
@*/
PETSC_EXTERN PetscErrorCode PetscMallocResetHIPHost(void)
{
  PetscFunctionBegin;
  PetscTrMalloc  = PetscMallocOld;
  PetscTrRealloc = PetscReallocOld;
  PetscTrFree    = PetscFreeOld;
  PetscFunctionReturn(0);
}
