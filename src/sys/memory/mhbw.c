#include <petscsys.h>             /*I   "petscsys.h"   I*/

#if defined(PETSC_HAVE_MEMKIND)
#include <hbwmalloc.h>
#endif

/*
   These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t,PetscBool,int,const char[],const char[],void**);
PETSC_EXTERN PetscErrorCode PetscFreeAlign(void*,int,const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscReallocAlign(size_t,int,const char[],const char[],void**);

/*
   PetscHBWMalloc - HBW malloc.

   Input Parameters:
   +   a   - number of bytes to allocate
   .   lineno - line number where used
   .   function - function calling routine
   -   filename  - file name where used

   Returns:
   double aligned pointer to requested storage, or null if not
   available.
*/
static PetscErrorCode PetscHBWMalloc(size_t a,PetscBool clear,int lineno,const char function[],const char filename[],void **result)
{
#if !defined(PETSC_HAVE_MEMKIND)
  return PetscMallocAlign(a,clear,lineno,function,filename,result);
#else
  if (!a) { *result = NULL; return 0; }
  /*
    The default policy is if insufficient memory is available from the high bandwidth memory
    fall back to standard memory. If we use the HBW_POLICY_BIND policy, errno is set to ENOMEM
    and the allocated pointer is set to NULL if there is not enough HWB memory available.
  */
  {
    int err = hbw_posix_memalign(result,PETSC_MEMALIGN,a);
    PetscCheckFalse(err || !*result,PETSC_COMM_SELF,PETSC_ERR_MEM,"HBW Memory requested %.0f",(PetscLogDouble)a);
  }
  return 0;
#endif
}

static PetscErrorCode PetscHBWFree(void *aa,int lineno,const char function[],const char filename[])
{
#if !defined(PETSC_HAVE_MEMKIND)
  return PetscFreeAlign(aa,lineno,function,filename);
#else
  hbw_free(aa);
  return 0;
#endif
}

static PetscErrorCode PetscHBWRealloc(size_t a,int lineno,const char function[],const char filename[],void **result)
{
#if !defined(PETSC_HAVE_MEMKIND)
  return PetscReallocAlign(a,lineno,function,filename,result);
#else
  if (!a) {
    int err = PetscFreeAlign(*result,lineno,function,filename);
    if (err) return err;
    *result = NULL;
    return 0;
  }
  *result = hbw_realloc(*result,a);
  PetscCheckFalse(!*result,PETSC_COMM_SELF,PETSC_ERR_MEM,"Memory requested %.0f",(PetscLogDouble)a);
  return 0;
#endif
}

PETSC_INTERN PetscErrorCode PetscSetUseHBWMalloc_Private(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscMallocSet(PetscHBWMalloc,PetscHBWFree,NULL));
  PetscTrRealloc = PetscHBWRealloc;
  PetscFunctionReturn(0);
}
