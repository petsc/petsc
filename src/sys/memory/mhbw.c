#include <petscsys.h>             /*I   "petscsys.h"   I*/

#if defined(PETSC_HAVE_MEMKIND)
#include <hbwmalloc.h>
#endif

/*
   These are defined in mal.c and ensure that malloced space is PetscScalar aligned
*/
extern PetscErrorCode PetscMallocAlign(size_t,int,const char[],const char[],void**);
extern PetscErrorCode PetscFreeAlign(void*,int,const char[],const char[]);

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
PetscErrorCode PetscHBWMalloc(size_t a,int lineno,const char function[],const char filename[],void **result)
{
#if !defined(PETSC_HAVE_MEMKIND)
  return PetscMallocAlign(a,lineno,function,filename,result);
#else
  if (!a) { *result = NULL; return 0; }
  /*
    The default policy is if insufficient memory is available from the high bandwidth memory
    fall back to standard memory. If we use the HBW_POLICY_BIND policy, errno is set to ENOMEM
    and the allocated pointer is set to NULL if there is not enough HWB memory available.
  */
  {
    int ierr = hbw_posix_memalign(result,PETSC_MEMALIGN,a);
    if (ierr || !*result) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"HBW Memory requested %.0f",(PetscLogDouble)a);
  }
  return 0;
#endif
}

PetscErrorCode PetscHBWFree(void *aa,int line,const char function[],const char file[])
{
#if !defined(PETSC_HAVE_MEMKIND)
  return PetscFreeAlign(aa,line,function,file);
#else
  hbw_free(aa);
  return 0;
#endif
}

PetscErrorCode PetscSetUseHBWMalloc_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMallocSet(PetscHBWMalloc,PetscHBWFree);CHKERRQ(ierr);
  PetscTrRealloc = NULL;
  PetscFunctionReturn(0);
}
