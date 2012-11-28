#include <petsc-private/fortranimpl.h>
#include <petscdmshell.h>       /*I    "petscdmshell.h"  I*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmshellsetcreatematrix_                DMSHELLSETCREATEMATRIX
#define dmshellsetcreateglobalvector_          DMSHELLSETCREATEGLOBALVECTOR_
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmshellsetcreatematrix_                dmshellsetcreatematrix
#define dmshellsetcreateglobalvector_          dmshellsetcreateglobalvector
#endif

/*
C routines are required for matrix and global vector creation.
We define C routines here that call the corresponding Fortran routine (stashed
in dm->fortran_func_pointers) that was set by the user.

dm->fortran_func_pointers usage:

0: ourcreatematrix
1: ourcreateglobalvector
*/

static PetscErrorCode ourcreatematrix(DM dm,MatType type,Mat *A)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(DM*,MatType*,Mat*,PetscErrorCode*))(((PetscObject)dm)->fortran_func_pointers[0]))(&dm,&type,A,&ierr);
  return ierr;
}

static PetscErrorCode ourcreateglobalvector(DM dm,Vec *v)
{
  PetscErrorCode ierr = 0;
  (*(PetscErrorCode (PETSC_STDCALL *)(DM*,Vec*,PetscErrorCode*))(((PetscObject)dm)->fortran_func_pointers[1]))(&dm,v,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

void PETSC_STDCALL dmshellsetcreatematrix_(DM *dm,void (PETSC_STDCALL *func)(DM*,MatType*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*dm,2);
  ((PetscObject)*dm)->fortran_func_pointers[0] = (PetscVoidFunction) func;
  *ierr = DMShellSetCreateMatrix(*dm,ourcreatematrix);
}

void PETSC_STDCALL dmshellsetcreateglobalvector_(DM *dm,void (PETSC_STDCALL *func)(DM*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*dm,2);
  ((PetscObject)*dm)->fortran_func_pointers[1] = (PetscVoidFunction) func;
  *ierr = DMShellSetCreateGlobalVector(*dm,ourcreateglobalvector);
}

EXTERN_C_END
