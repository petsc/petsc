/*$Id: vecreg.c,v 1.20 2001/03/23 23:21:18 balay Exp bsmith $*/

#include "src/vec/vecimpl.h"  /*I "petscvec.h" I*/

EXTERN_C_BEGIN
EXTERN int VecCreate_Seq(Vec);
EXTERN int VecCreate_MPI(Vec);
EXTERN int VecCreate_FETI(Vec);
EXTERN int VecCreate_Shared(Vec);
EXTERN int VecCreate_ESI(Vec);
EXTERN int VecCreate_PetscESI(Vec);
EXTERN_C_END


/*
    This is used by VecCreate() to make sure that at least one 
    VecRegisterAll() is called. In general, if there is more than one
    DLL, then VecRegisterAll() may be called several times.
*/
extern PetscTruth VecRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the Vec components in the PETSc package.

  Not Collective

  Level: advanced

.seealso:  VecRegisterDestroy()
@*/
int VecRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegisterDynamic(VEC_MPI,           path,"VecCreate_MPI",     VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VEC_SHARED,        path,"VecCreate_Shared",  VecCreate_Shared);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VEC_SEQ,           path,"VecCreate_Seq",     VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VEC_FETI,          path,"VecCreate_FETI",    VecCreate_FETI);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ESI) && defined(__cplusplus)
  ierr = VecRegisterDynamic(VEC_ESI,           path,"VecCreate_ESI",    VecCreate_ESI);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VEC_PETSCESI,      path,"VecCreate_PetscESI",    VecCreate_PetscESI);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


