/*$Id: vecreg.c,v 1.16 2000/05/10 16:40:00 bsmith Exp bsmith $*/

#include "src/vec/vecimpl.h"  /*I "petscvec.h" I*/

EXTERN_C_BEGIN
EXTERN int VecCreate_Seq(Vec);
EXTERN int VecCreate_MPI(Vec);
EXTERN int VecCreate_FETI(Vec);
EXTERN int VecCreate_Shared(Vec);
EXTERN_C_END


/*
    This is used by VecCreate() to make sure that at least one 
    VecRegisterAll() is called. In general, if there is more than one
    DLL, then VecRegisterAll() may be called several times.
*/
extern PetscTruth VecRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the Vec components in the PETSc package.

  Not Collective

  Level: advanced

.keywords: Vec, register, all

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
  PetscFunctionReturn(0);
}


