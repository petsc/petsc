/*$Id: vecreg.c,v 1.13 2000/04/09 04:35:20 bsmith Exp bsmith $*/

#include "src/vec/vecimpl.h"  /*I "vec.h" I*/

EXTERN_C_BEGIN
extern int VecCreate_Seq(Vec);
extern int VecCreate_MPI(Vec);
extern int VecCreate_Shared(Vec);
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
  PetscFunctionReturn(0);
}


