/*$Id: vecreg.c,v 1.9 1999/09/02 14:53:08 bsmith Exp bsmith $*/

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
extern int VecRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "VecRegisterAll"
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
  VecRegisterAllCalled = 1;

  ierr = VecRegister(VEC_MPI,           path,"VecCreate_MPI",     VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegister(VEC_SHARED,        path,"VecCreate_Shared",  VecCreate_Shared);CHKERRQ(ierr);
  ierr = VecRegister(VEC_SEQ,           path,"VecCreate_Seq",     VecCreate_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


