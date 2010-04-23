#define PETSCSNES_DLL

#include "private/snesimpl.h"  /*I  "petscsnes.h" I*/
#include "../src/mat/impls/mffd/mffdimpl.h"
#include "private/matimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDComputeJacobian"
/*@C
   MatMFFDComputeJacobian - Tells the matrix-free Jacobian object the new location at which
       Jacobian matrix vector products will be computed at, i.e. J(x) * a. The x is obtained
       from the SNES object (using SNESGetSolution()).

   Collective on SNES

   Input Parameters:
+   snes - the nonlinear solver context
.   x - the point at which the Jacobian vector products will be performed
.   jac - the matrix-free Jacobian object
.   B - either the same as jac or another matrix type (ignored)
.   flag - not relevent for matrix-free form
-   dummy - the user context (ignored)

   Level: developer

   Warning: 
      If MatMFFDSetBase() is ever called on jac then this routine will NO longer get 
    the x from the SNES object and MatMFFDSetBase() must from that point on be used to
    change the base vector x.

   Notes:
     This can be passed into SNESSetJacobian() when using a completely matrix-free solver,
     that is the B matrix is also the same matrix operator. This is used when you select
     -snes_mf but rarely used directly by users. (All this routine does is call MatAssemblyBegin/End() on
     the Mat jac.

.seealso: MatMFFDGetH(), MatCreateSNESMF(), MatCreateMFFD(), MATMFFD,
          MatMFFDSetHHistory(), MatMFFDSetFunctionError(), MatCreateMFFD(), SNESSetJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT MatMFFDComputeJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MFFD(Mat,MatAssemblyType);
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SNESMF"
/*
   MatAssemblyEnd_SNESMF - Calls MatAssemblyEnd_MFFD() and then sets the 
    base from the SNES context

*/
PetscErrorCode MatAssemblyEnd_SNESMF(Mat J,MatAssemblyType mt)
{
  PetscErrorCode ierr;
  MatMFFD        j = (MatMFFD)J->data;
  SNES           snes = (SNES)j->funcctx;
  Vec            u,f;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MFFD(J,mt);CHKERRQ(ierr);

  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMFFDSetBase(J,u,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetBase_MFFD(Mat,Vec,Vec);
/*
    This routine resets the MatAssemblyEnd() for the MatMFFD created from MatCreateSNESMF() so that it NO longer
  uses the solution in the SNES object to update the base. See the warning in MatCreateSNESMF().
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetBase_SNESMF"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetBase_SNESMF(Mat J,Vec U,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMFFDSetBase_MFFD(J,U,F);CHKERRQ(ierr);
  J->ops->assemblyend = MatAssemblyEnd_MFFD;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSNESMF"
/*@
   MatCreateSNESMF - Creates a matrix-free matrix context for use with
   a SNES solver.  This matrix can be used as the Jacobian argument for
   the routine SNESSetJacobian(). See MatCreateMFFD() for details on how
   the finite difference computation is done.

   Collective on SNES and Vec

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  J - the matrix-free matrix

   Level: advanced

   Warning: 
      If MatMFFDSetBase() is ever called on jac then this routine will NO longer get 
    the x from the SNES object and MatMFFDSetBase() must from that point on be used to
    change the base vector x.

   Notes: The difference between this routine and MatCreateMFFD() is that this matrix
     automatically gets the current base vector from the SNES object and not from an
     explicit call to MatMFFDSetBase().

.seealso: MatDestroy(), MatMFFDSetFunctionError(), MatMFFDDSSetUmin()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateMFFD(),
          MatMFFDGetH(), MatMFFDRegisterDynamic), MatMFFDComputeJacobian()
 
@*/
PetscErrorCode PETSCSNES_DLLEXPORT MatCreateSNESMF(SNES snes,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       n,N;

  PetscFunctionBegin;
  if (!snes->vec_func) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
  
  ierr = VecGetLocalSize(snes->vec_func,&n);CHKERRQ(ierr);
  ierr = VecGetSize(snes->vec_func,&N);CHKERRQ(ierr);
  ierr = MatCreateMFFD(((PetscObject)snes)->comm,n,n,N,N,J);CHKERRQ(ierr);
  ierr = MatMFFDSetFunction(*J,(PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunction,snes);CHKERRQ(ierr);
  (*J)->ops->assemblyend = MatAssemblyEnd_SNESMF;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)*J,"MatMFFDSetBase_C","MatMFFDSetBase_SNESMF",MatMFFDSetBase_SNESMF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






