
#include <petsc/private/snesimpl.h>  /*I  "petscsnes.h" I*/
#include <petscdm.h>                 /*I  "petscdm.h"   I*/
#include <../src/mat/impls/mffd/mffdimpl.h>
#include <petsc/private/matimpl.h>

/*@C
   MatMFFDComputeJacobian - Tells the matrix-free Jacobian object the new location at which
       Jacobian matrix vector products will be computed at, i.e. J(x) * a. The x is obtained
       from the SNES object (using SNESGetSolution()).

   Logically Collective on SNES

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
     This can be passed into SNESSetJacobian() as the Jacobian evaluation function argument
     when using a completely matrix-free solver,
     that is the B matrix is also the same matrix operator. This is used when you select
     -snes_mf but rarely used directly by users. (All this routine does is call MatAssemblyBegin/End() on
     the Mat jac.)

.seealso: MatMFFDGetH(), MatCreateSNESMF(), MatCreateMFFD(), MATMFFD,
          MatMFFDSetHHistory(), MatMFFDSetFunctionError(), MatCreateMFFD(), SNESSetJacobian()

@*/
PetscErrorCode  MatMFFDComputeJacobian(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatAssemblyEnd_MFFD(Mat,MatAssemblyType);
PETSC_EXTERN PetscErrorCode MatMFFDSetBase_MFFD(Mat,Vec,Vec);

/*
   MatAssemblyEnd_SNESMF - Calls MatAssemblyEnd_MFFD() and then sets the
    base from the SNES context

*/
static PetscErrorCode MatAssemblyEnd_SNESMF(Mat J,MatAssemblyType mt)
{
  PetscErrorCode ierr;
  MatMFFD        j    = (MatMFFD)J->data;
  SNES           snes = (SNES)j->ctx;
  Vec            u,f;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MFFD(J,mt);CHKERRQ(ierr);

  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  if (j->func == (PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunction) {
    ierr = SNESGetFunction(snes,&f,NULL,NULL);CHKERRQ(ierr);
    ierr = MatMFFDSetBase_MFFD(J,u,f);CHKERRQ(ierr);
  } else {
    /* f value known by SNES is not correct for other differencing function */
    ierr = MatMFFDSetBase_MFFD(J,u,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    This routine resets the MatAssemblyEnd() for the MatMFFD created from MatCreateSNESMF() so that it NO longer
  uses the solution in the SNES object to update the base. See the warning in MatCreateSNESMF().
*/
static PetscErrorCode  MatMFFDSetBase_SNESMF(Mat J,Vec U,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMFFDSetBase_MFFD(J,U,F);CHKERRQ(ierr);

  J->ops->assemblyend = MatAssemblyEnd_MFFD;
  PetscFunctionReturn(0);
}

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


   Notes:
     You can call SNESSetJacobian() with MatMFFDComputeJacobian() if you are using matrix and not a different
     preconditioner matrix

     If you wish to provide a different function to do differencing on to compute the matrix-free operator than
     that provided to SNESSetFunction() then call MatMFFDSetFunction() with your function after this call.

     The difference between this routine and MatCreateMFFD() is that this matrix
     automatically gets the current base vector from the SNES object and not from an
     explicit call to MatMFFDSetBase().

   Warning:
     If MatMFFDSetBase() is ever called on jac then this routine will NO longer get
     the x from the SNES object and MatMFFDSetBase() must from that point on be used to
     change the base vector x.

   Warning:
     Using a different function for the differencing will not work if you are using non-linear left preconditioning.


.seealso: MatDestroy(), MatMFFDSetFunction(), MatMFFDSetFunctionError(), MatMFFDDSSetUmin()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateMFFD(),
          MatMFFDGetH(), MatMFFDRegister(), MatMFFDComputeJacobian()

@*/
PetscErrorCode  MatCreateSNESMF(SNES snes,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       n,N;
  MatMFFD        mf;

  PetscFunctionBegin;
  if (snes->vec_func) {
    ierr = VecGetLocalSize(snes->vec_func,&n);CHKERRQ(ierr);
    ierr = VecGetSize(snes->vec_func,&N);CHKERRQ(ierr);
  } else if (snes->dm) {
    Vec tmp;
    ierr = DMGetGlobalVector(snes->dm,&tmp);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tmp,&n);CHKERRQ(ierr);
    ierr = VecGetSize(tmp,&N);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(snes->dm,&tmp);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() or SNESSetDM() first");
  ierr = MatCreateMFFD(PetscObjectComm((PetscObject)snes),n,n,N,N,J);CHKERRQ(ierr);
  mf      = (MatMFFD)(*J)->data;
  mf->ctx = snes;

  if (snes->npc && snes->npcside== PC_LEFT) {
    ierr = MatMFFDSetFunction(*J,(PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunctionDefaultNPC,snes);CHKERRQ(ierr);
  } else {
    ierr = MatMFFDSetFunction(*J,(PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunction,snes);CHKERRQ(ierr);
  }

  (*J)->ops->assemblyend = MatAssemblyEnd_SNESMF;

  ierr = PetscObjectComposeFunction((PetscObject)*J,"MatMFFDSetBase_C",MatMFFDSetBase_SNESMF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






