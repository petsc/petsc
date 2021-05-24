
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

/*@
    MatSNESMFGetSNES - returns the SNES associated with a matrix created with MatCreateSNESMF()

    Not collective

    Input Parameter:
.   J - the matrix

    Output Parameter:
.   snes - the SNES object

    Level: advanced

.seealso: MatCreateSNESMF()
@*/
PetscErrorCode MatSNESMFGetSNES(Mat J,SNES *snes)
{
  MatMFFD        j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&j);CHKERRQ(ierr);
  *snes = (SNES)j->ctx;
  PetscFunctionReturn(0);
}

/*
   MatAssemblyEnd_SNESMF - Calls MatAssemblyEnd_MFFD() and then sets the
    base from the SNES context

*/
static PetscErrorCode MatAssemblyEnd_SNESMF(Mat J,MatAssemblyType mt)
{
  PetscErrorCode ierr;
  MatMFFD        j;
  SNES           snes;
  Vec            u,f;
  DM             dm;
  DMSNES         dms;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&j);CHKERRQ(ierr);
  snes = (SNES)j->ctx;
  ierr = MatAssemblyEnd_MFFD(J,mt);CHKERRQ(ierr);

  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDMSNES(dm,&dms);CHKERRQ(ierr);
  if ((j->func == (PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunction) && !dms->ops->computemffunction) {
    ierr = SNESGetFunction(snes,&f,NULL,NULL);CHKERRQ(ierr);
    ierr = MatMFFDSetBase_MFFD(J,u,f);CHKERRQ(ierr);
  } else {
    /* f value known by SNES is not correct for other differencing function */
    ierr = MatMFFDSetBase_MFFD(J,u,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   MatAssemblyEnd_SNESMF_UseBase - Calls MatAssemblyEnd_MFFD() and then sets the
    base from the SNES context. This version will cause the base to be used for differencing
    even if the func is not SNESComputeFunction. See: MatSNESMFUseBase()

*/
static PetscErrorCode MatAssemblyEnd_SNESMF_UseBase(Mat J,MatAssemblyType mt)
{
  PetscErrorCode ierr;
  MatMFFD        j;
  SNES           snes;
  Vec            u,f;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MFFD(J,mt);CHKERRQ(ierr);
  ierr = MatShellGetContext(J,&j);CHKERRQ(ierr);
  snes = (SNES)j->ctx;
  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMFFDSetBase_MFFD(J,u,f);CHKERRQ(ierr);
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

static PetscErrorCode  MatSNESMFSetReuseBase_SNESMF(Mat J,PetscBool use)
{
  PetscFunctionBegin;
  if (use) {
    J->ops->assemblyend = MatAssemblyEnd_SNESMF_UseBase;
  } else {
    J->ops->assemblyend = MatAssemblyEnd_SNESMF;
  }
  PetscFunctionReturn(0);
}

/*@
    MatSNESMFSetReuseBase - Causes the base vector to be used for differencing even if the function provided to SNESSetFunction() is not the
                       same as that provided to MatMFFDSetFunction().

    Logically Collective on Mat

    Input Parameters:
+   J - the MatMFFD matrix
-   use - if true always reuse the base vector instead of recomputing f(u) even if the function in the MatSNESMF is
          not SNESComputeFunction()

    Notes:
    Care must be taken when using this routine to insure that the function provided to MatMFFDSetFunction(), call it F_MF() is compatible with
           with that provided to SNESSetFunction(), call it F_SNES(). That is, (F_MF(u + h*d) - F_SNES(u))/h has to approximate the derivative

    Developer Notes:
    This was provided for the MOOSE team who desired to have a SNESSetFunction() function that could change configurations due
                     to contacts while the function provided to MatMFFDSetFunction() cannot. Except for the possibility of changing the configuration
                     both functions compute the same mathematical function so the differencing makes sense.

    Level: advanced

.seealso: MatCreateSNESMF(), MatSNESMFGetReuseBase()
@*/
PetscErrorCode  MatSNESMFSetReuseBase(Mat J,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  ierr = PetscTryMethod(J,"MatSNESMFSetReuseBase_C",(Mat,PetscBool),(J,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSNESMFGetReuseBase_SNESMF(Mat J,PetscBool *use)
{
  PetscFunctionBegin;
  if (J->ops->assemblyend == MatAssemblyEnd_SNESMF_UseBase) *use = PETSC_TRUE;
  else *use = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
    MatSNESMFGetReuseBase - Determines if the base vector is to be used for differencing even if the function provided to SNESSetFunction() is not the
                       same as that provided to MatMFFDSetFunction().

    Logically Collective on Mat

    Input Parameter:
.   J - the MatMFFD matrix

    Output Parameter:
.   use - if true always reuse the base vector instead of recomputing f(u) even if the function in the MatSNESMF is
          not SNESComputeFunction()

    Notes:
    Care must be taken when using this routine to insure that the function provided to MatMFFDSetFunction(), call it F_MF() is compatible with
           with that provided to SNESSetFunction(), call it F_SNES(). That is, (F_MF(u + h*d) - F_SNES(u))/h has to approximate the derivative

    Developer Notes:
    This was provided for the MOOSE team who desired to have a SNESSetFunction() function that could change configurations due
                     to contacts while the function provided to MatMFFDSetFunction() cannot. Except for the possibility of changing the configuration
                     both functions compute the same mathematical function so the differencing makes sense.

    Level: advanced

.seealso: MatCreateSNESMF(), MatSNESMFSetReuseBase()
@*/
PetscErrorCode  MatSNESMFGetReuseBase(Mat J,PetscBool *use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  ierr = PetscUseMethod(J,"MatSNESMFGetReuseBase_C",(Mat,PetscBool*),(J,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateSNESMF - Creates a matrix-free matrix context for use with
   a SNES solver.  This matrix can be used as the Jacobian argument for
   the routine SNESSetJacobian(). See MatCreateMFFD() for details on how
   the finite difference computation is done.

   Collective on SNES

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
          MatMFFDGetH(), MatMFFDRegister(), MatMFFDComputeJacobian(), MatSNESMFSetReuseBase(), MatSNESMFGetReuseBase()

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
  ierr = MatShellGetContext(*J,&mf);CHKERRQ(ierr);
  mf->ctx = snes;

  if (snes->npc && snes->npcside== PC_LEFT) {
    ierr = MatMFFDSetFunction(*J,(PetscErrorCode (*)(void*,Vec,Vec))SNESComputeFunctionDefaultNPC,snes);CHKERRQ(ierr);
  } else {
    DM     dm;
    DMSNES dms;

    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMGetDMSNES(dm,&dms);CHKERRQ(ierr);
    ierr = MatMFFDSetFunction(*J,(PetscErrorCode (*)(void*,Vec,Vec))(dms->ops->computemffunction ? SNESComputeMFFunction : SNESComputeFunction),snes);CHKERRQ(ierr);
  }
  (*J)->ops->assemblyend = MatAssemblyEnd_SNESMF;

  ierr = PetscObjectComposeFunction((PetscObject)*J,"MatMFFDSetBase_C",MatMFFDSetBase_SNESMF);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)*J,"MatSNESMFSetReuseBase_C",MatSNESMFSetReuseBase_SNESMF);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)*J,"MatSNESMFGetReuseBase_C",MatSNESMFGetReuseBase_SNESMF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
