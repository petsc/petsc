#include <petsc/private/snesimpl.h>             /*I   "petscsnes.h"   I*/
#include <petscdm.h>

PetscErrorCode MatMultASPIN(Mat m,Vec X,Vec Y)
{
  void           *ctx;
  SNES           snes;
  PetscInt       n,i;
  VecScatter     *oscatter;
  SNES           *subsnes;
  PetscBool      match;
  MPI_Comm       comm;
  KSP            ksp;
  Vec            *x,*b;
  Vec            W;
  SNES           npc;
  Mat            subJ,subpJ;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(m,&ctx));
  snes = (SNES)ctx;
  CHKERRQ(SNESGetNPC(snes,&npc));
  CHKERRQ(SNESGetFunction(npc,&W,NULL,NULL));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)npc,SNESNASM,&match));
  if (!match) {
    CHKERRQ(PetscObjectGetComm((PetscObject)snes,&comm));
    SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"MatMultASPIN requires that the nonlinear preconditioner be Nonlinear additive Schwarz");
  }
  CHKERRQ(SNESNASMGetSubdomains(npc,&n,&subsnes,NULL,&oscatter,NULL));
  CHKERRQ(SNESNASMGetSubdomainVecs(npc,&n,&x,&b,NULL,NULL));

  CHKERRQ(VecSet(Y,0));
  CHKERRQ(MatMult(npc->jacobian_pre,X,W));

  for (i=0;i<n;i++) {
    CHKERRQ(VecScatterBegin(oscatter[i],W,b[i],INSERT_VALUES,SCATTER_FORWARD));
  }
  for (i=0;i<n;i++) {
    CHKERRQ(VecScatterEnd(oscatter[i],W,b[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecSet(x[i],0.));
    CHKERRQ(SNESGetJacobian(subsnes[i],&subJ,&subpJ,NULL,NULL));
    CHKERRQ(SNESGetKSP(subsnes[i],&ksp));
    CHKERRQ(KSPSetOperators(ksp,subJ,subpJ));
    CHKERRQ(KSPSolve(ksp,b[i],x[i]));
    CHKERRQ(VecScatterBegin(oscatter[i],x[i],Y,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(oscatter[i],x[i],Y,ADD_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_ASPIN(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(SNESDestroy(&snes->npc));
  /* reset NEWTONLS and free the data */
  CHKERRQ(SNESReset(snes));
  CHKERRQ(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESASPIN - Helper SNES type for Additive-Schwarz Preconditioned Inexact Newton

   Options Database:
+  -npc_snes_ - options prefix of the nonlinear subdomain solver (must be of type NASM)
.  -npc_sub_snes_ - options prefix of the subdomain nonlinear solves
.  -npc_sub_ksp_ - options prefix of the subdomain Krylov solver
-  -npc_sub_pc_ - options prefix of the subdomain preconditioner

    Notes:
    This routine sets up an instance of NETWONLS with nonlinear left preconditioning.  It differs from other
    similar functionality in SNES as it creates a linear shell matrix that corresponds to the product

    \sum_{i=0}^{N_b}J_b({X^b_{converged}})^{-1}J(X + \sum_{i=0}^{N_b}(X^b_{converged} - X^b))

    which is the ASPIN preconditioned matrix. Similar solvers may be constructed by having matrix-free differencing of
    nonlinear solves per linear iteration, but this is far more efficient when subdomain sparse-direct preconditioner
    factorizations are reused on each application of J_b^{-1}.

    The Krylov method used in this nonlinear solver is run with NO preconditioner, because the preconditioning is done
    at the nonlinear level, but the Jacobian for the original function must be provided (or calculated via coloring and
    finite differences automatically) in the Pmat location of SNESSetJacobian() because the action of the original Jacobian
    is needed by the shell matrix used to apply the Jacobian of the nonlinear preconditioned problem (see above).
    Note that since the Pmat is not used to construct a preconditioner it could be provided in a matrix-free form.
    The code for this implementation is a bit confusing because the Amat of SNESSetJacobian() applies the Jacobian of the
    nonlinearly preconditioned function Jacobian while the Pmat provides the Jacobian of the original user provided function.
    Note that the original SNES and nonlinear preconditioner preconditioner (see SNESGetNPC()), in this case NASM, share
    the same Jacobian matrices. SNESNASM computes the needed Jacobian in SNESNASMComputeFinalJacobian_Private().

   Level: intermediate

   References:
+  * - X. C. Cai and D. E. Keyes, "Nonlinearly preconditioned inexact Newton algorithms",  SIAM J. Sci. Comput., 24, 2002.
-  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNASM, SNESGetNPC(), SNESGetNPCSide()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_ASPIN(SNES snes)
{
  SNES           npc;
  KSP            ksp;
  PC             pc;
  Mat            aspinmat;
  Vec            F;
  PetscInt       n;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  /* set up the solver */
  CHKERRQ(SNESSetType(snes,SNESNEWTONLS));
  CHKERRQ(SNESSetNPCSide(snes,PC_LEFT));
  CHKERRQ(SNESSetFunctionType(snes,SNES_FUNCTION_PRECONDITIONED));
  CHKERRQ(SNESGetNPC(snes,&npc));
  CHKERRQ(SNESSetType(npc,SNESNASM));
  CHKERRQ(SNESNASMSetType(npc,PC_ASM_BASIC));
  CHKERRQ(SNESNASMSetComputeFinalJacobian(npc,PETSC_TRUE));
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(SNESGetLineSearch(snes,&linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    CHKERRQ(SNESLineSearchSetType(linesearch,SNESLINESEARCHBT));
  }

  /* set up the shell matrix */
  CHKERRQ(SNESGetFunction(snes,&F,NULL,NULL));
  CHKERRQ(VecGetLocalSize(F,&n));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)snes),n,n,PETSC_DECIDE,PETSC_DECIDE,snes,&aspinmat));
  CHKERRQ(MatSetType(aspinmat,MATSHELL));
  CHKERRQ(MatShellSetOperation(aspinmat,MATOP_MULT,(void(*)(void))MatMultASPIN));
  CHKERRQ(SNESSetJacobian(snes,aspinmat,NULL,NULL,NULL));
  CHKERRQ(MatDestroy(&aspinmat));

  snes->ops->destroy = SNESDestroy_ASPIN;

  PetscFunctionReturn(0);
}
