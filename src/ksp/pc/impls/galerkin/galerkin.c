
/*
      Defines a preconditioner defined by R^T S R
*/
#include <petsc/private/pcimpl.h>
#include <petscksp.h>         /*I "petscksp.h" I*/

typedef struct {
  KSP            ksp;
  Mat            R,P;
  Vec            b,x;
  PetscErrorCode (*computeasub)(PC,Mat,Mat,Mat*,void*);
  void           *computeasub_ctx;
} PC_Galerkin;

static PetscErrorCode PCApply_Galerkin(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  if (jac->R) {
    ierr = MatRestrict(jac->R,x,jac->b);CHKERRQ(ierr);
  } else {
    ierr = MatRestrict(jac->P,x,jac->b);CHKERRQ(ierr);
  }
  ierr = KSPSolve(jac->ksp,jac->b,jac->x);CHKERRQ(ierr);
  ierr = KSPCheckSolve(jac->ksp,pc,jac->x);CHKERRQ(ierr);
  if (jac->P) {
    ierr = MatInterpolate(jac->P,jac->x,y);CHKERRQ(ierr);
  } else {
    ierr = MatInterpolate(jac->R,jac->x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Galerkin(PC pc)
{
  PetscErrorCode ierr;
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscBool      a;
  Vec            *xx,*yy;

  PetscFunctionBegin;
  if (jac->computeasub) {
    Mat Ap;
    if (!pc->setupcalled) {
      ierr = (*jac->computeasub)(pc,pc->pmat,NULL,&Ap,jac->computeasub_ctx);CHKERRQ(ierr);
      ierr = KSPSetOperators(jac->ksp,Ap,Ap);CHKERRQ(ierr);
      ierr = MatDestroy(&Ap);CHKERRQ(ierr);
    } else {
      ierr = KSPGetOperators(jac->ksp,NULL,&Ap);CHKERRQ(ierr);
      ierr = (*jac->computeasub)(pc,pc->pmat,Ap,NULL,jac->computeasub_ctx);CHKERRQ(ierr);
    }
  }

  if (!jac->x) {
    ierr = KSPGetOperatorsSet(jac->ksp,&a,NULL);CHKERRQ(ierr);
    PetscAssertFalse(!a,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set operator of PCGALERKIN KSP with PCGalerkinGetKSP()/KSPSetOperators()");
    ierr   = KSPCreateVecs(jac->ksp,1,&xx,1,&yy);CHKERRQ(ierr);
    jac->x = *xx;
    jac->b = *yy;
    ierr   = PetscFree(xx);CHKERRQ(ierr);
    ierr   = PetscFree(yy);CHKERRQ(ierr);
  }
  PetscAssertFalse(!jac->R && !jac->P,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or interpolation of PCGALERKIN with PCGalerkinSetRestriction()/Interpolation()");
  /* should check here that sizes of R/P match size of a */

  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Galerkin(PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&jac->R);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->P);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->x);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->b);CHKERRQ(ierr);
  ierr = KSPReset(jac->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Galerkin(PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Galerkin(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Galerkin(PC pc,PetscViewer viewer)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  KSP on Galerkin follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
  }
  ierr = KSPView(jac->ksp,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGalerkinGetKSP_Galerkin(PC pc,KSP *ksp)
{
  PC_Galerkin *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  *ksp = jac->ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGalerkinSetRestriction_Galerkin(PC pc,Mat R)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscObjectReference((PetscObject)R);CHKERRQ(ierr);
  ierr   = MatDestroy(&jac->R);CHKERRQ(ierr);
  jac->R = R;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGalerkinSetInterpolation_Galerkin(PC pc,Mat P)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
  ierr   = MatDestroy(&jac->P);CHKERRQ(ierr);
  jac->P = P;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGalerkinSetComputeSubmatrix_Galerkin(PC pc,PetscErrorCode (*computeAsub)(PC,Mat,Mat,Mat*,void*),void *ctx)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  jac->computeasub     = computeAsub;
  jac->computeasub_ctx = ctx;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
/*@
   PCGalerkinSetRestriction - Sets the restriction operator for the "Galerkin-type" preconditioner

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  R - the restriction operator

   Notes:
    Either this or PCGalerkinSetInterpolation() or both must be called

   Level: Intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetInterpolation(), PCGalerkinGetKSP()

@*/
PetscErrorCode  PCGalerkinSetRestriction(PC pc,Mat R)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGalerkinSetRestriction_C",(PC,Mat),(pc,R));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCGalerkinSetInterpolation - Sets the interpolation operator for the "Galerkin-type" preconditioner

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  R - the interpolation operator

   Notes:
    Either this or PCGalerkinSetRestriction() or both must be called

   Level: Intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetRestriction(), PCGalerkinGetKSP()

@*/
PetscErrorCode  PCGalerkinSetInterpolation(PC pc,Mat P)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGalerkinSetInterpolation_C",(PC,Mat),(pc,P));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCGalerkinSetComputeSubmatrix - Provide a routine that will be called to compute the Galerkin submatrix

   Logically Collective

   Input Parameters:
+  pc - the preconditioner context
.  computeAsub - routine that computes the submatrix from the global matrix
-  ctx - context used by the routine, or NULL

   Calling sequence of computeAsub:
$    computeAsub(PC pc,Mat A, Mat Ap, Mat *cAP,void *ctx);

+  PC - the Galerkin PC
.  A - the matrix in the Galerkin PC
.  Ap - the computed submatrix from any previous computation, if NULL it has not previously been computed
.  cAp - the submatrix computed by this routine
-  ctx - optional user-defined function context

   Level: Intermediate

   Notes:
    Instead of providing this routine you can call PCGalerkinGetKSP() and then KSPSetOperators() to provide the submatrix,
          but that will not work for multiple KSPSolves with different matrices unless you call it for each solve.

          This routine is called each time the outer matrix is changed. In the first call the Ap argument is NULL and the routine should create the
          matrix and computes its values in cAp. On each subsequent call the routine should up the Ap matrix.

   Developer Notes:
    If the user does not call this routine nor call PCGalerkinGetKSP() and KSPSetOperators() then PCGalerkin could
                    could automatically compute the submatrix via calls to MatGalerkin() or MatRARt()

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetRestriction(), PCGalerkinSetInterpolation(), PCGalerkinGetKSP()

@*/
PetscErrorCode  PCGalerkinSetComputeSubmatrix(PC pc,PetscErrorCode (*computeAsub)(PC,Mat,Mat,Mat*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGalerkinSetComputeSubmatrix_C",(PC,PetscErrorCode (*)(PC,Mat,Mat,Mat*,void*),void*),(pc,computeAsub,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCGalerkinGetKSP - Gets the KSP object in the Galerkin PC.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  ksp - the KSP object

   Level: Intermediate

   Notes:
    Once you have called this routine you can call KSPSetOperators() on the resulting ksp to provide the operator for the Galerkin problem,
          an alternative is to use PCGalerkinSetComputeSubmatrix() to provide a routine that computes the submatrix as needed.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetRestriction(), PCGalerkinSetInterpolation(), PCGalerkinSetComputeSubmatrix()

@*/
PetscErrorCode  PCGalerkinGetKSP(PC pc,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(pc,"PCGalerkinGetKSP_C",(PC,KSP*),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Galerkin(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode ierr;
  const char     *prefix;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = KSPGetOptionsPrefix(jac->ksp,&prefix);CHKERRQ(ierr);
  ierr = PetscStrendswith(prefix,"galerkin_",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(jac->ksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(jac->ksp,"galerkin_");CHKERRQ(ierr);
  }

  ierr = PetscOptionsHead(PetscOptionsObject,"Galerkin options");CHKERRQ(ierr);
  if (jac->ksp) {
    ierr = KSPSetFromOptions(jac->ksp);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

/*MC
     PCGALERKIN - Build (part of) a preconditioner by P S R (where P is often R^T)

$   Use PCGalerkinSetRestriction(pc,R) and/or PCGalerkinSetInterpolation(pc,P) followed by
$   PCGalerkinGetKSP(pc,&ksp); KSPSetOperators(ksp,A,....)

   Level: intermediate

   Developer Note: If KSPSetOperators() has not been called on the inner KSP then PCGALERKIN could use MatRARt() or MatPtAP() to compute
                   the operators automatically.
                   Should there be a prefix for the inner KSP.
                   There is no KSPSetFromOptions_Galerkin() that calls KSPSetFromOptions() on the inner KSP

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCKSP, PCGalerkinSetRestriction(), PCGalerkinSetInterpolation(), PCGalerkinGetKSP()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Galerkin(PC pc)
{
  PetscErrorCode ierr;
  PC_Galerkin    *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);

  pc->ops->apply           = PCApply_Galerkin;
  pc->ops->setup           = PCSetUp_Galerkin;
  pc->ops->reset           = PCReset_Galerkin;
  pc->ops->destroy         = PCDestroy_Galerkin;
  pc->ops->view            = PCView_Galerkin;
  pc->ops->setfromoptions  = PCSetFromOptions_Galerkin;
  pc->ops->applyrichardson = NULL;

  ierr = KSPCreate(PetscObjectComm((PetscObject)pc),&jac->ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(jac->ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1);CHKERRQ(ierr);

  pc->data = (void*)jac;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetRestriction_C",PCGalerkinSetRestriction_Galerkin);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetInterpolation_C",PCGalerkinSetInterpolation_Galerkin);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinGetKSP_C",PCGalerkinGetKSP_Galerkin);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetComputeSubmatrix_C",PCGalerkinSetComputeSubmatrix_Galerkin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

