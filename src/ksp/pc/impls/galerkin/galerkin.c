
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
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  if (jac->R) {
    PetscCall(MatRestrict(jac->R,x,jac->b));
  } else {
    PetscCall(MatRestrict(jac->P,x,jac->b));
  }
  PetscCall(KSPSolve(jac->ksp,jac->b,jac->x));
  PetscCall(KSPCheckSolve(jac->ksp,pc,jac->x));
  if (jac->P) {
    PetscCall(MatInterpolate(jac->P,jac->x,y));
  } else {
    PetscCall(MatInterpolate(jac->R,jac->x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Galerkin(PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscBool      a;
  Vec            *xx,*yy;

  PetscFunctionBegin;
  if (jac->computeasub) {
    Mat Ap;
    if (!pc->setupcalled) {
      PetscCall((*jac->computeasub)(pc,pc->pmat,NULL,&Ap,jac->computeasub_ctx));
      PetscCall(KSPSetOperators(jac->ksp,Ap,Ap));
      PetscCall(MatDestroy(&Ap));
    } else {
      PetscCall(KSPGetOperators(jac->ksp,NULL,&Ap));
      PetscCall((*jac->computeasub)(pc,pc->pmat,Ap,NULL,jac->computeasub_ctx));
    }
  }

  if (!jac->x) {
    PetscCall(KSPGetOperatorsSet(jac->ksp,&a,NULL));
    PetscCheck(a,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set operator of PCGALERKIN KSP with PCGalerkinGetKSP()/KSPSetOperators()");
    PetscCall(KSPCreateVecs(jac->ksp,1,&xx,1,&yy));
    jac->x = *xx;
    jac->b = *yy;
    PetscCall(PetscFree(xx));
    PetscCall(PetscFree(yy));
  }
  PetscCheck(jac->R || jac->P,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or interpolation of PCGALERKIN with PCGalerkinSetRestriction()/Interpolation()");
  /* should check here that sizes of R/P match size of a */

  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Galerkin(PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&jac->R));
  PetscCall(MatDestroy(&jac->P));
  PetscCall(VecDestroy(&jac->x));
  PetscCall(VecDestroy(&jac->b));
  PetscCall(KSPReset(jac->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Galerkin(PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_Galerkin(pc));
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Galerkin(PC pc,PetscViewer viewer)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  KSP on Galerkin follow\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n"));
  }
  PetscCall(KSPView(jac->ksp,viewer));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)R));
  PetscCall(MatDestroy(&jac->R));
  jac->R = R;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGalerkinSetInterpolation_Galerkin(PC pc,Mat P)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)P));
  PetscCall(MatDestroy(&jac->P));
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCGALERKIN`,
          `PCGalerkinSetInterpolation()`, `PCGalerkinGetKSP()`

@*/
PetscErrorCode  PCGalerkinSetRestriction(PC pc,Mat R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGalerkinSetRestriction_C",(PC,Mat),(pc,R));
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCGALERKIN`,
          `PCGalerkinSetRestriction()`, `PCGalerkinGetKSP()`

@*/
PetscErrorCode  PCGalerkinSetInterpolation(PC pc,Mat P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGalerkinSetInterpolation_C",(PC,Mat),(pc,P));
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCGALERKIN`,
          `PCGalerkinSetRestriction()`, `PCGalerkinSetInterpolation()`, `PCGalerkinGetKSP()`

@*/
PetscErrorCode  PCGalerkinSetComputeSubmatrix(PC pc,PetscErrorCode (*computeAsub)(PC,Mat,Mat,Mat*,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGalerkinSetComputeSubmatrix_C",(PC,PetscErrorCode (*)(PC,Mat,Mat,Mat*,void*),void*),(pc,computeAsub,ctx));
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCGALERKIN`,
          `PCGalerkinSetRestriction()`, `PCGalerkinSetInterpolation()`, `PCGalerkinSetComputeSubmatrix()`

@*/
PetscErrorCode  PCGalerkinGetKSP(PC pc,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  PetscUseMethod(pc,"PCGalerkinGetKSP_C",(PC,KSP*),(pc,ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Galerkin(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Galerkin    *jac = (PC_Galerkin*)pc->data;
  const char     *prefix;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(KSPGetOptionsPrefix(jac->ksp,&prefix));
  PetscCall(PetscStrendswith(prefix,"galerkin_",&flg));
  if (!flg) {
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(KSPSetOptionsPrefix(jac->ksp,prefix));
    PetscCall(KSPAppendOptionsPrefix(jac->ksp,"galerkin_"));
  }

  PetscOptionsHeadBegin(PetscOptionsObject,"Galerkin options");
  if (jac->ksp) {
    PetscCall(KSPSetFromOptions(jac->ksp));
  }
  PetscOptionsHeadEnd();
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCSHELL`, `PCKSP`, `PCGalerkinSetRestriction()`, `PCGalerkinSetInterpolation()`, `PCGalerkinGetKSP()`

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Galerkin(PC pc)
{
  PC_Galerkin    *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));

  pc->ops->apply           = PCApply_Galerkin;
  pc->ops->setup           = PCSetUp_Galerkin;
  pc->ops->reset           = PCReset_Galerkin;
  pc->ops->destroy         = PCDestroy_Galerkin;
  pc->ops->view            = PCView_Galerkin;
  pc->ops->setfromoptions  = PCSetFromOptions_Galerkin;
  pc->ops->applyrichardson = NULL;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc),&jac->ksp));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp,pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1));

  pc->data = (void*)jac;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetRestriction_C",PCGalerkinSetRestriction_Galerkin));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetInterpolation_C",PCGalerkinSetInterpolation_Galerkin));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinGetKSP_C",PCGalerkinGetKSP_Galerkin));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGalerkinSetComputeSubmatrix_C",PCGalerkinSetComputeSubmatrix_Galerkin));
  PetscFunctionReturn(0);
}
