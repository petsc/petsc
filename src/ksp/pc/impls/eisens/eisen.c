
/*
   Defines a  Eisenstat trick SSOR  preconditioner. This uses about
 %50 of the usual amount of floating point ops used for SSOR + Krylov
 method. But it requires actually solving the preconditioned problem
 with both left and right preconditioning.
*/
#include <petsc/private/pcimpl.h>           /*I "petscpc.h" I*/

typedef struct {
  Mat       shell,A;
  Vec       b[2],diag;   /* temporary storage for true right hand side */
  PetscReal omega;
  PetscBool usediag;     /* indicates preconditioner should include diagonal scaling*/
} PC_Eisenstat;

static PetscErrorCode PCMult_Eisenstat(Mat mat,Vec b,Vec x)
{
  PC             pc;
  PC_Eisenstat   *eis;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&pc));
  eis  = (PC_Eisenstat*)pc->data;
  CHKERRQ(MatSOR(eis->A,b,eis->omega,SOR_EISENSTAT,0.0,1,1,x));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Eisenstat(PC pc,Vec x,Vec y)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;
  PetscBool      hasop;

  PetscFunctionBegin;
  if (eis->usediag) {
    CHKERRQ(MatHasOperation(pc->pmat,MATOP_MULT_DIAGONAL_BLOCK,&hasop));
    if (hasop) {
      CHKERRQ(MatMultDiagonalBlock(pc->pmat,x,y));
    } else {
      CHKERRQ(VecPointwiseMult(y,x,eis->diag));
    }
  } else CHKERRQ(VecCopy(x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolve_Eisenstat(PC pc,KSP ksp,Vec b,Vec x)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;
  PetscBool      nonzero;

  PetscFunctionBegin;
  if (pc->presolvedone < 2) {
    PetscCheckFalse(pc->mat != pc->pmat,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot have different mat and pmat");
    /* swap shell matrix and true matrix */
    eis->A  = pc->mat;
    pc->mat = eis->shell;
  }

  if (!eis->b[pc->presolvedone-1]) {
    CHKERRQ(VecDuplicate(b,&eis->b[pc->presolvedone-1]));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)eis->b[pc->presolvedone-1]));
  }

  /* if nonzero initial guess, modify x */
  CHKERRQ(KSPGetInitialGuessNonzero(ksp,&nonzero));
  if (nonzero) {
    CHKERRQ(VecCopy(x,eis->b[pc->presolvedone-1]));
    CHKERRQ(MatSOR(eis->A,eis->b[pc->presolvedone-1],eis->omega,SOR_APPLY_UPPER,0.0,1,1,x));
  }

  /* save true b, other option is to swap pointers */
  CHKERRQ(VecCopy(b,eis->b[pc->presolvedone-1]));

  /* modify b by (L + D/omega)^{-1} */
  CHKERRQ(MatSOR(eis->A,eis->b[pc->presolvedone-1],eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP),0.0,1,1,b));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPostSolve_Eisenstat(PC pc,KSP ksp,Vec b,Vec x)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  /* get back true b */
  CHKERRQ(VecCopy(eis->b[pc->presolvedone],b));

  /* modify x by (U + D/omega)^{-1} */
  CHKERRQ(VecCopy(x,eis->b[pc->presolvedone]));
  CHKERRQ(MatSOR(eis->A,eis->b[pc->presolvedone],eis->omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_BACKWARD_SWEEP),0.0,1,1,x));
  if (!pc->presolvedone) pc->mat = eis->A;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Eisenstat(PC pc)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&eis->b[0]));
  CHKERRQ(VecDestroy(&eis->b[1]));
  CHKERRQ(MatDestroy(&eis->shell));
  CHKERRQ(VecDestroy(&eis->diag));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Eisenstat(PC pc)
{
  PetscFunctionBegin;
  CHKERRQ(PCReset_Eisenstat(pc));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Eisenstat(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;
  PetscBool      set,flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Eisenstat SSOR options"));
  CHKERRQ(PetscOptionsReal("-pc_eisenstat_omega","Relaxation factor 0 < omega < 2","PCEisenstatSetOmega",eis->omega,&eis->omega,NULL));
  CHKERRQ(PetscOptionsBool("-pc_eisenstat_no_diagonal_scaling","Do not use standard diagonal scaling","PCEisenstatSetNoDiagonalScaling",eis->usediag ? PETSC_FALSE : PETSC_TRUE,&flg,&set));
  if (set) {
    CHKERRQ(PCEisenstatSetNoDiagonalScaling(pc,flg));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Eisenstat(PC pc,PetscViewer viewer)
{
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  omega = %g\n",(double)eis->omega));
    if (eis->usediag) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Using diagonal scaling (default)\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Not using diagonal scaling\n"));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Eisenstat(PC pc)
{
  PetscInt       M,N,m,n;
  PC_Eisenstat   *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    CHKERRQ(MatGetSize(pc->mat,&M,&N));
    CHKERRQ(MatGetLocalSize(pc->mat,&m,&n));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pc),&eis->shell));
    CHKERRQ(MatSetSizes(eis->shell,m,n,M,N));
    CHKERRQ(MatSetType(eis->shell,MATSHELL));
    CHKERRQ(MatSetUp(eis->shell));
    CHKERRQ(MatShellSetContext(eis->shell,pc));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)eis->shell));
    CHKERRQ(MatShellSetOperation(eis->shell,MATOP_MULT,(void (*)(void))PCMult_Eisenstat));
  }
  if (!eis->usediag) PetscFunctionReturn(0);
  if (!pc->setupcalled) {
    CHKERRQ(MatCreateVecs(pc->pmat,&eis->diag,NULL));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)eis->diag));
  }
  CHKERRQ(MatGetDiagonal(pc->pmat,eis->diag));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/

static PetscErrorCode  PCEisenstatSetOmega_Eisenstat(PC pc,PetscReal omega)
{
  PC_Eisenstat *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  PetscCheckFalse(omega >= 2.0 || omega <= 0.0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Relaxation out of range");
  eis->omega = omega;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCEisenstatSetNoDiagonalScaling_Eisenstat(PC pc,PetscBool flg)
{
  PC_Eisenstat *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  eis->usediag = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCEisenstatGetOmega_Eisenstat(PC pc,PetscReal *omega)
{
  PC_Eisenstat *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  *omega = eis->omega;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCEisenstatGetNoDiagonalScaling_Eisenstat(PC pc,PetscBool *flg)
{
  PC_Eisenstat *eis = (PC_Eisenstat*)pc->data;

  PetscFunctionBegin;
  *flg = eis->usediag;
  PetscFunctionReturn(0);
}

/*@
   PCEisenstatSetOmega - Sets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
.  -pc_eisenstat_omega <omega> - Sets omega

   Notes:
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner,
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

   Level: intermediate

.seealso: PCSORSetOmega()
@*/
PetscErrorCode  PCEisenstatSetOmega(PC pc,PetscReal omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,omega,2);
  CHKERRQ(PetscTryMethod(pc,"PCEisenstatSetOmega_C",(PC,PetscReal),(pc,omega)));
  PetscFunctionReturn(0);
}

/*@
   PCEisenstatSetNoDiagonalScaling - Causes the Eisenstat preconditioner
   not to do additional diagonal preconditioning. For matrices with a constant
   along the diagonal, this may save a small amount of work.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - PETSC_TRUE turns off diagonal scaling inside the algorithm

   Options Database Key:
.  -pc_eisenstat_no_diagonal_scaling - Activates PCEisenstatSetNoDiagonalScaling()

   Level: intermediate

   Note:
     If you use the KSPSetDiagonalScaling() or -ksp_diagonal_scale option then you will
   likley want to use this routine since it will save you some unneeded flops.

.seealso: PCEisenstatSetOmega()
@*/
PetscErrorCode  PCEisenstatSetNoDiagonalScaling(PC pc,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCEisenstatSetNoDiagonalScaling_C",(PC,PetscBool),(pc,flg)));
  PetscFunctionReturn(0);
}

/*@
   PCEisenstatGetOmega - Gets the SSOR relaxation coefficient, omega,
   to use with Eisenstat's trick (where omega = 1.0 by default).

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  omega - relaxation coefficient (0 < omega < 2)

   Options Database Key:
.  -pc_eisenstat_omega <omega> - Sets omega

   Notes:
   The Eisenstat trick implementation of SSOR requires about 50% of the
   usual amount of floating point operations used for SSOR + Krylov method;
   however, the preconditioned problem must be solved with both left
   and right preconditioning.

   To use SSOR without the Eisenstat trick, employ the PCSOR preconditioner,
   which can be chosen with the database options
$    -pc_type  sor  -pc_sor_symmetric

   Level: intermediate

.seealso: PCSORGetOmega(), PCEisenstatSetOmega()
@*/
PetscErrorCode  PCEisenstatGetOmega(PC pc,PetscReal *omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCEisenstatGetOmega_C",(PC,PetscReal*),(pc,omega)));
  PetscFunctionReturn(0);
}

/*@
   PCEisenstatGetNoDiagonalScaling - Tells if the Eisenstat preconditioner
   not to do additional diagonal preconditioning. For matrices with a constant
   along the diagonal, this may save a small amount of work.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flg - PETSC_TRUE means there is no diagonal scaling applied

   Options Database Key:
.  -pc_eisenstat_no_diagonal_scaling - Activates PCEisenstatSetNoDiagonalScaling()

   Level: intermediate

   Note:
     If you use the KSPSetDiagonalScaling() or -ksp_diagonal_scale option then you will
   likley want to use this routine since it will save you some unneeded flops.

.seealso: PCEisenstatGetOmega()
@*/
PetscErrorCode  PCEisenstatGetNoDiagonalScaling(PC pc,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCEisenstatGetNoDiagonalScaling_C",(PC,PetscBool*),(pc,flg)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolveChangeRHS_Eisenstat(PC pc, PetscBool* change)
{
  PetscFunctionBegin;
  *change = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/

/*MC
     PCEISENSTAT - An implementation of SSOR (symmetric successive over relaxation, symmetric Gauss-Seidel)
           preconditioning that incorporates Eisenstat's trick to reduce the amount of computation needed.

   Options Database Keys:
+  -pc_eisenstat_omega <omega> - Sets omega
-  -pc_eisenstat_no_diagonal_scaling - Activates PCEisenstatSetNoDiagonalScaling()

   Level: beginner

   Notes:
    Only implemented for the SeqAIJ matrix format.
          Not a true parallel SOR, in parallel this implementation corresponds to block
          Jacobi with SOR on each block.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCEisenstatSetNoDiagonalScaling(), PCEisenstatSetOmega(), PCSOR
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Eisenstat(PC pc)
{
  PC_Eisenstat   *eis;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&eis));

  pc->ops->apply           = PCApply_Eisenstat;
  pc->ops->presolve        = PCPreSolve_Eisenstat;
  pc->ops->postsolve       = PCPostSolve_Eisenstat;
  pc->ops->applyrichardson = NULL;
  pc->ops->setfromoptions  = PCSetFromOptions_Eisenstat;
  pc->ops->destroy         = PCDestroy_Eisenstat;
  pc->ops->reset           = PCReset_Eisenstat;
  pc->ops->view            = PCView_Eisenstat;
  pc->ops->setup           = PCSetUp_Eisenstat;

  pc->data     = eis;
  eis->omega   = 1.0;
  eis->b[0]    = NULL;
  eis->b[1]    = NULL;
  eis->diag    = NULL;
  eis->usediag = PETSC_TRUE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatSetOmega_C",PCEisenstatSetOmega_Eisenstat));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatSetNoDiagonalScaling_C",PCEisenstatSetNoDiagonalScaling_Eisenstat));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatGetOmega_C",PCEisenstatGetOmega_Eisenstat));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCEisenstatGetNoDiagonalScaling_C",PCEisenstatGetNoDiagonalScaling_Eisenstat));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",PCPreSolveChangeRHS_Eisenstat));
  PetscFunctionReturn(0);
}
