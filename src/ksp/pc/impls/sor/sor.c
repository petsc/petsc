/*
   Defines a  (S)SOR  preconditioner for any Mat implementation
*/
#include <petsc/private/pcimpl.h>               /*I "petscpc.h" I*/

typedef struct {
  PetscInt   its;         /* inner iterations, number of sweeps */
  PetscInt   lits;        /* local inner iterations, number of sweeps applied by the local matrix mat->A */
  MatSORType sym;         /* forward, reverse, symmetric etc. */
  PetscReal  omega;
  PetscReal  fshift;
} PC_SOR;

static PetscErrorCode PCDestroy_SOR(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_SOR(PC pc,Vec x,Vec y)
{
  PC_SOR         *jac = (PC_SOR*)pc->data;
  PetscInt       flag = jac->sym | SOR_ZERO_INITIAL_GUESS;

  PetscFunctionBegin;
  PetscCall(MatSOR(pc->pmat,x,jac->omega,(MatSORType)flag,jac->fshift,jac->its,jac->lits,y));
  PetscCall(MatFactorGetError(pc->pmat,(MatFactorError*)&pc->failedreason));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_SOR(PC pc,Vec x,Vec y)
{
  PC_SOR         *jac = (PC_SOR*)pc->data;
  PetscInt       flag = jac->sym | SOR_ZERO_INITIAL_GUESS;
  PetscBool      set,sym;

  PetscFunctionBegin;
  PetscCall(MatIsSymmetricKnown(pc->pmat,&set,&sym));
  PetscCheck(set && sym && (jac->sym == SOR_SYMMETRIC_SWEEP || jac->sym == SOR_LOCAL_SYMMETRIC_SWEEP),PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Can only apply transpose of SOR if matrix is symmetric and sweep is symmetric");
  PetscCall(MatSOR(pc->pmat,x,jac->omega,(MatSORType)flag,jac->fshift,jac->its,jac->lits,y));
  PetscCall(MatFactorGetError(pc->pmat,(MatFactorError*)&pc->failedreason));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_SOR(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_SOR         *jac = (PC_SOR*)pc->data;
  MatSORType     stype = jac->sym;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc,"Warning, convergence critera ignored, using %" PetscInt_FMT " iterations\n",its));
  if (guesszero) stype = (MatSORType) (stype | SOR_ZERO_INITIAL_GUESS);
  PetscCall(MatSOR(pc->pmat,b,jac->omega,stype,jac->fshift,its*jac->its,jac->lits,y));
  PetscCall(MatFactorGetError(pc->pmat,(MatFactorError*)&pc->failedreason));
  *outits = its;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_SOR(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_SOR         *jac = (PC_SOR*)pc->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"(S)SOR options");
  PetscCall(PetscOptionsReal("-pc_sor_omega","relaxation factor (0 < omega < 2)","PCSORSetOmega",jac->omega,&jac->omega,NULL));
  PetscCall(PetscOptionsReal("-pc_sor_diagonal_shift","Add to the diagonal entries","",jac->fshift,&jac->fshift,NULL));
  PetscCall(PetscOptionsInt("-pc_sor_its","number of inner SOR iterations","PCSORSetIterations",jac->its,&jac->its,NULL));
  PetscCall(PetscOptionsInt("-pc_sor_lits","number of local inner SOR iterations","PCSORSetIterations",jac->lits,&jac->lits,NULL));
  PetscCall(PetscOptionsBoolGroupBegin("-pc_sor_symmetric","SSOR, not SOR","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_SYMMETRIC_SWEEP));
  PetscCall(PetscOptionsBoolGroup("-pc_sor_backward","use backward sweep instead of forward","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_BACKWARD_SWEEP));
  PetscCall(PetscOptionsBoolGroup("-pc_sor_forward","use forward sweep","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_FORWARD_SWEEP));
  PetscCall(PetscOptionsBoolGroup("-pc_sor_local_symmetric","use SSOR separately on each processor","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_LOCAL_SYMMETRIC_SWEEP));
  PetscCall(PetscOptionsBoolGroup("-pc_sor_local_backward","use backward sweep locally","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_LOCAL_BACKWARD_SWEEP));
  PetscCall(PetscOptionsBoolGroupEnd("-pc_sor_local_forward","use forward sweep locally","PCSORSetSymmetric",&flg));
  if (flg) PetscCall(PCSORSetSymmetric(pc,SOR_LOCAL_FORWARD_SWEEP));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_SOR(PC pc,PetscViewer viewer)
{
  PC_SOR         *jac = (PC_SOR*)pc->data;
  MatSORType     sym  = jac->sym;
  const char     *sortype;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (sym & SOR_ZERO_INITIAL_GUESS) PetscCall(PetscViewerASCIIPrintf(viewer,"  zero initial guess\n"));
    if (sym == SOR_APPLY_UPPER)                                              sortype = "apply_upper";
    else if (sym == SOR_APPLY_LOWER)                                         sortype = "apply_lower";
    else if (sym & SOR_EISENSTAT)                                            sortype = "Eisenstat";
    else if ((sym & SOR_SYMMETRIC_SWEEP) == SOR_SYMMETRIC_SWEEP)             sortype = "symmetric";
    else if (sym & SOR_BACKWARD_SWEEP)                                       sortype = "backward";
    else if (sym & SOR_FORWARD_SWEEP)                                        sortype = "forward";
    else if ((sym & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) sortype = "local_symmetric";
    else if (sym & SOR_LOCAL_FORWARD_SWEEP)                                  sortype = "local_forward";
    else if (sym & SOR_LOCAL_BACKWARD_SWEEP)                                 sortype = "local_backward";
    else                                                                     sortype = "unknown";
    PetscCall(PetscViewerASCIIPrintf(viewer,"  type = %s, iterations = %" PetscInt_FMT ", local iterations = %" PetscInt_FMT ", omega = %g\n",sortype,jac->its,jac->lits,(double)jac->omega));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
static PetscErrorCode  PCSORSetSymmetric_SOR(PC pc,MatSORType flag)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  jac->sym = flag;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCSORSetOmega_SOR(PC pc,PetscReal omega)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  PetscCheck(omega > 0.0 && omega < 2.0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Relaxation out of range");
  jac->omega = omega;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCSORSetIterations_SOR(PC pc,PetscInt its,PetscInt lits)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  jac->its  = its;
  jac->lits = lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCSORGetSymmetric_SOR(PC pc,MatSORType *flag)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  *flag = jac->sym;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCSORGetOmega_SOR(PC pc,PetscReal *omega)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  *omega = jac->omega;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCSORGetIterations_SOR(PC pc,PetscInt *its,PetscInt *lits)
{
  PC_SOR *jac = (PC_SOR*)pc->data;

  PetscFunctionBegin;
  if (its)  *its = jac->its;
  if (lits) *lits = jac->lits;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
/*@
   PCSORGetSymmetric - Gets the form the SOR preconditioner is using;   backward, or forward relaxation.  The local variants perform SOR on
   each processor.  By default forward relaxation is used.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flag - one of the following
.vb
    SOR_FORWARD_SWEEP
    SOR_BACKWARD_SWEEP
    SOR_SYMMETRIC_SWEEP
    SOR_LOCAL_FORWARD_SWEEP
    SOR_LOCAL_BACKWARD_SWEEP
    SOR_LOCAL_SYMMETRIC_SWEEP
.ve

   Options Database Keys:
+  -pc_sor_symmetric - Activates symmetric version
.  -pc_sor_backward - Activates backward version
.  -pc_sor_local_forward - Activates local forward version
.  -pc_sor_local_symmetric - Activates local symmetric version
-  -pc_sor_local_backward - Activates local backward version

   Notes:
   To use the Eisenstat trick with SSOR, employ the PCEISENSTAT preconditioner,
   which can be chosen with the option
.  -pc_type eisenstat - Activates Eisenstat trick

   Level: intermediate

.seealso: PCEisenstatSetOmega(), PCSORSetIterations(), PCSORSetOmega(), PCSORSetSymmetric()
@*/
PetscErrorCode  PCSORGetSymmetric(PC pc,MatSORType *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCSORGetSymmetric_C",(PC,MatSORType*),(pc,flag));
  PetscFunctionReturn(0);
}

/*@
   PCSORGetOmega - Gets the SOR relaxation coefficient, omega
   (where omega = 1.0 by default).

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  omega - relaxation coefficient (0 < omega < 2).

   Options Database Key:
.  -pc_sor_omega <omega> - Sets omega

   Level: intermediate

.seealso: PCSORSetSymmetric(), PCSORSetIterations(), PCEisenstatSetOmega(), PCSORSetOmega()
@*/
PetscErrorCode  PCSORGetOmega(PC pc,PetscReal *omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCSORGetOmega_C",(PC,PetscReal*),(pc,omega));
  PetscFunctionReturn(0);
}

/*@
   PCSORGetIterations - Gets the number of inner iterations to
   be used by the SOR preconditioner. The default is 1.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  lits - number of local iterations, smoothings over just variables on processor
-  its - number of parallel iterations to use; each parallel iteration has lits local iterations

   Options Database Key:
+  -pc_sor_its <its> - Sets number of iterations
-  -pc_sor_lits <lits> - Sets number of local iterations

   Level: intermediate

   Notes:
    When run on one processor the number of smoothings is lits*its

.seealso: PCSORSetOmega(), PCSORSetSymmetric(), PCSORSetIterations()
@*/
PetscErrorCode  PCSORGetIterations(PC pc,PetscInt *its,PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCSORGetIterations_C",(PC,PetscInt*,PetscInt*),(pc,its,lits));
  PetscFunctionReturn(0);
}

/*@
   PCSORSetSymmetric - Sets the SOR preconditioner to use symmetric (SSOR),
   backward, or forward relaxation.  The local variants perform SOR on
   each processor.  By default forward relaxation is used.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - one of the following
.vb
    SOR_FORWARD_SWEEP
    SOR_BACKWARD_SWEEP
    SOR_SYMMETRIC_SWEEP
    SOR_LOCAL_FORWARD_SWEEP
    SOR_LOCAL_BACKWARD_SWEEP
    SOR_LOCAL_SYMMETRIC_SWEEP
.ve

   Options Database Keys:
+  -pc_sor_symmetric - Activates symmetric version
.  -pc_sor_backward - Activates backward version
.  -pc_sor_local_forward - Activates local forward version
.  -pc_sor_local_symmetric - Activates local symmetric version
-  -pc_sor_local_backward - Activates local backward version

   Notes:
   To use the Eisenstat trick with SSOR, employ the PCEISENSTAT preconditioner,
   which can be chosen with the option
.  -pc_type eisenstat - Activates Eisenstat trick

   Level: intermediate

.seealso: PCEisenstatSetOmega(), PCSORSetIterations(), PCSORSetOmega()
@*/
PetscErrorCode  PCSORSetSymmetric(PC pc,MatSORType flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,flag,2);
  PetscTryMethod(pc,"PCSORSetSymmetric_C",(PC,MatSORType),(pc,flag));
  PetscFunctionReturn(0);
}

/*@
   PCSORSetOmega - Sets the SOR relaxation coefficient, omega
   (where omega = 1.0 by default).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  omega - relaxation coefficient (0 < omega < 2).

   Options Database Key:
.  -pc_sor_omega <omega> - Sets omega

   Level: intermediate

   Note:
   If omega != 1, you will need to set the MAT_USE_INODES option to PETSC_FALSE on the matrix.

.seealso: PCSORSetSymmetric(), PCSORSetIterations(), PCEisenstatSetOmega(), MatSetOption()
@*/
PetscErrorCode  PCSORSetOmega(PC pc,PetscReal omega)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveReal(pc,omega,2);
  PetscTryMethod(pc,"PCSORSetOmega_C",(PC,PetscReal),(pc,omega));
  PetscFunctionReturn(0);
}

/*@
   PCSORSetIterations - Sets the number of inner iterations to
   be used by the SOR preconditioner. The default is 1.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  lits - number of local iterations, smoothings over just variables on processor
-  its - number of parallel iterations to use; each parallel iteration has lits local iterations

   Options Database Key:
+  -pc_sor_its <its> - Sets number of iterations
-  -pc_sor_lits <lits> - Sets number of local iterations

   Level: intermediate

   Notes:
    When run on one processor the number of smoothings is lits*its

.seealso: PCSORSetOmega(), PCSORSetSymmetric()
@*/
PetscErrorCode  PCSORSetIterations(PC pc,PetscInt its,PetscInt lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,its,2);
  PetscTryMethod(pc,"PCSORSetIterations_C",(PC,PetscInt,PetscInt),(pc,its,lits));
  PetscFunctionReturn(0);
}

/*MC
     PCSOR - (S)SOR (successive over relaxation, Gauss-Seidel) preconditioning

   Options Database Keys:
+  -pc_sor_symmetric - Activates symmetric version
.  -pc_sor_backward - Activates backward version
.  -pc_sor_forward - Activates forward version
.  -pc_sor_local_forward - Activates local forward version
.  -pc_sor_local_symmetric - Activates local symmetric version  (default version)
.  -pc_sor_local_backward - Activates local backward version
.  -pc_sor_omega <omega> - Sets omega
.  -pc_sor_diagonal_shift <shift> - shift the diagonal entries; useful if the matrix has zeros on the diagonal
.  -pc_sor_its <its> - Sets number of iterations   (default 1)
-  -pc_sor_lits <lits> - Sets number of local iterations  (default 1)

   Level: beginner

   Notes:
    Only implemented for the AIJ  and SeqBAIJ matrix formats.
          Not a true parallel SOR, in parallel this implementation corresponds to block
          Jacobi with SOR on each block.

          For AIJ matrix if a diagonal entry is zero (and the diagonal shift is zero) then by default the inverse of that
          zero will be used and hence the KSPSolve() will terminate with KSP_DIVERGED_NANORIF. If the option
          KSPSetErrorIfNotConverged() or -ksp_error_if_not_converged the code will terminate as soon as it detects the
          zero pivot.

          For SeqBAIJ matrices this implements point-block SOR, but the omega, its, lits options are not supported.

          For SeqBAIJ the diagonal blocks are inverted using dense LU with partial pivoting. If a zero pivot is detected
          the computation is stopped with an error

          If used with KSPRICHARDSON and no monitors the convergence test is skipped to improve speed, thus it always iterates
          the maximum number of iterations you've selected for KSP. It is usually used in this mode as a smoother for multigrid.

          If omega != 1, you will need to set the MAT_USE_INODES option to PETSC_FALSE on the matrix.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSORSetIterations(), PCSORSetSymmetric(), PCSORSetOmega(), PCEISENSTAT, MatSetOption()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_SOR(PC pc)
{
  PC_SOR         *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));

  pc->ops->apply           = PCApply_SOR;
  pc->ops->applytranspose  = PCApplyTranspose_SOR;
  pc->ops->applyrichardson = PCApplyRichardson_SOR;
  pc->ops->setfromoptions  = PCSetFromOptions_SOR;
  pc->ops->setup           = NULL;
  pc->ops->view            = PCView_SOR;
  pc->ops->destroy         = PCDestroy_SOR;
  pc->data                 = (void*)jac;
  jac->sym                 = SOR_LOCAL_SYMMETRIC_SWEEP;
  jac->omega               = 1.0;
  jac->fshift              = 0.0;
  jac->its                 = 1;
  jac->lits                = 1;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORSetSymmetric_C",PCSORSetSymmetric_SOR));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORSetOmega_C",PCSORSetOmega_SOR));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORSetIterations_C",PCSORSetIterations_SOR));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORGetSymmetric_C",PCSORGetSymmetric_SOR));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORGetOmega_C",PCSORGetOmega_SOR));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSORGetIterations_C",PCSORGetIterations_SOR));
  PetscFunctionReturn(0);
}
