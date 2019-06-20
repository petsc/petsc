#include <../src/ksp/pc/impls/deflation/deflation.h> /*I "petscpc.h" I*/  /* includes for fortran wrappers */

const char *const PCDeflationSpaceTypes[] = {
  "haar",
  "jacket-haar",
  "db2",
  "db4",
  "db8",
  "db16",
  "biorth22",
  "meyer",
  "aggregation",
  "slepc",
  "slepc-cheap",
  "user",
  "PCDeflationSpaceType",
  "PC_DEFLATION_SPACE_",
  0
};

static PetscErrorCode PCDeflationSetInitOnly_Deflation(PC pc,PetscBool flg)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  def->init = flg;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetInitOnly - Do only initialization step.
    Sets initial guess to the solution on the deflation space but do not apply deflation preconditioner.
    The additional preconditioner is still applied.

   Logically Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
-  flg - default PETSC_FALSE

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetInitOnly(PC pc,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetInitOnly_C",(PC,PetscBool),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode PCDeflationSetLvl_Deflation(PC pc,PetscInt current,PetscInt max)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  if (current) def->nestedlvl = current;
  def->maxnestedlvl = max;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetMaxLvl - Set maximum level of deflation.

   Logically Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
-  max - maximum deflation level

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetMaxLvl(PC pc,PetscInt max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,max,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetLvl_C",(PC,PetscInt,PetscInt),(pc,0,max));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetReductionFactor_Deflation(PC pc,PetscInt red)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  def->reductionfact = red;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetReductionFactor - Set reduction factor for PCTELESCOPE coarse problem solver.

   Logically Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
-  red - reduction factor (or PETSC_DETERMINE)

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetReductionFactor(PC pc,PetscInt red)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,red,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetReductionFactor_C",(PC,PetscInt),(pc,red));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetCorrectionFactor_Deflation(PC pc,PetscScalar fact)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  /* TODO PETSC_DETERMINE -> compute max eigenvalue with power method */
  def->correct = PETSC_TRUE;
  def->correctfact = fact;
  if (fact == PETSC_DEFAULT) {
    def->correctfact = 1.0;
  } else if (def->correct == 0.0) {
    def->correct = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetCorrectionFactor - Set coarse problem correction factor.
    The Preconditioner becomes P*M^{-1} + factor*Q.

   Logically Collective on PC

   Input Parameters:
+  pc   - the preconditioner context
-  fact - correction factor (set 0.0 to disable, PETSC_DEFAULT = 1.0)

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetCorrectionFactor(PC pc,PetscScalar fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,fact,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetCorrectionFactor_C",(PC,PetscScalar),(pc,fact));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetSpaceToCompute_Deflation(PC pc,PCDeflationSpaceType type,PetscInt size)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  if (type) def->spacetype = type;
  if (size > 0) def->spacesize = size;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetSpaceToCompute - Set deflation space type and size to compute.

   Logically Collective on PC

   Input Parameters:
+  pc   - the preconditioner context
.  type - deflation space type to compute (or PETSC_IGNORE)
-  size - size of the space to compute (or PETSC_DEFAULT)

   Notes:
    For wavelet-based deflation, size represents number of levels.
    The deflation space is computed in PCSetUP().

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetSpaceToCompute(PC pc,PCDeflationSpaceType type,PetscInt size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (type) PetscValidLogicalCollectiveEnum(pc,type,2);
  if (size > 0) PetscValidLogicalCollectiveInt(pc,size,3);
  ierr = PetscTryMethod(pc,"PCDeflationSetSpaceToCompute_C",(PC,PCDeflationSpaceType,PetscInt),(pc,type,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetSpace_Deflation(PC pc,Mat W,PetscBool transpose)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (transpose) {
    def->Wt = W;
    def->W = NULL;
  } else {
    def->W = W;
  }
  ierr = PetscObjectReference((PetscObject)W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetSpace - Set deflation space matrix (or its transpose).

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  W  - deflation matrix
-  tranpose - indicates that W is an explicit transpose of the deflation matrix

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetSpace(PC pc,Mat W,PetscBool transpose)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(W,MAT_CLASSID,2);
  PetscValidLogicalCollectiveBool(pc,transpose,3);
  ierr = PetscTryMethod(pc,"PCDeflationSetSpace_C",(PC,Mat,PetscBool),(pc,W,transpose));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetProjectionNullSpaceMat_Deflation(PC pc,Mat mat)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&def->WtA);CHKERRQ(ierr);
  def->WtA = mat;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetProjectionNullSpaceMat - Set projection null space matrix (W'*A).

   Not Collective

   Input Parameters:
+  pc  - preconditioner context
-  mat - projection null space matrix

   Level: developer

.seealso: PCDEFLATION
@*/
PetscErrorCode  PCDeflationSetProjectionNullSpaceMat(PC pc,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetProjectionNullSpaceMat_C",(PC,Mat),(pc,mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetCoarseMat_Deflation(PC pc,Mat mat)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&def->WtAW);CHKERRQ(ierr);
  def->WtAW = mat;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetCoarseMat - Set coarse problem Mat.

   Not Collective

   Input Parameters:
+  pc - preconditioner context
-  mat - coarse problem mat

   Level: developer

.seealso: PCDEFLATION
@*/
PetscErrorCode  PCDeflationSetCoarseMat(PC pc,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetCoarseMat_C",(PC,Mat),(pc,mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationGetCoarseKSP_Deflation(PC pc,KSP *ksp)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  *ksp = def->WtAWinv;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationGetCoarseKSP - Returns a pointer to the coarse problem KSP.

   Not Collective

   Input Parameters:
.  pc - preconditioner context

   Output Parameter:
.  ksp - coarse problem KSP context

   Level: developer

.seealso: PCDeflationSetCoarseKSP(), PCDEFLATION
@*/
PetscErrorCode  PCDeflationGetCoarseKSP(PC pc,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscTryMethod(pc,"PCDeflationGetCoarseKSP_C",(PC,KSP*),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetCoarseKSP_Deflation(PC pc,KSP ksp)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&def->WtAWinv);CHKERRQ(ierr);
  def->WtAWinv = ksp;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtAWinv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetCoarseKSP - Set coarse problem KSP.

   Collective on PC

   Input Parameters:
+  pc - preconditioner context
-  ksp - coarse problem KSP context

   Level: developer

.seealso: PCDeflationGetCoarseKSP(), PCDEFLATION
@*/
PetscErrorCode  PCDeflationSetCoarseKSP(PC pc,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(pc,1,ksp,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetCoarseKSP_C",(PC,KSP),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationGetPC_Deflation(PC pc,PC *apc)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  *apc = def->pc;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationGetPC - Returns a pointer to additional preconditioner.

   Not Collective

   Input Parameters:
.  pc  - the preconditioner context

   Output Parameter:
.  apc - additional preconditioner

   Level: advanced

.seealso: PCDeflationSetPC(), PCDEFLATION
@*/
PetscErrorCode PCDeflationGetPC(PC pc,PC *apc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(pc,2);
  ierr = PetscTryMethod(pc,"PCDeflationGetPC_C",(PC,PC*),(pc,apc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetPC_Deflation(PC pc,PC apc)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy(&def->pc);CHKERRQ(ierr);
  def->pc = apc;
  ierr = PetscObjectReference((PetscObject)apc);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetPC - Set additional preconditioner.

   Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
-  apc - additional preconditioner

   Level: developer

.seealso: PCDeflationGetPC(), PCDEFLATION
@*/
PetscErrorCode PCDeflationSetPC(PC pc,PC apc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(apc,PC_CLASSID,2);
  PetscCheckSameComm(pc,1,apc,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetPC_C",(PC,PC),(pc,apc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  x <- x + W*(W'*A*W)^{-1}*W'*r  = x + Q*r
*/
static PetscErrorCode PCPreSolve_Deflation(PC pc,KSP ksp,Vec b, Vec x)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  Mat              A;
  Vec              r,w1,w2;
  PetscBool        nonzero;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  w1 = def->workcoarse[0];
  w2 = def->workcoarse[1];
  r  = def->work;
  ierr = PCGetOperators(pc,NULL,&A);CHKERRQ(ierr);

  ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  if (nonzero) {
    ierr = MatMult(A,x,r);CHKERRQ(ierr);                          /*    r  <- b - Ax              */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);                            /*    r  <- b (x is 0)          */
  }

  if (def->Wt) {
    ierr = MatMult(def->Wt,r,w1);CHKERRQ(ierr);                   /*    w1 <- W'*r                */
  } else {
    ierr = MatMultHermitianTranspose(def->W,r,w1);CHKERRQ(ierr);  /*    w1 <- W'*r                */
  }
  ierr = KSPSolve(def->WtAWinv,w1,w2);CHKERRQ(ierr);              /*    w2 <- (W'*A*W)^{-1}*w1    */
  ierr = MatMult(def->W,w2,r);CHKERRQ(ierr);                      /*    r  <- W*w2                */
  ierr = VecAYPX(x,1.0,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  if (def->correct) {
    z <- M^{-1}r - W*(W'*A*W)^{-1}*(W'*A*M^{-1}r - l*W'*r) = (P*M^{-1} + l*Q)*r
  } else {
    z <- M^{-1}*r - W*(W'*A*W)^{-1}*W'*A*M{-1}*r = P*M^{-1}*r
  }
*/
static PetscErrorCode PCApply_Deflation(PC pc,Vec r,Vec z)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  Mat              A;
  Vec              u,w1,w2;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  w1 = def->workcoarse[0];
  w2 = def->workcoarse[1];
  u  = def->work;
  ierr = PCGetOperators(pc,NULL,&A);CHKERRQ(ierr);

  ierr = PCApply(def->pc,r,z);CHKERRQ(ierr);                          /*    z <- M^{-1}*r             */
  if (!def->init) {
    ierr = MatMult(def->WtA,z,w1);CHKERRQ(ierr);                      /*    w1 <- W'*A*z              */
    if (def->correct) {
      if (def->Wt) {
        ierr = MatMult(def->Wt,r,w2);CHKERRQ(ierr);                   /*    w2 <- W'*r                */
      } else {
        ierr = MatMultHermitianTranspose(def->W,r,w2);CHKERRQ(ierr);  /*    w2 <- W'*r                */
      }
      ierr = VecAXPY(w1,-1.0*def->correctfact,w2);CHKERRQ(ierr);      /*    w1 <- w1 - l*w2           */
    }
    ierr = KSPSolve(def->WtAWinv,w1,w2);CHKERRQ(ierr);                /*    w2 <- (W'*A*W)^{-1}*w1    */
    ierr = MatMult(def->W,w2,u);CHKERRQ(ierr);                        /*    u  <- W*w2                */
    ierr = VecAXPY(z,-1.0,u);CHKERRQ(ierr);                           /*    z  <- z - u               */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Deflation(PC pc)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;
  KSP              innerksp;
  PC               pcinner;
  Mat              Amat,nextDef=NULL,*mats;
  PetscInt         i,m,red,size,commsize;
  PetscBool        match,flgspd,transp=PETSC_FALSE;
  MatCompositeType ctype;
  MPI_Comm         comm;
  const char       *prefix;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (pc->setupcalled) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&Amat);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  /* compute a deflation space */
  if (def->W || def->Wt) {
    def->spacetype = PC_DEFLATION_SPACE_USER;
  } else {
    ierr = PCDeflationComputeSpace(pc);CHKERRQ(ierr);
  }

  /* nested deflation */
  if (def->W) {
    ierr = PetscObjectTypeCompare((PetscObject)def->W,MATCOMPOSITE,&match);CHKERRQ(ierr);
    if (match) {
      ierr = MatCompositeGetType(def->W,&ctype);CHKERRQ(ierr);
      ierr = MatCompositeGetNumberMat(def->W,&size);CHKERRQ(ierr);
    }
  } else {
    ierr = MatCreateHermitianTranspose(def->Wt,&def->W);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)def->Wt,MATCOMPOSITE,&match);CHKERRQ(ierr);
    if (match) {
      ierr = MatCompositeGetType(def->Wt,&ctype);CHKERRQ(ierr);
      ierr = MatCompositeGetNumberMat(def->Wt,&size);CHKERRQ(ierr);
    }
    transp = PETSC_TRUE;
  }
  if (match && ctype == MAT_COMPOSITE_MULTIPLICATIVE) {
    if (!transp) {
      if (def->nestedlvl < def->maxnestedlvl) {
        ierr = PetscMalloc1(size,&mats);CHKERRQ(ierr);
        for (i=0; i<size; i++) {
          ierr = MatCompositeGetMat(def->W,i,&mats[i]);CHKERRQ(ierr);
        }
        size -= 1;
        ierr = MatDestroy(&def->W);CHKERRQ(ierr);
        def->W = mats[size];
        ierr = PetscObjectReference((PetscObject)mats[size]);CHKERRQ(ierr);
        if (size > 1) {
          ierr = MatCreateComposite(comm,size,mats,&nextDef);CHKERRQ(ierr);
          ierr = MatCompositeSetType(nextDef,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
        } else {
          nextDef = mats[0];
          ierr = PetscObjectReference((PetscObject)mats[0]);CHKERRQ(ierr);
        }
        ierr = PetscFree(mats);CHKERRQ(ierr);
      } else {
        /* ierr = MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT);CHKERRQ(ierr); */
        ierr = MatCompositeMerge(def->W);CHKERRQ(ierr);
      }
    } else {
      if (def->nestedlvl < def->maxnestedlvl) {
        ierr = PetscMalloc1(size,&mats);CHKERRQ(ierr);
        for (i=0; i<size; i++) {
          ierr = MatCompositeGetMat(def->Wt,i,&mats[i]);CHKERRQ(ierr);
        }
        size -= 1;
        ierr = MatDestroy(&def->Wt);CHKERRQ(ierr);
        def->Wt = mats[0];
        ierr = PetscObjectReference((PetscObject)mats[0]);CHKERRQ(ierr);
        if (size > 1) {
          ierr = MatCreateComposite(comm,size,&mats[1],&nextDef);CHKERRQ(ierr);
          ierr = MatCompositeSetType(nextDef,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
        } else {
          nextDef = mats[1];
          ierr = PetscObjectReference((PetscObject)mats[1]);CHKERRQ(ierr);
        }
        ierr = PetscFree(mats);CHKERRQ(ierr);
      } else {
        /* ierr = MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT);CHKERRQ(ierr); */
        ierr = MatCompositeMerge(def->Wt);CHKERRQ(ierr);
      }
    }
  }

  if (transp) {
    ierr = MatDestroy(&def->W);CHKERRQ(ierr);
    ierr = MatHermitianTranspose(def->Wt,MAT_INITIAL_MATRIX,&def->W);CHKERRQ(ierr);
  }

  /* assemble WtA */
  if (!def->WtA) {
    if (def->Wt) {
      ierr = MatMatMult(def->Wt,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA);CHKERRQ(ierr);
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = MatHermitianTranspose(def->W,MAT_INITIAL_MATRIX,&def->Wt);CHKERRQ(ierr);
      ierr = MatMatMult(def->W,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA);CHKERRQ(ierr);
#else
      ierr = MatTransposeMatMult(def->W,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA);CHKERRQ(ierr);
#endif
    }
  }
  /* setup coarse problem */
  if (!def->WtAWinv) {
    ierr = MatGetSize(def->W,NULL,&m);CHKERRQ(ierr);
    if (!def->WtAW) {
      ierr = MatMatMult(def->WtA,def->W,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtAW);CHKERRQ(ierr);
      /* TODO create MatInheritOption(Mat,MatOption) */
      ierr = MatGetOption(Amat,MAT_SPD,&flgspd);CHKERRQ(ierr);
      ierr = MatSetOption(def->WtAW,MAT_SPD,flgspd);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      /* Check columns of W are not in kernel of A */
      PetscReal *norms;
      ierr = PetscMalloc1(m,&norms);CHKERRQ(ierr);
      ierr = MatGetColumnNorms(def->WtAW,NORM_INFINITY,norms);CHKERRQ(ierr);
      for (i=0; i<m; i++) {
        if (norms[i] < 100*PETSC_MACHINE_EPSILON) {
          SETERRQ1(comm,PETSC_ERR_SUP,"Column %D of W is in kernel of A.",i);
        }
      }
      ierr = PetscFree(norms);CHKERRQ(ierr);
#endif
    } else {
      ierr = MatGetOption(def->WtAW,MAT_SPD,&flgspd);CHKERRQ(ierr);
    }
    /* TODO use MATINV */
    ierr = KSPCreate(comm,&def->WtAWinv);CHKERRQ(ierr);
    ierr = KSPSetOperators(def->WtAWinv,def->WtAW,def->WtAW);CHKERRQ(ierr);
    ierr = KSPGetPC(def->WtAWinv,&pcinner);CHKERRQ(ierr);
    /* Setup KSP and PC */
    if (nextDef) { /* next level for multilevel deflation */
      innerksp = def->WtAWinv;
      /* set default KSPtype */
      if (!def->ksptype) {
        def->ksptype = KSPFGMRES;
        if (flgspd) { /* SPD system */
          def->ksptype = KSPFCG;
        }
      }
      ierr = KSPSetType(innerksp,def->ksptype);CHKERRQ(ierr); /* TODO iherit from KSP + tolerances */
      ierr = PCSetType(pcinner,PCDEFLATION);CHKERRQ(ierr); /* TODO create coarse preconditinoner M_c = WtMW ? */
      ierr = PCDeflationSetSpace(pcinner,nextDef,transp);CHKERRQ(ierr);
      ierr = PCDeflationSetLvl_Deflation(pcinner,def->nestedlvl+1,def->maxnestedlvl);CHKERRQ(ierr);
      /* inherit options */
      ((PC_Deflation*)(pcinner->data))->ksptype = def->ksptype;
      ((PC_Deflation*)(pcinner->data))->correct = def->correct;
      ierr = MatDestroy(&nextDef);CHKERRQ(ierr);
    } else { /* the last level */
      ierr = KSPSetType(def->WtAWinv,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pcinner,PCTELESCOPE);CHKERRQ(ierr);
      /* ugly hack to not have overwritten PCTELESCOPE */
      if (prefix) {
        ierr = KSPSetOptionsPrefix(def->WtAWinv,prefix);CHKERRQ(ierr);
      }
      ierr = KSPAppendOptionsPrefix(def->WtAWinv,"tel_");CHKERRQ(ierr);
      ierr = PCSetFromOptions(pcinner);CHKERRQ(ierr);
      /* Reduction factor choice */
      red = def->reductionfact;
      if (red < 0) {
        ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
        red  = ceil((float)commsize/ceil((float)m/commsize));
        ierr = PetscObjectTypeCompareAny((PetscObject)(def->WtAW),&match,MATSEQDENSE,MATMPIDENSE,MATDENSE,"");CHKERRQ(ierr);
        if (match) red = commsize;
        ierr = PetscInfo1(pc,"Auto choosing reduction factor %D\n",red);CHKERRQ(ierr); /* TODO add level? */
      }
      ierr = PCTelescopeSetReductionFactor(pcinner,red);CHKERRQ(ierr);
      ierr = PCSetUp(pcinner);CHKERRQ(ierr);
      ierr = PCTelescopeGetKSP(pcinner,&innerksp);CHKERRQ(ierr);
      if (innerksp) {
        ierr = KSPGetPC(innerksp,&pcinner);CHKERRQ(ierr);
        /* TODO Cholesky if flgspd? */
        ierr = PCSetType(pcinner,PCLU);CHKERRQ(ierr);
        //TODO remove explicit matSolverPackage
        if (commsize == red) {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU);CHKERRQ(ierr);
        } else {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
        }
      }
    }

    if (innerksp) {
      /* TODO use def_[lvl]_ if lvl > 0? */
      if (prefix) {
        ierr = KSPSetOptionsPrefix(innerksp,prefix);CHKERRQ(ierr);
      }
      ierr = KSPAppendOptionsPrefix(innerksp,"def_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(innerksp);CHKERRQ(ierr);
      ierr = KSPSetUp(innerksp);CHKERRQ(ierr);
    }
  }
  ierr = KSPSetFromOptions(def->WtAWinv);CHKERRQ(ierr);
  ierr = KSPSetUp(def->WtAWinv);CHKERRQ(ierr);

  if (!def->pc) {
    ierr = PCCreate(comm,&def->pc);CHKERRQ(ierr);
    ierr = PCSetOperators(def->pc,Amat,Amat);CHKERRQ(ierr);
    ierr = PCSetType(def->pc,PCNONE);CHKERRQ(ierr);
    if (prefix) {
      ierr = PCSetOptionsPrefix(def->pc,prefix);CHKERRQ(ierr);
    }
    ierr = PCAppendOptionsPrefix(def->pc,"def_pc_");CHKERRQ(ierr);
    ierr = PCSetFromOptions(def->pc);CHKERRQ(ierr);
    ierr = PCSetUp(def->pc);CHKERRQ(ierr);
  }

  /* create work vecs */
  ierr = MatCreateVecs(Amat,NULL,&def->work);CHKERRQ(ierr);
  ierr = KSPCreateVecs(def->WtAWinv,2,&def->workcoarse,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Deflation(PC pc)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&def->work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2,&def->workcoarse);CHKERRQ(ierr);
  ierr = MatDestroy(&def->W);CHKERRQ(ierr);
  ierr = MatDestroy(&def->Wt);CHKERRQ(ierr);
  ierr = MatDestroy(&def->WtA);CHKERRQ(ierr);
  ierr = MatDestroy(&def->WtAW);CHKERRQ(ierr);
  ierr = KSPDestroy(&def->WtAWinv);CHKERRQ(ierr);
  ierr = PCDestroy(&def->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PCDestroy_Deflation - Destroys the private context for the Deflation preconditioner
   that was created with PCCreate_Deflation().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_Deflation(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Deflation(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Deflation(PC pc,PetscViewer viewer)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;
  PetscInt          its;
  PetscBool         iascii;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (def->correct) {
      ierr = PetscViewerASCIIPrintf(viewer,"using CP correction, factor = %g+%gi\n",
                                    (double)PetscRealPart(def->correctfact),
                                    (double)PetscImaginaryPart(def->correctfact));CHKERRQ(ierr);
    }
    if (!def->nestedlvl) {
      ierr = PetscViewerASCIIPrintf(viewer,"deflation space type: %s\n",PCDeflationSpaceTypes[def->spacetype]);CHKERRQ(ierr);
    }

    ierr = PetscViewerASCIIPrintf(viewer,"--- Additional PC:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PCView(def->pc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"--- Coarse problem solver:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPGetTotalIterations(def->WtAWinv,&its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"total number of iterations: %D\n",its);CHKERRQ(ierr);
    ierr = KSPView(def->WtAWinv,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Deflation(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Deflation options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_init_only","Use only initialization step - Initdef","PCDeflationSetInitOnly",def->init,&def->init,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_max_lvl","Maximum of deflation levels","PCDeflationSetMaxLvl",def->maxnestedlvl,&def->maxnestedlvl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_reduction_factor","Reduction factor for coarse problem solution using PCTELESCOPE","PCDeflationSetReductionFactor",def->reductionfact,&def->reductionfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_correction","Add coarse problem correction Q to P","PCDeflationSetCorrectionFactor",def->correct,&def->correct,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pc_deflation_correction_factor","Set multiple of Q to use as coarse problem correction","PCDeflationSetCorrectionFactor",def->correctfact,&def->correctfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_deflation_compute_space","Compute deflation space","PCDeflationSetSpace",PCDeflationSpaceTypes,(PetscEnum)def->spacetype,(PetscEnum*)&def->spacetype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_compute_space_size","Set size of the deflation space to compute","PCDeflationSetSpace",def->spacesize,&def->spacesize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_space_extend","Extend deflation space instead of truncating (wavelets)","PCDeflation",def->extendsp,&def->extendsp,NULL);CHKERRQ(ierr);
//TODO add set function and fix manpages
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCDEFLATION - Deflation preconditioner shifts part of the spectrum to zero (deflates)
     or to a predefined value

   Options Database Key:
+    -pc_deflation_type <init,pre,post> - selects approach to deflation (default: pre)
-    -pc_jacobi_abs - use the absolute value of the diagonal entry

   Level: beginner

  Notes:
    todo

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCDeflationSetType(), PCDeflationSetSpace()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Deflation(PC pc)
{
  PC_Deflation   *def;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&def);CHKERRQ(ierr);
  pc->data = (void*)def;

  def->init          = PETSC_FALSE;
  def->correct       = PETSC_FALSE;
  def->correctfact   = 1.0;
  def->reductionfact = -1;
  def->spacetype     = PC_DEFLATION_SPACE_HAAR;
  def->spacesize     = 1;
  def->extendsp      = PETSC_FALSE;
  def->nestedlvl     = 0;
  def->maxnestedlvl  = 0;

  pc->ops->apply          = PCApply_Deflation;
  pc->ops->presolve       = PCPreSolve_Deflation;
  pc->ops->setup          = PCSetUp_Deflation;
  pc->ops->reset          = PCReset_Deflation;
  pc->ops->destroy        = PCDestroy_Deflation;
  pc->ops->setfromoptions = PCSetFromOptions_Deflation;
  pc->ops->view           = PCView_Deflation;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetInitOnly_C",PCDeflationSetInitOnly_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetLvl_C",PCDeflationSetLvl_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetReductionFactor_C",PCDeflationSetReductionFactor_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCorrectionFactor_C",PCDeflationSetCorrectionFactor_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpaceToCompute_C",PCDeflationSetSpaceToCompute_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetProjectionNullSpaceMat_C",PCDeflationSetProjectionNullSpaceMat_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCoarseMat_C",PCDeflationSetCoarseMat_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetCoarseKSP_C",PCDeflationGetCoarseKSP_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCoarseKSP_C",PCDeflationSetCoarseKSP_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetPC_C",PCDeflationGetPC_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetPC_C",PCDeflationSetPC_Deflation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

