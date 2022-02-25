#include <../src/ksp/pc/impls/deflation/deflation.h> /*I "petscksp.h" I*/  /* includes for fortran wrappers */

const char *const PCDeflationSpaceTypes[] = {
  "haar",
  "db2",
  "db4",
  "db8",
  "db16",
  "biorth22",
  "meyer",
  "aggregation",
  "user",
  "PCDeflationSpaceType",
  "PC_DEFLATION_SPACE_",
  NULL
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
    Sets initial guess to the solution on the deflation space but does not apply
    the deflation preconditioner. The additional preconditioner is still applied.

   Logically Collective

   Input Parameters:
+  pc  - the preconditioner context
-  flg - default PETSC_FALSE

   Options Database Keys:
.    -pc_deflation_init_only <false> - if true computes only the special guess

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

static PetscErrorCode PCDeflationSetLevels_Deflation(PC pc,PetscInt current,PetscInt max)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  if (current) def->lvl = current;
  def->maxlvl = max;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetLevels - Set the maximum level of deflation nesting.

   Logically Collective

   Input Parameters:
+  pc  - the preconditioner context
-  max - maximum deflation level

   Options Database Keys:
.    -pc_deflation_max_lvl <0> - maximum number of levels for multilevel deflation

   Level: intermediate

.seealso: PCDeflationSetSpaceToCompute(), PCDeflationSetSpace(), PCDEFLATION
@*/
PetscErrorCode PCDeflationSetLevels(PC pc,PetscInt max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,max,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetLevels_C",(PC,PetscInt,PetscInt),(pc,0,max));CHKERRQ(ierr);
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
   PCDeflationSetReductionFactor - Set reduction factor for the bottom PCTELESCOPE coarse problem solver.

   Logically Collective

   Input Parameters:
+  pc  - the preconditioner context
-  red - reduction factor (or PETSC_DETERMINE)

   Options Database Keys:
.    -pc_deflation_reduction_factor <\-1> - reduction factor on bottom level coarse problem for PCTELESCOPE

   Notes:
     Default is computed based on the size of the coarse problem.

   Level: intermediate

.seealso: PCTELESCOPE, PCDEFLATION
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
  if (def->correct == 0.0) {
    def->correct = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetCorrectionFactor - Set coarse problem correction factor.
    The Preconditioner becomes P*M^{-1} + fact*Q.

   Logically Collective

   Input Parameters:
+  pc   - the preconditioner context
-  fact - correction factor

   Options Database Keys:
+    -pc_deflation_correction        <false> - if true apply coarse problem correction
-    -pc_deflation_correction_factor <1.0>   - sets coarse problem correction factor

   Notes:
    Any non-zero fact enables the coarse problem correction.

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

   Logically Collective

   Input Parameters:
+  pc   - the preconditioner context
.  type - deflation space type to compute (or PETSC_IGNORE)
-  size - size of the space to compute (or PETSC_DEFAULT)

   Options Database Keys:
+    -pc_deflation_compute_space      <haar> - compute PCDeflationSpaceType deflation space
-    -pc_deflation_compute_space_size <1>    - size of the deflation space

   Notes:
    For wavelet-based deflation, size represents number of levels.

    The deflation space is computed in PCSetUp().

   Level: intermediate

.seealso: PCDeflationSetLevels(), PCDEFLATION
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
  /* possibly allows W' = Wt (which is valid but not tested) */
  ierr = PetscObjectReference((PetscObject)W);CHKERRQ(ierr);
  if (transpose) {
    ierr = MatDestroy(&def->Wt);CHKERRQ(ierr);
    def->Wt = W;
  } else {
    ierr = MatDestroy(&def->W);CHKERRQ(ierr);
    def->W = W;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetSpace - Set the deflation space matrix (or its (Hermitian) transpose).

   Logically Collective

   Input Parameters:
+  pc        - the preconditioner context
.  W         - deflation matrix
-  transpose - indicates that W is an explicit transpose of the deflation matrix

   Notes:
    Setting W as a multipliplicative MATCOMPOSITE enables use of the multilevel
    deflation. If W = W0*W1*W2*...*Wn, W0 is taken as the first deflation space and
    the coarse problem (W0'*A*W0)^{-1} is again preconditioned by deflation with
    W1 as the deflation matrix. This repeats until the maximum level set by
    PCDeflationSetLevels() is reached or there are no more matrices available.
    If there are matrices left after reaching the maximum level,
    they are merged into a deflation matrix ...*W{n-1}*Wn.

   Level: intermediate

.seealso: PCDeflationSetLevels(), PCDEFLATION
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
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&def->WtA);CHKERRQ(ierr);
  def->WtA = mat;
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetProjectionNullSpaceMat - Set the projection null space matrix (W'*A).

   Collective

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
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&def->WtAW);CHKERRQ(ierr);
  def->WtAW = mat;
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetCoarseMat - Set the coarse problem Mat.

   Collective

   Input Parameters:
+  pc  - preconditioner context
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
   PCDeflationGetCoarseKSP - Returns the coarse problem KSP.

   Not Collective

   Input Parameters:
.  pc - preconditioner context

   Output Parameters:
.  ksp - coarse problem KSP context

   Level: advanced

.seealso: PCDEFLATION
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

static PetscErrorCode PCDeflationGetPC_Deflation(PC pc,PC *apc)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  *apc = def->pc;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationGetPC - Returns the additional preconditioner M^{-1}.

   Not Collective

   Input Parameters:
.  pc  - the preconditioner context

   Output Parameters:
.  apc - additional preconditioner

   Level: advanced

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationGetPC(PC pc,PC *apc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(pc,1);
  ierr = PetscTryMethod(pc,"PCDeflationGetPC_C",(PC,PC*),(pc,apc));CHKERRQ(ierr);
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
  PetscInt         i,m,red,size;
  PetscMPIInt      commsize;
  PetscBool        match,flgspd,transp=PETSC_FALSE;
  MatCompositeType ctype;
  MPI_Comm         comm;
  char             prefix[128]="";
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (pc->setupcalled) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&Amat);CHKERRQ(ierr);
  if (!def->lvl && !def->prefix) {
    ierr = PCGetOptionsPrefix(pc,&def->prefix);CHKERRQ(ierr);
  }
  if (def->lvl) {
    ierr = PetscSNPrintf(prefix,sizeof(prefix),"%d_",(int)def->lvl);CHKERRQ(ierr);
  }

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
      if (def->lvl < def->maxlvl) {
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
        /* TODO test merge side performance */
        /* ierr = MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT);CHKERRQ(ierr); */
        ierr = MatCompositeMerge(def->W);CHKERRQ(ierr);
      }
    } else {
      if (def->lvl < def->maxlvl) {
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
      ierr = MatMatMult(def->Wt,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA);CHKERRQ(ierr);
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
      if (PetscDefined(USE_DEBUG)) {
        /* Check columns of W are not in kernel of A */
        PetscReal *norms;
        ierr = PetscMalloc1(m,&norms);CHKERRQ(ierr);
        ierr = MatGetColumnNorms(def->WtAW,NORM_INFINITY,norms);CHKERRQ(ierr);
        for (i=0; i<m; i++) {
          if (norms[i] < 100*PETSC_MACHINE_EPSILON) {
            SETERRQ(comm,PETSC_ERR_SUP,"Column %D of W is in kernel of A.",i);
          }
        }
        ierr = PetscFree(norms);CHKERRQ(ierr);
      }
    } else {
      ierr = MatGetOption(def->WtAW,MAT_SPD,&flgspd);CHKERRQ(ierr);
    }
    /* TODO use MATINV ? */
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
      ierr = PCDeflationSetLevels_Deflation(pcinner,def->lvl+1,def->maxlvl);CHKERRQ(ierr);
      /* inherit options */
      if (def->prefix) ((PC_Deflation*)(pcinner->data))->prefix = def->prefix;
      ((PC_Deflation*)(pcinner->data))->init          = def->init;
      ((PC_Deflation*)(pcinner->data))->ksptype       = def->ksptype;
      ((PC_Deflation*)(pcinner->data))->correct       = def->correct;
      ((PC_Deflation*)(pcinner->data))->correctfact   = def->correctfact;
      ((PC_Deflation*)(pcinner->data))->reductionfact = def->reductionfact;
      ierr = MatDestroy(&nextDef);CHKERRQ(ierr);
    } else { /* the last level */
      ierr = KSPSetType(def->WtAWinv,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pcinner,PCTELESCOPE);CHKERRQ(ierr);
      /* do not overwrite PCTELESCOPE */
      if (def->prefix) {
        ierr = KSPSetOptionsPrefix(def->WtAWinv,def->prefix);CHKERRQ(ierr);
      }
      ierr = KSPAppendOptionsPrefix(def->WtAWinv,"deflation_tel_");CHKERRQ(ierr);
      ierr = PCSetFromOptions(pcinner);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pcinner,PCTELESCOPE,&match);CHKERRQ(ierr);
      PetscCheckFalse(!match,comm,PETSC_ERR_SUP,"User can not owerwrite PCTELESCOPE on bottom level, use reduction factor = 1 instead.");
      /* Reduction factor choice */
      red = def->reductionfact;
      if (red < 0) {
        ierr = MPI_Comm_size(comm,&commsize);CHKERRMPI(ierr);
        red  = PetscCeilInt(commsize,PetscCeilInt(m,commsize));
        ierr = PetscObjectTypeCompareAny((PetscObject)(def->WtAW),&match,MATSEQDENSE,MATMPIDENSE,MATDENSE,"");CHKERRQ(ierr);
        if (match) red = commsize;
        ierr = PetscInfo(pc,"Auto choosing reduction factor %D\n",red);CHKERRQ(ierr);
      }
      ierr = PCTelescopeSetReductionFactor(pcinner,red);CHKERRQ(ierr);
      ierr = PCSetUp(pcinner);CHKERRQ(ierr);
      ierr = PCTelescopeGetKSP(pcinner,&innerksp);CHKERRQ(ierr);
      if (innerksp) {
        ierr = KSPGetPC(innerksp,&pcinner);CHKERRQ(ierr);
        ierr = PCSetType(pcinner,PCLU);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUPERLU)
        ierr = MatGetFactorAvailable(def->WtAW,MATSOLVERSUPERLU,MAT_FACTOR_LU,&match);CHKERRQ(ierr);
        if (match) {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU);CHKERRQ(ierr);
        }
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
        ierr = MatGetFactorAvailable(def->WtAW,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&match);CHKERRQ(ierr);
        if (match) {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
        }
#endif
      }
    }

    if (innerksp) {
      if (def->prefix) {
        ierr = KSPSetOptionsPrefix(innerksp,def->prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(innerksp,"deflation_");CHKERRQ(ierr);
      } else {
        ierr = KSPSetOptionsPrefix(innerksp,"deflation_");CHKERRQ(ierr);
      }
      ierr = KSPAppendOptionsPrefix(innerksp,prefix);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(innerksp);CHKERRQ(ierr);
      ierr = KSPSetUp(innerksp);CHKERRQ(ierr);
    }
  }
  ierr = KSPSetFromOptions(def->WtAWinv);CHKERRQ(ierr);
  ierr = KSPSetUp(def->WtAWinv);CHKERRQ(ierr);

  /* create preconditioner */
  if (!def->pc) {
    ierr = PCCreate(comm,&def->pc);CHKERRQ(ierr);
    ierr = PCSetOperators(def->pc,Amat,Amat);CHKERRQ(ierr);
    ierr = PCSetType(def->pc,PCNONE);CHKERRQ(ierr);
    if (def->prefix) {
      ierr = PCSetOptionsPrefix(def->pc,def->prefix);CHKERRQ(ierr);
    }
    ierr = PCAppendOptionsPrefix(def->pc,"deflation_");CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(def->pc,prefix);CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(def->pc,"pc_");CHKERRQ(ierr);
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
    if (!def->lvl) {
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
  ierr = PetscOptionsInt("-pc_deflation_levels","Maximum of deflation levels","PCDeflationSetLevels",def->maxlvl,&def->maxlvl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_reduction_factor","Reduction factor for coarse problem solution using PCTELESCOPE","PCDeflationSetReductionFactor",def->reductionfact,&def->reductionfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_correction","Add coarse problem correction Q to P","PCDeflationSetCorrectionFactor",def->correct,&def->correct,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pc_deflation_correction_factor","Set multiple of Q to use as coarse problem correction","PCDeflationSetCorrectionFactor",def->correctfact,&def->correctfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_deflation_compute_space","Compute deflation space","PCDeflationSetSpace",PCDeflationSpaceTypes,(PetscEnum)def->spacetype,(PetscEnum*)&def->spacetype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_compute_space_size","Set size of the deflation space to compute","PCDeflationSetSpace",def->spacesize,&def->spacesize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_space_extend","Extend deflation space instead of truncating (wavelets)","PCDeflation",def->extendsp,&def->extendsp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   PCDEFLATION - Deflation preconditioner shifts (deflates) part of the spectrum to zero or to a predefined value.

   Options Database Keys:
+    -pc_deflation_init_only          <false> - if true computes only the special guess
.    -pc_deflation_max_lvl            <0>     - maximum number of levels for multilevel deflation
.    -pc_deflation_reduction_factor <\-1>     - reduction factor on bottom level coarse problem for PCTELESCOPE (default based on the size of the coarse problem)
.    -pc_deflation_correction         <false> - if true apply coarse problem correction
.    -pc_deflation_correction_factor  <1.0>   - sets coarse problem correction factor
.    -pc_deflation_compute_space      <haar>  - compute PCDeflationSpaceType deflation space
-    -pc_deflation_compute_space_size <1>     - size of the deflation space (corresponds to number of levels for wavelet-based deflation)

   Notes:
    Given a (complex - transpose is always Hermitian) full rank deflation matrix W, the deflation (introduced in [1,2])
    preconditioner uses projections Q = W*(W'*A*W)^{-1}*W' and P = I - Q*A, where A is pmat.

    The deflation computes initial guess x0 = x_{-1} - Q*r_{-1}, which is the solution on the deflation space.
    If PCDeflationSetInitOnly() or -pc_deflation_init_only is set to PETSC_TRUE (InitDef scheme), the application of the
    preconditioner consists only of application of the additional preconditioner M^{-1}. Otherwise, the preconditioner
    application consists of P*M^{-1} + factor*Q. The first part of the preconditioner (PM^{-1}) shifts some eigenvalues
    to zero while the addition of the coarse problem correction (factor*Q) makes the preconditioner to shift some
    eigenvalues to the given factor. The InitDef scheme is recommended for deflation using high accuracy estimates
    of eigenvectors of A when it exhibits similar convergence to the full deflation but is cheaper.

    The deflation matrix is by default automatically computed. The type of deflation matrix and its size to compute can
    be controlled by PCDeflationSetSpaceToCompute() or -pc_deflation_compute_space and -pc_deflation_compute_space_size.
    User can set an arbitrary deflation space matrix with PCDeflationSetSpace(). If the deflation matrix
    is a multiplicative MATCOMPOSITE, a multilevel deflation [3] is used. The first matrix in the composite is used as the
    deflation matrix, and the coarse problem (W'*A*W)^{-1} is solved by KSPFCG (if A is MAT_SPD) or KSPFGMRES preconditioned
    by deflation with deflation matrix being the next matrix in the MATCOMPOSITE. This scheme repeats until the maximum
    level is reached or there are no more matrices. If the maximum level is reached, the remaining matrices are merged
    (multiplied) to create the last deflation matrix. The maximum level defaults to 0 and can be set by
    PCDeflationSetLevels() or by -pc_deflation_levels.

    The coarse problem KSP can be controlled from the command line with prefix -deflation_ for the first level and -deflation_[lvl-1]
    from the second level onward. You can also use
    PCDeflationGetCoarseKSP() to control it from code. The bottom level KSP defaults to
    KSPPREONLY with PCLU direct solver (MATSOLVERSUPERLU/MATSOLVERSUPERLU_DIST if available) wrapped into PCTELESCOPE.
    For convenience, the reduction factor can be set by PCDeflationSetReductionFactor()
    or -pc_deflation_recduction_factor. The default is chosen heuristically based on the coarse problem size.

    The additional preconditioner can be controlled from command line with prefix -deflation_[lvl]_pc (same rules used for
    coarse problem KSP apply for [lvl]_ part of prefix), e.g., -deflation_1_pc_pc_type bjacobi. You can also use
    PCDeflationGetPC() to control the additional preconditioner from code. It defaults to PCNONE.

    The coarse problem correction term (factor*Q) can be turned on by -pc_deflation_correction and the factor value can
    be set by pc_deflation_correction_factor or by PCDeflationSetCorrectionFactor(). The coarse problem can
    significantly improve convergence when the deflation coarse problem is not solved with high enough accuracy. We
    recommend setting factor to some eigenvalue, e.g., the largest eigenvalue so that the preconditioner does not create
    an isolated eigenvalue.

    The options are automatically inherited from the previous deflation level.

    The preconditioner supports KSPMonitorDynamicTolerance(). This is useful for the multilevel scheme for which we also
    recommend limiting the number of iterations for the coarse problems.

    See section 3 of [4] for additional references and decription of the algorithm when used for conjugate gradients.
    Section 4 describes some possible choices for the deflation space.

   Developer Notes:
     Contributed by Jakub Kruzik (PERMON), Institute of Geonics of the Czech
     Academy of Sciences and VSB - TU Ostrava.

     Developed from PERMON code used in [4] while on a research stay with
     Prof. Reinhard Nabben at the Institute of Mathematics, TU Berlin.

   References:
+    [1] - A. Nicolaides. "Deflation of conjugate gradients with applications to boundary value problems", SIAM J. Numer. Anal. 24.2, 1987.
.    [2] - Z. Dostal. "Conjugate gradient method with preconditioning by projector", Int J. Comput. Math. 23.3-4, 1988.
.    [3] - Y. A. Erlangga and R. Nabben. "Multilevel Projection-Based Nested Krylov Iteration for Boundary Value Problems", SIAM J. Sci. Comput. 30.3, 2008.
-    [4] - J. Kruzik "Implementation of the Deflated Variants of the Conjugate Gradient Method", Master's thesis, VSB-TUO, 2018 - http://dspace5.vsb.cz/bitstream/handle/10084/130303/KRU0097_USP_N2658_2612T078_2018.pdf

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCDeflationSetInitOnly(), PCDeflationSetLevels(), PCDeflationSetReductionFactor(),
           PCDeflationSetCorrectionFactor(), PCDeflationSetSpaceToCompute(),
           PCDeflationSetSpace(), PCDeflationSpaceType, PCDeflationSetProjectionNullSpaceMat(),
           PCDeflationSetCoarseMat(), PCDeflationGetCoarseKSP(), PCDeflationGetPC()
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
  def->lvl           = 0;
  def->maxlvl        = 0;
  def->W             = NULL;
  def->Wt            = NULL;

  pc->ops->apply          = PCApply_Deflation;
  pc->ops->presolve       = PCPreSolve_Deflation;
  pc->ops->setup          = PCSetUp_Deflation;
  pc->ops->reset          = PCReset_Deflation;
  pc->ops->destroy        = PCDestroy_Deflation;
  pc->ops->setfromoptions = PCSetFromOptions_Deflation;
  pc->ops->view           = PCView_Deflation;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetInitOnly_C",PCDeflationSetInitOnly_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetLevels_C",PCDeflationSetLevels_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetReductionFactor_C",PCDeflationSetReductionFactor_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCorrectionFactor_C",PCDeflationSetCorrectionFactor_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpaceToCompute_C",PCDeflationSetSpaceToCompute_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetProjectionNullSpaceMat_C",PCDeflationSetProjectionNullSpaceMat_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCoarseMat_C",PCDeflationSetCoarseMat_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetCoarseKSP_C",PCDeflationGetCoarseKSP_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetPC_C",PCDeflationGetPC_Deflation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

