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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetInitOnly_C",(PC,PetscBool),(pc,flg)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,max,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetLevels_C",(PC,PetscInt,PetscInt),(pc,0,max)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,red,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetReductionFactor_C",(PC,PetscInt),(pc,red)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,fact,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetCorrectionFactor_C",(PC,PetscScalar),(pc,fact)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (type) PetscValidLogicalCollectiveEnum(pc,type,2);
  if (size > 0) PetscValidLogicalCollectiveInt(pc,size,3);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetSpaceToCompute_C",(PC,PCDeflationSpaceType,PetscInt),(pc,type,size)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetSpace_Deflation(PC pc,Mat W,PetscBool transpose)
{
  PC_Deflation   *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  /* possibly allows W' = Wt (which is valid but not tested) */
  CHKERRQ(PetscObjectReference((PetscObject)W));
  if (transpose) {
    CHKERRQ(MatDestroy(&def->Wt));
    def->Wt = W;
  } else {
    CHKERRQ(MatDestroy(&def->W));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(W,MAT_CLASSID,2);
  PetscValidLogicalCollectiveBool(pc,transpose,3);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetSpace_C",(PC,Mat,PetscBool),(pc,W,transpose)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetProjectionNullSpaceMat_Deflation(PC pc,Mat mat)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)mat));
  CHKERRQ(MatDestroy(&def->WtA));
  def->WtA = mat;
  CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtA));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetProjectionNullSpaceMat_C",(PC,Mat),(pc,mat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDeflationSetCoarseMat_Deflation(PC pc,Mat mat)
{
  PC_Deflation     *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)mat));
  CHKERRQ(MatDestroy(&def->WtAW));
  def->WtAW = mat;
  CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)def->WtAW));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationSetCoarseMat_C",(PC,Mat),(pc,mat)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationGetCoarseKSP_C",(PC,KSP*),(pc,ksp)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(pc,1);
  CHKERRQ(PetscTryMethod(pc,"PCDeflationGetPC_C",(PC,PC*),(pc,apc)));
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

  PetscFunctionBegin;
  w1 = def->workcoarse[0];
  w2 = def->workcoarse[1];
  r  = def->work;
  CHKERRQ(PCGetOperators(pc,NULL,&A));

  CHKERRQ(KSPGetInitialGuessNonzero(ksp,&nonzero));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  if (nonzero) {
    CHKERRQ(MatMult(A,x,r));                          /*    r  <- b - Ax              */
    CHKERRQ(VecAYPX(r,-1.0,b));
  } else {
    CHKERRQ(VecCopy(b,r));                            /*    r  <- b (x is 0)          */
  }

  if (def->Wt) {
    CHKERRQ(MatMult(def->Wt,r,w1));                   /*    w1 <- W'*r                */
  } else {
    CHKERRQ(MatMultHermitianTranspose(def->W,r,w1));  /*    w1 <- W'*r                */
  }
  CHKERRQ(KSPSolve(def->WtAWinv,w1,w2));              /*    w2 <- (W'*A*W)^{-1}*w1    */
  CHKERRQ(MatMult(def->W,w2,r));                      /*    r  <- W*w2                */
  CHKERRQ(VecAYPX(x,1.0,r));
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

  PetscFunctionBegin;
  w1 = def->workcoarse[0];
  w2 = def->workcoarse[1];
  u  = def->work;
  CHKERRQ(PCGetOperators(pc,NULL,&A));

  CHKERRQ(PCApply(def->pc,r,z));                          /*    z <- M^{-1}*r             */
  if (!def->init) {
    CHKERRQ(MatMult(def->WtA,z,w1));                      /*    w1 <- W'*A*z              */
    if (def->correct) {
      if (def->Wt) {
        CHKERRQ(MatMult(def->Wt,r,w2));                   /*    w2 <- W'*r                */
      } else {
        CHKERRQ(MatMultHermitianTranspose(def->W,r,w2));  /*    w2 <- W'*r                */
      }
      CHKERRQ(VecAXPY(w1,-1.0*def->correctfact,w2));      /*    w1 <- w1 - l*w2           */
    }
    CHKERRQ(KSPSolve(def->WtAWinv,w1,w2));                /*    w2 <- (W'*A*W)^{-1}*w1    */
    CHKERRQ(MatMult(def->W,w2,u));                        /*    u  <- W*w2                */
    CHKERRQ(VecAXPY(z,-1.0,u));                           /*    z  <- z - u               */
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

  PetscFunctionBegin;
  if (pc->setupcalled) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetOperators(pc,NULL,&Amat));
  if (!def->lvl && !def->prefix) {
    CHKERRQ(PCGetOptionsPrefix(pc,&def->prefix));
  }
  if (def->lvl) {
    CHKERRQ(PetscSNPrintf(prefix,sizeof(prefix),"%d_",(int)def->lvl));
  }

  /* compute a deflation space */
  if (def->W || def->Wt) {
    def->spacetype = PC_DEFLATION_SPACE_USER;
  } else {
    CHKERRQ(PCDeflationComputeSpace(pc));
  }

  /* nested deflation */
  if (def->W) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)def->W,MATCOMPOSITE,&match));
    if (match) {
      CHKERRQ(MatCompositeGetType(def->W,&ctype));
      CHKERRQ(MatCompositeGetNumberMat(def->W,&size));
    }
  } else {
    CHKERRQ(MatCreateHermitianTranspose(def->Wt,&def->W));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)def->Wt,MATCOMPOSITE,&match));
    if (match) {
      CHKERRQ(MatCompositeGetType(def->Wt,&ctype));
      CHKERRQ(MatCompositeGetNumberMat(def->Wt,&size));
    }
    transp = PETSC_TRUE;
  }
  if (match && ctype == MAT_COMPOSITE_MULTIPLICATIVE) {
    if (!transp) {
      if (def->lvl < def->maxlvl) {
        CHKERRQ(PetscMalloc1(size,&mats));
        for (i=0; i<size; i++) {
          CHKERRQ(MatCompositeGetMat(def->W,i,&mats[i]));
        }
        size -= 1;
        CHKERRQ(MatDestroy(&def->W));
        def->W = mats[size];
        CHKERRQ(PetscObjectReference((PetscObject)mats[size]));
        if (size > 1) {
          CHKERRQ(MatCreateComposite(comm,size,mats,&nextDef));
          CHKERRQ(MatCompositeSetType(nextDef,MAT_COMPOSITE_MULTIPLICATIVE));
        } else {
          nextDef = mats[0];
          CHKERRQ(PetscObjectReference((PetscObject)mats[0]));
        }
        CHKERRQ(PetscFree(mats));
      } else {
        /* TODO test merge side performance */
        /* CHKERRQ(MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT)); */
        CHKERRQ(MatCompositeMerge(def->W));
      }
    } else {
      if (def->lvl < def->maxlvl) {
        CHKERRQ(PetscMalloc1(size,&mats));
        for (i=0; i<size; i++) {
          CHKERRQ(MatCompositeGetMat(def->Wt,i,&mats[i]));
        }
        size -= 1;
        CHKERRQ(MatDestroy(&def->Wt));
        def->Wt = mats[0];
        CHKERRQ(PetscObjectReference((PetscObject)mats[0]));
        if (size > 1) {
          CHKERRQ(MatCreateComposite(comm,size,&mats[1],&nextDef));
          CHKERRQ(MatCompositeSetType(nextDef,MAT_COMPOSITE_MULTIPLICATIVE));
        } else {
          nextDef = mats[1];
          CHKERRQ(PetscObjectReference((PetscObject)mats[1]));
        }
        CHKERRQ(PetscFree(mats));
      } else {
        /* CHKERRQ(MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT)); */
        CHKERRQ(MatCompositeMerge(def->Wt));
      }
    }
  }

  if (transp) {
    CHKERRQ(MatDestroy(&def->W));
    CHKERRQ(MatHermitianTranspose(def->Wt,MAT_INITIAL_MATRIX,&def->W));
  }

  /* assemble WtA */
  if (!def->WtA) {
    if (def->Wt) {
      CHKERRQ(MatMatMult(def->Wt,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA));
    } else {
#if defined(PETSC_USE_COMPLEX)
      CHKERRQ(MatHermitianTranspose(def->W,MAT_INITIAL_MATRIX,&def->Wt));
      CHKERRQ(MatMatMult(def->Wt,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA));
#else
      CHKERRQ(MatTransposeMatMult(def->W,Amat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtA));
#endif
    }
  }
  /* setup coarse problem */
  if (!def->WtAWinv) {
    CHKERRQ(MatGetSize(def->W,NULL,&m));
    if (!def->WtAW) {
      CHKERRQ(MatMatMult(def->WtA,def->W,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtAW));
      /* TODO create MatInheritOption(Mat,MatOption) */
      CHKERRQ(MatGetOption(Amat,MAT_SPD,&flgspd));
      CHKERRQ(MatSetOption(def->WtAW,MAT_SPD,flgspd));
      if (PetscDefined(USE_DEBUG)) {
        /* Check columns of W are not in kernel of A */
        PetscReal *norms;
        CHKERRQ(PetscMalloc1(m,&norms));
        CHKERRQ(MatGetColumnNorms(def->WtAW,NORM_INFINITY,norms));
        for (i=0; i<m; i++) {
          if (norms[i] < 100*PETSC_MACHINE_EPSILON) {
            SETERRQ(comm,PETSC_ERR_SUP,"Column %D of W is in kernel of A.",i);
          }
        }
        CHKERRQ(PetscFree(norms));
      }
    } else {
      CHKERRQ(MatGetOption(def->WtAW,MAT_SPD,&flgspd));
    }
    /* TODO use MATINV ? */
    CHKERRQ(KSPCreate(comm,&def->WtAWinv));
    CHKERRQ(KSPSetOperators(def->WtAWinv,def->WtAW,def->WtAW));
    CHKERRQ(KSPGetPC(def->WtAWinv,&pcinner));
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
      CHKERRQ(KSPSetType(innerksp,def->ksptype)); /* TODO iherit from KSP + tolerances */
      CHKERRQ(PCSetType(pcinner,PCDEFLATION)); /* TODO create coarse preconditinoner M_c = WtMW ? */
      CHKERRQ(PCDeflationSetSpace(pcinner,nextDef,transp));
      CHKERRQ(PCDeflationSetLevels_Deflation(pcinner,def->lvl+1,def->maxlvl));
      /* inherit options */
      if (def->prefix) ((PC_Deflation*)(pcinner->data))->prefix = def->prefix;
      ((PC_Deflation*)(pcinner->data))->init          = def->init;
      ((PC_Deflation*)(pcinner->data))->ksptype       = def->ksptype;
      ((PC_Deflation*)(pcinner->data))->correct       = def->correct;
      ((PC_Deflation*)(pcinner->data))->correctfact   = def->correctfact;
      ((PC_Deflation*)(pcinner->data))->reductionfact = def->reductionfact;
      CHKERRQ(MatDestroy(&nextDef));
    } else { /* the last level */
      CHKERRQ(KSPSetType(def->WtAWinv,KSPPREONLY));
      CHKERRQ(PCSetType(pcinner,PCTELESCOPE));
      /* do not overwrite PCTELESCOPE */
      if (def->prefix) {
        CHKERRQ(KSPSetOptionsPrefix(def->WtAWinv,def->prefix));
      }
      CHKERRQ(KSPAppendOptionsPrefix(def->WtAWinv,"deflation_tel_"));
      CHKERRQ(PCSetFromOptions(pcinner));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pcinner,PCTELESCOPE,&match));
      PetscCheckFalse(!match,comm,PETSC_ERR_SUP,"User can not owerwrite PCTELESCOPE on bottom level, use reduction factor = 1 instead.");
      /* Reduction factor choice */
      red = def->reductionfact;
      if (red < 0) {
        CHKERRMPI(MPI_Comm_size(comm,&commsize));
        red  = PetscCeilInt(commsize,PetscCeilInt(m,commsize));
        CHKERRQ(PetscObjectTypeCompareAny((PetscObject)(def->WtAW),&match,MATSEQDENSE,MATMPIDENSE,MATDENSE,""));
        if (match) red = commsize;
        CHKERRQ(PetscInfo(pc,"Auto choosing reduction factor %D\n",red));
      }
      CHKERRQ(PCTelescopeSetReductionFactor(pcinner,red));
      CHKERRQ(PCSetUp(pcinner));
      CHKERRQ(PCTelescopeGetKSP(pcinner,&innerksp));
      if (innerksp) {
        CHKERRQ(KSPGetPC(innerksp,&pcinner));
        CHKERRQ(PCSetType(pcinner,PCLU));
#if defined(PETSC_HAVE_SUPERLU)
        CHKERRQ(MatGetFactorAvailable(def->WtAW,MATSOLVERSUPERLU,MAT_FACTOR_LU,&match));
        if (match) {
          CHKERRQ(PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU));
        }
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
        CHKERRQ(MatGetFactorAvailable(def->WtAW,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&match));
        if (match) {
          CHKERRQ(PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU_DIST));
        }
#endif
      }
    }

    if (innerksp) {
      if (def->prefix) {
        CHKERRQ(KSPSetOptionsPrefix(innerksp,def->prefix));
        CHKERRQ(KSPAppendOptionsPrefix(innerksp,"deflation_"));
      } else {
        CHKERRQ(KSPSetOptionsPrefix(innerksp,"deflation_"));
      }
      CHKERRQ(KSPAppendOptionsPrefix(innerksp,prefix));
      CHKERRQ(KSPSetFromOptions(innerksp));
      CHKERRQ(KSPSetUp(innerksp));
    }
  }
  CHKERRQ(KSPSetFromOptions(def->WtAWinv));
  CHKERRQ(KSPSetUp(def->WtAWinv));

  /* create preconditioner */
  if (!def->pc) {
    CHKERRQ(PCCreate(comm,&def->pc));
    CHKERRQ(PCSetOperators(def->pc,Amat,Amat));
    CHKERRQ(PCSetType(def->pc,PCNONE));
    if (def->prefix) {
      CHKERRQ(PCSetOptionsPrefix(def->pc,def->prefix));
    }
    CHKERRQ(PCAppendOptionsPrefix(def->pc,"deflation_"));
    CHKERRQ(PCAppendOptionsPrefix(def->pc,prefix));
    CHKERRQ(PCAppendOptionsPrefix(def->pc,"pc_"));
    CHKERRQ(PCSetFromOptions(def->pc));
    CHKERRQ(PCSetUp(def->pc));
  }

  /* create work vecs */
  CHKERRQ(MatCreateVecs(Amat,NULL,&def->work));
  CHKERRQ(KSPCreateVecs(def->WtAWinv,2,&def->workcoarse,0,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Deflation(PC pc)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&def->work));
  CHKERRQ(VecDestroyVecs(2,&def->workcoarse));
  CHKERRQ(MatDestroy(&def->W));
  CHKERRQ(MatDestroy(&def->Wt));
  CHKERRQ(MatDestroy(&def->WtA));
  CHKERRQ(MatDestroy(&def->WtAW));
  CHKERRQ(KSPDestroy(&def->WtAWinv));
  CHKERRQ(PCDestroy(&def->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Deflation(PC pc)
{
  PetscFunctionBegin;
  CHKERRQ(PCReset_Deflation(pc));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Deflation(PC pc,PetscViewer viewer)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;
  PetscInt          its;
  PetscBool         iascii;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (def->correct) {
      ierr = PetscViewerASCIIPrintf(viewer,"using CP correction, factor = %g+%gi\n",
                                    (double)PetscRealPart(def->correctfact),
                                    (double)PetscImaginaryPart(def->correctfact));CHKERRQ(ierr);
    }
    if (!def->lvl) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"deflation space type: %s\n",PCDeflationSpaceTypes[def->spacetype]));
    }

    CHKERRQ(PetscViewerASCIIPrintf(viewer,"--- Additional PC:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PCView(def->pc,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));

    CHKERRQ(PetscViewerASCIIPrintf(viewer,"--- Coarse problem solver:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(KSPGetTotalIterations(def->WtAWinv,&its));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"total number of iterations: %D\n",its));
    CHKERRQ(KSPView(def->WtAWinv,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Deflation(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Deflation      *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Deflation options"));
  CHKERRQ(PetscOptionsBool("-pc_deflation_init_only","Use only initialization step - Initdef","PCDeflationSetInitOnly",def->init,&def->init,NULL));
  CHKERRQ(PetscOptionsInt("-pc_deflation_levels","Maximum of deflation levels","PCDeflationSetLevels",def->maxlvl,&def->maxlvl,NULL));
  CHKERRQ(PetscOptionsInt("-pc_deflation_reduction_factor","Reduction factor for coarse problem solution using PCTELESCOPE","PCDeflationSetReductionFactor",def->reductionfact,&def->reductionfact,NULL));
  CHKERRQ(PetscOptionsBool("-pc_deflation_correction","Add coarse problem correction Q to P","PCDeflationSetCorrectionFactor",def->correct,&def->correct,NULL));
  CHKERRQ(PetscOptionsScalar("-pc_deflation_correction_factor","Set multiple of Q to use as coarse problem correction","PCDeflationSetCorrectionFactor",def->correctfact,&def->correctfact,NULL));
  CHKERRQ(PetscOptionsEnum("-pc_deflation_compute_space","Compute deflation space","PCDeflationSetSpace",PCDeflationSpaceTypes,(PetscEnum)def->spacetype,(PetscEnum*)&def->spacetype,NULL));
  CHKERRQ(PetscOptionsInt("-pc_deflation_compute_space_size","Set size of the deflation space to compute","PCDeflationSetSpace",def->spacesize,&def->spacesize,NULL));
  CHKERRQ(PetscOptionsBool("-pc_deflation_space_extend","Extend deflation space instead of truncating (wavelets)","PCDeflation",def->extendsp,&def->extendsp,NULL));
  CHKERRQ(PetscOptionsTail());
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
+  * - A. Nicolaides. "Deflation of conjugate gradients with applications to boundary value problems", SIAM J. Numer. Anal. 24.2, 1987.
.  * - Z. Dostal. "Conjugate gradient method with preconditioning by projector", Int J. Comput. Math. 23.3-4, 1988.
.  * - Y. A. Erlangga and R. Nabben. "Multilevel Projection-Based Nested Krylov Iteration for Boundary Value Problems", SIAM J. Sci. Comput. 30.3, 2008.
-  * - J. Kruzik "Implementation of the Deflated Variants of the Conjugate Gradient Method", Master's thesis, VSB-TUO, 2018 - http://dspace5.vsb.cz/bitstream/handle/10084/130303/KRU0097_USP_N2658_2612T078_2018.pdf

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

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&def));
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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetInitOnly_C",PCDeflationSetInitOnly_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetLevels_C",PCDeflationSetLevels_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetReductionFactor_C",PCDeflationSetReductionFactor_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCorrectionFactor_C",PCDeflationSetCorrectionFactor_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpaceToCompute_C",PCDeflationSetSpaceToCompute_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetProjectionNullSpaceMat_C",PCDeflationSetProjectionNullSpaceMat_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetCoarseMat_C",PCDeflationSetCoarseMat_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetCoarseKSP_C",PCDeflationGetCoarseKSP_Deflation));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetPC_C",PCDeflationGetPC_Deflation));
  PetscFunctionReturn(0);
}
