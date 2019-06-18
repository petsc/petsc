
/*  --------------------------------------------------------------------

     This file implements a Deflation preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Deflation (e.g., PCCreate_Deflation, PCApply_Deflation).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(),
     so the application code interface remains identical for all
     preconditioners.

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(),
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft()
     and PCApplySymmetricRight().  The Deflation implementation is
     PCApplySymmetricLeftOrRight_Deflation().

    -------------------------------------------------------------------- */

/*
   Include files needed for the Deflation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

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
  "DdefSpaceType",
  "Ddef_SPACE_",
  0
};


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

/* TODO create PCDeflationSetSpaceTranspose? */
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

   Logically Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
-  apc - additional preconditioner

   Level: advanced

.seealso: PCDEFLATION
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
    ierr = MatMult(A,x,r);CHKERRQ(ierr);               /*    r  <- b - Ax              */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);                 /*    r  <- b (x is 0)          */
  }

  ierr = MatMultTranspose(def->W,r,w1);CHKERRQ(ierr);  /*    w1 <- W'*r                */
  ierr = KSPSolve(def->WtAWinv,w1,w2);CHKERRQ(ierr);   /*    w2 <- (W'*A*W)^{-1}*w1    */
  ierr = MatMult(def->W,w2,r);CHKERRQ(ierr);           /*    r  <- W*w2                */
  ierr = VecAYPX(x,1.0,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  if (def->correct) {
    z <- r - W*(W'*A*W)^{-1}*W'*(A*r -r) = (P-Q)*M^{-1}*r
  } else {
    z <- r - W*(W'*A*W)^{-1}*W'*A*r = P*M^{-1}*r
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

  ierr = PCApply(def->pc,r,z);CHKERRQ(ierr);
  if (!def->init) {
    if (!def->AW) {
      ierr = MatMult(A,z,u);CHKERRQ(ierr);                                        /*    u  <- A*z                 */
      if (def->correct) ierr = VecAXPY(u,-1.0*def->correctfact,z);CHKERRQ(ierr);  /*    u  <- A*z -z              */
      ierr = MatMultTranspose(def->W,u,w1);CHKERRQ(ierr);                         /*    w1 <- W'*u                */
    } else {
      /* ONLY if A SYM */
      ierr = MatMultTranspose(def->AW,z,w1);CHKERRQ(ierr);                        /*    w1  <- W'*A*r             */
      if (def->correct) {
        ierr = MatMultTranspose(def->W,z,w2);CHKERRQ(ierr);                       /*    w2 <- W'*u                */
        ierr = VecAXPY(w1,-1.0*def->correctfact,w2);CHKERRQ(ierr);                /*    w1 <- w1 - w2             */
      }
    }
    ierr = KSPSolve(def->WtAWinv,w1,w2);CHKERRQ(ierr);                            /*    w2 <- (W'*A*W)^{-1}*w1    */
    ierr = MatMult(def->W,w2,u);CHKERRQ(ierr);                                    /*    u  <- W*w2                */
    ierr = VecAXPY(z,-1.0,u);CHKERRQ(ierr);                                       /*    z  <- z - u               */
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
    ierr = MatCreateTranspose(def->Wt,&def->W);CHKERRQ(ierr);
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
    ierr = MatTranspose(def->Wt,MAT_INITIAL_MATRIX,&def->W);CHKERRQ(ierr);
  }


  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  /* setup coarse problem */
  if (!def->WtAWinv) {
    ierr = MatGetSize(def->W,NULL,&m);CHKERRQ(ierr);
    if (!def->WtAW) {
      /* TODO add implicit product version ? */
      if (!def->AW) {
        ierr = MatPtAP(Amat,def->W,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtAW);CHKERRQ(ierr);
      } else {
        ierr = MatTransposeMatMult(def->W,def->AW,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&def->WtAW);CHKERRQ(ierr);
      }
      /* TODO create MatInheritOption(Mat,MatOption) */
      ierr = MatGetOption(Amat,MAT_SPD,&flgspd);CHKERRQ(ierr);
      ierr = MatSetOption(def->WtAW,MAT_SPD,flgspd);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      /* Check WtAW is not sigular */
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
      ierr = KSPSetType(innerksp,def->ksptype);CHKERRQ(ierr); /* TODO iherit from KSP */
      ierr = PCSetType(pcinner,PCDEFLATION);CHKERRQ(ierr); /* TODO create coarse preconditinoner M_c = WtMW ? */
      ierr = PCDeflationSetSpace(pcinner,nextDef,transp);CHKERRQ(ierr);
      ierr = PCDeflationSetLvl_Deflation(pcinner,def->nestedlvl+1,def->maxnestedlvl);CHKERRQ(ierr);
      /* inherit options TODO if not set */
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
  ierr = MatDestroy(&def->AW);CHKERRQ(ierr);
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
  PetscBool         iascii;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    //if (cg->adaptiveconv) {
    //  ierr = PetscViewerASCIIPrintf(viewer,"  DCG: using adaptive precision for inner solve with C=%.1e\n",cg->adaptiveconst);CHKERRQ(ierr);
    //}
    if (def->correct) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using CP correction\n");CHKERRQ(ierr);
    }
    if (!def->nestedlvl) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Deflation space type: %s\n",PCDeflationSpaceTypes[def->spacetype]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  DCG %s\n",def->extendsp ? "extended" : "truncated");CHKERRQ(ierr);
    }

    ierr = PetscViewerASCIIPrintf(viewer,"--- Additional PC\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PCView(def->pc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"--- Coarse solver\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
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
  ierr = PetscOptionsEnum("-pc_deflation_compute_space","Compute deflation space","PCDeflationSetSpace",PCDeflationSpaceTypes,(PetscEnum)def->spacetype,(PetscEnum*)&def->spacetype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_compute_space_size","Set size of the deflation space to compute","PCDeflationSetSpace",def->spacesize,&def->spacesize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_space_extend","Extend deflation space instead of truncating (wavelets)","PCDeflation",def->extendsp,&def->extendsp,NULL);CHKERRQ(ierr);
//TODO add set function and fix manpages
  ierr = PetscOptionsBool("-pc_deflation_initdef","Use only initialization step - Initdef","PCDeflation",def->init,&def->init,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_deflation_correct","Add coarse problem correction Q to P","PCDeflation",def->correct,&def->correct,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_deflation_correct_val","Set multiple of Q to use as coarse problem correction","PCDeflation",def->correctfact,&def->correctfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_redfact","Reduction factor for coarse problem solution","PCDeflation",def->reductionfact,&def->reductionfact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_deflation_max_nested_lvl","Maximum of nested deflation levels","PCDeflation",def->maxnestedlvl,&def->maxnestedlvl,NULL);CHKERRQ(ierr);
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

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_Deflation;
  pc->ops->applytranspose      = PCApply_Deflation;
  pc->ops->presolve            = PCPreSolve_Deflation;
  pc->ops->postsolve           = 0;
  pc->ops->setup               = PCSetUp_Deflation;
  pc->ops->reset               = PCReset_Deflation;
  pc->ops->destroy             = PCDestroy_Deflation;
  pc->ops->setfromoptions      = PCSetFromOptions_Deflation;
  pc->ops->view                = PCView_Deflation;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetLvl_C",PCDeflationSetLvl_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetPC_C",PCDeflationSetPC_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetCoarseKSP_C",PCDeflationGetCoarseKSP_Deflation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

