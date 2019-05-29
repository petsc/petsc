
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

const char *const PCDeflationTypes[]    = {"INIT","PRE","POST","PCDeflationType","PC_DEFLATION_",0};

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
  def->nestedlvl = current;
  def->maxnestedlvl = max;
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetMaxLvl - Set maximum level of deflation.

   Logically Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
.  max - maximum deflation level

   Level: intermediate

.seealso: PCDEFLATION
@*/
PetscErrorCode PCDeflationSetMaxLvl(PC pc,PetscInt max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(pc,max,2);
  ierr = PetscTryMethod(pc,"PCDeflationSetLvl_C",(PC,PetscInt,PetscInt),(pc,0,max));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TODO CP corection?
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
    z <- r - W*(W'*A*W)^{-1}*W'*(A*r -r) = (P-Q)*r
  } else {
    z <- r - W*(W'*A*W)^{-1}*W'*A*r = P*r
  }
*/
static PetscErrorCode PCApply_Deflation(PC pc,Vec r, Vec z)
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

  if (!def->AW) {
    ierr = MatMult(A,r,u);CHKERRQ(ierr);                       /*    u  <- A*r                 */
    if (def->correct) ierr = VecAXPY(u,-1.0,r);CHKERRQ(ierr);  /*    u  <- A*r -r              */
    ierr = MatMultTranspose(def->W,u,w1);CHKERRQ(ierr);        /*    w1 <- W'*u                */
  } else {
    ierr = MatMultTranspose(def->AW,u,w1);CHKERRQ(ierr);       /*    u  <- A*r                 */
    if (def->correct) {
      ierr = MatMultTranspose(def->W,r,w2);CHKERRQ(ierr);      /*    w2 <- W'*u                */
      ierr = VecAXPY(w1,-1.0,w2);CHKERRQ(ierr);                /*    w1 <- w1 - w2             */
    }
  }
  ierr = KSPSolve(def->WtAWinv,w1,w2);CHKERRQ(ierr);           /*    w2 <- (W'*A*W)^{-1}*w1    */
  ierr = MatMult(def->W,w2,u);CHKERRQ(ierr);                   /*    u  <- W*w2                */
  ierr = VecWAXPY(z,-1.0,u,r);CHKERRQ(ierr);                   /*    z  <- r - u               */
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCDeflationSetType_Deflation(PC pc,PCDeflationType type)
{
  PC_Deflation *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  def->init = PETSC_FALSE;
  def->pre = PETSC_FALSE;
  if (type == PC_DEFLATION_POST) {
    //pc->ops->postsolve = PCPostSolve_Deflation;
    pc->ops->presolve = 0;
  } else {
    pc->ops->presolve = PCPreSolve_Deflation;
    pc->ops->postsolve = 0;
    if (type == PC_DEFLATION_INIT) {
      def->init = PETSC_TRUE;
      pc->ops->apply = 0;
    } else {
      def->pre  = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationSetType - Causes the deflation preconditioner to use only a special
    initial gues or pre/post solve solution update

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - PC_DEFLATION_PRE, PC_DEFLATION_INIT, PC_DEFLATION_POST

   Options Database Key:
.  -pc_deflation_type <pre,init,post>

   Level: intermediate

   Concepts: Deflation preconditioner

.seealso: PCDeflationGetType()
@*/
PetscErrorCode  PCDeflationSetType(PC pc,PCDeflationType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCDeflationSetType_C",(PC,PCDeflationType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCDeflationGetType_Deflation(PC pc,PCDeflationType *type)
{
  PC_Deflation *def = (PC_Deflation*)pc->data;

  PetscFunctionBegin;
  if (def->init) {
    *type = PC_DEFLATION_INIT;
  } else if (def->pre) {
    *type = PC_DEFLATION_PRE;
  } else {
    *type = PC_DEFLATION_POST;
  }
  PetscFunctionReturn(0);
}

/*@
   PCDeflationGetType - Gets how the diagonal matrix is produced for the preconditioner

   Not Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
-  type - PC_DEFLATION_PRE, PC_DEFLATION_INIT, PC_DEFLATION_POST

   Level: intermediate

   Concepts: Deflation preconditioner

.seealso: PCDeflationSetType()
@*/
PetscErrorCode  PCDeflationGetType(PC pc,PCDeflationType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCDeflationGetType_C",(PC,PCDeflationType*),(pc,type));CHKERRQ(ierr);
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
    ierr = PetscMalloc1(size,&mats);CHKERRQ(ierr);
    if (!transp) {
      for (i=0; i<size; i++) {
        ierr = MatCompositeGetMat(def->W,i,&mats[i]);CHKERRQ(ierr);
        //ierr = PetscObjectReference((PetscObject)mats[i]);CHKERRQ(ierr);
      }
      if (def->nestedlvl < def->maxnestedlvl) {
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
      } else {
        /* ierr = MatCompositeSetMergeType(def->W,MAT_COMPOSITE_MERGE_LEFT);CHKERRQ(ierr); */
        ierr = MatCompositeMerge(def->W);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(mats);CHKERRQ(ierr);
  }

  /* setup coarse problem */
  if (!def->WtAWinv) {
    ierr = MatGetSize(def->W,NULL,&m);CHKERRQ(ierr); /* TODO works for W MatTranspose? */
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
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPCreate(comm,&def->WtAWinv);CHKERRQ(ierr);
    ierr = KSPSetOperators(def->WtAWinv,def->WtAW,def->WtAW);CHKERRQ(ierr);
    ierr = KSPSetType(def->WtAWinv,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(def->WtAWinv,&pcinner);CHKERRQ(ierr);
    ierr = PCSetType(pcinner,PCTELESCOPE);CHKERRQ(ierr);
    /* ugly hack to not have overwritten PCTELESCOPE */
    if (prefix) {
      ierr = KSPSetOptionsPrefix(def->WtAWinv,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(def->WtAWinv,"tel_");CHKERRQ(ierr);
    } else {
      ierr = KSPSetOptionsPrefix(def->WtAWinv,"tel_");CHKERRQ(ierr);
    }
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
      /* Setup KSP and PC */
      if (nextDef) { /* next level for multilevel deflation */
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
        ((PC_Deflation*)(pcinner))->ksptype = def->ksptype;
        ((PC_Deflation*)(pcinner))->correct = def->correct;
        ((PC_Deflation*)(pcinner))->adaptiveconv = def->adaptiveconv;
        ((PC_Deflation*)(pcinner))->adaptiveconst = def->adaptiveconst;
        ierr = MatDestroy(&nextDef);CHKERRQ(ierr);
      } else { /* the last level */
        ierr = KSPSetType(innerksp,KSPGMRES);CHKERRQ(ierr);
        /* TODO Cholesky if flgspd? */
        ierr = PCSetType(pcinner,PCLU);CHKERRQ(ierr);
        //TODO remove explicit matSolverPackage
        if (commsize == red) {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU);CHKERRQ(ierr);
        } else {
          ierr = PCFactorSetMatSolverType(pcinner,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
        }
      }
      /* TODO use def_[lvl]_ if lvl > 0? */
      if (prefix) {
        ierr = KSPSetOptionsPrefix(innerksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(innerksp,"def_");CHKERRQ(ierr);
      } else {
        ierr = KSPSetOptionsPrefix(innerksp,"def_");CHKERRQ(ierr);
      }
    }
    /* TODO: check if WtAWinv is KSP and move following from this if */
    ierr = KSPSetFromOptions(def->WtAWinv);CHKERRQ(ierr);
    //if (def->adaptiveconv) {
    //  PetscReal *rnorm;
    //  PetscNew(&rnorm);
    //  ierr = KSPSetConvergenceTest(def->WtAWinv,KSPDCGConvergedAdaptive_DCG,rnorm,NULL);CHKERRQ(ierr);
    //}
    ierr = KSPSetUp(def->WtAWinv);CHKERRQ(ierr);
  }

  /* create work vecs */
  ierr = MatCreateVecs(Amat,NULL,&def->work);CHKERRQ(ierr);
  ierr = KSPCreateVecs(def->WtAWinv,2,&def->workcoarse,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCReset_Deflation(PC pc)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Deflation(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Deflation      *jac = (PC_Deflation*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg;
  PCDeflationType   deflt,type;

  PetscFunctionBegin;
  ierr = PCDeflationGetType(pc,&deflt);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Deflation options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_jacobi_type","How to construct diagonal matrix","PCDeflationSetType",PCDeflationTypes,(PetscEnum)deflt,(PetscEnum*)&type,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCDeflationSetType(pc,type);CHKERRQ(ierr);
  }
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
  def->pre           = PETSC_TRUE;
  def->correct       = PETSC_FALSE;
  def->truenorm      = PETSC_TRUE;
  def->reductionfact = -1;
  def->spacetype     = PC_DEFLATION_SPACE_HAAR;
  def->spacesize     = 1;
  def->extendsp      = PETSC_FALSE;
  def->nestedlvl     = 0;
  def->maxnestedlvl  = 0;
  def->adaptiveconv  = PETSC_FALSE;
  def->adaptiveconst = 1.0;

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
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetType_C",PCDeflationSetType_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationGetType_C",PCDeflationGetType_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetSpace_C",PCDeflationSetSpace_Deflation);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCDeflationSetLvl_C",PCDeflationSetLvl_Deflation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

