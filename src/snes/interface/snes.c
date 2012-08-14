
#include <petsc-private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscdmshell.h>                /*I "petscdmshell.h" I*/
#include <petscsys.h>                    /*I "petscsys.h" I*/

PetscBool  SNESRegisterAllCalled = PETSC_FALSE;
PetscFList SNESList              = PETSC_NULL;

/* Logging support */
PetscClassId  SNES_CLASSID;
PetscLogEvent  SNES_Solve, SNES_FunctionEval, SNES_JacobianEval, SNES_GSEval;

#undef __FUNCT__
#define __FUNCT__ "SNESSetErrorIfNotConverged"
/*@
   SNESSetErrorIfNotConverged - Causes SNESSolve() to generate an error if the solver has not converged.

   Logically Collective on SNES

   Input Parameters:
+  snes - iterative context obtained from SNESCreate()
-  flg - PETSC_TRUE indicates you want the error generated

   Options database keys:
.  -snes_error_if_not_converged : this takes an optional truth value (0/1/no/yes/true/false)

   Level: intermediate

   Notes:
    Normally PETSc continues if a linear solver fails to converge, you can call SNESGetConvergedReason() after a SNESSolve() 
    to determine if it has converged.

.keywords: SNES, set, initial guess, nonzero

.seealso: SNESGetErrorIfNotConverged(), KSPGetErrorIfNotConverged(), KSPSetErrorIFNotConverged()
@*/
PetscErrorCode  SNESSetErrorIfNotConverged(SNES snes,PetscBool  flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flg,2);
  snes->errorifnotconverged = flg;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetErrorIfNotConverged"
/*@
   SNESGetErrorIfNotConverged - Will SNESSolve() generate an error if the solver does not converge?

   Not Collective

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Output Parameter:
.  flag - PETSC_TRUE if it will generate an error, else PETSC_FALSE

   Level: intermediate

.keywords: SNES, set, initial guess, nonzero

.seealso:  SNESSetErrorIfNotConverged(), KSPGetErrorIfNotConverged(), KSPSetErrorIFNotConverged()
@*/
PetscErrorCode  SNESGetErrorIfNotConverged(SNES snes,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = snes->errorifnotconverged;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFunctionDomainError"
/*@
   SNESSetFunctionDomainError - tells SNES that the input vector to your FormFunction is not
     in the functions domain. For example, negative pressure.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Level: advanced

.keywords: SNES, view

.seealso: SNESCreate(), SNESSetFunction()
@*/
PetscErrorCode  SNESSetFunctionDomainError(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->domainerror = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESGetFunctionDomainError"
/*@
   SNESGetFunctionDomainError - Gets the status of the domain error after a call to SNESComputeFunction;

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Output Parameters:
.  domainerror Set to PETSC_TRUE if there's a domain error; PETSC_FALSE otherwise.

   Level: advanced

.keywords: SNES, view

.seealso: SNESSetFunctionDomainError, SNESComputeFunction()
@*/
PetscErrorCode  SNESGetFunctionDomainError(SNES snes, PetscBool *domainerror)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(domainerror, 2);
  *domainerror = snes->domainerror;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESView"
/*@C
   SNESView - Prints the SNES data structure.

   Collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  viewer - visualization context

   Options Database Key:
.  -snes_view - Calls SNESView() at end of SNESSolve()

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.keywords: SNES, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  SNESView(SNES snes,PetscViewer viewer)
{
  SNESKSPEW           *kctx;
  PetscErrorCode      ierr;
  KSP                 ksp;
  SNESLineSearch      linesearch;
  PetscBool           iascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)snes)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(snes,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)snes,viewer,"SNES Object");CHKERRQ(ierr);
    if (snes->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*snes->ops->view)(snes,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, maximum function evaluations=%D\n",snes->max_its,snes->max_funcs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%G, absolute=%G, solution=%G\n",
                 snes->rtol,snes->abstol,snes->stol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",snes->linear_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",snes->nfuncs);CHKERRQ(ierr);
    if (snes->gridsequence) {
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of grid sequence refinements=%D\n",snes->gridsequence);CHKERRQ(ierr);
    }
    if (snes->ksp_ewconv) {
      kctx = (SNESKSPEW *)snes->kspconvctx;
      if (kctx) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Eisenstat-Walker computation of KSP relative tolerance (version %D)\n",kctx->version);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    rtol_0=%G, rtol_max=%G, threshold=%G\n",kctx->rtol_0,kctx->rtol_max,kctx->threshold);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    gamma=%G, alpha=%G, alpha2=%G\n",kctx->gamma,kctx->alpha,kctx->alpha2);CHKERRQ(ierr);
      }
    }
    if (snes->lagpreconditioner == -1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioned is never rebuilt\n");CHKERRQ(ierr);
    } else if (snes->lagpreconditioner > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioned is rebuilt every %D new Jacobians\n",snes->lagpreconditioner);CHKERRQ(ierr);
    }
    if (snes->lagjacobian == -1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Jacobian is never rebuilt\n");CHKERRQ(ierr);
    } else if (snes->lagjacobian > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Jacobian is rebuilt every %D SNES iterations\n",snes->lagjacobian);CHKERRQ(ierr);
    }
  } else if (isstring) {
    const char *type;
    ierr = SNESGetType(snes,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-3.3s",type);CHKERRQ(ierr);
  }
  if (snes->pc && snes->usespc) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(snes->pc, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  if (snes->usesksp) {
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  if (snes->linesearch) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchView(linesearch, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  We retain a list of functions that also take SNES command 
  line options. These are called at the end SNESSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
static PetscInt numberofsetfromoptions;
static PetscErrorCode (*othersetfromoptions[MAXSETFROMOPTIONS])(SNES);

#undef __FUNCT__  
#define __FUNCT__ "SNESAddOptionsChecker"
/*@C
  SNESAddOptionsChecker - Adds an additional function to check for SNES options.

  Not Collective

  Input Parameter:
. snescheck - function that checks for options

  Level: developer

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode  SNESAddOptionsChecker(PetscErrorCode (*snescheck)(SNES))
{
  PetscFunctionBegin;
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Too many options checkers, only %D allowed", MAXSETFROMOPTIONS);
  }
  othersetfromoptions[numberofsetfromoptions++] = snescheck;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  SNESDefaultMatrixFreeCreate2(SNES,Vec,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUpMatrixFree_Private"
static PetscErrorCode SNESSetUpMatrixFree_Private(SNES snes, PetscBool  hasOperator, PetscInt version)
{
  Mat            J;
  KSP            ksp;
  PC             pc;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  if(!snes->vec_func && (snes->jacobian || snes->jacobian_pre)) {
    Mat A = snes->jacobian, B = snes->jacobian_pre;
    ierr = MatGetVecs(A ? A : B, PETSC_NULL,&snes->vec_func);CHKERRQ(ierr);
  }

  if (version == 1) {
    ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
    ierr = MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix);CHKERRQ(ierr);
    ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  } else if (version == 2) {
    if (!snes->vec_func) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_REAL_SINGLE) && !defined(PETSC_USE_REAL___FLOAT128)
    ierr = SNESDefaultMatrixFreeCreate2(snes,snes->vec_func,&J);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "matrix-free operator rutines (version 2)");
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "matrix-free operator rutines, only version 1 and 2");
  
  ierr = PetscInfo1(snes,"Setting default matrix-free operator routines (version %D)\n", version);CHKERRQ(ierr);
  if (hasOperator) {
    /* This version replaces the user provided Jacobian matrix with a
       matrix-free version but still employs the user-provided preconditioner matrix. */
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
  } else {
    /* This version replaces both the user-provided Jacobian and the user-
       provided preconditioner matrix with the default matrix free version. */
    void *functx;
    ierr = SNESGetFunction(snes,PETSC_NULL,PETSC_NULL,&functx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,functx);CHKERRQ(ierr);
    /* Force no preconditioner */
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscInfo(snes,"Setting default matrix-free preconditioner routines\nThat is no preconditioner is being used\n");CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_SNESVecSol"
static PetscErrorCode DMRestrictHook_SNESVecSol(DM dmfine,Mat Restrict,Vec Rscale,Mat Inject,DM dmcoarse,void *ctx)
{
  SNES snes = (SNES)ctx;
  PetscErrorCode ierr;
  Vec Xfine,Xfine_named = PETSC_NULL,Xcoarse;

  PetscFunctionBegin;
  if (PetscLogPrintInfo) {
    PetscInt finelevel,coarselevel,fineclevel,coarseclevel;
    ierr = DMGetRefineLevel(dmfine,&finelevel);CHKERRQ(ierr);
    ierr = DMGetCoarsenLevel(dmfine,&fineclevel);CHKERRQ(ierr);
    ierr = DMGetRefineLevel(dmcoarse,&coarselevel);CHKERRQ(ierr);
    ierr = DMGetCoarsenLevel(dmcoarse,&coarseclevel);CHKERRQ(ierr);
    ierr = PetscInfo4(dmfine,"Restricting SNES solution vector from level %D-%D to level %D-%D\n",finelevel,fineclevel,coarselevel,coarseclevel);CHKERRQ(ierr);
  }
  if (dmfine == snes->dm) Xfine = snes->vec_sol;
  else {
    ierr = DMGetNamedGlobalVector(dmfine,"SNESVecSol",&Xfine_named);CHKERRQ(ierr);
    Xfine = Xfine_named;
  }
  ierr = DMGetNamedGlobalVector(dmcoarse,"SNESVecSol",&Xcoarse);CHKERRQ(ierr);
  ierr = MatRestrict(Restrict,Xfine,Xcoarse);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Xcoarse,Xcoarse,Rscale);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(dmcoarse,"SNESVecSol",&Xcoarse);CHKERRQ(ierr);
  if (Xfine_named) {ierr = DMRestoreNamedGlobalVector(dmfine,"SNESVecSol",&Xfine_named);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_SNESVecSol"
static PetscErrorCode DMCoarsenHook_SNESVecSol(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCoarsenHookAdd(dmc,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPComputeOperators_SNES"
/* This may be called to rediscretize the operator on levels of linear multigrid. The DM shuffle is so the user can
 * safely call SNESGetDM() in their residual evaluation routine. */
static PetscErrorCode KSPComputeOperators_SNES(KSP ksp,Mat A,Mat B,MatStructure *mstruct,void *ctx)
{
  SNES snes = (SNES)ctx;
  PetscErrorCode ierr;
  Mat Asave = A,Bsave = B;
  Vec X,Xnamed = PETSC_NULL;
  DM dmsave;

  PetscFunctionBegin;
  dmsave = snes->dm;
  ierr = KSPGetDM(ksp,&snes->dm);CHKERRQ(ierr);
  if (dmsave == snes->dm) X = snes->vec_sol; /* We are on the finest level */
  else {                                     /* We are on a coarser level, this vec was initialized using a DM restrict hook */
    ierr = DMGetNamedGlobalVector(snes->dm,"SNESVecSol",&Xnamed);CHKERRQ(ierr);
    X = Xnamed;
  }
  ierr = SNESComputeJacobian(snes,X,&A,&B,mstruct);CHKERRQ(ierr);
  if (A != Asave || B != Bsave) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_SUP,"No support for changing matrices at this time");
  if (Xnamed) {
    ierr = DMRestoreNamedGlobalVector(snes->dm,"SNESVecSol",&Xnamed);CHKERRQ(ierr);
  }
  snes->dm = dmsave;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUpMatrices"
/*@
   SNESSetUpMatrices - ensures that matrices are available for SNES, to be called by SNESSetUp_XXX()

   Collective

   Input Arguments:
.  snes - snes to configure

   Level: developer

.seealso: SNESSetUp()
@*/
PetscErrorCode SNESSetUpMatrices(SNES snes)
{
  PetscErrorCode ierr;
  DM             dm;
  SNESDM         sdm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computejacobian) {
    SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESDM not improperly configured");
  } else if (!snes->jacobian && sdm->computejacobian == MatMFFDComputeJacobian) {
    Mat J;
    void *functx;
    ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
    ierr = MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix);CHKERRQ(ierr);
    ierr = MatSetFromOptions(J);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes,PETSC_NULL,PETSC_NULL,&functx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,functx);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  } else if (snes->mf_operator && !snes->jacobian_pre && !snes->jacobian) {
    Mat J,B;
    ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
    ierr = MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix);CHKERRQ(ierr);
    ierr = MatSetFromOptions(J);CHKERRQ(ierr);
    ierr = DMCreateMatrix(snes->dm,MATAIJ,&B);CHKERRQ(ierr);
    /* sdm->computejacobian was already set to reach here */
    ierr = SNESSetJacobian(snes,J,B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else if (!snes->jacobian_pre) {
    Mat J,B;
    J = snes->jacobian;
    ierr = DMCreateMatrix(snes->dm,MATAIJ,&B);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J?J:B,B,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  {
    KSP ksp;
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp,KSPComputeOperators_SNES,snes);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(snes->dm,DMCoarsenHook_SNESVecSol,DMRestrictHook_SNESVecSol,snes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions"
/*@
   SNESSetFromOptions - Sets various SNES and KSP parameters from user options.

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Options Database Keys:
+  -snes_type <type> - ls, tr, ngmres, ncg, richardson, qn, vi, fas
.  -snes_stol - convergence tolerance in terms of the norm
                of the change in the solution between steps
.  -snes_atol <abstol> - absolute tolerance of residual norm
.  -snes_rtol <rtol> - relative decrease in tolerance norm from initial
.  -snes_max_it <max_it> - maximum number of iterations
.  -snes_max_funcs <max_funcs> - maximum number of function evaluations
.  -snes_max_fail <max_fail> - maximum number of line search failures allowed before stopping, default is none
.  -snes_max_linear_solve_fail - number of linear solver failures before SNESSolve() stops
.  -snes_lag_preconditioner <lag> - how often preconditioner is rebuilt (use -1 to never rebuild)
.  -snes_lag_jacobian <lag> - how often Jacobian is rebuilt (use -1 to never rebuild)
.  -snes_trtol <trtol> - trust region tolerance
.  -snes_no_convergence_test - skip convergence test in nonlinear
                               solver; hence iterations will continue until max_it
                               or some other criterion is reached. Saves expense
                               of convergence test
.  -snes_monitor <optional filename> - prints residual norm at each iteration. if no
                                       filename given prints to stdout
.  -snes_monitor_solution - plots solution at each iteration
.  -snes_monitor_residual - plots residual (not its norm) at each iteration
.  -snes_monitor_solution_update - plots update to solution at each iteration
.  -snes_monitor_draw - plots residual norm at each iteration
.  -snes_fd - use finite differences to compute Jacobian; very slow, only for testing
.  -snes_mf_ksp_monitor - if using matrix-free multiply then print h at each KSP iteration
-  -snes_converged_reason - print the reason for convergence/divergence after each solve

    Options Database for Eisenstat-Walker method:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   To see all options, run your program with the -help option or consult
   the <A href="../../docs/manual.pdf#nameddest=ch_snes">SNES chapter of the users manual</A>.

   Level: beginner

.keywords: SNES, nonlinear, set, options, database

.seealso: SNESSetOptionsPrefix()
@*/
PetscErrorCode  SNESSetFromOptions(SNES snes)
{
  PetscBool               flg,mf,mf_operator,pcset;
  PetscInt                i,indx,lag,mf_version,grids;
  MatStructure            matflag;
  const char              *deft = SNESLS;
  const char              *convtests[] = {"default","skip"};
  SNESKSPEW               *kctx = NULL;
  char                    type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer             monviewer;
  PetscErrorCode          ierr;
  const char              *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  if (!SNESRegisterAllCalled) {ierr = SNESRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscObjectOptionsBegin((PetscObject)snes);CHKERRQ(ierr);
    if (((PetscObject)snes)->type_name) { deft = ((PetscObject)snes)->type_name; }
    ierr = PetscOptionsList("-snes_type","Nonlinear solver method","SNESSetType",SNESList,deft,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetType(snes,type);CHKERRQ(ierr);
    } else if (!((PetscObject)snes)->type_name) {
      ierr = SNESSetType(snes,deft);CHKERRQ(ierr);
    }
    /* not used here, but called so will go into help messaage */
    ierr = PetscOptionsName("-snes_view","Print detailed information on solver used","SNESView",0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_stol","Stop if step length less than","SNESSetTolerances",snes->stol,&snes->stol,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_atol","Stop if function norm less than","SNESSetTolerances",snes->abstol,&snes->abstol,0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_rtol","Stop if decrease in function norm less than","SNESSetTolerances",snes->rtol,&snes->rtol,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_it","Maximum iterations","SNESSetTolerances",snes->max_its,&snes->max_its,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_funcs","Maximum function evaluations","SNESSetTolerances",snes->max_funcs,&snes->max_funcs,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_fail","Maximum nonlinear step failures","SNESSetMaxNonlinearStepFailures",snes->maxFailures,&snes->maxFailures,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_linear_solve_fail","Maximum failures in linear solves allowed","SNESSetMaxLinearSolveFailures",snes->maxLinearSolveFailures,&snes->maxLinearSolveFailures,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-snes_error_if_not_converged","Generate error if solver does not converge","SNESSetErrorIfNotConverged",snes->errorifnotconverged,&snes->errorifnotconverged,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_lag_preconditioner","How often to rebuild preconditioner","SNESSetLagPreconditioner",snes->lagpreconditioner,&lag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetLagPreconditioner(snes,lag);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-snes_lag_jacobian","How often to rebuild Jacobian","SNESSetLagJacobian",snes->lagjacobian,&lag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetLagJacobian(snes,lag);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-snes_grid_sequence","Use grid sequencing to generate initial guess","SNESSetGridSequence",snes->gridsequence,&grids,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetGridSequence(snes,grids);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEList("-snes_convergence_test","Convergence test","SNESSetConvergenceTest",convtests,2,"default",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0: ierr = SNESSetConvergenceTest(snes,SNESDefaultConverged,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); break;
      case 1: ierr = SNESSetConvergenceTest(snes,SNESSkipConverged,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);    break;
      }
    }

    ierr = PetscOptionsBool("-snes_converged_reason","Print reason for converged or diverged","SNESSolve",snes->printreason,&snes->printreason,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsEList("-snes_norm_type","SNES Norm type","SNESSetNormType",SNESNormTypes,5,"function",&indx,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SNESSetNormType(snes,(SNESNormType)indx);CHKERRQ(ierr); }

    kctx = (SNESKSPEW *)snes->kspconvctx;

    ierr = PetscOptionsBool("-snes_ksp_ew","Use Eisentat-Walker linear system convergence test","SNESKSPSetUseEW",snes->ksp_ewconv,&snes->ksp_ewconv,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_ksp_ew_version","Version 1, 2 or 3","SNESKSPSetParametersEW",kctx->version,&kctx->version,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtol0","0 <= rtol0 < 1","SNESKSPSetParametersEW",kctx->rtol_0,&kctx->rtol_0,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtolmax","0 <= rtolmax < 1","SNESKSPSetParametersEW",kctx->rtol_max,&kctx->rtol_max,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_gamma","0 <= gamma <= 1","SNESKSPSetParametersEW",kctx->gamma,&kctx->gamma,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha","1 < alpha <= 2","SNESKSPSetParametersEW",kctx->alpha,&kctx->alpha,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha2","alpha2","SNESKSPSetParametersEW",kctx->alpha2,&kctx->alpha2,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_threshold","0 < threshold < 1","SNESKSPSetParametersEW",kctx->threshold,&kctx->threshold,0);CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_cancel","Remove all monitors","SNESMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorCancel(snes);CHKERRQ(ierr);}

    ierr = PetscOptionsString("-snes_monitor","Monitor norm of function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSet(snes,SNESMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_monitor_range","Monitor range of elements of function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESMonitorSet(snes,SNESMonitorRange,0,0);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_ratiomonitor","Monitor ratios of norms of function","SNESMonitorSetRatio","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSetRatio(snes,monviewer);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_monitor_short","Monitor norm of function (fewer digits)","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSet(snes,SNESMonitorDefaultShort,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_monitor_python","Use Python function","SNESMonitorSet",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscPythonMonitorSet((PetscObject)snes,monfilename);CHKERRQ(ierr);}

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_solution","Plot solution at each iteration","SNESMonitorSolution",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorSolution,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_solution_update","Plot correction at each iteration","SNESMonitorSolutionUpdate",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorSolutionUpdate,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_residual","Plot residual at each iteration","SNESMonitorResidual",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorResidual,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_draw","Plot function norm at each iteration","SNESMonitorLG",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_monitor_range_draw","Plot function range at each iteration","SNESMonitorLG",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorLGRange,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);}

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_fd","Use finite differences (slow) to compute Jacobian","SNESDefaultComputeJacobian",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      void *functx;
      ierr = SNESGetFunction(snes,PETSC_NULL,PETSC_NULL,&functx);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,SNESDefaultComputeJacobian,functx);CHKERRQ(ierr);
      ierr = PetscInfo(snes,"Setting default finite difference Jacobian matrix\n");CHKERRQ(ierr);
    }

    mf = mf_operator = PETSC_FALSE;
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_mf_operator","Use a Matrix-Free Jacobian with user-provided preconditioner matrix","MatCreateSNESMF",PETSC_FALSE,&mf_operator,&flg);CHKERRQ(ierr);
    if (flg && mf_operator) {
      snes->mf_operator = PETSC_TRUE;
      mf = PETSC_TRUE;
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-snes_mf","Use a Matrix-Free Jacobian with no preconditioner matrix","MatCreateSNESMF",PETSC_FALSE,&mf,&flg);CHKERRQ(ierr);
    if (!flg && mf_operator) mf = PETSC_TRUE;
    mf_version = 1;
    ierr = PetscOptionsInt("-snes_mf_version","Matrix-Free routines version 1 or 2","None",mf_version,&mf_version,0);CHKERRQ(ierr);


    /* GS Options */
    ierr = PetscOptionsInt("-snes_gs_sweeps","Number of sweeps of GS to apply","SNESComputeGS",snes->gssweeps,&snes->gssweeps,PETSC_NULL);CHKERRQ(ierr);

    for(i = 0; i < numberofsetfromoptions; i++) {
      ierr = (*othersetfromoptions[i])(snes);CHKERRQ(ierr);
    }

    if (snes->ops->setfromoptions) {
      ierr = (*snes->ops->setfromoptions)(snes);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)snes);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (mf) { ierr = SNESSetUpMatrixFree_Private(snes, mf_operator, mf_version);CHKERRQ(ierr); }

  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
  ierr = KSPGetOperators(snes->ksp,PETSC_NULL,PETSC_NULL,&matflag);
  ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,matflag);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(snes->ksp);CHKERRQ(ierr);

  if (!snes->linesearch) {
    ierr = SNESGetSNESLineSearch(snes, &snes->linesearch);CHKERRQ(ierr);
  }
  ierr = SNESLineSearchSetFromOptions(snes->linesearch);CHKERRQ(ierr);

  /* if someone has set the SNES PC type, create it. */
  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(optionsprefix, "-npc_snes_type", &pcset);CHKERRQ(ierr);
  if (pcset && (!snes->pc)) {
    ierr = SNESGetPC(snes, &snes->pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetComputeApplicationContext"
/*@
   SNESSetComputeApplicationContext - Sets an optional function to compute a user-defined context for 
   the nonlinear solvers.  

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  compute - function to compute the context
-  destroy - function to destroy the context

   Level: intermediate

.keywords: SNES, nonlinear, set, application, context

.seealso: SNESGetApplicationContext(), SNESSetComputeApplicationContext(), SNESGetApplicationContext()
@*/
PetscErrorCode  SNESSetComputeApplicationContext(SNES snes,PetscErrorCode (*compute)(SNES,void**),PetscErrorCode (*destroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->ops->usercompute = compute;
  snes->ops->userdestroy = destroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetApplicationContext"
/*@
   SNESSetApplicationContext - Sets the optional user-defined context for 
   the nonlinear solvers.  

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  usrP - optional user context

   Level: intermediate

.keywords: SNES, nonlinear, set, application, context

.seealso: SNESGetApplicationContext()
@*/
PetscErrorCode  SNESSetApplicationContext(SNES snes,void *usrP)
{
  PetscErrorCode ierr;
  KSP            ksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr       = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr       = KSPSetApplicationContext(ksp,usrP);CHKERRQ(ierr);
  snes->user = usrP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetApplicationContext"
/*@
   SNESGetApplicationContext - Gets the user-defined context for the 
   nonlinear solvers.  

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.keywords: SNES, nonlinear, get, application, context

.seealso: SNESSetApplicationContext()
@*/
PetscErrorCode  SNESGetApplicationContext(SNES snes,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *(void**)usrP = snes->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetIterationNumber"
/*@
   SNESGetIterationNumber - Gets the number of nonlinear iterations completed
   at this time.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  iter - iteration number

   Notes:
   For example, during the computation of iteration 2 this would return 1.

   This is useful for using lagged Jacobians (where one does not recompute the 
   Jacobian at each SNES iteration). For example, the code
.vb
      ierr = SNESGetIterationNumber(snes,&it);
      if (!(it % 2)) {
        [compute Jacobian here]
      }
.ve
   can be used in your ComputeJacobian() function to cause the Jacobian to be
   recomputed every second SNES iteration.

   Level: intermediate

.keywords: SNES, nonlinear, get, iteration, number, 

.seealso:   SNESGetFunctionNorm(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESGetIterationNumber(SNES snes,PetscInt* iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(iter,2);
  *iter = snes->iter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetIterationNumber"
/*@
   SNESSetIterationNumber - Sets the current iteration number.

   Not Collective

   Input Parameter:
.  snes - SNES context
.  iter - iteration number

   Level: developer

.keywords: SNES, nonlinear, set, iteration, number, 

.seealso:   SNESGetFunctionNorm(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESSetIterationNumber(SNES snes,PetscInt iter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = iter;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetFunctionNorm"
/*@
   SNESGetFunctionNorm - Gets the norm of the current function that was set
   with SNESSSetFunction().

   Collective on SNES

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  fnorm - 2-norm of function

   Level: intermediate

.keywords: SNES, nonlinear, get, function, norm

.seealso: SNESGetFunction(), SNESGetIterationNumber(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESGetFunctionNorm(SNES snes,PetscReal *fnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidScalarPointer(fnorm,2);
  *fnorm = snes->norm;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSetFunctionNorm"
/*@
   SNESSetFunctionNorm - Sets the 2-norm of the current function computed using VecNorm().

   Collective on SNES

   Input Parameter:
.  snes - SNES context
.  fnorm - 2-norm of function

   Level: developer

.keywords: SNES, nonlinear, set, function, norm

.seealso: SNESSetFunction(), SNESSetIterationNumber(), VecNorm().
@*/
PetscErrorCode  SNESSetFunctionNorm(SNES snes,PetscReal fnorm)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNonlinearStepFailures"
/*@
   SNESGetNonlinearStepFailures - Gets the number of unsuccessful steps
   attempted by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of unsuccessful steps attempted

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetMaxNonlinearStepFailures()
@*/
PetscErrorCode  SNESGetNonlinearStepFailures(SNES snes,PetscInt* nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMaxNonlinearStepFailures"
/*@
   SNESSetMaxNonlinearStepFailures - Sets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, set, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESGetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()
@*/
PetscErrorCode  SNESSetMaxNonlinearStepFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->maxFailures = maxFails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetMaxNonlinearStepFailures"
/*@
   SNESGetMaxNonlinearStepFailures - Gets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()
 
@*/
PetscErrorCode  SNESGetMaxNonlinearStepFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNumberFunctionEvals"
/*@
   SNESGetNumberFunctionEvals - Gets the number of user provided function evaluations
     done by SNES.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  nfuncs - number of evaluations

   Level: intermediate

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures()
@*/
PetscErrorCode  SNESGetNumberFunctionEvals(SNES snes, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = snes->nfuncs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLinearSolveFailures"
/*@
   SNESGetLinearSolveFailures - Gets the number of failed (non-converged)
   linear solvers.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of failed solves

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures()
@*/
PetscErrorCode  SNESGetLinearSolveFailures(SNES snes,PetscInt* nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numLinearSolveFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMaxLinearSolveFailures"
/*@
   SNESSetMaxLinearSolveFailures - the number of failed linear solve attempts
   allowed before SNES returns with a diverged reason of SNES_DIVERGED_LINEAR_SOLVE

   Logically Collective on SNES

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum allowed linear solve failures

   Level: intermediate

   Notes: By default this is 0; that is SNES returns on the first failed linear solve

.keywords: SNES, nonlinear, set, maximum, unsuccessful, steps

.seealso: SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode  SNESSetMaxLinearSolveFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveInt(snes,maxFails,2);
  snes->maxLinearSolveFailures = maxFails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetMaxLinearSolveFailures"
/*@
   SNESGetMaxLinearSolveFailures - gets the maximum number of linear solve failures that
     are allowed before SNES terminates

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful solves allowed

   Level: intermediate

   Notes: By default this is 1; that is SNES returns on the first failed linear solve

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(),
@*/
PetscErrorCode  SNESGetMaxLinearSolveFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxLinearSolveFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLinearSolveIterations"
/*@
   SNESGetLinearSolveIterations - Gets the total number of linear iterations
   used by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, linear, iterations

.seealso:  SNESGetIterationNumber(), SNESGetFunctionNorm(), SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures()
@*/
PetscErrorCode  SNESGetLinearSolveIterations(SNES snes,PetscInt* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = snes->linear_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetKSP"
/*@
   SNESGetKSP - Returns the KSP context for a SNES solver.

   Not Collective, but if SNES object is parallel, then KSP object is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  ksp - the KSP context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   PC contexts as well.

   Level: beginner

.keywords: SNES, nonlinear, get, KSP, context

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode  SNESGetKSP(SNES snes,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(ksp,2);

  if (!snes->ksp) {
    ierr = KSPCreate(((PetscObject)snes)->comm,&snes->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)snes->ksp,(PetscObject)snes,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->ksp);CHKERRQ(ierr);
  }
  *ksp = snes->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetKSP"
/*@
   SNESSetKSP - Sets a KSP context for the SNES object to use

   Not Collective, but the SNES and KSP objects must live on the same MPI_Comm

   Input Parameters:
+  snes - the SNES context
-  ksp - the KSP context

   Notes:
   The SNES object already has its KSP object, you can obtain with SNESGetKSP()
   so this routine is rarely needed.

   The KSP object that is already in the SNES object has its reference count
   decreased by one.

   Level: developer

.keywords: SNES, nonlinear, get, KSP, context

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode  SNESSetKSP(SNES snes,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(snes,1,ksp,2);
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  if (snes->ksp) {ierr = PetscObjectDereference((PetscObject)snes->ksp);CHKERRQ(ierr);}
  snes->ksp = ksp;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "SNESPublish_Petsc"
static PetscErrorCode SNESPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

/* -----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate"
/*@
   SNESCreate - Creates a nonlinear solver context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator

   Output Parameter:
.  outsnes - the new SNES context

   Options Database Keys:
+   -snes_mf - Activates default matrix-free Jacobian-vector products,
               and no preconditioning matrix
.   -snes_mf_operator - Activates default matrix-free Jacobian-vector
               products, and a user-provided preconditioning matrix
               as set by SNESSetJacobian()
-   -snes_fd - Uses (slow!) finite differences to compute Jacobian

   Level: beginner

.keywords: SNES, nonlinear, create, context

.seealso: SNESSolve(), SNESDestroy(), SNES, SNESSetLagPreconditioner()

@*/
PetscErrorCode  SNESCreate(MPI_Comm comm,SNES *outsnes)
{
  PetscErrorCode      ierr;
  SNES                snes;
  SNESKSPEW           *kctx;

  PetscFunctionBegin;
  PetscValidPointer(outsnes,2);
  *outsnes = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SNESInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(snes,_p_SNES,struct _SNESOps,SNES_CLASSID,0,"SNES","Nonlinear solver","SNES",comm,SNESDestroy,SNESView);CHKERRQ(ierr);

  snes->ops->converged    = SNESDefaultConverged;
  snes->usesksp           = PETSC_TRUE;
  snes->tolerancesset     = PETSC_FALSE;
  snes->max_its           = 50;
  snes->max_funcs         = 10000;
  snes->norm              = 0.0;
  snes->normtype          = SNES_NORM_FUNCTION;
  snes->rtol              = 1.e-8;
  snes->ttol              = 0.0;
  snes->abstol            = 1.e-50;
  snes->stol              = 1.e-8;
  snes->deltatol          = 1.e-12;
  snes->nfuncs            = 0;
  snes->numFailures       = 0;
  snes->maxFailures       = 1;
  snes->linear_its        = 0;
  snes->lagjacobian       = 1;
  snes->lagpreconditioner = 1;
  snes->numbermonitors    = 0;
  snes->data              = 0;
  snes->setupcalled       = PETSC_FALSE;
  snes->ksp_ewconv        = PETSC_FALSE;
  snes->nwork             = 0;
  snes->work              = 0;
  snes->nvwork            = 0;
  snes->vwork             = 0;
  snes->conv_hist_len     = 0;
  snes->conv_hist_max     = 0;
  snes->conv_hist         = PETSC_NULL;
  snes->conv_hist_its     = PETSC_NULL;
  snes->conv_hist_reset   = PETSC_TRUE;
  snes->vec_func_init_set = PETSC_FALSE;
  snes->norm_init         = 0.;
  snes->norm_init_set     = PETSC_FALSE;
  snes->reason            = SNES_CONVERGED_ITERATING;
  snes->gssweeps          = 1;

  snes->numLinearSolveFailures = 0;
  snes->maxLinearSolveFailures = 1;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  ierr = PetscNewLog(snes,SNESKSPEW,&kctx);CHKERRQ(ierr);
  snes->kspconvctx  = (void*)kctx;
  kctx->version     = 2;
  kctx->rtol_0      = .3; /* Eisenstat and Walker suggest rtol_0=.5, but
                             this was too large for some test cases */
  kctx->rtol_last   = 0.0;
  kctx->rtol_max    = .9;
  kctx->gamma       = 1.0;
  kctx->alpha       = .5*(1.0 + PetscSqrtReal(5.0));
  kctx->alpha2      = kctx->alpha;
  kctx->threshold   = .1;
  kctx->lresid_last = 0.0;
  kctx->norm_last   = 0.0;

  *outsnes = snes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFunction"
/*@C
   SNESSetFunction - Sets the function evaluation routine and function
   vector for use by the SNES routines in solving systems of nonlinear
   equations.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  r - vector to store function value
.  func - function evaluation routine
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,Vec f,void *ctx);

+  snes - the SNES context
.  x - state at which to evaluate residual
.  f - vector to put residual
-  ctx - optional user-defined function context 

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
   where f'(x) denotes the Jacobian matrix and f(x) is the function.

   Level: beginner

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetPicard()
@*/
PetscErrorCode  SNESSetFunction(SNES snes,Vec r,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (r) {
    PetscValidHeaderSpecific(r,VEC_CLASSID,2);
    PetscCheckSameComm(snes,1,r,2);
    ierr = PetscObjectReference((PetscObject)r);CHKERRQ(ierr);
    ierr = VecDestroy(&snes->vec_func);CHKERRQ(ierr);
    snes->vec_func = r;
  }
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetFunction(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSetInitialFunction"
/*@C
   SNESSetInitialFunction - Sets the function vector to be used as the
   function norm at the initialization of the method.  In some
   instances, the user has precomputed the function before calling
   SNESSolve.  This function allows one to avoid a redundant call
   to SNESComputeFunction in that case.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  f - vector to store function value

   Notes:
   This should not be modified during the solution procedure.

   This is used extensively in the SNESFAS hierarchy and in nonlinear preconditioning.

   Level: developer

.keywords: SNES, nonlinear, set, function

.seealso: SNESSetFunction(), SNESComputeFunction(), SNESSetInitialFunctionNorm()
@*/
PetscErrorCode  SNESSetInitialFunction(SNES snes, Vec f)
{
  PetscErrorCode ierr;
  Vec            vec_func;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(f,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,f,2);
  ierr = SNESGetFunction(snes,&vec_func,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCopy(f, vec_func);CHKERRQ(ierr);
  snes->vec_func_init_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSetInitialFunctionNorm"
/*@C
   SNESSetInitialFunctionNorm - Sets the function norm to be used as the function norm
   at the initialization of the  method.  In some instances, the user has precomputed
   the function and its norm before calling SNESSolve.  This function allows one to
   avoid a redundant call to SNESComputeFunction() and VecNorm() in that case.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  fnorm - the norm of F as set by SNESSetInitialFunction()

   This is used extensively in the SNESFAS hierarchy and in nonlinear preconditioning.

   Level: developer

.keywords: SNES, nonlinear, set, function, norm

.seealso: SNESSetFunction(), SNESComputeFunction(), SNESSetInitialFunction()
@*/
PetscErrorCode  SNESSetInitialFunctionNorm(SNES snes, PetscReal fnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->norm_init = fnorm;
  snes->norm_init_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetNormType"
/*@
   SNESSetNormType - Sets the SNESNormType used in covergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normtype - the type of the norm used

   Notes:
   Only certain SNES methods support certain SNESNormTypes.  Most require evaluation
   of the nonlinear function and the taking of its norm at every iteration to
   even ensure convergence at all.  However, methods such as custom Gauss-Seidel methods
   (SNESGS) and the like do not require the norm of the function to be computed, and therfore
   may either be monitored for convergence or not.  As these are often used as nonlinear
   preconditioners, monitoring the norm of their error is not a useful enterprise within
   their solution.

   Level: developer

.keywords: SNES, nonlinear, set, function, norm, type

.seealso: SNESGetNormType(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormType
@*/
PetscErrorCode  SNESSetNormType(SNES snes, SNESNormType normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  snes->normtype = normtype;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESGetNormType"
/*@
   SNESGetNormType - Gets the SNESNormType used in covergence and monitoring
   of the SNES method.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  normtype - the type of the norm used

   Level: advanced

.keywords: SNES, nonlinear, set, function, norm, type

.seealso: SNESSetNormType(), SNESComputeFunction(), VecNorm(), SNESSetFunction(), SNESSetInitialFunction(), SNESNormType
@*/
PetscErrorCode  SNESGetNormType(SNES snes, SNESNormType *normtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *normtype = snes->normtype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetGS"
/*@C
   SNESSetGS - Sets the user nonlinear Gauss-Seidel routine for
   use with composed nonlinear solvers.

   Input Parameters:
+  snes   - the SNES context
.  gsfunc - function evaluation routine
-  ctx    - [optional] user-defined context for private data for the
            smoother evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,Vec b,void *ctx);

+  X   - solution vector
.  B   - RHS vector
-  ctx - optional user-defined Gauss-Seidel context

   Notes:
   The GS routines are used by the composed nonlinear solver to generate
    a problem appropriate update to the solution, particularly FAS.

   Level: intermediate

.keywords: SNES, nonlinear, set, Gauss-Seidel

.seealso: SNESGetFunction(), SNESComputeGS()
@*/
PetscErrorCode SNESSetGS(SNES snes,PetscErrorCode (*gsfunc)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetGS(dm,gsfunc,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetGSSweeps"
/*@
   SNESSetGSSweeps - Sets the number of sweeps of GS to use.

   Input Parameters:
+  snes   - the SNES context
-  sweeps  - the number of sweeps of GS to perform.

   Level: intermediate

.keywords: SNES, nonlinear, set, Gauss-Siedel

.seealso: SNESSetGS(), SNESGetGS(), SNESSetPC(), SNESGetGSSweeps()
@*/

PetscErrorCode SNESSetGSSweeps(SNES snes, PetscInt sweeps) {
  PetscFunctionBegin;
  snes->gssweeps = sweeps;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESGetGSSweeps"
/*@
   SNESGetGSSweeps - Gets the number of sweeps GS will use.

   Input Parameters:
.  snes   - the SNES context

   Output Parameters:
.  sweeps  - the number of sweeps of GS to perform.

   Level: intermediate

.keywords: SNES, nonlinear, set, Gauss-Siedel

.seealso: SNESSetGS(), SNESGetGS(), SNESSetPC(), SNESSetGSSweeps()
@*/
PetscErrorCode SNESGetGSSweeps(SNES snes, PetscInt * sweeps) {
  PetscFunctionBegin;
  *sweeps = snes->gssweeps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESPicardComputeFunction"
PetscErrorCode  SNESPicardComputeFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;
  DM dm;
  SNESDM sdm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  /*  A(x)*x - b(x) */
  if (sdm->computepfunction) {
    ierr = (*sdm->computepfunction)(snes,x,f,sdm->pctx);CHKERRQ(ierr);
  } else if (snes->dm) {
    ierr = DMComputeFunction(snes->dm,x,f);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetPicard() or SNESSetDM() before SNESPicardComputeFunction to provide Picard function.");
  }

  if (sdm->computepjacobian) {
    ierr = (*sdm->computepjacobian)(snes,x,&snes->jacobian,&snes->jacobian_pre,&snes->matstruct,sdm->pctx);CHKERRQ(ierr);
  } else if (snes->dm) {
    ierr = DMComputeJacobian(snes->dm,x,snes->jacobian,snes->jacobian_pre,&snes->matstruct);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetPicard() or SNESSetDM() before SNESPicardComputeFunction to provide Picard matrix.");
  }

  ierr = VecView(x,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
  ierr = VecView(f,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
  ierr = VecScale(f,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(snes->jacobian_pre,x,f,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESPicardComputeJacobian"
PetscErrorCode  SNESPicardComputeJacobian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscFunctionBegin;
  /* the jacobian matrix should be pre-filled in SNESPicardComputeFunction */
  *flag = snes->matstruct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetPicard"
/*@C
   SNESSetPicard - Use SNES to solve the semilinear-system A(x) x = b(x) via a Picard type iteration (Picard linearization) 

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  r - vector to store function value
.  func - function evaluation routine
.  jmat - normally the same as mat but you can pass another matrix for which you compute the Jacobian of A(x) x - b(x) (see jmat below)
.  mat - matrix to store A
.  mfunc  - function to compute matrix value
-  ctx - [optional] user-defined context for private data for the
         function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,Vec f,void *ctx);

+  f - function vector
-  ctx - optional user-defined function context 

   Calling sequence of mfunc:
$     mfunc (SNES snes,Vec x,Mat *jmat,Mat *mat,int *flag,void *ctx);

+  x - input vector
.  jmat - Form Jacobian matrix of A(x) x - b(x) if available, not there is really no reason to use it in this way since then you can just use SNESSetJacobian(), 
          normally just pass mat in this location
.  mat - form A(x) matrix
.  flag - flag indicating information about the preconditioner matrix
   structure (same as flag in KSPSetOperators()), one of SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER
-  ctx - [optional] user-defined Jacobian context

   Notes:
    One can call SNESSetPicard() or SNESSetFunction() (and possibly SNESSetJacobian()) but cannot call both

$     Solves the equation A(x) x = b(x) via the defect correction algorithm A(x^{n}) (x^{n+1} - x^{n}) = b(x^{n}) - A(x^{n})x^{n}
$     Note that when an exact solver is used this corresponds to the "classic" Picard A(x^{n}) x^{n+1} = b(x^{n}) iteration.

     Run with -snes_mf_operator to solve the system with Newton's method using A(x^{n}) to construct the preconditioner.

   We implement the defect correction form of the Picard iteration because it converges much more generally when inexact linear solvers are used then 
   the direct Picard iteration A(x^n) x^{n+1} = b(x^n)

   There is some controversity over the definition of a Picard iteration for nonlinear systems but almost everyone agrees that it involves a linear solve and some
   believe it is the iteration  A(x^{n}) x^{n+1} = b(x^{n}) hence we use the name Picard. If anyone has an authoritative  reference that defines the Picard iteration 
   different please contact us at petsc-dev@mcs.anl.gov and we'll have an entirely new argument :-).

   Level: beginner

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESSetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESGetPicard(), SNESLineSearchPreCheckPicard()
@*/
PetscErrorCode  SNESSetPicard(SNES snes,Vec r,PetscErrorCode (*func)(SNES,Vec,Vec,void*),Mat jmat, Mat mat, PetscErrorCode (*mfunc)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = DMSNESSetPicard(dm,func,mfunc,ctx);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,SNESPicardComputeFunction,ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,jmat,mat,SNESPicardComputeJacobian,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESGetPicard"
/*@C
   SNESGetPicard - Returns the context for the Picard iteration

   Not Collective, but Vec is parallel if SNES is parallel. Collective if Vec is requested, but has not been created yet.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
+  r - the function (or PETSC_NULL)
.  func - the function (or PETSC_NULL)
.  jmat - the picard matrix (or PETSC_NULL)
.  mat  - the picard preconditioner matrix (or PETSC_NULL)
.  mfunc - the function for matrix evaluation (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, function

.seealso: SNESSetPicard, SNESGetFunction, SNESGetJacobian, SNESGetDM
@*/
PetscErrorCode  SNESGetPicard(SNES snes,Vec *r,PetscErrorCode (**func)(SNES,Vec,Vec,void*),Mat *jmat, Mat *mat, PetscErrorCode (**mfunc)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetFunction(snes,r,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,jmat,mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetPicard(dm,func,mfunc,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetComputeInitialGuess"
/*@C
   SNESSetComputeInitialGuess - Sets a routine used to compute an initial guess for the problem

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - function evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,void *ctx);

.  f - function vector
-  ctx - optional user-defined function context 

   Level: intermediate

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian()
@*/
PetscErrorCode  SNESSetComputeInitialGuess(SNES snes,PetscErrorCode (*func)(SNES,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) snes->ops->computeinitialguess = func;
  if (ctx)  snes->initialguessP            = ctx;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESGetRhs"
/*@C
   SNESGetRhs - Gets the vector for solving F(x) = rhs. If rhs is not set
   it assumes a zero right hand side.

   Logically Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  rhs - the right hand side vector or PETSC_NULL if the right hand side vector is null

   Level: intermediate

.keywords: SNES, nonlinear, get, function, right hand side

.seealso: SNESGetSolution(), SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
@*/
PetscErrorCode  SNESGetRhs(SNES snes,Vec *rhs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(rhs,2);
  *rhs = snes->vec_rhs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunction"
/*@
   SNESComputeFunction - Calls the function that has been set with
                         SNESSetFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() is typically used within nonlinear solvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction(), SNESGetFunction()
@*/
PetscErrorCode  SNESComputeFunction(SNES snes,Vec x,Vec y)
{
  PetscErrorCode ierr;
  DM             dm;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);
  VecValidValues(x,2,PETSC_TRUE);

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(SNES_FunctionEval,snes,x,y,0);CHKERRQ(ierr);
  if (sdm->computefunction) {
    PetscStackPush("SNES user function");
    ierr = (*sdm->computefunction)(snes,x,y,sdm->functionctx);CHKERRQ(ierr);
    PetscStackPop;
  } else if (snes->dm) {
    ierr = DMComputeFunction(snes->dm,x,y);CHKERRQ(ierr);
  } else if (snes->vec_rhs) {
    ierr = MatMult(snes->jacobian, x, y);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetFunction() or SNESSetDM() before SNESComputeFunction(), likely called from SNESSolve().");
  if (snes->vec_rhs) {
    ierr = VecAXPY(y,-1.0,snes->vec_rhs);CHKERRQ(ierr);
  }
  snes->nfuncs++;
  ierr = PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0);CHKERRQ(ierr);
  VecValidValues(y,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeGS"
/*@
   SNESComputeGS - Calls the Gauss-Seidel function that has been set with
                   SNESSetGS().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  x - input vector
-  b - rhs vector

   Output Parameter:
.  x - new solution vector

   Notes:
   SNESComputeGS() is typically used within composed nonlinear solver
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetGS(), SNESComputeFunction()
@*/
PetscErrorCode  SNESComputeGS(SNES snes,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt i;
  DM dm;
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  if(b) PetscCheckSameComm(snes,1,b,3);
  if (b) VecValidValues(b,2,PETSC_TRUE);
  ierr = PetscLogEventBegin(SNES_GSEval,snes,x,b,0);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (sdm->computegs) {
    for (i = 0; i < snes->gssweeps; i++) {
      PetscStackPush("SNES user GS");
      ierr = (*sdm->computegs)(snes,x,b,sdm->gsctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetGS() before SNESComputeGS(), likely called from SNESSolve().");
  ierr = PetscLogEventEnd(SNES_GSEval,snes,x,b,0);CHKERRQ(ierr);
  VecValidValues(x,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESComputeJacobian"
/*@
   SNESComputeJacobian - Computes the Jacobian matrix that has been
   set with SNESSetJacobian().

   Collective on SNES and Mat

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameters:
+  A - Jacobian matrix
.  B - optional preconditioning matrix
-  flag - flag indicating matrix structure (one of, SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER)

  Options Database Keys: 
+    -snes_lag_preconditioner <lag>
.    -snes_lag_jacobian <lag>
.    -snes_compare_explicit - Compare the computed Jacobian to the finite difference Jacobian and output the differences
.    -snes_compare_explicit_draw  - Compare the computed Jacobian to the finite difference Jacobian and draw the result
.    -snes_compare_explicit_contour  - Compare the computed Jacobian to the finite difference Jacobian and draw a contour plot with the result
.    -snes_compare_operator  - Make the comparison options above use the operator instead of the preconditioning matrix
.    -snes_compare_coloring - Compute the finite differece Jacobian using coloring and display norms of difference
.    -snes_compare_coloring_display - Compute the finite differece Jacobian using coloring and display verbose differences
.    -snes_compare_coloring_threshold - Display only those matrix entries that differ by more than a given threshold
.    -snes_compare_coloring_threshold_atol - Absolute tolerance for difference in matrix entries to be displayed by -snes_compare_coloring_threshold
.    -snes_compare_coloring_threshold_rtol - Relative tolerance for difference in matrix entries to be displayed by -snes_compare_coloring_threshold
.    -snes_compare_coloring_draw - Compute the finite differece Jacobian using coloring and draw differences
-    -snes_compare_coloring_draw_contour - Compute the finite differece Jacobian using coloring and show contours of matrices and differences


   Notes: 
   Most users should not need to explicitly call this routine, as it 
   is used internally within the nonlinear solvers. 

   See KSPSetOperators() for important information about setting the
   flag parameter.

   Level: developer

.keywords: SNES, compute, Jacobian, matrix

.seealso:  SNESSetJacobian(), KSPSetOperators(), MatStructure, SNESSetLagPreconditioner(), SNESSetLagJacobian()
@*/
PetscErrorCode  SNESComputeJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  DM             dm;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidPointer(flg,5);
  PetscCheckSameComm(snes,1,X,2);
  VecValidValues(X,2,PETSC_TRUE);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computejacobian) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_USER,"Must call SNESSetJacobian(), DMSNESSetJacobian(), DMDASNESSetJacobianLocal(), etc");

  /* make sure that MatAssemblyBegin/End() is called on A matrix if it is matrix free */

  if (snes->lagjacobian == -2) {
    snes->lagjacobian = -1;
    ierr = PetscInfo(snes,"Recomputing Jacobian/preconditioner because lag is -2 (means compute Jacobian, but then never again) \n");CHKERRQ(ierr);
  } else if (snes->lagjacobian == -1) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo(snes,"Reusing Jacobian/preconditioner because lag is -1\n");CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)*A,MATMFFD,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  } else if (snes->lagjacobian > 1 && snes->iter % snes->lagjacobian) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo2(snes,"Reusing Jacobian/preconditioner because lag is %D and SNES iteration is %D\n",snes->lagjacobian,snes->iter);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)*A,MATMFFD,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  *flg = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscLogEventBegin(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);
  PetscStackPush("SNES user Jacobian function");
  ierr = (*sdm->computejacobian)(snes,X,A,B,flg,sdm->jacobianctx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);

  if (snes->lagpreconditioner == -2) {
    ierr = PetscInfo(snes,"Rebuilding preconditioner exactly once since lag is -2\n");CHKERRQ(ierr);
    snes->lagpreconditioner = -1;
  } else if (snes->lagpreconditioner == -1) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo(snes,"Reusing preconditioner because lag is -1\n");CHKERRQ(ierr);
  } else if (snes->lagpreconditioner > 1 && snes->iter % snes->lagpreconditioner) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo2(snes,"Reusing preconditioner because lag is %D and SNES iteration is %D\n",snes->lagpreconditioner,snes->iter);CHKERRQ(ierr);
  }

  /* make sure user returned a correct Jacobian and preconditioner */
  /* PetscValidHeaderSpecific(*A,MAT_CLASSID,3);
    PetscValidHeaderSpecific(*B,MAT_CLASSID,4);   */
  {
    PetscBool flag = PETSC_FALSE,flag_draw = PETSC_FALSE,flag_contour = PETSC_FALSE,flag_operator = PETSC_FALSE;
    ierr  = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_explicit",&flag,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_explicit_draw",&flag_draw,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_explicit_draw_contour",&flag_contour,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_operator",&flag_operator,PETSC_NULL);CHKERRQ(ierr);
    if (flag || flag_draw || flag_contour) {
      Mat Bexp_mine = PETSC_NULL,Bexp,FDexp;
      MatStructure mstruct;
      PetscViewer vdraw,vstdout;
      PetscBool flg;
      if (flag_operator) {
        ierr = MatComputeExplicitOperator(*A,&Bexp_mine);CHKERRQ(ierr);
        Bexp = Bexp_mine;
      } else {
        /* See if the preconditioning matrix can be viewed and added directly */
        ierr = PetscObjectTypeCompareAny((PetscObject)*B,&flg,MATSEQAIJ,MATMPIAIJ,MATSEQDENSE,MATMPIDENSE,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPIBAIJ,"");CHKERRQ(ierr);
        if (flg) Bexp = *B;
        else {
          /* If the "preconditioning" matrix is itself MATSHELL or some other type without direct support */
          ierr = MatComputeExplicitOperator(*B,&Bexp_mine);CHKERRQ(ierr);
          Bexp = Bexp_mine;
        }
      }
      ierr = MatConvert(Bexp,MATSAME,MAT_INITIAL_MATRIX,&FDexp);CHKERRQ(ierr);
      ierr = SNESDefaultComputeJacobian(snes,X,&FDexp,&FDexp,&mstruct,NULL);CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(((PetscObject)snes)->comm,&vstdout);CHKERRQ(ierr);
      if (flag_draw || flag_contour) {
        ierr = PetscViewerDrawOpen(((PetscObject)snes)->comm,0,"Explicit Jacobians",PETSC_DECIDE,PETSC_DECIDE,300,300,&vdraw);CHKERRQ(ierr);
        if (flag_contour) {ierr = PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);}
      } else vdraw = PETSC_NULL;
      ierr = PetscViewerASCIIPrintf(vstdout,"Explicit %s\n",flag_operator?"Jacobian":"preconditioning Jacobian");CHKERRQ(ierr);
      if (flag) {ierr = MatView(Bexp,vstdout);CHKERRQ(ierr);}
      if (vdraw) {ierr = MatView(Bexp,vdraw);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(vstdout,"Finite difference Jacobian\n");CHKERRQ(ierr);
      if (flag) {ierr = MatView(FDexp,vstdout);CHKERRQ(ierr);}
      if (vdraw) {ierr = MatView(FDexp,vdraw);CHKERRQ(ierr);}
      ierr = MatAYPX(FDexp,-1.0,Bexp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vstdout,"User-provided matrix minus finite difference Jacobian\n");CHKERRQ(ierr);
      if (flag) {ierr = MatView(FDexp,vstdout);CHKERRQ(ierr);}
      if (vdraw) {              /* Always use contour for the difference */
        ierr = PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
        ierr = MatView(FDexp,vdraw);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(vdraw);CHKERRQ(ierr);
      }
      if (flag_contour) {ierr = PetscViewerPopFormat(vdraw);CHKERRQ(ierr);}
      ierr = PetscViewerDestroy(&vdraw);CHKERRQ(ierr);
      ierr = MatDestroy(&Bexp_mine);CHKERRQ(ierr);
      ierr = MatDestroy(&FDexp);CHKERRQ(ierr);
    }
  }
  {
    PetscBool flag = PETSC_FALSE,flag_display = PETSC_FALSE,flag_draw = PETSC_FALSE,flag_contour = PETSC_FALSE,flag_threshold = PETSC_FALSE;
    PetscReal threshold_atol = PETSC_SQRT_MACHINE_EPSILON,threshold_rtol = 10*PETSC_SQRT_MACHINE_EPSILON;
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_coloring",&flag,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_coloring_display",&flag_display,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_coloring_draw",&flag_draw,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_coloring_draw_contour",&flag_contour,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold",&flag_threshold,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold_rtol",&threshold_rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(((PetscObject)snes)->prefix,"-snes_compare_coloring_threshold_atol",&threshold_atol,PETSC_NULL);CHKERRQ(ierr);
    if (flag || flag_display || flag_draw || flag_contour || flag_threshold) {
      Mat Bfd;
      MatStructure mstruct;
      PetscViewer vdraw,vstdout;
      ISColoring iscoloring;
      MatFDColoring matfdcoloring;
      PetscErrorCode (*func)(SNES,Vec,Vec,void*);
      void *funcctx;
      PetscReal norm1,norm2,normmax;

      ierr = MatDuplicate(*B,MAT_DO_NOT_COPY_VALUES,&Bfd);CHKERRQ(ierr);
      ierr = MatGetColoring(Bfd,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(Bfd,iscoloring,&matfdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

      /* This method of getting the function is currently unreliable since it doesn't work for DM local functions. */
      ierr = SNESGetFunction(snes,PETSC_NULL,&func,&funcctx);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode(*)(void))func,funcctx);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)matfdcoloring,((PetscObject)snes)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)matfdcoloring,"coloring_");CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringApply(Bfd,matfdcoloring,X,&mstruct,snes);CHKERRQ(ierr);
      ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);

      ierr = PetscViewerASCIIGetStdout(((PetscObject)snes)->comm,&vstdout);CHKERRQ(ierr);
      if (flag_draw || flag_contour) {
        ierr = PetscViewerDrawOpen(((PetscObject)snes)->comm,0,"Colored Jacobians",PETSC_DECIDE,PETSC_DECIDE,300,300,&vdraw);CHKERRQ(ierr);
        if (flag_contour) {ierr = PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);}
      } else vdraw = PETSC_NULL;
      ierr = PetscViewerASCIIPrintf(vstdout,"Explicit preconditioning Jacobian\n");CHKERRQ(ierr);
      if (flag_display) {ierr = MatView(*B,vstdout);CHKERRQ(ierr);}
      if (vdraw) {ierr = MatView(*B,vdraw);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(vstdout,"Colored Finite difference Jacobian\n");CHKERRQ(ierr);
      if (flag_display) {ierr = MatView(Bfd,vstdout);CHKERRQ(ierr);}
      if (vdraw) {ierr = MatView(Bfd,vdraw);CHKERRQ(ierr);}
      ierr = MatAYPX(Bfd,-1.0,*B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(Bfd,NORM_1,&norm1);CHKERRQ(ierr);
      ierr = MatNorm(Bfd,NORM_FROBENIUS,&norm2);CHKERRQ(ierr);
      ierr = MatNorm(Bfd,NORM_MAX,&normmax);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vstdout,"User-provided matrix minus finite difference Jacobian, norm1=%G normFrob=%G normmax=%G\n",norm1,norm2,normmax);CHKERRQ(ierr);
      if (flag_display) {ierr = MatView(Bfd,vstdout);CHKERRQ(ierr);}
      if (vdraw) {              /* Always use contour for the difference */
        ierr = PetscViewerPushFormat(vdraw,PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
        ierr = MatView(Bfd,vdraw);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(vdraw);CHKERRQ(ierr);
      }
      if (flag_contour) {ierr = PetscViewerPopFormat(vdraw);CHKERRQ(ierr);}

      if (flag_threshold) {
        PetscInt bs,rstart,rend,i;
        ierr = MatGetBlockSize(*B,&bs);CHKERRQ(ierr);
        ierr = MatGetOwnershipRange(*B,&rstart,&rend);CHKERRQ(ierr);
        for (i=rstart; i<rend; i++) {
          const PetscScalar *ba,*ca;
          const PetscInt *bj,*cj;
          PetscInt bn,cn,j,maxentrycol = -1,maxdiffcol = -1,maxrdiffcol = -1;
          PetscReal maxentry = 0,maxdiff = 0,maxrdiff = 0;
          ierr = MatGetRow(*B,i,&bn,&bj,&ba);CHKERRQ(ierr);
          ierr = MatGetRow(Bfd,i,&cn,&cj,&ca);CHKERRQ(ierr);
          if (bn != cn) SETERRQ(((PetscObject)*A)->comm,PETSC_ERR_PLIB,"Unexpected different nonzero pattern in -snes_compare_coloring_threshold");
          for (j=0; j<bn; j++) {
            PetscReal rdiff = PetscAbsScalar(ca[j]) / (threshold_atol + threshold_rtol*PetscAbsScalar(ba[j]));
            if (PetscAbsScalar(ba[j]) > PetscAbs(maxentry)) {
              maxentrycol = bj[j];
              maxentry = PetscRealPart(ba[j]);
            }
            if (PetscAbsScalar(ca[j]) > PetscAbs(maxdiff)) {
              maxdiffcol = bj[j];
              maxdiff = PetscRealPart(ca[j]);
            }
            if (rdiff > maxrdiff) {
              maxrdiffcol = bj[j];
              maxrdiff = rdiff;
            }
          }
          if (maxrdiff > 1) {
            ierr = PetscViewerASCIIPrintf(vstdout,"row %D (maxentry=%G at %D, maxdiff=%G at %D, maxrdiff=%G at %D):",i,maxentry,maxentrycol,maxdiff,maxdiffcol,maxrdiff,maxrdiffcol);CHKERRQ(ierr);
            for (j=0; j<bn; j++) {
              PetscReal rdiff;
              rdiff = PetscAbsScalar(ca[j]) / (threshold_atol + threshold_rtol*PetscAbsScalar(ba[j]));
              if (rdiff > 1) {
                ierr = PetscViewerASCIIPrintf(vstdout," (%D,%G:%G)",bj[j],PetscRealPart(ba[j]),PetscRealPart(ca[j]));CHKERRQ(ierr);
              }
            }
            ierr = PetscViewerASCIIPrintf(vstdout,"\n",i,maxentry,maxdiff,maxrdiff);CHKERRQ(ierr);
          }
          ierr = MatRestoreRow(*B,i,&bn,&bj,&ba);CHKERRQ(ierr);
          ierr = MatRestoreRow(Bfd,i,&cn,&cj,&ca);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerDestroy(&vdraw);CHKERRQ(ierr);
      ierr = MatDestroy(&Bfd);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetJacobian"
/*@C
   SNESSetJacobian - Sets the function to compute Jacobian as well as the
   location to store the matrix.

   Logically Collective on SNES and Mat

   Input Parameters:
+  snes - the SNES context
.  A - Jacobian matrix
.  B - preconditioner matrix (usually same as the Jacobian)
.  func - Jacobian evaluation routine (if PETSC_NULL then SNES retains any previously set value)
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL) (if PETSC_NULL then SNES retains any previously set value)

   Calling sequence of func:
$     func (SNES snes,Vec x,Mat *A,Mat *B,int *flag,void *ctx);

+  x - input vector
.  A - Jacobian matrix
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
   structure (same as flag in KSPSetOperators()), one of SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER
-  ctx - [optional] user-defined Jacobian context

   Notes: 
   See KSPSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   If the A matrix and B matrix are different you must call MatAssemblyBegin/End() on
   each matrix.

   If using SNESDefaultComputeJacobianColor() to assemble a Jacobian, the ctx argument
   must be a MatFDColoring.

   Other defect-correction schemes can be used by computing a different matrix in place of the Jacobian.  One common
   example is to use the "Picard linearization" which only differentiates through the highest order parts of each term.

   Level: beginner

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: KSPSetOperators(), SNESSetFunction(), MatMFFDComputeJacobian(), SNESDefaultComputeJacobianColor(), MatStructure
@*/
PetscErrorCode  SNESSetJacobian(SNES snes,Mat A,Mat B,PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(snes,1,A,2);
  if (B) PetscCheckSameComm(snes,1,B,3);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetJacobian(dm,func,ctx);CHKERRQ(ierr);
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&snes->jacobian);CHKERRQ(ierr);
    snes->jacobian = A;
  }
  if (B) {
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    ierr = MatDestroy(&snes->jacobian_pre);CHKERRQ(ierr);
    snes->jacobian_pre = B;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetJacobian"
/*@C
   SNESGetJacobian - Returns the Jacobian matrix and optionally the user 
   provided context for evaluating the Jacobian.

   Not Collective, but Mat object will be parallel if SNES object is

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  A - location to stash Jacobian matrix (or PETSC_NULL)
.  B - location to stash preconditioner matrix (or PETSC_NULL)
.  func - location to put Jacobian function (or PETSC_NULL)
-  ctx - location to stash Jacobian ctx (or PETSC_NULL)

   Level: advanced

.seealso: SNESSetJacobian(), SNESComputeJacobian()
@*/
PetscErrorCode SNESGetJacobian(SNES snes,Mat *A,Mat *B,PetscErrorCode (**func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (A)    *A    = snes->jacobian;
  if (B)    *B    = snes->jacobian_pre;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->computejacobian;
  if (ctx)  *ctx  = sdm->jacobianctx;
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp"
/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Notes:
   For basic use of the SNES solvers the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().  However, if one wishes to control this
   phase separately, SNESSetUp() should be called after SNESCreate()
   and optional routines of the form SNESSetXXX(), but before SNESSolve().

   Level: advanced

.keywords: SNES, nonlinear, setup

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
PetscErrorCode  SNESSetUp(SNES snes)
{
  PetscErrorCode ierr;
  DM             dm;
  SNESDM         sdm;
  SNESLineSearch              linesearch;
  SNESLineSearch              pclinesearch;
  void                        *lsprectx,*lspostctx;
  SNESLineSearchPreCheckFunc  lsprefunc;
  SNESLineSearchPostCheckFunc lspostfunc;
  PetscErrorCode              (*func)(SNES,Vec,Vec,void*);
  Vec                         f,fpc;
  void                        *funcctx;
  PetscErrorCode              (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
  void                        *jacctx;
  Mat                         A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (snes->setupcalled) PetscFunctionReturn(0);

  if (!((PetscObject)snes)->type_name) {
    ierr = SNESSetType(snes,SNESLS);CHKERRQ(ierr);
  }

  ierr = SNESGetFunction(snes,&snes->vec_func,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (snes->vec_func == snes->vec_sol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be function vector");
  if (snes->vec_rhs  == snes->vec_sol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be right hand side vector");

  if (!snes->vec_sol_update /* && snes->vec_sol */) {
    ierr = VecDuplicate(snes->vec_sol,&snes->vec_sol_update);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->vec_sol_update);CHKERRQ(ierr);
  }

  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESSetUpLegacy(dm);CHKERRQ(ierr); /* To be removed when function routines are taken out of the DM package */
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computefunction) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must provide a residual function with SNESSetFunction(), DMSNESSetFunction(), DMDASNESSetFunctionLocal(), etc");
  if (!snes->vec_func) {
    ierr = DMCreateGlobalVector(dm,&snes->vec_func);CHKERRQ(ierr);
  }

  if (!snes->ksp) {ierr = SNESGetKSP(snes, &snes->ksp);CHKERRQ(ierr);}

  if (!snes->linesearch) {ierr = SNESGetSNESLineSearch(snes, &snes->linesearch);}

  if (snes->ops->usercompute && !snes->user) {
    ierr = (*snes->ops->usercompute)(snes,(void**)&snes->user);CHKERRQ(ierr);
  }

  if (snes->pc) {
    /* copy the DM over */
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = SNESSetDM(snes->pc,dm);CHKERRQ(ierr);

    /* copy the legacy SNES context not related to the DM over*/
    ierr = SNESGetFunction(snes,&f,&func,&funcctx);CHKERRQ(ierr);
    ierr = VecDuplicate(f,&fpc);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes->pc,fpc,func,funcctx);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes,&A,&B,&jac,&jacctx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes->pc,A,B,jac,jacctx);CHKERRQ(ierr);
    ierr = VecDestroy(&fpc);CHKERRQ(ierr);

    /* copy the function pointers over */
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes,(PetscObject)snes->pc);CHKERRQ(ierr);

     /* default to 1 iteration */
    ierr = SNESSetTolerances(snes->pc, 0.0, 0.0, 0.0, 1, snes->pc->max_funcs);CHKERRQ(ierr);
    ierr = SNESSetNormType(snes->pc, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes->pc);CHKERRQ(ierr);

    /* copy the line search context over */
    ierr = SNESGetSNESLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESGetSNESLineSearch(snes->pc,&pclinesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPreCheck(linesearch,&lsprefunc,&lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchGetPostCheck(linesearch,&lspostfunc,&lspostctx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPreCheck(pclinesearch,lsprefunc,lsprectx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetPostCheck(pclinesearch,lspostfunc,lspostctx);CHKERRQ(ierr);
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)pclinesearch);CHKERRQ(ierr);
  }

  if (snes->ops->setup) {
    ierr = (*snes->ops->setup)(snes);CHKERRQ(ierr);
  }

  snes->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset"
/*@
   SNESReset - Resets a SNES context to the snessetupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on SNES

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Level: intermediate

   Notes: Also calls the application context destroy routine set with SNESSetComputeApplicationContext() 

.keywords: SNES, destroy

.seealso: SNESCreate(), SNESSetUp(), SNESSolve()
@*/
PetscErrorCode  SNESReset(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (snes->ops->userdestroy && snes->user) {
    ierr       = (*snes->ops->userdestroy)((void**)&snes->user);CHKERRQ(ierr);
    snes->user = PETSC_NULL;
  }
  if (snes->pc) {
    ierr = SNESReset(snes->pc);CHKERRQ(ierr);
  }

  if (snes->ops->reset) {
    ierr = (*snes->ops->reset)(snes);CHKERRQ(ierr);
  }
  if (snes->ksp) {
    ierr = KSPReset(snes->ksp);CHKERRQ(ierr);
  }

  if (snes->linesearch) {
    ierr = SNESLineSearchReset(snes->linesearch);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&snes->vec_rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->vec_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->vec_sol_update);CHKERRQ(ierr);
  ierr = VecDestroy(&snes->vec_func);CHKERRQ(ierr);
  ierr = MatDestroy(&snes->jacobian);CHKERRQ(ierr);
  ierr = MatDestroy(&snes->jacobian_pre);CHKERRQ(ierr);
  ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(snes->nvwork,&snes->vwork);CHKERRQ(ierr);
  snes->nwork = snes->nvwork = 0;
  snes->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy"
/*@
   SNESDestroy - Destroys the nonlinear solver context that was created
   with SNESCreate().

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Level: beginner

.keywords: SNES, nonlinear, destroy

.seealso: SNESCreate(), SNESSolve()
@*/
PetscErrorCode  SNESDestroy(SNES *snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*snes) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*snes),SNES_CLASSID,1);
  if (--((PetscObject)(*snes))->refct > 0) {*snes = 0; PetscFunctionReturn(0);}

  ierr = SNESReset((*snes));CHKERRQ(ierr);
  ierr = SNESDestroy(&(*snes)->pc);CHKERRQ(ierr);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish((*snes));CHKERRQ(ierr);
  if ((*snes)->ops->destroy) {ierr = (*((*snes))->ops->destroy)((*snes));CHKERRQ(ierr);}

  ierr = DMDestroy(&(*snes)->dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&(*snes)->ksp);CHKERRQ(ierr);
  ierr = SNESLineSearchDestroy(&(*snes)->linesearch);CHKERRQ(ierr);

  ierr = PetscFree((*snes)->kspconvctx);CHKERRQ(ierr);
  if ((*snes)->ops->convergeddestroy) {
    ierr = (*(*snes)->ops->convergeddestroy)((*snes)->cnvP);CHKERRQ(ierr);
  }
  if ((*snes)->conv_malloc) {
    ierr = PetscFree((*snes)->conv_hist);CHKERRQ(ierr);
    ierr = PetscFree((*snes)->conv_hist_its);CHKERRQ(ierr);
  }
  ierr = SNESMonitorCancel((*snes));CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(snes);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetLagPreconditioner"
/*@
   SNESSetLagPreconditioner - Determines when the preconditioner is rebuilt in the nonlinear solve.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys: 
.    -snes_lag_preconditioner <lag>

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1
   If  -1 is used before the very first nonlinear solve the preconditioner is still built because there is no previous preconditioner to use

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian()

@*/
PetscErrorCode  SNESSetLagPreconditioner(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (lag < -2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  if (!lag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  PetscValidLogicalCollectiveInt(snes,lag,2);
  snes->lagpreconditioner = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetGridSequence"
/*@
   SNESSetGridSequence - sets the number of steps of grid sequencing that SNES does

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  steps - the number of refinements to do, defaults to 0

   Options Database Keys: 
.    -snes_grid_sequence <steps>

   Level: intermediate

   Notes:
   Use SNESGetSolution() to extract the fine grid solution after grid sequencing.

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian()

@*/
PetscErrorCode  SNESSetGridSequence(SNES snes,PetscInt steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveInt(snes,steps,2);
  snes->gridsequence = steps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLagPreconditioner"
/*@
   SNESGetLagPreconditioner - Indicates how often the preconditioner is rebuilt

   Not Collective

   Input Parameter:
.  snes - the SNES context
 
   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys: 
.    -snes_lag_preconditioner <lag>

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagPreconditioner()

@*/
PetscErrorCode  SNESGetLagPreconditioner(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *lag = snes->lagpreconditioner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetLagJacobian"
/*@
   SNESSetLagJacobian - Determines when the Jacobian is rebuilt in the nonlinear solve. See SNESSetLagPreconditioner() for determining how
     often the preconditioner is rebuilt.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 means rebuild at next chance but then never again

   Options Database Keys: 
.    -snes_lag_jacobian <lag>

   Notes:
   The default is 1
   The Jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1
   If  -1 is used before the very first nonlinear solve the CODE WILL FAIL! because no Jacobian is used, use -2 to indicate you want it recomputed
   at the next Newton step but never again (unless it is reset to another value)

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagPreconditioner(), SNESGetLagJacobian()

@*/
PetscErrorCode  SNESSetLagJacobian(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (lag < -2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  if (!lag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  PetscValidLogicalCollectiveInt(snes,lag,2);
  snes->lagjacobian = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLagJacobian"
/*@
   SNESGetLagJacobian - Indicates how often the Jacobian is rebuilt. See SNESGetLagPreconditioner() to determine when the preconditioner is rebuilt

   Not Collective

   Input Parameter:
.  snes - the SNES context
 
   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc.

   Options Database Keys: 
.    -snes_lag_jacobian <lag>

   Notes:
   The default is 1
   The jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagJacobian(), SNESSetLagPreconditioner(), SNESGetLagPreconditioner()

@*/
PetscErrorCode  SNESGetLagJacobian(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *lag = snes->lagjacobian;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetTolerances"
/*@
   SNESSetTolerances - Sets various parameters used in convergence tests.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  abstol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations

   Options Database Keys: 
+    -snes_atol <abstol> - Sets abstol
.    -snes_rtol <rtol> - Sets rtol
.    -snes_stol <stol> - Sets stol
.    -snes_max_it <maxit> - Sets maxit
-    -snes_max_funcs <maxf> - Sets maxf

   Notes:
   The default maximum number of iterations is 50.
   The default maximum number of function evaluations is 1000.

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance()
@*/
PetscErrorCode  SNESSetTolerances(SNES snes,PetscReal abstol,PetscReal rtol,PetscReal stol,PetscInt maxit,PetscInt maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,abstol,2);
  PetscValidLogicalCollectiveReal(snes,rtol,3);
  PetscValidLogicalCollectiveReal(snes,stol,4);
  PetscValidLogicalCollectiveInt(snes,maxit,5);
  PetscValidLogicalCollectiveInt(snes,maxf,6);

  if (abstol != PETSC_DEFAULT) {
    if (abstol < 0.0) SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %G must be non-negative",abstol);
    snes->abstol = abstol;
  }
  if (rtol != PETSC_DEFAULT) {
    if (rtol < 0.0 || 1.0 <= rtol) SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %G must be non-negative and less than 1.0",rtol);
    snes->rtol = rtol;
  }
  if (stol != PETSC_DEFAULT) {
    if (stol < 0.0) SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Step tolerance %G must be non-negative",stol);
    snes->stol = stol;
  }
  if (maxit != PETSC_DEFAULT) {
    if (maxit < 0) SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxit);
    snes->max_its = maxit;
  }
  if (maxf != PETSC_DEFAULT) {
    if (maxf < 0) SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of function evaluations %D must be non-negative",maxf);
    snes->max_funcs = maxf;
  }
  snes->tolerancesset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetTolerances"
/*@
   SNESGetTolerances - Gets various parameters used in convergence tests.

   Not Collective

   Input Parameters:
+  snes - the SNES context
.  atol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.keywords: SNES, nonlinear, get, convergence, tolerances

.seealso: SNESSetTolerances()
@*/
PetscErrorCode  SNESGetTolerances(SNES snes,PetscReal *atol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit,PetscInt *maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (atol)  *atol  = snes->abstol;
  if (rtol)  *rtol  = snes->rtol;
  if (stol)  *stol  = snes->stol;
  if (maxit) *maxit = snes->max_its;
  if (maxf)  *maxf  = snes->max_funcs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetTrustRegionTolerance"
/*@
   SNESSetTrustRegionTolerance - Sets the trust region parameter tolerance.  

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  tol - tolerance
   
   Options Database Key: 
.  -snes_trtol <tol> - Sets tol

   Level: intermediate

.keywords: SNES, nonlinear, set, trust region, tolerance

.seealso: SNESSetTolerances()
@*/
PetscErrorCode  SNESSetTrustRegionTolerance(SNES snes,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,tol,2);
  snes->deltatol = tol;
  PetscFunctionReturn(0);
}

/* 
   Duplicate the lg monitors for SNES from KSP; for some reason with 
   dynamic libraries things don't work under Sun4 if we just use 
   macros instead of functions
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLG"
PetscErrorCode  SNESMonitorLG(SNES snes,PetscInt it,PetscReal norm,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = KSPMonitorLG((KSP)snes,it,norm,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGCreate"
PetscErrorCode  SNESMonitorLGCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGCreate(host,label,x,y,m,n,draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGDestroy"
PetscErrorCode  SNESMonitorLGDestroy(PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode  SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);
#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRange"
PetscErrorCode  SNESMonitorLGRange(SNES snes,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscErrorCode   ierr;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;
  PetscFunctionBegin;
  if (!monctx) {
    MPI_Comm    comm;

    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    v      = PETSC_VIEWER_DRAW_(comm);
  }
  ierr   = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr   = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr   = PetscDrawSetTitle(draw,"Residual norm");CHKERRQ(ierr);
  x = (PetscReal) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,1,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"% elemts > .2*max elemt");CHKERRQ(ierr);
  ierr =  SNESMonitorRange_Private(snes,n,&per);CHKERRQ(ierr);
  x = (PetscReal) n;
  y = 100.0*per;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,2,&lg);CHKERRQ(ierr);
  if (!n) {prev = rnorm;ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/prev;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,3,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/(prev*per);
  if (n > 2) { /*skip initial crazy value */
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  }
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }
  prev = rnorm;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRangeCreate"
PetscErrorCode  SNESMonitorLGRangeCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGCreate(host,label,x,y,m,n,draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRangeDestroy"
PetscErrorCode  SNESMonitorLGRangeDestroy(PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMonitor"
/*@
   SNESMonitor - runs the user provided monitor routines, if they exist

   Collective on SNES

   Input Parameters:
+  snes - nonlinear solver context obtained from SNESCreate()
.  iter - iteration number
-  rnorm - relative norm of the residual

   Notes:
   This routine is called by the SNES implementations.
   It does not typically need to be called by the user.

   Level: developer

.seealso: SNESMonitorSet()
@*/
PetscErrorCode  SNESMonitor(SNES snes,PetscInt iter,PetscReal rnorm)
{
  PetscErrorCode ierr;
  PetscInt       i,n = snes->numbermonitors;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = (*snes->monitor[i])(snes,iter,rnorm,snes->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorSet"
/*@C
   SNESMonitorSet - Sets an ADDITIONAL function that is to be used at every
   iteration of the nonlinear solver to display the iteration's 
   progress.   

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
          monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling sequence of func:
$     int func(SNES snes,PetscInt its, PetscReal norm,void *mctx)

+    snes - the SNES context
.    its - iteration number
.    norm - 2-norm function value (may be estimated)
-    mctx - [optional] monitoring context

   Options Database Keys:
+    -snes_monitor        - sets SNESMonitorDefault()
.    -snes_monitor_draw    - sets line graph monitor,
                            uses SNESMonitorLGCreate()
-    -snes_monitor_cancel - cancels all monitors that have
                            been hardwired into a code by 
                            calls to SNESMonitorSet(), but
                            does not cancel those set via
                            the options database.

   Notes: 
   Several different monitoring routines may be set by calling
   SNESMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Fortran notes: Only a single monitor function can be set for each SNES object

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESMonitorDefault(), SNESMonitorCancel()
@*/
PetscErrorCode  SNESMonitorSet(SNES snes,PetscErrorCode (*monitor)(SNES,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (snes->numbermonitors >= MAXSNESMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  for (i=0; i<snes->numbermonitors;i++) {
    if (monitor == snes->monitor[i] && monitordestroy == snes->monitordestroy[i] && mctx == snes->monitorcontext[i]) {
      if (monitordestroy) {
        ierr = (*monitordestroy)(&mctx);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
  }
  snes->monitor[snes->numbermonitors]           = monitor;
  snes->monitordestroy[snes->numbermonitors]    = monitordestroy;
  snes->monitorcontext[snes->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorCancel"
/*@C
   SNESMonitorCancel - Clears all the monitor functions for a SNES object.

   Logically Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Options Database Key:
.  -snes_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to SNESMonitorSet(), but does not cancel those 
    set via the options database

   Notes: 
   There is no way to clear one specific monitor from a SNES object.

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESMonitorDefault(), SNESMonitorSet()
@*/
PetscErrorCode  SNESMonitorCancel(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  for (i=0; i<snes->numbermonitors; i++) {
    if (snes->monitordestroy[i]) {
      ierr = (*snes->monitordestroy[i])(&snes->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  snes->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetConvergenceTest"
/*@C
   SNESSetConvergenceTest - Sets the function that is to be used 
   to test for convergence of the nonlinear iterative solution.   

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - routine to test for convergence
.  cctx - [optional] context for private data for the convergence routine  (may be PETSC_NULL)
-  destroy - [optional] destructor for the context (may be PETSC_NULL; PETSC_NULL_FUNCTION in Fortran)

   Calling sequence of func:
$     PetscErrorCode func (SNES snes,PetscInt it,PetscReal xnorm,PetscReal gnorm,PetscReal f,SNESConvergedReason *reason,void *cctx)

+    snes - the SNES context
.    it - current iteration (0 is the first and is before any Newton step)
.    cctx - [optional] convergence context
.    reason - reason for convergence/divergence
.    xnorm - 2-norm of current iterate
.    gnorm - 2-norm of current step
-    f - 2-norm of function

   Level: advanced

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESDefaultConverged(), SNESSkipConverged()
@*/
PetscErrorCode  SNESSetConvergenceTest(SNES snes,PetscErrorCode (*func)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void *cctx,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!func) func = SNESSkipConverged;
  if (snes->ops->convergeddestroy) {
    ierr = (*snes->ops->convergeddestroy)(snes->cnvP);CHKERRQ(ierr);
  }
  snes->ops->converged        = func;
  snes->ops->convergeddestroy = destroy;
  snes->cnvP                  = cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergedReason"
/*@
   SNESGetConvergedReason - Gets the reason the SNES iteration was stopped.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see SNESConvergedReason or the 
            manual pages for the individual convergence tests for complete lists

   Level: intermediate

   Notes: Can only be called after the call the SNESSolve() is complete.

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESSetConvergenceTest(), SNESConvergedReason
@*/
PetscErrorCode  SNESGetConvergedReason(SNES snes,SNESConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = snes->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetConvergenceHistory"
/*@
   SNESSetConvergenceHistory - Sets the array used to hold the convergence history.

   Logically Collective on SNES

   Input Parameters:
+  snes - iterative context obtained from SNESCreate()
.  a   - array to hold history, this array will contain the function norms computed at each step
.  its - integer array holds the number of linear iterations for each solve.
.  na  - size of a and its
-  reset - PETSC_TRUE indicates each new nonlinear solve resets the history counter to zero,
           else it continues storing new values for new nonlinear solves after the old ones

   Notes:
   If 'a' and 'its' are PETSC_NULL then space is allocated for the history. If 'na' PETSC_DECIDE or PETSC_DEFAULT then a
   default array of length 10000 is allocated.

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: SNES, set, convergence, history

.seealso: SNESGetConvergenceHistory()

@*/
PetscErrorCode  SNESSetConvergenceHistory(SNES snes,PetscReal a[],PetscInt its[],PetscInt na,PetscBool  reset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (na)  PetscValidScalarPointer(a,2);
  if (its) PetscValidIntPointer(its,3);
  if (na == PETSC_DECIDE || na == PETSC_DEFAULT || !a) {
    if (na == PETSC_DECIDE || na == PETSC_DEFAULT) na = 1000;
    ierr = PetscMalloc(na*sizeof(PetscReal),&a);CHKERRQ(ierr);
    ierr = PetscMalloc(na*sizeof(PetscInt),&its);CHKERRQ(ierr);
    snes->conv_malloc   = PETSC_TRUE;
  }
  snes->conv_hist       = a;
  snes->conv_hist_its   = its;
  snes->conv_hist_max   = na;
  snes->conv_hist_len   = 0;
  snes->conv_hist_reset = reset;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergenceHistoryMatlab"
mxArray *SNESGetConvergenceHistoryMatlab(SNES snes)
{
  mxArray        *mat;
  PetscInt       i;
  PetscReal      *ar;

  PetscFunctionBegin;
  mat  = mxCreateDoubleMatrix(snes->conv_hist_len,1,mxREAL);
  ar   = (PetscReal*) mxGetData(mat);
  for (i=0; i<snes->conv_hist_len; i++) {
    ar[i] = snes->conv_hist[i];
  }
  PetscFunctionReturn(mat);
}
EXTERN_C_END
#endif


#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergenceHistory"
/*@C
   SNESGetConvergenceHistory - Gets the array used to hold the convergence history.

   Not Collective

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Output Parameters:
.  a   - array to hold history
.  its - integer array holds the number of linear iterations (or
         negative if not converged) for each solve.
-  na  - size of a and its

   Notes:
    The calling sequence for this routine in Fortran is
$   call SNESGetConvergenceHistory(SNES snes, integer na, integer ierr)

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: SNES, get, convergence, history

.seealso: SNESSetConvergencHistory()

@*/
PetscErrorCode  SNESGetConvergenceHistory(SNES snes,PetscReal *a[],PetscInt *its[],PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (a)   *a   = snes->conv_hist;
  if (its) *its = snes->conv_hist_its;
  if (na)  *na  = snes->conv_hist_len;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUpdate"
/*@C
  SNESSetUpdate - Sets the general-purpose update function called
  at the beginning of every iteration of the nonlinear solve. Specifically
  it is called just before the Jacobian is "evaluated".

  Logically Collective on SNES

  Input Parameters:
. snes - The nonlinear solver context
. func - The function

  Calling sequence of func:
. func (SNES snes, PetscInt step);

. step - The current step of the iteration

  Level: advanced

  Note: This is NOT what one uses to update the ghost points before a function evaluation, that should be done at the beginning of your FormFunction()
        This is not used by most users.

.keywords: SNES, update

.seealso SNESDefaultUpdate(), SNESSetJacobian(), SNESSolve()
@*/
PetscErrorCode  SNESSetUpdate(SNES snes, PetscErrorCode (*func)(SNES, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID,1);
  snes->ops->update = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultUpdate"
/*@
  SNESDefaultUpdate - The default update function which does nothing.

  Not collective

  Input Parameters:
. snes - The nonlinear solver context
. step - The current step of the iteration

  Level: intermediate

.keywords: SNES, update
.seealso SNESSetUpdate(), SNESDefaultRhsBC(), SNESDefaultShortolutionBC()
@*/
PetscErrorCode  SNESDefaultUpdate(SNES snes, PetscInt step)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESScaleStep_Private"
/*
   SNESScaleStep_Private - Scales a step so that its length is less than the
   positive parameter delta.

    Input Parameters:
+   snes - the SNES context
.   y - approximate solution of linear system
.   fnorm - 2-norm of current function
-   delta - trust region size

    Output Parameters:
+   gpnorm - predicted function norm at the new point, assuming local 
    linearization.  The value is zero if the step lies within the trust 
    region, and exceeds zero otherwise.
-   ynorm - 2-norm of the step

    Note:
    For non-trust region methods such as SNESLS, the parameter delta 
    is set to be the maximum allowable step size.  

.keywords: SNES, nonlinear, scale, step
*/
PetscErrorCode SNESScaleStep_Private(SNES snes,Vec y,PetscReal *fnorm,PetscReal *delta,PetscReal *gpnorm,PetscReal *ynorm)
{
  PetscReal      nrm;
  PetscScalar    cnorm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscCheckSameComm(snes,1,y,2);

  ierr = VecNorm(y,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > *delta) {
     nrm = *delta/nrm;
     *gpnorm = (1.0 - nrm)*(*fnorm);
     cnorm = nrm;
     ierr = VecScale(y,cnorm);CHKERRQ(ierr);
     *ynorm = *delta;
  } else {
     *gpnorm = 0.0;
     *ynorm = nrm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSolve"
/*@C
   SNESSolve - Solves a nonlinear system F(x) = b.
   Call SNESSolve() after calling SNESCreate() and optional routines of the form SNESSetXXX().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  b - the constant part of the equation F(x) = b, or PETSC_NULL to use zero.
-  x - the solution vector.

   Notes:
   The user should initialize the vector,x, with the initial guess
   for the nonlinear solve prior to calling SNESSolve.  In particular,
   to employ an initial guess of zero, the user should explicitly set
   this vector to zero by calling VecSet().

   Level: beginner

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESDestroy(), SNESSetFunction(), SNESSetJacobian(), SNESSetGridSequence(), SNESGetSolution()
@*/
PetscErrorCode  SNESSolve(SNES snes,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscInt       grid;
  Vec            xcreated = PETSC_NULL;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (x) PetscCheckSameComm(snes,1,x,3);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (b) PetscCheckSameComm(snes,1,b,2);

  if (!x) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&xcreated);CHKERRQ(ierr);
    x    = xcreated;
  }

  for (grid=0; grid<snes->gridsequence; grid++) {ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm));CHKERRQ(ierr);}
  for (grid=0; grid<snes->gridsequence+1; grid++) {

    /* set solution vector */
    if (!grid) {ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);}
    ierr = VecDestroy(&snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol = x;
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

    /* set affine vector if provided */
    if (b) { ierr = PetscObjectReference((PetscObject)b);CHKERRQ(ierr); }
    ierr = VecDestroy(&snes->vec_rhs);CHKERRQ(ierr);
    snes->vec_rhs = b;

    ierr = SNESSetUp(snes);CHKERRQ(ierr);

    if (!grid) {
      if (snes->ops->computeinitialguess) {
        ierr = (*snes->ops->computeinitialguess)(snes,snes->vec_sol,snes->initialguessP);CHKERRQ(ierr);
      } else if (snes->dm) {
        PetscBool ig;
        ierr = DMHasInitialGuess(snes->dm,&ig);CHKERRQ(ierr);
        if (ig) {
          ierr = DMComputeInitialGuess(snes->dm,snes->vec_sol);CHKERRQ(ierr);
        }
      }
    }

    if (snes->conv_hist_reset) snes->conv_hist_len = 0;
    snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;

    ierr = PetscLogEventBegin(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
    ierr = (*snes->ops->solve)(snes);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
    if (snes->domainerror){
      snes->reason      = SNES_DIVERGED_FUNCTION_DOMAIN;
      snes->domainerror = PETSC_FALSE;
    }
    if (!snes->reason) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(((PetscObject)snes)->prefix,"-snes_test_local_min",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg && !PetscPreLoadingOn) { ierr = SNESTestLocalMin(snes);CHKERRQ(ierr); }
    if (snes->printreason) {
      ierr = PetscViewerASCIIAddTab(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm),((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      if (snes->reason > 0) {
        ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm),"Nonlinear solve converged due to %s iterations %D\n",SNESConvergedReasons[snes->reason],snes->iter);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm),"Nonlinear solve did not converge due to %s iterations %D\n",SNESConvergedReasons[snes->reason],snes->iter);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISubtractTab(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm),((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }

    if (snes->errorifnotconverged && snes->reason < 0) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_NOT_CONVERGED,"SNESSolve has not converged");
    if (grid <  snes->gridsequence) {
      DM  fine;
      Vec xnew;
      Mat interp;

      ierr = DMRefine(snes->dm,((PetscObject)snes)->comm,&fine);CHKERRQ(ierr);
      if (!fine) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_INCOMP,"DMRefine() did not perform any refinement, cannot continue grid sequencing");
      ierr = DMCreateInterpolation(snes->dm,fine,&interp,PETSC_NULL);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(fine,&xnew);CHKERRQ(ierr);
      ierr = MatInterpolate(interp,x,xnew);CHKERRQ(ierr);
      ierr = DMInterpolate(snes->dm,interp,fine);CHKERRQ(ierr);
      ierr = MatDestroy(&interp);CHKERRQ(ierr);
      x    = xnew;

      ierr = SNESReset(snes);CHKERRQ(ierr);
      ierr = SNESSetDM(snes,fine);CHKERRQ(ierr);
      ierr = DMDestroy(&fine);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm));CHKERRQ(ierr);
    }
  }
  /* monitoring and viewing */
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetString(((PetscObject)snes)->prefix,"-snes_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = SNESView(snes,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetString(((PetscObject)snes)->prefix,"-snes_view_solution_vtk",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerCreate(((PetscObject)snes)->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
    ierr = VecView(snes->vec_sol,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&xcreated);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------- Internal routines for SNES Package --------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetType"
/*@C
   SNESSetType - Sets the method for the nonlinear solver.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  type - a known method

   Options Database Key:
.  -snes_type <type> - Sets the method; use -help for a list
   of available methods (for instance, ls or tr)

   Notes:
   See "petsc/include/petscsnes.h" for available methods (for instance)
+    SNESLS - Newton's method with line search
     (systems of nonlinear equations)
.    SNESTR - Newton's method with trust region
     (systems of nonlinear equations)

  Normally, it is best to use the SNESSetFromOptions() command and then
  set the SNES solver type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many nonlinear solvers.
  The SNESSetType() routine is provided for those situations where it
  is necessary to set the nonlinear solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of solver changes during the execution of the program,
  and the user's application is taking responsibility for choosing the
  appropriate method.

  Level: intermediate

.keywords: SNES, set, type

.seealso: SNESType, SNESCreate()

@*/
PetscErrorCode  SNESSetType(SNES snes,const SNESType type)
{
  PetscErrorCode ierr,(*r)(SNES);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)snes,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(SNESList,((PetscObject)snes)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SNES type %s",type);
  /* Destroy the previous private SNES context */
  if (snes->ops->destroy) {
    ierr = (*(snes)->ops->destroy)(snes);CHKERRQ(ierr);
    snes->ops->destroy = PETSC_NULL;
  }
  /* Reinitialize function pointers in SNESOps structure */
  snes->ops->setup          = 0;
  snes->ops->solve          = 0;
  snes->ops->view           = 0;
  snes->ops->setfromoptions = 0;
  snes->ops->destroy        = 0;
  /* Call the SNESCreate_XXX routine for this particular Nonlinear solver */
  snes->setupcalled = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)snes,type);CHKERRQ(ierr);
  ierr = (*r)(snes);CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
  if (PetscAMSPublishAll) {
    ierr = PetscObjectAMSPublish((PetscObject)snes);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0); 
}


/* --------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterDestroy"
/*@
   SNESRegisterDestroy - Frees the list of nonlinear solvers that were
   registered by SNESRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: SNES, nonlinear, register, destroy

.seealso: SNESRegisterAll(), SNESRegisterAll()
@*/
PetscErrorCode  SNESRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&SNESList);CHKERRQ(ierr);
  SNESRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetType"
/*@C
   SNESGetType - Gets the SNES method type and name (as a string).

   Not Collective

   Input Parameter:
.  snes - nonlinear solver context

   Output Parameter:
.  type - SNES method (a character string)

   Level: intermediate

.keywords: SNES, nonlinear, get, type, name
@*/
PetscErrorCode  SNESGetType(SNES snes,const SNESType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)snes)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolution"
/*@
   SNESGetSolution - Returns the vector where the approximate solution is
   stored. This is the fine grid solution when using SNESSetGridSequence().

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

   Level: intermediate

.keywords: SNES, nonlinear, get, solution

.seealso:  SNESGetSolutionUpdate(), SNESGetFunction()
@*/
PetscErrorCode  SNESGetSolution(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolutionUpdate"
/*@
   SNESGetSolutionUpdate - Returns the vector where the solution update is
   stored. 

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution update

   Level: advanced

.keywords: SNES, nonlinear, get, solution, update

.seealso: SNESGetSolution(), SNESGetFunction()
@*/
PetscErrorCode  SNESGetSolutionUpdate(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol_update;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetFunction"
/*@C
   SNESGetFunction - Returns the vector where the function is stored.

   Not Collective, but Vec is parallel if SNES is parallel. Collective if Vec is requested, but has not been created yet.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
+  r - the function (or PETSC_NULL)
.  func - the function (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, function

.seealso: SNESSetFunction(), SNESGetSolution()
@*/
PetscErrorCode  SNESGetFunction(SNES snes,Vec *r,PetscErrorCode (**func)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (r) {
    if (!snes->vec_func) {
      if (snes->vec_rhs) {
        ierr = VecDuplicate(snes->vec_rhs,&snes->vec_func);CHKERRQ(ierr);
      } else if (snes->vec_sol) {
        ierr = VecDuplicate(snes->vec_sol,&snes->vec_func);CHKERRQ(ierr);
      } else if (snes->dm) {
        ierr = DMCreateGlobalVector(snes->dm,&snes->vec_func);CHKERRQ(ierr);
      }
    }
    *r = snes->vec_func;
  }
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetFunction(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESGetGS - Returns the GS function and context.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
+  gsfunc - the function (or PETSC_NULL)
-  ctx    - the function context (or PETSC_NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, function

.seealso: SNESSetGS(), SNESGetFunction()
@*/

#undef __FUNCT__
#define __FUNCT__ "SNESGetGS"
PetscErrorCode SNESGetGS (SNES snes, PetscErrorCode(**func)(SNES, Vec, Vec, void*), void ** ctx) 
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMSNESGetGS(dm,func,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetOptionsPrefix"
/*@C
   SNESSetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Logically Collective on SNES

   Input Parameter:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: SNES, set, options, prefix, database

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode  SNESSetOptionsPrefix(SNES snes,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
  if (snes->linesearch) {
    ierr = SNESGetSNESLineSearch(snes,&snes->linesearch);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)snes->linesearch,prefix);CHKERRQ(ierr);
  }
  ierr = KSPSetOptionsPrefix(snes->ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESAppendOptionsPrefix"
/*@C
   SNESAppendOptionsPrefix - Appends to the prefix used for searching for all 
   SNES options in the database.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: SNES, append, options, prefix, database

.seealso: SNESGetOptionsPrefix()
@*/
PetscErrorCode  SNESAppendOptionsPrefix(SNES snes,const char prefix[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
  if (snes->linesearch) {
    ierr = SNESGetSNESLineSearch(snes,&snes->linesearch);CHKERRQ(ierr);
    ierr = PetscObjectAppendOptionsPrefix((PetscObject)snes->linesearch,prefix);CHKERRQ(ierr);
  }
  ierr = KSPAppendOptionsPrefix(snes->ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetOptionsPrefix"
/*@C
   SNESGetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: SNES, get, options, prefix, database

.seealso: SNESAppendOptionsPrefix()
@*/
PetscErrorCode  SNESGetOptionsPrefix(SNES snes,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESRegister"
/*@C
  SNESRegister - See SNESRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  SNESRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(SNES))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&SNESList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESTestLocalMin"
PetscErrorCode  SNESTestLocalMin(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       N,i,j;
  Vec            u,uh,fh;
  PetscScalar    value;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uh);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&fh);CHKERRQ(ierr);

  /* currently only works for sequential */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Testing FormFunction() for local min\n");
  ierr = VecGetSize(u,&N);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    ierr = VecCopy(u,uh);CHKERRQ(ierr);
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"i = %D\n",i);CHKERRQ(ierr);
    for (j=-10; j<11; j++) {
      value = PetscSign(j)*exp(PetscAbs(j)-10.0);
      ierr  = VecSetValue(uh,i,value,ADD_VALUES);CHKERRQ(ierr);
      ierr  = SNESComputeFunction(snes,uh,fh);CHKERRQ(ierr);
      ierr  = VecNorm(fh,NORM_2,&norm);CHKERRQ(ierr);
      ierr  = PetscPrintf(PETSC_COMM_WORLD,"       j norm %D %18.16e\n",j,norm);CHKERRQ(ierr);
      value = -value;
      ierr  = VecSetValue(uh,i,value,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&uh);CHKERRQ(ierr);
  ierr = VecDestroy(&fh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPSetUseEW"
/*@
   SNESKSPSetUseEW - Sets SNES use Eisenstat-Walker method for
   computing relative tolerance for linear solvers within an inexact
   Newton method.

   Logically Collective on SNES

   Input Parameters:
+  snes - SNES context
-  flag - PETSC_TRUE or PETSC_FALSE

    Options Database:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2 
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton

.seealso: SNESKSPGetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPSetUseEW(SNES snes,PetscBool  flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveBool(snes,flag,2);
  snes->ksp_ewconv = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPGetUseEW"
/*@
   SNESKSPGetUseEW - Gets if SNES is using Eisenstat-Walker method
   for computing relative tolerance for linear solvers within an
   inexact Newton method.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  flag - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton

.seealso: SNESKSPSetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPGetUseEW(SNES snes, PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = snes->ksp_ewconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPSetParametersEW"
/*@
   SNESKSPSetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Logically Collective on SNES
 
   Input Parameters:
+    snes - SNES context
.    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation
             (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Note:
   Version 3 was contributed by Luis Chacon, June 2006.

   Use PETSC_DEFAULT to retain the default for any of the parameters.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", Utah State University Math. Stat. Dept. Res. 
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput. 

.keywords: SNES, KSP, Eisenstat, Walker, set, parameters

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPGetParametersEW()
@*/
PetscErrorCode  SNESKSPSetParametersEW(SNES snes,PetscInt version,PetscReal rtol_0,PetscReal rtol_max,
							    PetscReal gamma,PetscReal alpha,PetscReal alpha2,PetscReal threshold)
{
  SNESKSPEW *kctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  if (!kctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  PetscValidLogicalCollectiveInt(snes,version,2);
  PetscValidLogicalCollectiveReal(snes,rtol_0,3);
  PetscValidLogicalCollectiveReal(snes,rtol_max,4);
  PetscValidLogicalCollectiveReal(snes,gamma,5);
  PetscValidLogicalCollectiveReal(snes,alpha,6);
  PetscValidLogicalCollectiveReal(snes,alpha2,7);
  PetscValidLogicalCollectiveReal(snes,threshold,8);

  if (version != PETSC_DEFAULT)   kctx->version   = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0    = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max  = rtol_max;
  if (gamma != PETSC_DEFAULT)     kctx->gamma     = gamma;
  if (alpha != PETSC_DEFAULT)     kctx->alpha     = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2    = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;
  
  if (kctx->version < 1 || kctx->version > 3) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 and 3 are supported: %D",kctx->version);
  }
  if (kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_0 < 1.0: %G",kctx->rtol_0);
  }
  if (kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_max (%G) < 1.0\n",kctx->rtol_max);
  }
  if (kctx->gamma < 0.0 || kctx->gamma > 1.0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= gamma (%G) <= 1.0\n",kctx->gamma);
  }
  if (kctx->alpha <= 1.0 || kctx->alpha > 2.0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"1.0 < alpha (%G) <= 2.0\n",kctx->alpha);
  }
  if (kctx->threshold <= 0.0 || kctx->threshold >= 1.0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"0.0 < threshold (%G) < 1.0\n",kctx->threshold);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPGetParametersEW"
/*@
   SNESKSPGetParametersEW - Gets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Not Collective
 
   Input Parameters:
     snes - SNES context

   Output Parameters:
+    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation
             (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Level: advanced

.keywords: SNES, KSP, Eisenstat, Walker, get, parameters

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode  SNESKSPGetParametersEW(SNES snes,PetscInt *version,PetscReal *rtol_0,PetscReal *rtol_max,
							    PetscReal *gamma,PetscReal *alpha,PetscReal *alpha2,PetscReal *threshold)
{
  SNESKSPEW *kctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  if (!kctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  if(version)   *version   = kctx->version;
  if(rtol_0)    *rtol_0    = kctx->rtol_0;
  if(rtol_max)  *rtol_max  = kctx->rtol_max;
  if(gamma)     *gamma     = kctx->gamma;
  if(alpha)     *alpha     = kctx->alpha;
  if(alpha2)    *alpha2    = kctx->alpha2;
  if(threshold) *threshold = kctx->threshold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPEW_PreSolve"
static PetscErrorCode SNESKSPEW_PreSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PetscReal      rtol=PETSC_DEFAULT,stol;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context exists");
  if (!snes->iter) { /* first time in, so use the original user rtol */
    rtol = kctx->rtol_0;
  } else {
    if (kctx->version == 1) {
      rtol = (snes->norm - kctx->lresid_last)/kctx->norm_last;
      if (rtol < 0.0) rtol = -rtol;
      stol = pow(kctx->rtol_last,kctx->alpha2);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 2) {
      rtol = kctx->gamma * pow(snes->norm/kctx->norm_last,kctx->alpha);
      stol = kctx->gamma * pow(kctx->rtol_last,kctx->alpha);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 3) {/* contributed by Luis Chacon, June 2006. */
      rtol = kctx->gamma * pow(snes->norm/kctx->norm_last,kctx->alpha);
      /* safeguard: avoid sharp decrease of rtol */
      stol = kctx->gamma*pow(kctx->rtol_last,kctx->alpha);
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
      /* safeguard: avoid oversolving */
      stol = kctx->gamma*(snes->ttol)/snes->norm;
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 or 3 are supported: %D",kctx->version);
  }
  /* safeguard: avoid rtol greater than one */
  rtol = PetscMin(rtol,kctx->rtol_max);
  ierr = KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PetscInfo3(snes,"iter %D, Eisenstat-Walker (version %D) KSP rtol=%G\n",snes->iter,kctx->version,rtol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPEW_PostSolve"
static PetscErrorCode SNESKSPEW_PostSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PCSide         pcside;
  Vec            lres;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context exists");
  ierr = KSPGetTolerances(ksp,&kctx->rtol_last,0,0,0);CHKERRQ(ierr);
  ierr = SNESGetFunctionNorm(snes,&kctx->norm_last);CHKERRQ(ierr);
  if (kctx->version == 1) {
    ierr = KSPGetPCSide(ksp,&pcside);CHKERRQ(ierr);
    if (pcside == PC_RIGHT) { /* XXX Should we also test KSP_UNPRECONDITIONED_NORM ? */
      /* KSP residual is true linear residual */
      ierr = KSPGetResidualNorm(ksp,&kctx->lresid_last);CHKERRQ(ierr);
    } else {
      /* KSP residual is preconditioned residual */
      /* compute true linear residual norm */
      ierr = VecDuplicate(b,&lres);CHKERRQ(ierr);
      ierr = MatMult(snes->jacobian,x,lres);CHKERRQ(ierr);
      ierr = VecAYPX(lres,-1.0,b);CHKERRQ(ierr);
      ierr = VecNorm(lres,NORM_2,&kctx->lresid_last);CHKERRQ(ierr);
      ierr = VecDestroy(&lres);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNES_KSPSolve"
PetscErrorCode SNES_KSPSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->ksp_ewconv) { ierr = SNESKSPEW_PreSolve(snes,ksp,b,x);CHKERRQ(ierr);  }
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  if (snes->ksp_ewconv) { ierr = SNESKSPEW_PostSolve(snes,ksp,b,x);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetDM"
/*@
   SNESSetDM - Sets the DM that may be used by some preconditioners

   Logically Collective on SNES

   Input Parameters:
+  snes - the preconditioner context
-  dm - the dm

   Level: intermediate


.seealso: SNESGetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  SNESSetDM(SNES snes,DM dm)
{
  PetscErrorCode ierr;
  KSP            ksp;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (dm) {ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);}
  if (snes->dm) {               /* Move the SNESDM context over to the new DM unless the new DM already has one */
    PetscContainer oldcontainer,container;
    ierr = PetscObjectQuery((PetscObject)snes->dm,"SNESDM",(PetscObject*)&oldcontainer);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)dm,"SNESDM",(PetscObject*)&container);CHKERRQ(ierr);
    if (oldcontainer && !container) {
      ierr = DMSNESCopyContext(snes->dm,dm);CHKERRQ(ierr);
      ierr = DMSNESGetContext(snes->dm,&sdm);CHKERRQ(ierr);
      if (sdm->originaldm == snes->dm) { /* Grant write privileges to the replacement DM */
        sdm->originaldm = dm;
      }
    }
    ierr = DMDestroy(&snes->dm);CHKERRQ(ierr);
  }
  snes->dm = dm;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_FALSE);CHKERRQ(ierr);
  if (snes->pc) {
    ierr = SNESSetDM(snes->pc, snes->dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetDM"
/*@
   SNESGetDM - Gets the DM that may be used by some preconditioners

   Not Collective but DM obtained is parallel on SNES

   Input Parameter:
. snes - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate


.seealso: SNESSetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  SNESGetDM(SNES snes,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (!snes->dm) {
    ierr = DMShellCreate(((PetscObject)snes)->comm,&snes->dm);CHKERRQ(ierr);
  }
  *dm = snes->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetPC"
/*@
  SNESSetPC - Sets the nonlinear preconditioner to be used.

  Collective on SNES

  Input Parameters:
+ snes - iterative context obtained from SNESCreate()
- pc   - the preconditioner object

  Notes:
  Use SNESGetPC() to retrieve the preconditioner context (for example,
  to configure it using the API).

  Level: developer

.keywords: SNES, set, precondition
.seealso: SNESGetPC()
@*/
PetscErrorCode SNESSetPC(SNES snes, SNES pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(pc, SNES_CLASSID, 2);
  PetscCheckSameComm(snes, 1, pc, 2);
  ierr = PetscObjectReference((PetscObject) pc);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes->pc);CHKERRQ(ierr);
  snes->pc = pc;
  ierr = PetscLogObjectParent(snes, snes->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetPC"
/*@
  SNESGetPC - Returns a pointer to the nonlinear preconditioning context set with SNESSetPC().

  Not Collective

  Input Parameter:
. snes - iterative context obtained from SNESCreate()

  Output Parameter:
. pc - preconditioner context

  Level: developer

.keywords: SNES, get, preconditioner
.seealso: SNESSetPC()
@*/
PetscErrorCode SNESGetPC(SNES snes, SNES *pc)
{
  PetscErrorCode              ierr;
  const char                  *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(pc, 2);
  if (!snes->pc) {
    ierr = SNESCreate(((PetscObject) snes)->comm,&snes->pc);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)snes->pc,(PetscObject)snes,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->pc);CHKERRQ(ierr);
    ierr = SNESGetOptionsPrefix(snes,&optionsprefix);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(snes->pc,optionsprefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(snes->pc,"npc_");CHKERRQ(ierr);
  }
  *pc = snes->pc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetSNESLineSearch"
/*@
  SNESSetSNESLineSearch - Sets the linesearch on the SNES instance.

  Collective on SNES

  Input Parameters:
+ snes - iterative context obtained from SNESCreate()
- linesearch   - the linesearch object

  Notes:
  Use SNESGetSNESLineSearch() to retrieve the preconditioner context (for example,
  to configure it using the API).

  Level: developer

.keywords: SNES, set, linesearch
.seealso: SNESGetSNESLineSearch()
@*/
PetscErrorCode SNESSetSNESLineSearch(SNES snes, SNESLineSearch linesearch)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 2);
  PetscCheckSameComm(snes, 1, linesearch, 2);
  ierr = PetscObjectReference((PetscObject) linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchDestroy(&snes->linesearch);CHKERRQ(ierr);
  snes->linesearch = linesearch;
  ierr = PetscLogObjectParent(snes, snes->linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGetSNESLineSearch"
/*@C
  SNESGetSNESLineSearch - Returns a pointer to the line search context set with SNESSetLineSearch()
  or creates a default line search instance associated with the SNES and returns it.

  Not Collective

  Input Parameter:
. snes - iterative context obtained from SNESCreate()

  Output Parameter:
. linesearch - linesearch context

  Level: developer

.keywords: SNES, get, linesearch
.seealso: SNESSetSNESLineSearch()
@*/
PetscErrorCode SNESGetSNESLineSearch(SNES snes, SNESLineSearch *linesearch)
{
  PetscErrorCode ierr;
  const char     *optionsprefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(linesearch, 2);
  if (!snes->linesearch) {
    ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
    ierr = SNESLineSearchCreate(((PetscObject) snes)->comm, &snes->linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSNES(snes->linesearch, snes);CHKERRQ(ierr);
    ierr = SNESLineSearchAppendOptionsPrefix(snes->linesearch, optionsprefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject) snes->linesearch, (PetscObject) snes, 1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes, snes->linesearch);CHKERRQ(ierr);
  }
  *linesearch = snes->linesearch;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <mex.h>

typedef struct {char *funcname; mxArray *ctx;} SNESMatlabContext;

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunction_Matlab"
/*
   SNESComputeFunction_Matlab - Calls the function that has been set with
                         SNESSetFunctionMatlab().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() is typically used within nonlinear solvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction(), SNESGetFunction()
*/
PetscErrorCode  SNESComputeFunction_Matlab(SNES snes,Vec x,Vec y, void *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx = (SNESMatlabContext *)ctx;
  int               nlhs = 1,nrhs = 5;
  mxArray	    *plhs[1],*prhs[5];
  long long int     lx = 0,ly = 0,ls = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);

  /* call Matlab function in ctx with arguments x and y */

  ierr = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&ly,&y,sizeof(x));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)ly);
  prhs[3] =  mxCreateString(sctx->funcname);
  prhs[4] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESComputeFunctionInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESSetFunctionMatlab"
/*
   SNESSetFunctionMatlab - Sets the function evaluation routine and function 
   vector for use by the SNES routines in solving systems of nonlinear
   equations from MATLAB. Here the function is a string containing the name of a MATLAB function

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  r - vector to store function value
-  func - function evaluation routine

   Calling sequence of func:
$    func (SNES snes,Vec x,Vec f,void *ctx);


   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
   where f'(x) denotes the Jacobian matrix and f(x) is the function.

   Level: beginner

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
*/
PetscErrorCode  SNESSetFunctionMatlab(SNES snes,Vec r,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(SNESMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /* 
     This should work, but it doesn't 
  sctx->ctx = ctx; 
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = SNESSetFunction(snes,r,SNESComputeFunction_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESComputeJacobian_Matlab"
/*
   SNESComputeJacobian_Matlab - Calls the function that has been set with
                         SNESSetJacobianMatlab().  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  x - input vector
.  A, B - the matrices
-  ctx - user context

   Output Parameter:
.  flag - structure of the matrix

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction(), SNESGetFunction()
@*/
PetscErrorCode  SNESComputeJacobian_Matlab(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *flag, void *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx = (SNESMatlabContext *)ctx;
  int               nlhs = 2,nrhs = 6;
  mxArray	    *plhs[2],*prhs[6];
  long long int     lx = 0,lA = 0,ls = 0, lB = 0;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);

  /* call Matlab function in ctx with arguments x and y */

  ierr = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lA,A,sizeof(x));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lB,B,sizeof(x));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)lx);
  prhs[2] =  mxCreateDoubleScalar((double)lA);
  prhs[3] =  mxCreateDoubleScalar((double)lB);
  prhs[4] =  mxCreateString(sctx->funcname);
  prhs[5] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESComputeJacobianInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  *flag   =  (MatStructure) mxGetScalar(plhs[1]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(plhs[0]);
  mxDestroyArray(plhs[1]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESSetJacobianMatlab"
/*
   SNESSetJacobianMatlab - Sets the Jacobian function evaluation routine and two empty Jacobian matrices
   vector for use by the SNES routines in solving systems of nonlinear
   equations from MATLAB. Here the function is a string containing the name of a MATLAB function

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  A,B - Jacobian matrices
.  func - function evaluation routine
-  ctx - user context

   Calling sequence of func:
$    flag = func (SNES snes,Vec x,Mat A,Mat B,void *ctx);


   Level: developer

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
*/
PetscErrorCode  SNESSetJacobianMatlab(SNES snes,Mat A,Mat B,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(SNESMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /* 
     This should work, but it doesn't 
  sctx->ctx = ctx; 
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobian_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitor_Matlab"
/*
   SNESMonitor_Matlab - Calls the function that has been set with SNESMonitorSetMatlab().  

   Collective on SNES

.seealso: SNESSetFunction(), SNESGetFunction()
@*/
PetscErrorCode  SNESMonitor_Matlab(SNES snes,PetscInt it, PetscReal fnorm, void *ctx)
{
  PetscErrorCode  ierr;
  SNESMatlabContext *sctx = (SNESMatlabContext *)ctx;
  int             nlhs = 1,nrhs = 6;
  mxArray	  *plhs[1],*prhs[6];
  long long int   lx = 0,ls = 0;
  Vec             x=snes->vec_sol;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr); 
  ierr = PetscMemcpy(&lx,&x,sizeof(x));CHKERRQ(ierr); 
  prhs[0] =  mxCreateDoubleScalar((double)ls);
  prhs[1] =  mxCreateDoubleScalar((double)it);
  prhs[2] =  mxCreateDoubleScalar((double)fnorm);
  prhs[3] =  mxCreateDoubleScalar((double)lx);
  prhs[4] =  mxCreateString(sctx->funcname);
  prhs[5] =  sctx->ctx;
  ierr    =  mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESMonitorInternal");CHKERRQ(ierr);
  ierr    =  mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(prhs[4]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorSetMatlab"
/*
   SNESMonitorSetMatlab - Sets the monitor function from MATLAB

   Level: developer

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
*/
PetscErrorCode  SNESMonitorSetMatlab(SNES snes,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr = PetscMalloc(sizeof(SNESMatlabContext),&sctx);CHKERRQ(ierr);
  ierr = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  /* 
     This should work, but it doesn't 
  sctx->ctx = ctx; 
  mexMakeArrayPersistent(sctx->ctx);
  */
  sctx->ctx = mxDuplicateArray(ctx);
  ierr = SNESMonitorSet(snes,SNESMonitor_Matlab,sctx,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
