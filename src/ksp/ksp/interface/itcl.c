#define PETSCKSP_DLL

/*
    Code for setting KSP options from the options database.
*/

#include "src/ksp/ksp/kspimpl.h"  /*I "petscksp.h" I*/
#include "petscsys.h"

/*
       We retain a list of functions that also take KSP command 
    line options. These are called at the end KSPSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
PetscInt numberofsetfromoptions = 0;
PetscErrorCode (*othersetfromoptions[MAXSETFROMOPTIONS])(KSP) = {0};

extern PetscTruth KSPRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "KSPAddOptionsChecker"
/*@C
    KSPAddOptionsChecker - Adds an additional function to check for KSP options.

    Not Collective

    Input Parameter:
.   kspcheck - function that checks for options

    Level: developer

.keywords: KSP, add, options, checker

.seealso: KSPSetFromOptions()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPAddOptionsChecker(PetscErrorCode (*kspcheck)(KSP))
{
  PetscFunctionBegin;
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many options checkers, only 5 allowed");
  }

  othersetfromoptions[numberofsetfromoptions++] = kspcheck;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetOptionsPrefix"
/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different KSP contexts, one could call
.vb
      KSPSetOptionsPrefix(ksp1,"sys1_")
      KSPSetOptionsPrefix(ksp2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
      -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
.ve

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPAppendOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PCSetOptionsPrefix(ksp->pc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNCT__  
#define __FUNCT__ "KSPAppendOptionsPrefix"
/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: KSP, append, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPAppendOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PCAppendOptionsPrefix(ksp->pc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetOptionsPrefix"
/*@C
   KSPGetOptionsPrefix - Gets the prefix used for searching for all 
   KSP options in the database.

   Not Collective

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOptionsPrefix(KSP ksp,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions"
/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov type. 

   Collective on KSP

   Input Parameters:
.  ksp - the Krylov space context

   Options Database Keys:
+   -ksp_max_it - maximum number of linear iterations
.   -ksp_rtol rtol - relative tolerance used in default determination of convergence, i.e.
                if residual norm decreases by this factor than convergence is declared
.   -ksp_atol abstol - absolute tolerance used in default convergence test, i.e. if residual 
                norm is less than this then convergence is declared
.   -ksp_divtol tol - if residual norm increases by this factor than divergence is declared
.   -ksp_converged_use_initial_residual_norm - see KSPDefaultConvergedSetUIRNorm()
.   -ksp_converged_use_min_initial_residual_norm - see KSPDefaultConvergedSetUMIRNorm()
.   -ksp_norm_type - none - skip norms used in convergence tests (useful only when not using 
$                       convergence test (say you always want to run with 5 iterations) to 
$                       save on communication overhead
$                    preconditioned - default for left preconditioning 
$                    unpreconditioned - see KSPSetNormType()
$                    natural - see KSPSetNormType()
.    -ksp_constant_null_space - assume the operator (matrix) has the constant vector in its null space
.    -ksp_test_null_space - tests the null space set with KSPSetNullSpace() to see if it truly is a null space
.   -ksp_knoll - compute initial guess by applying the preconditioner to the right hand side
.   -ksp_cancelmonitors - cancel all previous convergene monitor routines set
.   -ksp_monitor <optional filename> - print residual norm at each iteration
.   -ksp_xmonitor - plot residual norm at each iteration
.   -ksp_vecmonitor - plot solution at each iteration
-   -ksp_singmonitor - monitor extremem singular values at each iteration

   Notes:  
   To see all options, run your program with the -help option
   or consult the users manual.

   Level: beginner

.keywords: KSP, set, from, options, database

.seealso: 
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFromOptions(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  char           type[256], monfilename[PETSC_MAX_PATH_LEN];
  const char     *stype[] = {"none","preconditioned","unpreconditioned","natural"};
  PetscViewer    monviewer;
  PetscTruth     flg,flag;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PCSetFromOptions(ksp->pc);CHKERRQ(ierr);

  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(ksp->comm,ksp->prefix,"Krylov Method (KSP) Options","KSP");CHKERRQ(ierr);
    ierr = PetscOptionsList("-ksp_type","Krylov method","KSPSetType",KSPList,(char*)(ksp->type_name?ksp->type_name:KSPGMRES),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetType(ksp,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!ksp->type_name) {
      ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
    }

    ierr = PetscOptionsInt("-ksp_max_it","Maximum number of iterations","KSPSetTolerances",ksp->max_it,&ksp->max_it,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_rtol","Relative decrease in residual norm","KSPSetTolerances",ksp->rtol,&ksp->rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_atol","Absolute value of residual norm","KSPSetTolerances",ksp->abstol,&ksp->abstol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_divtol","Residual norm increase cause divergence","KSPSetTolerances",ksp->divtol,&ksp->divtol,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsName("-ksp_converged_use_initial_residual_norm","Use initial residual residual norm for computing relative convergence","KSPDefaultConvergedSetUIRNorm",&flag);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUIRNorm(ksp);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-ksp_converged_use_min_initial_residual_norm","Use minimum of initial residual norm and b for computing relative convergence","KSPDefaultConvergedSetUMIRNorm",&flag);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUMIRNorm(ksp);CHKERRQ(ierr);}

    ierr = PetscOptionsTruth("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,
                                  &ksp->guess_knoll,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsEList("-ksp_norm_type","KSP Norm type","KSPSetNormType",stype,4,"preconditioned",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0:
        ierr = KSPSetNormType(ksp,KSP_NO_NORM);CHKERRQ(ierr);
        ierr = KSPSetConvergenceTest(ksp,KSPSkipConverged,0);CHKERRQ(ierr);
        break;
      case 1:
        ierr = KSPSetNormType(ksp,KSP_PRECONDITIONED_NORM);CHKERRQ(ierr);
        break;
      case 2:
        ierr = KSPSetNormType(ksp,KSP_UNPRECONDITIONED_NORM);CHKERRQ(ierr);
        break;
      case 3:
        ierr = KSPSetNormType(ksp,KSP_NATURAL_NORM);CHKERRQ(ierr);
        break;
      }
    }

    ierr = PetscOptionsName("-ksp_diagonal_scale","Diagonal scale matrix before building preconditioner","KSPSetDiagonalScale",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScale(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ksp_diagonal_scale_fix","Fix diagonaled scaled matrix after solve","KSPSetDiagonalScaleFix",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScaleFix(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }


    ierr = PetscOptionsName("-ksp_constant_null_space","Add constant null space to Krylov solver","KSPSetNullSpace",&flg);CHKERRQ(ierr);
    if (flg) {
      MatNullSpace nsp;

      ierr = MatNullSpaceCreate(ksp->comm,PETSC_TRUE,0,0,&nsp);CHKERRQ(ierr);
      ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(nsp);CHKERRQ(ierr);
    }

    /* option is actually checked in KSPSetUp() */
    if (ksp->nullsp) {
      ierr = PetscOptionsName("-ksp_test_null_space","Is provided null space correct","None",&flg);CHKERRQ(ierr);
    }

    /*
      Prints reason for convergence or divergence of each linear solve
    */
    ierr = PetscOptionsName("-ksp_converged_reason","Print reason for converged or diverged","KSPSolve",&flg);CHKERRQ(ierr);
    if (flg) {
      ksp->printreason = PETSC_TRUE;
    }

    ierr = PetscOptionsName("-ksp_cancelmonitors","Remove any hardwired monitor routines","KSPClearMonitor",&flg);CHKERRQ(ierr);
    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to KSPSetFromOptions()
    */
    if (flg) {
      ierr = KSPClearMonitor(ksp);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor","Monitor preconditioned residual norm","KSPSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(ksp->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Plots the vector solution 
    */
    ierr = PetscOptionsName("-ksp_vecmonitor","Monitor solution graphically","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetMonitor(ksp,KSPVecViewMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned and true residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_truemonitor","Monitor preconditioned residual norm","KSPSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(ksp->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPTrueMonitor,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints extreme eigenvalue estimates at each iteration
    */
    ierr = PetscOptionsString("-ksp_singmonitor","Monitor singular values","KSPSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(ksp->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPSingularValueMonitor,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm with fewer digits
    */
    ierr = PetscOptionsString("-ksp_smonitor","Monitor preconditioned residual norm with fewer digits","KSPSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(ksp->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPDefaultSMonitor,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned residual norm
    */
    ierr = PetscOptionsName("-ksp_xmonitor","Monitor graphically preconditioned residual norm","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetMonitor(ksp,KSPLGMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned and true residual norm
    */
    ierr = PetscOptionsName("-ksp_xtruemonitor","Monitor graphically true residual norm","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg){
      ierr = KSPSetMonitor(ksp,KSPLGTrueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsTruthGroupBegin("-ksp_left_pc","Use left preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_LEFT);CHKERRQ(ierr); }
    ierr = PetscOptionsTruthGroup("-ksp_right_pc","Use right preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_RIGHT);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-ksp_symmetric_pc","Use symmetric (factorized) preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_SYMMETRIC);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-ksp_compute_singularvalues","Compute singular values of preconditioned operator","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_compute_eigenvalues","Compute eigenvalues of preconditioned operator","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }

    for(i = 0; i < numberofsetfromoptions; i++) {
      ierr = (*othersetfromoptions[i])(ksp);CHKERRQ(ierr);
    }

    if (ksp->ops->setfromoptions) {
      ierr = (*ksp->ops->setfromoptions)(ksp);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ksp_view","View linear solver parameters","KSPView",&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
