/*
     The basic KSP routines, Create, View etc. are here.
*/
#include "src/ksp/ksp/kspimpl.h"      /*I "petscksp.h" I*/
#include "petscsys.h"

/* Logging support */
PetscCookie KSP_COOKIE = 0;
PetscEvent  KSP_GMRESOrthogonalization = 0, KSP_SetUp = 0, KSP_Solve = 0;


PetscTruth KSPRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "KSPView"
/*@C 
   KSPView - Prints the KSP data structure.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  viewer - visualization context

   Options Database Keys:
.  -ksp_view - print the ksp data structure at the end of a KSPSolve call

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.keywords: KSP, view

.seealso: PCView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode KSPView(KSP ksp,PetscViewer viewer)
{
  char           *type;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(ksp->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(ksp,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
    if (ksp->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP Object:(%s)\n",ksp->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"KSP Object:\n");CHKERRQ(ierr);
    }
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (ksp->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ksp->guess_zero) {ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, initial guess is zero\n",ksp->max_it);CHKERRQ(ierr);}
    else                 {ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D\n", ksp->max_it);CHKERRQ(ierr);}
    if (ksp->guess_knoll) {ierr = PetscViewerASCIIPrintf(viewer,"  using preconditioner applied to right hand side for initial guess\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances:  relative=%g, absolute=%g, divergence=%g\n",ksp->rtol,ksp->abstol,ksp->divtol);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT)          {ierr = PetscViewerASCIIPrintf(viewer,"  right preconditioning\n");CHKERRQ(ierr);}
    else if (ksp->pc_side == PC_SYMMETRIC) {ierr = PetscViewerASCIIPrintf(viewer,"  symmetric preconditioning\n");CHKERRQ(ierr);}
    else                                   {ierr = PetscViewerASCIIPrintf(viewer,"  left preconditioning\n");CHKERRQ(ierr);}
  } else {
    if (ksp->ops->view) {
      ierr = (*ksp->ops->view)(ksp,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PCView(ksp->pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered KSP routines
*/
PetscFList KSPList = 0;

#undef __FUNCT__  
#define __FUNCT__ "KSPSetNormType"
/*@C
   KSPSetNormType - Sets the norm that is used for convergence testing.

   Collective on KSP

   Input Parameter:
+  ksp - Krylov solver context
-  normtype - one of 
$   KSP_NO_NORM - skips computing the norm, this should only be used if you are using
$                 the Krylov method as a smoother with a fixed small number of iterations.
$                 You must also call KSPSetConvergenceTest(ksp,KSPSkipConverged,PETSC_NULL);
$                 supported only by CG, Richardson, Bi-CG-stab, CR, and CGS methods.
$   KSP_PRECONDITIONED_NORM - the default for left preconditioned solves, uses the l2 norm
$                 of the preconditioned residual
$   KSP_UNPRECONDITIONED_NORM - uses the l2 norm of the true b - Ax residual, supported only by
$                 CG, CHEBYCHEV, and RICHARDSON  
$   KSP_NATURAL_NORM - supported  by cg, cr, and cgs 


   Options Database Key:
.   -ksp_norm_type <none,preconditioned,unpreconditioned,natural>

   Notes: 
   Currently only works with the CG, Richardson, Bi-CG-stab, CR, and CGS methods.

   Level: advanced

.keywords: KSP, create, context, norms

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSPSkipConverged()                               
@*/
PetscErrorCode KSPSetNormType(KSP ksp,KSPNormType normtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ksp->normtype = normtype;
  if (normtype == KSP_NO_NORM) {
    ierr = PetscLogInfo((ksp,"KSPSetNormType:Warning seting KSPNormType to skip computing the norm\n\
  make sure you set the KSP convergence test to KSPSkipConvergence\n"));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPPublish_Petsc"
static PetscErrorCode KSPPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetOperators"
/*@
   KSPSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Collective on KSP and Mat

   Input Parameters:
+  ksp - the KSP context
.  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually the
          same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

   Notes: 
   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Pmat is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Pmat has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Pmat does not have the same nonzero structure.

    Caution:
    If you specify SAME_NONZERO_PATTERN, PETSc believes your assertion
    and does not check the structure of the matrix.  If you erroneously
    claim that the structure is the same when it actually is not, the new
    preconditioner will not function correctly.  Thus, use this optimization
    feature carefully!

    If in doubt about whether your preconditioner matrix has changed
    structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

    Level: beginner

.keywords: KSP, set, operators, matrix, preconditioner, linear system

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPGetOperators()
@*/
PetscErrorCode KSPSetOperators(KSP ksp,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_COOKIE,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_COOKIE,3);
  ierr = PCSetOperators(ksp->pc,Amat,Pmat,flag);CHKERRQ(ierr);
  if (ksp->setupcalled > 1) ksp->setupcalled = 1;  /* so that next solve call will call setup */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetOperators"
/*@
   KSPGetOperators - Gets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Collective on KSP and Mat

   Input Parameter:
.  ksp - the KSP context

   Output Parameters:
+  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually the
          same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

    Level: intermediate

.keywords: KSP, set, get, operators, matrix, preconditioner, linear system

.seealso: KSPSolve(), KSPGetPC(), PCGetOperators(), PCSetOperators(), KSPSetOperators()
@*/
PetscErrorCode KSPGetOperators(KSP ksp,Mat *Amat,Mat *Pmat,MatStructure *flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PCGetOperators(ksp->pc,Amat,Pmat,flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPCreate"
/*@C
   KSPCreate - Creates the default KSP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ksp - location to put the KSP context

   Notes:
   The default KSP type is GMRES with a restart of 30, using modified Gram-Schmidt
   orthogonalization.

   Level: beginner

.keywords: KSP, create, context

.seealso: KSPSetUp(), KSPSolve(), KSPDestroy(), KSP
@*/
PetscErrorCode KSPCreate(MPI_Comm comm,KSP *inksp)
{
  KSP            ksp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(inksp,2);
  *inksp = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = KSPInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(ksp,_p_KSP,struct _KSPOps,KSP_COOKIE,-1,"KSP",comm,KSPDestroy,KSPView);CHKERRQ(ierr);
  *inksp             = ksp;
  ksp->bops->publish = KSPPublish_Petsc;

  ksp->type          = -1;
  ksp->max_it        = 10000;
  ksp->pc_side       = PC_LEFT;
  ksp->rtol          = 1.e-5;
  ksp->abstol          = 1.e-50;
  ksp->divtol        = 1.e4;

  ksp->normtype            = KSP_PRECONDITIONED_NORM;
  ksp->rnorm               = 0.0;
  ksp->its                 = 0;
  ksp->guess_zero          = PETSC_TRUE;
  ksp->calc_sings          = PETSC_FALSE;
  ksp->res_hist            = PETSC_NULL;
  ksp->res_hist_len        = 0;
  ksp->res_hist_max        = 0;
  ksp->res_hist_reset      = PETSC_TRUE;
  ksp->numbermonitors      = 0;
  ksp->converged           = KSPDefaultConverged;
  ksp->ops->buildsolution  = KSPDefaultBuildSolution;
  ksp->ops->buildresidual  = KSPDefaultBuildResidual;

  ksp->ops->setfromoptions = 0;

  ksp->vec_sol         = 0;
  ksp->vec_rhs         = 0;
  ksp->pc              = 0;

  ksp->ops->solve      = 0;
  ksp->ops->setup      = 0;
  ksp->ops->destroy    = 0;

  ksp->data            = 0;
  ksp->nwork           = 0;
  ksp->work            = 0;

  ksp->cnvP            = 0;

  ksp->reason          = KSP_CONVERGED_ITERATING;

  ksp->setupcalled     = 0;
  ierr = PetscPublishAll(ksp);CHKERRQ(ierr);
  ierr = PCCreate(comm,&ksp->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "KSPSetType"
/*@C
   KSPSetType - Builds KSP for a particular solver. 

   Collective on KSP

   Input Parameters:
+  ksp      - the Krylov space context
-  type - a known method

   Options Database Key:
.  -ksp_type  <method> - Sets the method; use -help for a list 
    of available methods (for instance, cg or gmres)

   Notes:  
   See "petsc/include/petscksp.h" for available methods (for instance,
   KSPCG or KSPGMRES).

  Normally, it is best to use the KSPSetFromOptions() command and
  then set the KSP type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different Krylov methods.
  The KSPSetType() routine is provided for those situations where it
  is necessary to set the iterative solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of iterative solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  not for beginners.

  Level: intermediate

.keywords: KSP, set, method

.seealso: PCSetType(), KSPType

@*/
PetscErrorCode KSPSetType(KSP ksp,const KSPType type)
{
  PetscErrorCode ierr,(*r)(KSP);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)ksp,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (ksp->data) {
    /* destroy the old private KSP context */
    ierr = (*ksp->ops->destroy)(ksp);CHKERRQ(ierr);
    ksp->data = 0;
  }
  /* Get the function pointers for the iterative method requested */
  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr =  PetscFListFind(ksp->comm,KSPList,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown KSP type given: %s",type);
  ksp->setupcalled = 0;
  ierr = (*r)(ksp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)ksp,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPRegisterDestroy"
/*@C
   KSPRegisterDestroy - Frees the list of KSP methods that were
   registered by KSPRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: KSP, register, destroy

.seealso: KSPRegisterDynamic(), KSPRegisterAll()
@*/
PetscErrorCode KSPRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (KSPList) {
    ierr = PetscFListDestroy(&KSPList);CHKERRQ(ierr);
    KSPList = 0;
  }
  KSPRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetType"
/*@C
   KSPGetType - Gets the KSP type as a string from the KSP object.

   Not Collective

   Input Parameter:
.  ksp - Krylov context 

   Output Parameter:
.  name - name of KSP method 

   Level: intermediate

.keywords: KSP, get, method, name

.seealso: KSPSetType()
@*/
PetscErrorCode KSPGetType(KSP ksp,KSPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ksp->type_name;
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
.   -ksp_monitor - print residual norm at each iteration
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
PetscErrorCode KSPSetFromOptions(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  char           type[256];
  const char     *stype[] = {"none","preconditioned","unpreconditioned","natural"};
  PetscTruth     flg;

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
    ierr = PetscOptionsLogical("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,
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
    ierr = PetscOptionsName("-ksp_monitor","Monitor preconditioned residual norm","KSPSetMonitor",&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
    ierr = PetscOptionsName("-ksp_truemonitor","Monitor true (unpreconditioned) residual norm","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetMonitor(ksp,KSPTrueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Prints extreme eigenvalue estimates at each iteration
    */
    ierr = PetscOptionsName("-ksp_singmonitor","Monitor singular values","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPSingularValueMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
    }
    /*
      Prints preconditioned residual norm with fewer digits
    */
    ierr = PetscOptionsName("-ksp_smonitor","Monitor preconditioned residual norm with fewer digitis","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetMonitor(ksp,KSPDefaultSMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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

    ierr = PetscOptionsLogicalGroupBegin("-ksp_left_pc","Use left preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_LEFT);CHKERRQ(ierr); }
    ierr = PetscOptionsLogicalGroup("-ksp_right_pc","Use right preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_RIGHT);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-ksp_symmetric_pc","Use symmetric (factorized) preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_SYMMETRIC);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-ksp_compute_singularvalues","Compute singular values of preconditioned operator","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_compute_eigenvalues","Compute eigenvalues of preconditioned operator","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues","KSPSetComputeSingularValues",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }

    if (ksp->ops->setfromoptions) {
      ierr = (*ksp->ops->setfromoptions)(ksp);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ksp_view","View linear solver parameters","KSPView",&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPRegister"
/*@C
  KSPRegister - See KSPRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode KSPRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(KSP))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&KSPList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetNullSpace"
/*@C
  KSPSetNullSpace - Sets the null space of the operator

  Collective on KSP

  Input Parameters:
+  ksp - the Krylov space object
-  nullsp - the null space of the operator

  Level: advanced

.seealso: KSPSetOperators(), MatNullSpaceCreate(), KSPGetNullSpace()
@*/
PetscErrorCode KSPSetNullSpace(KSP ksp,MatNullSpace nullsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->nullsp) {
    ierr = MatNullSpaceDestroy(ksp->nullsp);CHKERRQ(ierr);
  }
  ksp->nullsp = nullsp;
  ierr = PetscObjectReference((PetscObject)ksp->nullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetNullSpace"
/*@C
  KSPGetNullSpace - Gets the null space of the operator

  Collective on KSP

  Input Parameters:
+  ksp - the Krylov space object
-  nullsp - the null space of the operator

  Level: advanced

.seealso: KSPSetOperators(), MatNullSpaceCreate(), KSPSetNullSpace()
@*/
PetscErrorCode KSPGetNullSpace(KSP ksp,MatNullSpace *nullsp)
{
  PetscFunctionBegin;
  *nullsp = ksp->nullsp;
  PetscFunctionReturn(0);
}

