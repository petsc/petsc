#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h" 

PetscTruth TaoSolverRegisterAllCalled = PETSC_FALSE;
PetscFList TaoSolverList = PETSC_NULL;

PetscCookie TAOSOLVER_DLL TAOSOLVER_COOKIE;
PetscLogEvent TaoSolver_Solve, TaoSolver_ObjectiveEval, TaoSolver_GradientEval, TaoSolver_ObjGradientEval, TaoSolver_HessianEval, TaoSolver_JacobianEval;

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve"
/*@ 
  TaoSolverSolve - Solves an optimization problem min F(x) s.t. l <= x <= u

  Collective on TaoSolver
  
  Input Parameters:
. tao - the TaoSolver context

  Notes:
  The user must set up the TaoSolver with calls to TaoSolverSetInitialVector(),
  TaoSolverSetObjective(),
  TaoSolverSetGradient(), and (if using 2nd order method) TaoSolverSetHessian().

  .seealso: TaoSolverCreate(), TaoSolverSetObjective(), TaoSolverSetGradient(), TaoSolverSetHessian()
  @*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSolve(TaoSolver tao)
{
  PetscErrorCode ierr;
  TaoFunctionBegin;
  TaoValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);

  ierr = TaoGetSolution(tao,&xx);CHKERRQ(ierr);
  ierr = TaoSetUp(tao);CHKERRQ(ierr);
  ierr = TaoResetStatistics(tao); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(TaoSolver_Solve,tao,0,0,0); CHKERRQ(ierr);
  if (tao->ops->solve){ ierr = (*tao->ops->solve)(tao);CHKERRQ(ierr); }
  ierr = PetscLogEventEnd(TaoSolver_Solve,tao,0,0,0); CHKERRQ(ierr);

  if (tao->viewtao) { ierr = TaoView(tao);CHKERRQ(ierr); }
  if (tao->viewksptao) { ierr = TaoViewLinearSolver(tao);CHKERRQ(ierr); }

  if (tao->printreason) { 
      if (tao->reason > 0) {
	  ierr = PetscPrintf(((PetscObject)tao)->comm,"TAO solve converged due to %s\n",TaoSolverConvergedReasons[tao->reason]); CHKERRQ(ierr);
      } else {
	  ierr = PetscPrintf(((PetscObject)tao)->comm,"TAO solve did not converge due to %s\n",TaoSolverConvergedReasons[tao->reason]); CHKERRQ(ierr);
      }
  }
  

  TaoFunctionReturn(0);

    
}


/*@ 
  TaoSolverSetUp - Sets up the internal data structures for the later use
  of a Tao solver

  Collective on tao
  
  Input Parameters:
. tao - the TAO context

  Notes:
  The user will not need to explicitly call TaoSolverSetUp(), as it will 
  automatically be called in TaoSolverSolve().  However, if the user
  desires to call it explicitly, it should come after TaoSolverCreate()
  and TaoSolverSetXXX(), but before TaoSolverSolve().

  Level: advanced

.seealso: TaoSolverCreate(), TaoSolverSolve()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetUp(TaoSolver tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAOSOLVER_COOKIE,1); 
  if (tao->setupcalled) PetscFunctionReturn(0);
  
  if (solver->ops->setup) {
    ierr = (*solver->ops->setup)(tao); CHKERRQ(ierr);
  }

  tao->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy"
/*@ 
  TaoSolverDestroy - Destroys the TAO context that was created with 
  TaoSolverCreate()

  Collective on TaoSolver

  Input Parameter
. tao - the TaoSolver context

  Level: beginner

.seealse: TaoSolverCreate(), TaoSolverSolve()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDestroy(TaoSolver tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);

  ierr = PetscObjectDepublish(tao); CHKERRQ(ierr);
  
  if (tao->ops->destroy) {
      ierr = (*tao->ops->destroy)(tao); CHKERRQ(ierr);
  }
  ierr = TaoSolverMonitorCancel(tao); CHKERRQ(ierr);
  if (tao->ops->convergeddestroy) {
      ierr = (*tao->ops->convergeddestroy)(tao->cnvP); CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(tao); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions"
/*@
  TaoSolverSetFromOptions -Sets various TaoSolver parameters from user
  options.

  Collective on TaoSolver

  Input Paremeter:
. tao - the TaoSolver solver context

  Options Database Keys:
+ -tao_method <method> - The algorithm that TAO uses (tao_lmvm, tao_nls, etc.)
. -tao_fatol <fatol>
. -tao_frtol <frtol>
. -tao_gatol <gatol>
. -tao_grtol <grtol>
. -tao_gttol <gttol>
. -tao_catol <catol>
. -tao_cttol <cttol>
. -tao_no_convergence_test
. -tao_monitor
. -tao_smonitor
. -tao_vecmonitor
. -tao_vecmonitor_update
. -tao_xmonitor
. -tao_fd
- -tao_fdgrad

  Notes:
  To see all options, run your program with the -help option or consult the 
  user's manual

  Level: beginner
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetFromOptions(TaoSolver tao)
{
    PetscTruth ierr;
    const char *default_method = TAO_LMVM;
    char method[256];
    PetscTruth flg;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);

    ierr = PetscOptionsBegin(((PetscObject)tao)->comm, ((PetscObject)tao)->prefix,"TaoSolver options","TaoSolver"); CHKERRQ(ierr);
    {
	if (!TaoSolverRegisterAllCalled) {
	    ierr = TaoSolverRegisterAll(PETSC_NULL); CHKERRQ(ierr);
	}
	if (((PetscObject)tao)->type_name) {
	    default_type = ((PetscObject)tao)->type_name;
	}
	/* Check for method from options */
	ierr = PetscOptionsList("-tao_method","Tao Solver method","TaoSolverSetType",TaoSolverList,default_method,method,256,&flg); CHKERRQ(ierr);
	if (flg) {
	    ierr = TaoSolverSetMethod(tao,method); CHKERRQ(ierr);
	} else if (!((PetscObject)tao)->type_name) {
	    ierr = TaoSolverSetMethod(tao,default_method);
	}
	if (tao->ops->setfromoptions) {
	    ierr = (*tao->ops->setfromoptions)(tao); CHKERRQ(ierr);
	}
    }
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverView"
/*@
  TaoSolverView - Prints information about the TaoSolver
 
  Collective on TaoSolver

  InputParameters:
+ tao - the TaoSolver context
- viewer - visualization context

  Options Database Key:
. -tao_view - Calls TaoSolverView() at the end of TaoSolverSolve()

  Notes:
  The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

  Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode TASOLVER_DLLEXPORT TaoSolverView(TaoSolver tao, PetscViewer viewer)
{
    PetscErrorCode ierr;
    PetscTruth isascii,isstring;
    const TaoSolverType method;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
    PetscCheckSameComm(tao,1,viewer,2);

    PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii); CHKERRQ(ierr);
    PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring); CHKERRQ(ierr);
    if (isascii) {
	if (((PetscObject)snes)->prefix) {
	    ierr = PetscViewerASCIIPrintf(viewer,"TaoSolver Object:(%s)\n",((PetscObject)tao)->prefix); CHKERRQ(ierr);
        } else {
	    ierr = PetscViewerASCIIPrintf(viewer,"TaoSolver Object:\n"); CHKERRQ(ierr); CHKERRQ(ierr);
	}
	ierr = TaoSolverGetMethod(tao,&method);
	if (method) {
	    ierr = PetscViewerASCIIPrintf(viewer,"  method: %s\n",method); CHKERRQ(ierr);
	} else {
	    ierr = PetscViewerASCIIPrintf(viewer,"  method: not set yet\n"); CHKERRQ(ierr);
	}
	if (tao->ops->view) {
	    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
	    ierr = (*tao->ops->view)(tao,viewer); CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
	}
	ierr=PetscViewerASCIIPrintf(viewer,"  convergence tolerances: fatol=%g,",tao->fatol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," frtol=%g\n",tao->frtol);CHKERRQ(ierr);

	ierr=PetscViewerASCIIPrintf(viewer,"  convergence tolerances: gatol=%g,",tao->gatol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," trtol=%g,",tao->trtol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," gttol=%g\n",tao->gttol);CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(viewer,"  Residual in Function/Gradient:=%e\n",tao->norm);CHKERRQ(ierr);

	if (tao->cnorm>0 || tao->catol>0 || tao->crtol>0){
	    ierr=PetscViewerASCIIPrintf(viewer,"  convergence tolerances:");CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer," catol=%g,",tao->catol);CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer," crtol=%g\n",tao->crtol);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"  Residual in Constraints:=%e\n",tao->cnorm);CHKERRQ(ierr);
	}

	if (tao->trtol>0){
	    ierr=PetscViewerASCIIPrintf(viewer,"  convergence tolerances: trtol=%g\n",tao->trtol);CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer,"  Final step size/trust region radius:=%g\n",tao->step);CHKERRQ(ierr);
	}

	if (tao->fmin>-1.e25){
	    ierr=PetscViewerASCIIPrintf(viewer,"  convergence tolerances: function minimum=%g\n"
					,tao->fmin);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer,"  Objective value=%e\n",
				      tao->fc);CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(viewer,"  total number of iterations=%d,          ",
				      tao->iter);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"              (max: %d)\n",tao->max_its);CHKERRQ(ierr);

	if (tao->nfuncs>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%d,",
					  tao->nfuncs);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"                max: %d\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->ngrads>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of gradient evaluations=%d,",
					  tao->ngrads);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"                max: %d\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->nfgrads>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of function/gradient evaluations=%d,",
					  tao->nfgrads);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"    (max: %d)\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->nhesss>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of Hessian evaluations=%d\n",
					  tao->nhesss);CHKERRQ(ierr);
	}
	if (tao->linear_its>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total Krylov method iterations=%d\n",
					  tao->linear_its);CHKERRQ(ierr);
	}
	if (tao->nvfunc>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of constraint function evaluations=%d\n",
					  tao->nvfunc);CHKERRQ(ierr);
	}
	if (tao->njac>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total number of Jacobian evaluations=%d\n",
					  tao->njac);CHKERRQ(ierr);
	}

	if (tao->reason>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  Solution found\n");CHKERRQ(ierr);
	} else {
	    ierr = PetscViewerASCIIPrintf(viewer,"  Solver terminated: %d\n",tao->reason);CHKERRQ(ierr);
	}
	
    } else if (isstring) {
	ierr = TaoSolverGetType(tao,&method); CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(viewer," %-3.3s",method); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetTolerances"
/*@
  TaoSolverSetTolerances - Sets parameters used in TAO convergence tests

  Collective on TaoSolver

  Input Parameters
+ tao - the TaoSolver context
. fatol - absolute convergence tolerance
. frtol - relative convergence tolerance
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by a this factor

  Options Database Keys:
+ -tao_fatol <fatol> - Sets fatol
. -tao_frtol <frtol> - Sets frtol
. -tao_gatol <catol> - Sets gatol
. -tao_grtol <catol> - Sets gatol
- .tao_gttol <crtol> - Sets gttol

  Absolute Stopping Criteria
$ f_{k+1} <= f_k + fatol

  Relative Stopping Criteria
$ f_{k+1} <= f_k + frtol*|f_k|

  Notes: Use PETSC_DEFAULT to leave one or more tolerances unchanged.

  Level: beginner

.seealso: TaoSetMaximumIterates(), 
          TaoSetMaximumFunctionEvaluations(), TaoGetTolerances(),
          TaoSetConstraintTolerances()

@*/
PetscErrorCode TaoSolverSetTolerances(TaoSolver tao, PetscReal fatol, PetscReal frtol, PetscReal gatol, PetscReal grtol, PetscReal gttol)
{
    PetscErrorCode ierr;
    TaoFunctionBegin;
    TaoValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
    
    if (fatol != PETSC_DEFAULT) {
      if (fatol<0) {
	ierr = PetscInfo(tao,"Tried to set negative fatol -- ignored.");
	CHKERRQ(ierr);
      } else {
	tao->fatol = PetscMax(0,fatol);
      }
    }
    
    if (frtol != PETSC_DEFAULT) {
      if (frtol<0) {
	ierr = PetscInfo(tao,"Tried to set negative frtol -- ignored.");
	CHKERRQ(ierr);
      } else {
	tao->frtol = PetscMax(0,frtol);
      }
    }

    if (gatol != PETSC_DEFAULT) {
      if (gatol<0) {
	ierr = PetscInfo(tao,"Tried to set negative gatol -- ignored.");
	CHKERRQ(ierr);
      } else {
	tao->gatol = PetscMax(0,gatol);
      }
    }

    if (grtol != PETSC_DEFAULT) {
      if (grtol<0) {
	ierr = PetscInfo(tao,"Tried to set negative grtol -- ignored.");
	CHKERRQ(ierr);
      } else {
	tao->grtol = PetscMax(0,grtol);
      }
    }

    if (gttol != PETSC_DEFAULT) {
      if (gttol<0) {
	ierr = PetscInfo(tao,"Tried to set negative gttol -- ignored.");
	CHKERRQ(ierr);
      } else {
	tao->gttol = PetscMax(0,gttol);
      }
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverGetTolerances"
/*@
  TaoSolverGetTolerances - gets the current values of tolerances

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
. fatol
. frtol
. gatol
. grtol
- gttol

  Notes: Use PETSC_NULL as an argument to ignore one or more tolerances.
.seealse TaoSolverSetTolerances()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetTolerances(TaoSolver tao, PetscReal *fatol, PetscReal *frtol, PetscReal *gatol, PetscReal *grtol, PetscReal *gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);
  if (fatol) *fatol=tao->fatol;
  if (frtol) *frtol=tao->frtol;
  if (gatol) *gatol=tao->gatol;
  if (grtol) *grtol=tao->grtol;
  if (gttol) *gttol=tao->gttol;
  PetscFunctionReturn(0);
}



