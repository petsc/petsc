#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h" /*I "taosolver.h" I*/

PetscBool TaoSolverRegisterAllCalled = PETSC_FALSE;
PetscFList TaoSolverList = PETSC_NULL;

PetscClassId TAOSOLVER_CLASSID;
PetscLogEvent TaoSolver_Solve, TaoSolver_ObjectiveEval, TaoSolver_GradientEval, TaoSolver_ObjGradientEval, TaoSolver_HessianEval, TaoSolver_ConstraintsEval, TaoSolver_JacobianEval;



static const char *TAO_SUBSET[64] = {
  "subvec","mask","matrixfree"
};

#undef __FUNCT__
#define __FUNCT__ "TaoCreate"
/*@
  TaoCreate - Creates a TAO solver

  Collective on MPI_Comm

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newtao - the new TaoSolver context

  Available methods include:
+    tao_nls - Newton's method with line search for unconstrained minimization
.    tao_ntr - Newton's method with trust region for unconstrained minimization
.    tao_ntl - Newton's method with trust region, line search for unconstrained minimization
.    tao_lmvm - Limited memory variable metric method for unconstrained minimization
.    tao_cg - Nonlinear conjugate gradient method for unconstrained minimization
.    tao_nm - Nelder-Mead algorithm for derivate-free unconstrained minimization
.    tao_pounder - Model-based algorithm for derivate-free unconstrained minimiz
ation
.    tao_tron - Newton Trust Region method for bound constrained minimization
.    tao_gpcg - Newton Trust Region method for quadratic bound constrained minimization
.    tao_blmvm - Limited memory variable metric method for bound constrained minimization
-    tao_pounders - Model-based algorithm pounder extended for nonlinear least squares 

   Options Database Keys:
+   -tao_method - select which method TAO should use
-   -tao_type - identical to -tao_method

   Level: beginner

.seealso: TaoSolve(), TaoDestroy()
@*/
PetscErrorCode TaoCreate(MPI_Comm comm, TaoSolver *newtao)
{
    PetscErrorCode ierr;
    TaoSolver tao;
    
    PetscFunctionBegin;
    PetscValidPointer(newtao,2);
    *newtao = PETSC_NULL;

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
    ierr = TaoInitializePackage(PETSC_NULL); CHKERRQ(ierr);
#endif

    ierr = PetscHeaderCreate(tao,_p_TaoSolver, struct _TaoSolverOps, TAOSOLVER_CLASSID,0,"TaoSolver",0,0,comm,TaoDestroy,TaoView); CHKERRQ(ierr);
    
    tao->ops->computeobjective=0;
    tao->ops->computeobjectiveandgradient=0;
    tao->ops->computegradient=0;
    tao->ops->computehessian=0;
    tao->ops->computeseparableobjective=0;
    tao->ops->computeconstraints=0;
    tao->ops->computejacobian=0;
    tao->ops->convergencetest=TaoDefaultConvergenceTest;
    tao->ops->convergencedestroy=0;
    tao->ops->computedual=0;
    tao->ops->setup=0;
    tao->ops->solve=0;
    tao->ops->view=0;
    tao->ops->setfromoptions=0;
    tao->ops->destroy=0;

    tao->solution=PETSC_NULL;
    tao->gradient=PETSC_NULL;
    tao->sep_objective = PETSC_NULL;
    tao->constraints=PETSC_NULL;
    tao->stepdirection=PETSC_NULL;
    tao->XL = PETSC_NULL;
    tao->XU = PETSC_NULL;
    tao->hessian = PETSC_NULL;
    tao->hessian_pre = PETSC_NULL;
    tao->jacobian = PETSC_NULL;
    tao->jacobian_pre = PETSC_NULL;
    tao->jacobian_state = PETSC_NULL;
    tao->jacobian_state_pre = PETSC_NULL;
    tao->jacobian_state_inv = PETSC_NULL;
    tao->jacobian_design = PETSC_NULL;
    tao->jacobian_design_pre = PETSC_NULL;
    tao->state_is = PETSC_NULL;
    tao->design_is = PETSC_NULL;

    tao->max_it     = 10000;
    tao->max_funcs   = 10000;
    tao->fatol       = 1e-8;
    tao->frtol       = 1e-8;
    tao->gatol       = 1e-8;
    tao->grtol       = 1e-8;
    tao->gttol       = 0.0;
    tao->catol       = 0.0;
    tao->crtol       = 0.0;
    tao->xtol        = 0.0;
    tao->steptol       = 0.0;
    tao->trust0      = TAO_INFINITY;
    tao->fmin        = -1e100;
    tao->hist_reset = PETSC_TRUE;
    tao->hist_max = 0;
    tao->hist_len = 0;
    tao->hist_obj = PETSC_NULL;
    tao->hist_resid = PETSC_NULL;
    tao->hist_cnorm = PETSC_NULL;

    tao->numbermonitors=0;
    tao->viewsolution=PETSC_FALSE;
    tao->viewhessian=PETSC_FALSE;
    tao->viewgradient=PETSC_FALSE;
    tao->viewjacobian=PETSC_FALSE;
    tao->viewconstraints = PETSC_FALSE;
    tao->viewtao = PETSC_FALSE;
    
    ierr = TaoResetStatistics(tao); CHKERRQ(ierr);


    *newtao = tao; 
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolve"
/*@ 
  TaoSolve - Solves an optimization problem min F(x) s.t. l <= x <= u

  Collective on TaoSolver
  
  Input Parameters:
. tao - the TaoSolver context

  Notes:
  The user must set up the TaoSolver with calls to TaoSetInitialVector(),
  TaoSetObjectiveRoutine(),
  TaoSetGradientRoutine(), and (if using 2nd order method) TaoSetHessianRoutine().

  Level: beginner
  .seealso: TaoCreate(), TaoSetObjectiveRoutine(), TaoSetGradientRoutine(), TaoSetHessianRoutine()
 @*/
PetscErrorCode TaoSolve(TaoSolver tao)
{
  PetscErrorCode ierr;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscViewer    viewer;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);

  ierr = TaoSetUp(tao);CHKERRQ(ierr);
  ierr = TaoResetStatistics(tao); CHKERRQ(ierr);
  if (tao->linesearch) {
    ierr = TaoLineSearchReset(tao->linesearch); CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(TaoSolver_Solve,tao,0,0,0); CHKERRQ(ierr);
  if (tao->ops->solve){ ierr = (*tao->ops->solve)(tao);CHKERRQ(ierr); }
  ierr = PetscLogEventEnd(TaoSolver_Solve,tao,0,0,0); CHKERRQ(ierr);


  
  ierr = PetscOptionsGetString(((PetscObject)tao)->prefix,"-tao_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)tao)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = TaoView(tao,viewer);CHKERRQ(ierr); 
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  

  if (tao->printreason) { 
      if (tao->reason > 0) {
	  ierr = PetscPrintf(((PetscObject)tao)->comm,"TAO solve converged due to %s\n",TaoSolverTerminationReasons[tao->reason]); CHKERRQ(ierr);
      } else {
	  ierr = PetscPrintf(((PetscObject)tao)->comm,"TAO solve did not converge due to %s\n",TaoSolverTerminationReasons[tao->reason]); CHKERRQ(ierr);
      }
  }
  

  PetscFunctionReturn(0);

    
}


#undef __FUNCT__
#define __FUNCT__ "TaoSetUp"
/*@ 
  TaoSetUp - Sets up the internal data structures for the later use
  of a Tao solver

  Collective on tao
  
  Input Parameters:
. tao - the TAO context

  Notes:
  The user will not need to explicitly call TaoSetUp(), as it will 
  automatically be called in TaoSolve().  However, if the user
  desires to call it explicitly, it should come after TaoCreate()
  and TaoSetXXX(), but before TaoSolve().

  Level: advanced

.seealso: TaoCreate(), TaoSolve()
@*/
PetscErrorCode TaoSetUp(TaoSolver tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAOSOLVER_CLASSID,1); 
  if (tao->setupcalled) PetscFunctionReturn(0);

  if (!tao->solution) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetInitialVector");
  }
  if (tao->ops->setup) {
    ierr = (*tao->ops->setup)(tao); CHKERRQ(ierr);
  }
  
  tao->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy"
/*@ 
  TaoDestroy - Destroys the TAO context that was created with 
  TaoCreate()

  Collective on TaoSolver

  Input Parameter:
. tao - the TaoSolver context

  Level: beginner

.seealso: TaoCreate(), TaoSolve()
@*/
PetscErrorCode TaoDestroy(TaoSolver *tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!*tao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*tao,TAOSOLVER_CLASSID,1);

  if (--((PetscObject)*tao)->refct > 0) {*tao=0;PetscFunctionReturn(0);}

  ierr = PetscObjectDepublish(*tao); CHKERRQ(ierr);
  
  if ((*tao)->ops->destroy) {
    ierr = (*((*tao))->ops->destroy)(*tao); CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&(*tao)->ksp); CHKERRQ(ierr);
  ierr = TaoLineSearchDestroy(&(*tao)->linesearch); CHKERRQ(ierr);

  if ((*tao)->ops->convergencedestroy) {
    ierr = (*(*tao)->ops->convergencedestroy)((*tao)->cnvP); CHKERRQ(ierr);
    if ((*tao)->jacobian_state_inv) {
      ierr = MatDestroy(&(*tao)->jacobian_state_inv); CHKERRQ(ierr);
      (*tao)->jacobian_state_inv = PETSC_NULL;
    }
  }
  ierr = VecDestroy(&(*tao)->solution); CHKERRQ(ierr);
  ierr = VecDestroy(&(*tao)->gradient); CHKERRQ(ierr);

  ierr = VecDestroy(&(*tao)->XL); CHKERRQ(ierr);
  ierr = VecDestroy(&(*tao)->XU); CHKERRQ(ierr);
  ierr = VecDestroy(&(*tao)->stepdirection); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->hessian_pre); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->hessian); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_pre); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_state_pre); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_state); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_state_inv); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_design_pre); CHKERRQ(ierr);
  ierr = MatDestroy(&(*tao)->jacobian_design); CHKERRQ(ierr);
  ierr = ISDestroy(&(*tao)->state_is); CHKERRQ(ierr);
  ierr = ISDestroy(&(*tao)->design_is); CHKERRQ(ierr);
  ierr = TaoCancelMonitors(*tao); CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(tao); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetFromOptions"
/*@
  TaoSetFromOptions - Sets various TaoSolver parameters from user
  options.

  Collective on TaoSolver

  Input Paremeter:
. tao - the TaoSolver solver context

  Options Database Keys:
+ -tao_method <type> - The algorithm that TAO uses (tao_lmvm, tao_nls, etc.)
. -tao_fatol <fatol> - absolute error tolerance in function value
. -tao_frtol <frtol> - relative error tolerance in function value
. -tao_gatol <gatol> - absolute error tolerance for ||gradient||
. -tao_grtol <grtol> - relative error tolerance for ||gradient||
. -tao_gttol <gttol> - reduction of ||gradient|| relative to initial gradient
. -tao_max_it <max> - sets maximum number of iterations
. -tao_max_funcs <max> - sets maximum number of function evaluations
. -tao_fmin <fmin> - stop if function value reaches fmin
. -tao_steptol <tol> - stop if trust region radius less than <tol>
. -tao_trust0 <t> - initial trust region radius
. -tao_monitor - prints function value and residual at each iteration
. -tao_smonitor - same as tao_monitor, but truncates very small values
. -tao_cmonitor - prints function value, residual, and constraint norm at each iteration
. -tao_view_solution - prints solution vector at each iteration
. -tao_view_separableobjective - prints separable objective vector at each iteration
. -tao_view_step - prints step direction vector at each iteration
. -tao_view_gradient - prints gradient vector at each iteration
. -tao_draw_solution - graphically view solution vector at each iteration
. -tao_draw_step - graphically view step vector at each iteration
. -tao_draw_gradient - graphically view gradient at each iteration
. -tao_fd_gradient - use gradient computed with finite differences
. -tao_cancelmonitors - cancels all monitors (except those set with command line)
. -tao_view - prints information about the TaoSolver after solving
- -tao_converged_reason - prints the reason TAO stopped iterating

  Notes:
  To see all options, run your program with the -help option or consult the 
  user's manual. Should be called after TaoCreate() but before TaoSolve()

  Level: beginner
@*/
PetscErrorCode TaoSetFromOptions(TaoSolver tao)
{
    PetscErrorCode ierr;
    const TaoSolverType default_type = "tao_lmvm";
    const char *prefix;
    char type[256], monfilename[PETSC_MAX_PATH_LEN];
    PetscViewer monviewer;
    PetscBool flg;
    MPI_Comm comm;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    ierr = PetscObjectGetComm((PetscObject)tao,&comm); CHKERRQ(ierr);
    ierr = TaoGetOptionsPrefix(tao,&prefix);
    /* So no warnings are given about unused options */
    ierr = PetscOptionsHasName(prefix,"-tao_ksp_type",&flg);
    ierr = PetscOptionsHasName(prefix,"-tao_pc_type",&flg);
    ierr = PetscOptionsHasName(prefix,"-tao_ls_type",&flg);
    

    ierr = PetscOptionsBegin(comm, ((PetscObject)tao)->prefix,"TaoSolver options","TaoSolver"); CHKERRQ(ierr);
    {
							

	if (!TaoSolverRegisterAllCalled) {
	    ierr = TaoSolverRegisterAll(PETSC_NULL); CHKERRQ(ierr);
	}
	if (((PetscObject)tao)->type_name) {
	    default_type = ((PetscObject)tao)->type_name;
	}
	/* Check for type from options */
	ierr = PetscOptionsList("-tao_type","Tao Solver type","TaoSetType",TaoSolverList,default_type,type,256,&flg); CHKERRQ(ierr);
	if (flg) {
	    ierr = TaoSetType(tao,type); CHKERRQ(ierr);
	} else {
	  ierr = PetscOptionsList("-tao_method","Tao Solver type","TaoSetType",TaoSolverList,default_type,type,256,&flg); CHKERRQ(ierr);
	  if (flg) {
  	    ierr = TaoSetType(tao,type); CHKERRQ(ierr);
	  } else if (!((PetscObject)tao)->type_name) {
	    ierr = TaoSetType(tao,default_type);
	  }
	}

	ierr = PetscOptionsBool("-tao_view","view TaoSolver info after each minimization has completed","TaoView",PETSC_FALSE,&tao->viewtao,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_fatol","Stop if solution within","TaoSetTolerances",tao->fatol,&tao->fatol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_frtol","Stop if relative solution within","TaoSetTolerances",tao->frtol,&tao->frtol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_catol","Stop if constraints violations within","TaoSetTolerances",tao->catol,&tao->catol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_crtol","Stop if relative contraint violations within","TaoSetTolerances",tao->crtol,&tao->crtol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_gatol","Stop if norm of gradient less than","TaoSetTolerances",tao->gatol,&tao->gatol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_grtol","Stop if norm of gradient divided by the function value is less than","TaoSetTolerances",tao->grtol,&tao->grtol,&flg);CHKERRQ(ierr); 
	ierr = PetscOptionsReal("-tao_gttol","Stop if the norm of the gradient is less than the norm of the initial gradient times tol","TaoSetTolerances",tao->gttol,&tao->gttol,&flg);CHKERRQ(ierr); 
	ierr = PetscOptionsInt("-tao_max_it","Stop if iteration number exceeds",
			    "TaoSetMaximumIterations",tao->max_it,&tao->max_it,
			    &flg);CHKERRQ(ierr);
	ierr = PetscOptionsInt("-tao_max_funcs","Stop if number of function evaluations exceeds","TaoSetMaximumFunctionEvaluations",tao->max_funcs,&tao->max_funcs,&flg); CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_fmin","Stop if function less than","TaoSetFunctionLowerBound",tao->fmin,&tao->fmin,&flg); CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_steptol","Stop if step size or trust region radius less than","",tao->steptol,&tao->steptol,&flg);CHKERRQ(ierr);
	ierr = PetscOptionsReal("-tao_trust0","Initial trust region radius","TaoSetTrustRegionRadius",tao->trust0,&tao->trust0,&flg);CHKERRQ(ierr); 

	ierr = PetscOptionsString("-tao_view_solution","view solution vector after each evaluation","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoSolutionMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy); CHKERRQ(ierr);
	}

	ierr = PetscOptionsBool("-tao_converged_reason","Print reason for TAO termination","TaoSolve",flg,&flg,PETSC_NULL); CHKERRQ(ierr);
	if (flg) {
	  tao->printreason = PETSC_TRUE;
	}
	ierr = PetscOptionsString("-tao_view_gradient","view gradient vector after each evaluation","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoGradientMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy); CHKERRQ(ierr);
	}

  
	ierr = PetscOptionsString("-tao_view_stepdirection","view step direction vector after each iteration","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoStepDirectionMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy); CHKERRQ(ierr);
	}

	ierr = PetscOptionsString("-tao_view_separableobjective","view separable objective vector after each evaluation","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoSeparableObjectiveMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy); CHKERRQ(ierr);
	}
	
	ierr = PetscOptionsString("-tao_monitor","Use the default convergence monitor","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoDefaultMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
	}

	ierr = PetscOptionsString("-tao_smonitor","Use the short convergence monitor","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoDefaultSMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
	}

	ierr = PetscOptionsString("-tao_cmonitor","Use the default convergence monitor with constraint norm","TaoSetMonitor","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
	if (flg) {
	  ierr = PetscViewerASCIIOpen(comm,monfilename,&monviewer); CHKERRQ(ierr);
	  ierr = TaoSetMonitor(tao,TaoDefaultCMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
	}


	ierr = PetscOptionsBool("-tao_cancelmonitors","cancel all monitors and call any registered destroy routines","TaoCancelMonitors",PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr); 
	if (flg) {ierr = TaoCancelMonitors(tao);CHKERRQ(ierr);} 

	ierr = PetscOptionsBool("-tao_draw_solution","Plot solution vector at each iteration","TaoSetMonitor",PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr);
	if (flg) {
	  ierr = TaoSetMonitor(tao,TaoDrawSolutionMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
	} 

	ierr = PetscOptionsBool("-tao_draw_step","plots step direction at each iteration","TaoSetMonitor",PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr);
	if (flg) {
	  ierr = TaoSetMonitor(tao,TaoDrawStepMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
	} 

	ierr = PetscOptionsBool("-tao_draw_gradient","plots gradient at each iteration","TaoSetMonitor",PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr);
	if (flg) {
	  ierr = TaoSetMonitor(tao,TaoDrawGradientMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
	}
	ierr = PetscOptionsBool("-tao_fd_gradient","compute gradient using finite differences","TaoDefaultComputeGradient",PETSC_FALSE,&flg,PETSC_NULL); CHKERRQ(ierr);
	if (flg) {
	  ierr = TaoSetGradientRoutine(tao,TaoDefaultComputeGradient,PETSC_NULL); CHKERRQ(ierr);
	}
	ierr = PetscOptionsEList("-tao_subset_type","subset type", "", TAO_SUBSET, TAO_SUBSET_TYPES,TAO_SUBSET[tao->subset_type], &tao->subset_type, 0); CHKERRQ(ierr);

	if (tao->ops->setfromoptions) {
	    ierr = (*tao->ops->setfromoptions)(tao); CHKERRQ(ierr);
	}
 
    }    
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "TaoView"
/*@C
  TaoView - Prints information about the TaoSolver
 
  Collective on TaoSolver

  InputParameters:
+ tao - the TaoSolver context
- viewer - visualization context

  Options Database Key:
. -tao_view - Calls TaoView() at the end of TaoSolve()

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
PetscErrorCode TaoView(TaoSolver tao, PetscViewer viewer)
{
    PetscErrorCode ierr;
    PetscBool isascii,isstring;
    const TaoSolverType type;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    if (!viewer) {
      ierr = PetscViewerASCIIGetStdout(((PetscObject)tao)->comm,&viewer); CHKERRQ(ierr);
    }
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(tao,1,viewer,2);

    ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii); CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring); CHKERRQ(ierr);
    if (isascii) {
        ierr = PetscObjectPrintClassNamePrefixType((PetscObject)tao,viewer,"TaoSolver"); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);

	if (tao->ops->view) {
	    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
	    ierr = (*tao->ops->view)(tao,viewer); CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
	}
	if (tao->linesearch) {
	  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)(tao->linesearch),viewer,"TaoLineSearch"); CHKERRQ(ierr);
	}
	if (tao->ksp) {
	  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)(tao->ksp),viewer,"KSP Solver"); CHKERRQ(ierr);
	  ierr = PetscViewerASCIIPrintf(viewer,"total KSP iterations: %D\n",tao->ksp_its); CHKERRQ(ierr);
	}
	if (tao->XL || tao->XU) {
	  ierr = PetscViewerASCIIPrintf(viewer,"Active Set subset type: %s\n",TAO_SUBSET[tao->subset_type]);
	}
	  
	ierr=PetscViewerASCIIPrintf(viewer,"convergence tolerances: fatol=%G,",tao->fatol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," frtol=%G\n",tao->frtol);CHKERRQ(ierr);

	ierr=PetscViewerASCIIPrintf(viewer,"convergence tolerances: gatol=%G,",tao->gatol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," steptol=%G,",tao->steptol);CHKERRQ(ierr);
	ierr=PetscViewerASCIIPrintf(viewer," gttol=%G\n",tao->gttol);CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(viewer,"Residual in Function/Gradient:=%G\n",tao->residual);CHKERRQ(ierr);

	if (tao->cnorm>0 || tao->catol>0 || tao->crtol>0){
	    ierr=PetscViewerASCIIPrintf(viewer,"convergence tolerances:");CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer," catol=%G,",tao->catol);CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer," crtol=%G\n",tao->crtol);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"Residual in Constraints:=%G\n",tao->cnorm);CHKERRQ(ierr);
	}

	if (tao->trust < tao->steptol){
	    ierr=PetscViewerASCIIPrintf(viewer,"convergence tolerances: steptol=%G\n",tao->steptol);CHKERRQ(ierr);
	    ierr=PetscViewerASCIIPrintf(viewer,"Final trust region radius:=%G\n",tao->trust);CHKERRQ(ierr);
	}

	if (tao->fmin>-1.e25){
	    ierr=PetscViewerASCIIPrintf(viewer,"convergence tolerances: function minimum=%G\n"
					,tao->fmin);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer,"Objective value=%G\n",
				      tao->fc);CHKERRQ(ierr);

	ierr = PetscViewerASCIIPrintf(viewer,"total number of iterations=%D,          ",
				      tao->niter);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"              (max: %D)\n",tao->max_it);CHKERRQ(ierr);

	if (tao->nfuncs>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of function evaluations=%D,",
					  tao->nfuncs);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"                max: %D\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->ngrads>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of gradient evaluations=%D,",
					  tao->ngrads);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"                max: %D\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->nfuncgrads>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of function/gradient evaluations=%D,",
					  tao->nfuncgrads);CHKERRQ(ierr);
	    ierr = PetscViewerASCIIPrintf(viewer,"    (max: %D)\n",
					  tao->max_funcs);CHKERRQ(ierr);
	}
	if (tao->nhess>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of Hessian evaluations=%D\n",
					  tao->nhess);CHKERRQ(ierr);
	}
/*	if (tao->linear_its>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"  total Krylov method iterations=%D\n",
					  tao->linear_its);CHKERRQ(ierr);
					  }*/
	if (tao->nconstraints>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of constraint function evaluations=%D\n",
					  tao->nconstraints);CHKERRQ(ierr);
	}
	if (tao->njac>0){
	    ierr = PetscViewerASCIIPrintf(viewer,"total number of Jacobian evaluations=%D\n",
					  tao->njac);CHKERRQ(ierr);
	}

	if (tao->reason>0){
	    ierr = PetscViewerASCIIPrintf(viewer,    "Solution converged: ");CHKERRQ(ierr);
	    switch (tao->reason) {
		case TAO_CONVERGED_FATOL:
		    ierr = PetscViewerASCIIPrintf(viewer,"estimated f(x)-f(X*) <= fatol\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_FRTOL:
		    ierr = PetscViewerASCIIPrintf(viewer,"estimated |f(x)-f(X*)|/|f(X*)| <= frtol\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_GATOL:
		    ierr = PetscViewerASCIIPrintf(viewer," ||g(X)|| <= gatol\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_GRTOL:
		    ierr = PetscViewerASCIIPrintf(viewer," ||g(X)||/|f(X)| <= grtol\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_GTTOL:
		    ierr = PetscViewerASCIIPrintf(viewer," ||g(X)||/||g(X0)|| <= gttol\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_STEPTOL:
		    ierr = PetscViewerASCIIPrintf(viewer," Steptol -- step size small\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_MINF:
		    ierr = PetscViewerASCIIPrintf(viewer," Minf --  f < fmin\n"); CHKERRQ(ierr);
		    break;
		case TAO_CONVERGED_USER:
		    ierr = PetscViewerASCIIPrintf(viewer," User Terminated\n"); CHKERRQ(ierr);
		    break;
		default:
  		    ierr = PetscViewerASCIIPrintf(viewer,"\n"); CHKERRQ(ierr);
		    break;
	    }		
	    
	} else {
	    ierr = PetscViewerASCIIPrintf(viewer,"Solver terminated: %D",tao->reason);CHKERRQ(ierr);
	    switch (tao->reason) {
	    case TAO_DIVERGED_MAXITS:
	      ierr = PetscViewerASCIIPrintf(viewer," Maximum Iterations\n");
	      CHKERRQ(ierr);
	      break;
	    case TAO_DIVERGED_NAN:
	      ierr = PetscViewerASCIIPrintf(viewer," NAN or Inf encountered\n");
	      CHKERRQ(ierr);
	      break;
	    case TAO_DIVERGED_MAXFCN:
	      ierr = PetscViewerASCIIPrintf(viewer," Maximum Function Evaluations\n");
	      CHKERRQ(ierr);
	      break;
	    case TAO_DIVERGED_LS_FAILURE:
	      ierr = PetscViewerASCIIPrintf(viewer," Line Search Failure\n");
	      CHKERRQ(ierr);
	      break;
	    case TAO_DIVERGED_TR_REDUCTION:
	      ierr = PetscViewerASCIIPrintf(viewer," Trust Region too small\n");
	      CHKERRQ(ierr);
	      break;
	    case TAO_DIVERGED_USER:
	      ierr = PetscViewerASCIIPrintf(viewer," User Terminated\n");
	      CHKERRQ(ierr);
	      break;
	    default:
	      ierr = PetscViewerASCIIPrintf(viewer,"\n"); CHKERRQ(ierr);
	      break;
	    }
	}
	ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    } else if (isstring) {
	ierr = TaoGetType(tao,&type); CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(viewer," %-3.3s",type); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetTolerances"
/*@
  TaoSetTolerances - Sets parameters used in TAO convergence tests

  Collective on TaoSolver

  Input Parameters
+ tao - the TaoSolver context
. fatol - absolute convergence tolerance
. frtol - relative convergence tolerance
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by this factor

  Options Database Keys:
+ -tao_fatol <fatol> - Sets fatol
. -tao_frtol <frtol> - Sets frtol
. -tao_gatol <catol> - Sets gatol
. -tao_grtol <catol> - Sets grtol
- .tao_gttol <crtol> - Sets gttol

  Stopping Criteria 
$ f(X) - f(X*) (estimated)            <= fatol 
$ |f(X) - f(X*)| (estimated) / |f(X)| <= frtol
$ ||g(X)||                            <= gatol
$ ||g(X)|| / |f(X)|                   <= grtol
$ ||g(X)|| / ||g(X0)||                <= gttol

  Notes: Use PETSC_DEFAULT to leave one or more tolerances unchanged.

  Level: beginner

.seealso: TaoGetTolerances()

@*/
PetscErrorCode TaoSetTolerances(TaoSolver tao, PetscReal fatol, PetscReal frtol, PetscReal gatol, PetscReal grtol, PetscReal gttol)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    
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
#define __FUNCT__ "TaoSetFunctionLowerBound"
/*@
   TaoSetFunctionLowerBound - Sets a bound on the solution objective value.
   When an approximate solution with an objective value below this number
   has been found, the solver will terminate.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
-  fmin - the tolerance

   Options Database Keys: 
.    -tao_fmin <fmin> - sets the minimum function value

   Level: intermediate

.keywords: options, View, Bounds,

.seealso: TaoSetTolerances()
@*/
PetscErrorCode TaoSetFunctionLowerBound(TaoSolver tao,PetscReal fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->fmin = fmin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoGetFunctionLowerBound"
/*@
   TaoGetFunctionLowerBound - Sets a bound on the solution objective value.
   When an approximate solution with an objective value below this number
   has been found, the solver will terminate.

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver solver context

   OutputParameters:
.  fmin - the tolerance

   Level: intermediate

.keywords: options, View, Bounds,

.seealso: TaoSetFunctionLowerBound()
@*/
PetscErrorCode TaoGetFunctionLowerBound(TaoSolver tao,PetscReal *fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *fmin = tao->fmin;
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "TaoSetMaximumFunctionEvaluations"
/*@
   TaoSetMaximumFunctionEvaluations - Sets a maximum number of 
   function evaluations.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
-  nfcn - the maximum number of function evaluations (>=0)

   Options Database Keys: 
.    -tao_max_funcs <nfcn> - sets the maximum number of function evaluations

   Level: intermediate

.keywords: options, Iterate,  convergence

.seealso: TaoSetTolerances(), TaoSetMaximumIterations()
@*/

PetscErrorCode TaoSetMaximumFunctionEvaluations(TaoSolver tao,PetscInt nfcn)
{
  PetscInt zero=0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->max_funcs = PetscMax(zero,nfcn);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoGetMaximumFunctionEvaluations"
/*@
   TaoGetMaximumFunctionEvaluations - Sets a maximum number of 
   function evaluations.

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver solver context

   Output Parameters
.  nfcn - the maximum number of function evaluations

   Level: intermediate

.keywords: options, Iterate,  convergence

.seealso: TaoSetMaximumFunctionEvaluations(), TaoGetMaximumIterations()
@*/

PetscErrorCode TaoGetMaximumFunctionEvaluations(TaoSolver tao,PetscInt *nfcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *nfcn = tao->max_funcs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSetMaximumIterations"
/*@
   TaoSetMaximumIterations - Sets a maximum number of iterates.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
-  maxits - the maximum number of iterates (>=0)

   Options Database Keys: 
.    -tao_max_it <its> - sets the maximum number of iterations

   Level: intermediate

.keywords: options, Iterate, convergence

.seealso: TaoSetTolerances(), TaoSetMaximumFunctionEvaluations()
@*/
PetscErrorCode TaoSetMaximumIterations(TaoSolver tao,PetscInt maxits)
{
  PetscInt zero=0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->max_it = PetscMax(zero,maxits);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetMaximumIterations"
/*@
   TaoGetMaximumIterations - Sets a maximum number of iterates.

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver solver context

   Output Parameters:
.  maxits - the maximum number of iterates 

   Level: intermediate

.keywords: options, Iterate, convergence

.seealso: TaoSetMaximumIterations(), TaoGetMaximumFunctionEvaluations()
@*/
PetscErrorCode TaoGetMaximumIterations(TaoSolver tao,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *maxits = tao->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetInitialTrustRegionRadius"
/*@
   TaoSetInitialTrustRegionRadius - Sets the initial trust region radius.

   Collective on TaoSolver

   Input Parameter:
+  tao - a TAO optimization solver
-  radius - the trust region radius

   Level: intermediate

   Options Database Key:
.  -tao_trust0

.keywords: trust region

.seealso: TaoGetTrustRegionRadius(), TaoSetTrustRegionTolerance()
@*/
PetscErrorCode TaoSetInitialTrustRegionRadius(TaoSolver tao, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->trust0 = PetscMax(0.0,radius);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetInitialTrustRegionRadius"
/*@
   TaoGetInitialTrustRegionRadius - Sets the initial trust region radius.

   Collective on TaoSolver

   Input Parameter:
.  tao - a TAO optimization solver

   Output Parameter:
.  radius - the trust region radius

   Level: intermediate

.keywords: trust region

.seealso: TaoSetTrustRegionRadius()
@*/
PetscErrorCode TaoGetInitialTrustRegionRadius(TaoSolver tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *radius = tao->trust0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGetCurrentTrustRegionRadius"
/*@
   TaoGetCurrentTrustRegionRadius - Gets the currenttrust region radius.

   Collective on TaoSolver

   Input Parameter:
.  tao - a TAO optimization solver

   Output Parameter:
.  radius - the trust region radius

   Level: intermediate

.keywords: trust region

.seealso: TaoSetTrustRegionRadius(), TaoGetInitialTrustRegionRadius()
@*/
PetscErrorCode TaoGetCurrentTrustRegionRadius(TaoSolver tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *radius = tao->trust;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetTolerances"
/*@
  TaoGetTolerances - gets the current values of tolerances

  Collective on TaoSolver

  Input Parameters:
. tao - the TaoSolver context
  
  Output Parameters:
+ fatol - absolute convergence tolerance
. frtol - relative convergence tolerance
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by a this factor

  Notes: Use PETSC_NULL as an argument to ignore one or more tolerances.
.seealso TaoSetTolerances()
 
  Level: intermediate
@*/
PetscErrorCode TaoGetTolerances(TaoSolver tao, PetscReal *fatol, PetscReal *frtol, PetscReal *gatol, PetscReal *grtol, PetscReal *gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (fatol) *fatol=tao->fatol;
  if (frtol) *frtol=tao->frtol;
  if (gatol) *gatol=tao->gatol;
  if (grtol) *grtol=tao->grtol;
  if (gttol) *gttol=tao->gttol;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetKSP"
/*@
  TaoGetKSP - Gets the linear solver used by the optimization solver.
  Application writers should use TaoGetKSP if they need direct access
  to the PETSc KSP object.
  
   Input Parameters:
.  tao - the TAO solver

   Output Parameters:
.  ksp - the KSP linear solver used in the optimization solver

   Level: intermediate

.keywords: Application
@*/
PetscErrorCode TaoGetKSP(TaoSolver tao, KSP *ksp) {
  PetscFunctionBegin;
  *ksp = tao->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGetLineSearch"
/*@
  TaoGetLineSearch - Gets the line search used by the optimization solver.
  Application writers should use TaoGetLineSearch if they need direct access
  to the TaoLineSearch object.
  
   Input Parameters:
.  tao - the TAO solver

   Output Parameters:
.  ls - the line search used in the optimization solver

   Level: intermediate

.keywords: Application
@*/
PetscErrorCode TaoGetLineSearch(TaoSolver tao, TaoLineSearch *ls) {
  PetscFunctionBegin;
  *ls = tao->linesearch;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoAddLineSearchCounts"
/*@
  TaoAddLineSearchCounts - Adds the number of function evaluations spent
  in the line search to the running total.
  
   Input Parameters:
+  tao - the TAO solver
-  ls - the line search used in the optimization solver

   Level: developer

.seealso: TaoLineSearchApply()

.keywords: Application
@*/
PetscErrorCode TaoAddLineSearchCounts(TaoSolver tao) {
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nfeval,ngeval,nfgeval;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (tao->linesearch) {
    ierr = TaoLineSearchIsUsingTaoSolverRoutines(tao->linesearch,&flg);
    if (flg == PETSC_FALSE) {
      ierr = TaoLineSearchGetNumberFunctionEvaluations(tao->linesearch,&nfeval,
						       &ngeval,&nfgeval); CHKERRQ(ierr);
      tao->nfuncs+=nfeval;
      tao->ngrads+=ngeval;
      tao->nfuncgrads+=nfgeval;
    }
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetSolutionVector"
/*@
  TaoGetSolutionVector - Returns the vector with the current TAO solution

  Input Parameter:
. tao - the TaoSolver context

  Output Parameter:
. X - the current solution

  Level: intermediate
 
  Note:  The returned vector will be the same object that was passed into TaoSetInitialSolution()
@*/
PetscErrorCode TaoGetSolutionVector(TaoSolver tao, Vec *X)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    *X = tao->solution;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoGetGradientVector"
/*@
  TaoGetGradientVector - Returns the vector with the current TAO gradient

  Input Parameter:
. tao - the TaoSolver context

  Output Parameter:
. G - the current solution

  Level: intermediate
@*/
PetscErrorCode TaoGetGradientVector(TaoSolver tao, Vec *G)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    *G = tao->gradient;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoResetStatistics"
/*@
   TaoResetStatistics - Initialize the statistics used by TAO for all of the solvers.
   These statistics include the iteration number, residual norms, and convergence status.
   This routine gets called before solving each optimization problem.

   Collective on TaoSolver

   Input Parameters:
.  solver - the TaoSolver context

   Level: developer

.keywords: options, defaults

.seealso: TaoCreate(), TaoSolve()
@*/
PetscErrorCode TaoResetStatistics(TaoSolver tao)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->niter        = 0;
    tao->nfuncs       = 0;
    tao->nfuncgrads   = 0;
    tao->ngrads       = 0;
    tao->nhess        = 0;
    tao->njac         = 0;
    tao->nconstraints = 0;
    tao->ksp_its      = 0;
    tao->reason       = TAO_CONTINUE_ITERATING;
    tao->residual     = 0.0;
    tao->cnorm        = 0.0;
    tao->step         = 0.0;
    tao->lsflag       = PETSC_FALSE;
    if (tao->hist_reset) tao->hist_len=0;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSetConvergenceTest"
/*@C
  TaoSetConvergenceTest - Sets the function that is to be used to test
  for convergence o fthe iterative minimization solution.  The new convergence
  testing routine will replace TAO's default convergence test.

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver object
. conv - the routine to test for convergence
- ctx - [optional] context for private data for the convergence routine
        (may be PETSC_NULL)

  Calling sequence of conv:
$   PetscErrorCode conv(TaoSolver tao, void *ctx)

+ tao - the TaoSolver object
- ctx - [optional] convergence context

  Note: The new convergence testing routine should call TaoSetTerminationReason().

  Level: advanced

.seealse: TaoSetTerminationReason(), TaoGetSolutionStatus(), TaoGetTolerances(), TaoSetMonitor

@*/
PetscErrorCode TaoSetConvergenceTest(TaoSolver tao, PetscErrorCode (*conv)(TaoSolver,void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  (tao)->ops->convergencetest = conv;
  (tao)->cnvP = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetMonitor"
/*@C
   TaoSetMonitor - Sets an ADDITIONAL function that is to be used at every
   iteration of the unconstrained minimization solver to display the iteration's 
   progress.   

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
.  mymonitor - monitoring routine
-  mctx - [optional] user-defined context for private data for the 
          monitor routine (may be PETSC_NULL)

   Calling sequence of mymonitor:
$     int mymonitor(TaoSolver tao,void *mctx)

+    tao - the TaoSolver solver context
-    mctx - [optional] monitoring context


   Options Database Keys:
+    -tao_monitor        - sets TaoDefaultMonitor()
.    -tao_smonitor       - sets short monitor
.    -tao_cmonitor       - same as smonitor plus constraint norm
.    -tao_view_solution   - view solution at each iteration
.    -tao_view_gradient   - view gradient at each iteration
.    -tao_view_separableobjective - view separable objective function at each iteration
-    -tao_cancelmonitors - cancels all monitors that have been hardwired into a code by calls to TaoSetMonitor(), but does not cancel those set via the options database.


   Notes: 
   Several different monitoring routines may be set by calling
   TaoSetMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.keywords: options, monitor, View

.seealso: TaoDefaultMonitor(), TaoCancelMonitors(),  TaoSetDestroyRoutine()
@*/
PetscErrorCode TaoSetMonitor(TaoSolver tao, PetscErrorCode (*func)(TaoSolver, void*), void *ctx,PetscErrorCode (*dest)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (tao->numbermonitors >= MAXTAOMONITORS) {
      SETERRQ1(PETSC_COMM_SELF,1,"Cannot attach another monitor -- max=",MAXTAOMONITORS);
  }
  tao->monitor[tao->numbermonitors] = func;
  tao->monitorcontext[tao->numbermonitors] = ctx;
  tao->monitordestroy[tao->numbermonitors] = dest;
  ++tao->numbermonitors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoCancelMonitors"
/*@
   TaoCancelMonitors - Clears all the monitor functions for a TaoSolver object.

   Collective on TaoSolver

   Input Parameters:
.  tao - the TaoSolver solver context

   Options Database:
.  -tao_cancelmonitors - cancels all monitors that have been hardwired
    into a code by calls to TaoSetMonitor(), but does not cancel those 
    set via the options database

   Notes: 
   There is no way to clear one specific monitor from a TaoSolver object.

   Level: advanced

.keywords: options, monitor, View

.seealso: TaoDefaultMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoCancelMonitors(TaoSolver tao)
{
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  for (i=0;i<tao->numbermonitors;i++) {
    if (tao->monitordestroy[i]) {
      ierr = (*tao->monitordestroy[i])(&tao->monitorcontext[i]); CHKERRQ(ierr);
    }
  }
  tao->numbermonitors=0;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "TaoDefaultMonitor"
/*@C
   TaoDefaultMonitor - Default routine for monitoring progress of the 
   TaoSolver solvers (default).  This monitor prints the function value and gradient
   norm at each iteration.  It can be turned on from the command line using the 
   -tao_smonitor option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_monitor

   Level: advanced

.seealso: TaoDefaultSMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoDefaultMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt its;
  PetscReal fct,gnorm;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  its=tao->niter;
  fct=tao->fc;
  gnorm=tao->residual;
  ierr=PetscViewerASCIIPrintf(viewer,"iter = %3D,",its); CHKERRQ(ierr);
  ierr=PetscViewerASCIIPrintf(viewer," Function value: %G,",fct); CHKERRQ(ierr);
  ierr=PetscViewerASCIIPrintf(viewer,"  Residual: %G \n",gnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "TaoDefaultSMonitor"
/*@C
   TaoDefaultSMonitor - Default routine for monitoring progress of the 

   same as TaoDefaultMonitor() except
   it prints fewer digits of the residual as the residual gets smaller.
   This is because the later digits are meaningless and are often 
   different on different machines; by using this routine different 
   machines will usually generate the same output.
   
   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_smonitor

   Level: advanced

.seealso: TaoDefaultMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoDefaultSMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt its;
  PetscReal fct,gnorm;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  its=tao->niter;
  fct=tao->fc;
  gnorm=tao->residual;
  ierr=PetscViewerASCIIPrintf(viewer,"iter = %3D,",its); CHKERRQ(ierr);
  ierr=PetscViewerASCIIPrintf(viewer," Function value %G,",fct); CHKERRQ(ierr);
  if (gnorm > 1.e-6) {
    ierr=PetscViewerASCIIPrintf(viewer," Residual: %G \n",gnorm);CHKERRQ(ierr);
  } else if (gnorm > 1.e-11) {
    ierr=PetscViewerASCIIPrintf(viewer," Residual: < 1.0e-6 \n"); CHKERRQ(ierr);
  } else {
    ierr=PetscViewerASCIIPrintf(viewer," Residual: < 1.0e-11 \n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDefaultCMonitor"
/*@C
   TaoDefaultCMonitor - Default routine for monitoring progress of the 

   same as TaoDefaultMonitor() except
   it prints the norm of the constraints function.
   
   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_cmonitor

   Level: advanced

.seealso: TaoDefaultMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoDefaultCMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt its;
  PetscReal fct,gnorm;
  PetscViewer viewer;

  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  its=tao->niter;
  fct=tao->fc;
  gnorm=tao->residual;
  ierr=PetscViewerASCIIPrintf(viewer,"iter = %D,",its); CHKERRQ(ierr);
  ierr=PetscViewerASCIIPrintf(viewer," Function value: %G,",fct); CHKERRQ(ierr);
  ierr=PetscViewerASCIIPrintf(viewer,"  Residual: %G ",gnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Constraint: %G \n",tao->cnorm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





#undef __FUNCT__
#define __FUNCT__ "TaoSolutionMonitor"
/*@C
   TaoSolutionMonitor - Views the solution at each iteration
   It can be turned on from the command line using the 
   -tao_view_solution option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_view_solution

   Level: advanced

.seealso: TaoDefaultSMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoSolutionMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  ierr = VecView(tao->solution, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGradientMonitor"
/*@C
   TaoGradientMonitor - Views the gradient at each iteration
   It can be turned on from the command line using the 
   -tao_view_gradient option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_view_gradient

   Level: advanced

.seealso: TaoDefaultSMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoGradientMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  ierr = VecView(tao->gradient, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoStepDirectionMonitor"
/*@C
   TaoStepDirectionMonitor - Views the gradient at each iteration
   It can be turned on from the command line using the 
   -tao_view_gradient option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_view_gradient

   Level: advanced

.seealso: TaoDefaultSMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoStepDirectionMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  ierr = VecView(tao->stepdirection, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDrawSolutionMonitor"
/*@C
   TaoDrawSolutionMonitor - Plots the solution at each iteration
   It can be turned on from the command line using the 
   -tao_draw_solution option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_draw_solution

   Level: advanced

.seealso: TaoSolutionMonitor(), TaoSetMonitor(), TaoDrawGradientMonitor
@*/
PetscErrorCode TaoDrawSolutionMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer = (PetscViewer) ctx;
  MPI_Comm comm;
  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscObjectGetComm((PetscObject)tao,&comm); CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(tao->solution, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDrawGradientMonitor"
/*@C
   TaoDrawGradientMonitor - Plots the gradient at each iteration
   It can be turned on from the command line using the 
   -tao_draw_gradient option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_draw_gradient

   Level: advanced

.seealso: TaoGradientMonitor(), TaoSetMonitor(), TaoDrawSolutionMonitor
@*/
PetscErrorCode TaoDrawGradientMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer = (PetscViewer)ctx;
  MPI_Comm comm;
  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscObjectGetComm((PetscObject)tao,&comm); CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(tao->gradient, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDrawStepMonitor"
/*@C
   TaoDrawStepMonitor - Plots the step direction at each iteration
   It can be turned on from the command line using the 
   -tao_draw_step option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_draw_step

   Level: advanced

.seealso: TaoSetMonitor(), TaoDrawSolutionMonitor
@*/
PetscErrorCode TaoDrawStepMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer = (PetscViewer)(ctx);
  MPI_Comm comm;
  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscObjectGetComm((PetscObject)tao,&comm); CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(tao->stepdirection, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSeparableObjectiveMonitor"
/*@C
   TaoSeparableObjectiveMonitor - Views the separable objective function at each iteration
   It can be turned on from the command line using the 
   -tao_view_separableobjective option

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  ctx - PetscViewer context or PETSC_NULL

   Options Database Keys:
.  -tao_view_separableobjective

   Level: advanced

.seealso: TaoDefaultSMonitor(), TaoSetMonitor()
@*/
PetscErrorCode TaoSeparableObjectiveMonitor(TaoSolver tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscFunctionBegin;
  if (ctx) {
    viewer = (PetscViewer)ctx;
  } else {
    viewer = PETSC_VIEWER_STDOUT_(((PetscObject)tao)->comm);
  }
  ierr = VecView(tao->sep_objective,viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDefaultConvergenceTest"
/*@C
   TaoDefaultConvergenceTest - Determines whether the solver should continue iterating
   or terminate. 

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  dummy - unused dummy context

   Output Parameter:
.  reason - for terminating

   Notes:
   This routine checks the residual in the optimality conditions, the 
   relative residual in the optimity conditions, the number of function
   evaluations, and the function value to test convergence.  Some
   solvers may use different convergence routines.

   Level: developer

.seealso: TaoSetTolerances(),TaoGetTerminationReason(),TaoSetTerminationReason()
@*/

PetscErrorCode TaoDefaultConvergenceTest(TaoSolver tao,void *dummy)
{
  PetscInt niter=tao->niter, nfuncs=PetscMax(tao->nfuncs,tao->nfuncgrads);
  PetscInt max_funcs=tao->max_funcs;
  PetscReal gnorm=tao->residual, gnorm0=tao->gnorm0;
  PetscReal f=tao->fc, steptol=tao->steptol,trradius=tao->step;
  PetscReal gatol=tao->gatol,grtol=tao->grtol,gttol=tao->gttol;
  PetscReal fatol=tao->fatol,frtol=tao->frtol,catol=tao->catol,crtol=tao->crtol;
  PetscReal fmin=tao->fmin, cnorm=tao->cnorm, cnorm0=tao->cnorm0;
  PetscReal gnorm2;
  TaoSolverTerminationReason reason=tao->reason;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAOSOLVER_CLASSID,1);
  
  if (reason != TAO_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }
  gnorm2=gnorm*gnorm;

  if (PetscIsInfOrNanReal(f)) {
    ierr = PetscInfo(tao,"Failed to converged, function value is Inf or NaN\n"); CHKERRQ(ierr);
    reason = TAO_DIVERGED_NAN;
  } else if (f <= fmin && cnorm <=catol) {
    ierr = PetscInfo2(tao,"Converged due to function value %G < minimum function value %G\n", f,fmin); CHKERRQ(ierr);
    reason = TAO_CONVERGED_MINF;
  } else if (gnorm2 <= fatol && cnorm <=catol) {
    ierr = PetscInfo2(tao,"Converged due to estimated f(X) - f(X*) = %G < %G\n",gnorm2,fatol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_FATOL;
  } else if (f != 0 && gnorm2 / PetscAbsReal(f)<= frtol && cnorm/PetscMax(cnorm0,1.0) <= crtol) {
    ierr = PetscInfo2(tao,"Converged due to estimated |f(X)-f(X*)|/f(X) = %G < %G\n",gnorm2/PetscAbsReal(f),frtol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_FRTOL;
  } else if (gnorm<= gatol && cnorm <=catol) {
    ierr = PetscInfo2(tao,"Converged due to residual norm ||g(X)||=%G < %G\n",gnorm,gatol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_GATOL;
  } else if ( f!=0 && PetscAbsReal(gnorm/f) <= grtol && cnorm <= crtol) {
    ierr = PetscInfo2(tao,"Converged due to residual ||g(X)||/|f(X)| =%G < %G\n",gnorm/f,grtol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_GRTOL;
  } else if (gnorm0 != 0 && gnorm/gnorm0 <= gttol && cnorm <= crtol) {
    ierr = PetscInfo2(tao,"Converged due to relative residual norm ||g(X)||/||g(X0)|| = %G < %G\n",gnorm/gnorm0,gttol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_GTTOL;
  } else if (nfuncs > max_funcs){
    ierr = PetscInfo2(tao,"Exceeded maximum number of function evaluations: %D > %D\n", nfuncs,max_funcs); CHKERRQ(ierr);
    reason = TAO_DIVERGED_MAXFCN;
  } else if ( tao->lsflag != 0 ){
    ierr = PetscInfo(tao,"Tao Line Search failure.\n"); CHKERRQ(ierr);
    reason = TAO_DIVERGED_LS_FAILURE;
  } else if (trradius < steptol && niter > 0){
    ierr = PetscInfo2(tao,"Trust region/step size too small: %G < %G\n", trradius,steptol); CHKERRQ(ierr);
    reason = TAO_CONVERGED_STEPTOL;
  } else if (niter > tao->max_it) {
    ierr = PetscInfo2(tao,"Exceeded maximum number of iterations: %D > %D\n",niter,tao->max_it); CHKERRQ(ierr);
    reason = TAO_DIVERGED_MAXITS;
  } else {
    reason = TAO_CONTINUE_ITERATING;
  }
  tao->reason = reason;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetOptionsPrefix"
/*@C
   TaoSetOptionsPrefix - Sets the prefix used for searching for all
   TAO options in the database.


   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver context
-  prefix - the prefix string to prepend to all TAO option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   For example, to distinguish between the runtime options for two
   different TAO solvers, one could call
.vb
      TaoSetOptionsPrefix(tao1,"sys1_")
      TaoSetOptionsPrefix(tao2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_tao_method blmvm -sys1_tao_gtol 1.e-3
      -sys2_tao_method lmvm  -sys2_tao_gtol 1.e-4
.ve


   Level: advanced

.keywords: options

.seealso: TaoAppendOptionsPrefix(), TaoGetOptionsPrefix()
@*/

PetscErrorCode TaoSetOptionsPrefix(TaoSolver tao, const char p[])
{
  PetscObjectSetOptionsPrefix((PetscObject)tao,p);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoAppendOptionsPrefix"
/*@C
   TaoAppendOptionsPrefix - Appends to the prefix used for searching for all
   TAO options in the database.


   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
-  prefix - the prefix string to prepend to all TAO option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.


   Level: advanced

.keywords: options

.seealso: TaoSetOptionsPrefix(), TaoGetOptionsPrefix()
@*/
PetscErrorCode TaoAppendOptionsPrefix(TaoSolver tao, const char p[])
{
  PetscObjectAppendOptionsPrefix((PetscObject)tao,p);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGetOptionsPrefix"
/*@C
  TaoGetOptionsPrefix - Gets the prefix used for searching for all 
  TAO options in the database

  Not Collective

  Input Parameters:
. tao - the TaoSolver context
  
  Output Parameters:
. prefix - pointer to the prefix string used is returned

  Notes: On the fortran side, the user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

  Level: advanced

.keywords: options

.seealso: TaoSetOptionsPrefix(), TaoAppendOptionsPrefix()
@*/
PetscErrorCode TaoGetOptionsPrefix(TaoSolver tao, const char *p[])
{
   PetscObjectGetOptionsPrefix((PetscObject)tao,p);
   return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetDefaultKSPType"
/*@
  TaoSetDefaultKSPType - Sets the default KSP type if a KSP object
  is created.

  Collective on TaoSolver

  InputParameters:
+ tao - the TaoSolver context
- ktype - the KSP type TAO will use by default

  Note: Some solvers may require a particular KSP type and will not work 
  correctly if the default value is changed

  Options Database Key:
- tao_ksp_type

  Level: advanced
@*/
PetscErrorCode TaoSetDefaultKSPType(TaoSolver tao, KSPType ktype)
{
  const char *prefix=0;
  char *option=0;
  size_t n1,n2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  ierr = TaoGetOptionsPrefix(tao,&prefix); CHKERRQ(ierr);
  ierr = PetscStrlen(prefix,&n1);
  ierr = PetscStrlen("_ksp_type",&n2);
  ierr = PetscMalloc(n1+n2+1,&option);
  ierr = PetscStrncpy(option,prefix,n1+1);
  ierr = PetscStrncat(option,"_ksp_type",n2+1);
  ierr = PetscOptionsSetValue(option,ktype); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetDefaulLineSearchType"
/*@
  TaoSetDefaultLineSearchType - Sets the default LineSearch type if a LineSearch object
  is created.

  Collective on TaoSolver

  InputParameters:
+ tao - the TaoSolver context
- lstype - the line search type TAO will use by default

  Note: Some solvers may require a particular line search type and will not work 
  correctly if the default value is changed

  Options Database Key:
- tao_ls_type

  Level: advanced
@*/
PetscErrorCode TaoSetDefaultLineSearchType(TaoSolver tao, TaoLineSearchType lstype)
{
  const char *prefix=0;
  char *option=0;
  size_t n1,n2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  ierr = TaoGetOptionsPrefix(tao,&prefix); CHKERRQ(ierr);
  ierr = PetscStrlen(prefix,&n1);
  ierr = PetscStrlen("_ls_type",&n2);
  ierr = PetscMalloc(n1+n2+1,&option);
  ierr = PetscStrncpy(option,prefix,n1+1);
  ierr = PetscStrncat(option,"_ls_type",n2+1);
  ierr = PetscOptionsSetValue(option,lstype); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetDefaultPCType"
/*@
  TaoSetDefaultPCType - Sets the default PC type if a PC object
  is created.

  Collective on TaoSolver

  InputParameters:
+ tao - the TaoSolver context
- pctype - the preconditioner type TAO will use by default

  Note: Some solvers may require a particular PC type and will not work 
  correctly if the default value is changed

  Options Database Key:
- tao_pc_type

  Level: advanced
@*/
PetscErrorCode TaoSetDefaultPCType(TaoSolver tao, PCType pctype)
{
  
  const char *prefix=0;
  char *option=0;
  size_t n1,n2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  ierr = TaoGetOptionsPrefix(tao,&prefix); CHKERRQ(ierr);
  ierr = PetscStrlen(prefix,&n1);
  ierr = PetscStrlen("_pc_type",&n2);
  ierr = PetscMalloc(n1+n2+1,&option);
  ierr = PetscStrncpy(option,prefix,n1+1);
  ierr = PetscStrncat(option,"_pc_type",n2+1);
  ierr = PetscOptionsSetValue(option,pctype); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetType"
/*@C
   TaoSetType - Sets the method for the unconstrained minimization solver.  

   Collective on TaoSolver

   Input Parameters:
+  solver - the TaoSolver solver context
-  type - a known method

   Options Database Key:
+  -tao_method <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_method tao_lmvm" or 
   "-tao_method tao_tron")
-  -tao_type <type> - identical to -tao_method

   Available methods include:
+    tao_nls - Newton's method with line search for unconstrained minimization
.    tao_ntr - Newton's method with trust region for unconstrained minimization
.    tao_ntl - Newton's method with trust region, line search for unconstrained minimization
.    tao_lmvm - Limited memory variable metric method for unconstrained minimization
.    tao_cg - Nonlinear conjugate gradient method for unconstrained minimization
.    tao_nm - Nelder-Mead algorithm for derivate-free unconstrained minimization
.    tao_tron - Newton Trust Region method for bound constrained minimization
.    tao_gpcg - Newton Trust Region method for quadratic bound constrained minimization
.    tao_blmvm - Limited memory variable metric method for bound constrained minimization
+    tao_pounders - Model-based algorithm pounder extended for nonlinear least squares 

  Level: intermediate

.seealso: TaoCreate(), TaoGetMethod(), TaoGetType()

@*/
PetscErrorCode TaoSetType(TaoSolver tao, const TaoSolverType type)
{
    PetscErrorCode ierr;
    PetscErrorCode (*create_xxx)(TaoSolver);
    PetscBool  issame;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    
    ierr = PetscTypeCompare((PetscObject)tao,type,&issame); CHKERRQ(ierr);
    if (issame) PetscFunctionReturn(0);

    ierr = PetscFListFind(TaoSolverList,((PetscObject)tao)->comm, type, PETSC_TRUE, (void(**)(void))&create_xxx); CHKERRQ(ierr);
    if (!create_xxx) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested TaoSolver type %s",type);
    }
    

    /* Destroy the existing solver information */
    if (tao->ops->destroy) {
	ierr = (*tao->ops->destroy)(tao); CHKERRQ(ierr);
    }
    ierr = KSPDestroy(&tao->ksp); CHKERRQ(ierr);
    ierr = TaoLineSearchDestroy(&tao->linesearch); CHKERRQ(ierr);
    ierr = VecDestroy(&tao->gradient); CHKERRQ(ierr);
    ierr = VecDestroy(&tao->stepdirection); CHKERRQ(ierr);

    tao->ops->setup = 0;
    tao->ops->solve = 0;
    tao->ops->view  = 0;
    tao->ops->setfromoptions = 0;
    tao->ops->destroy = 0;

    tao->setupcalled = PETSC_FALSE;

    ierr = (*create_xxx)(tao); CHKERRQ(ierr);
    ierr = PetscObjectChangeTypeName((PetscObject)tao,type); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
    
}
#undef __FUNCT__
#define __FUNCT__ "TaoSolverRegister"
/*MC
   TaoSolverRegister - Adds a method to the TAO package for unconstrained minimization.

   Synopsis:
   TaoSolverRegister(char *name_solver,char *path,char *name_Create,int (*routine_Create)(TaoSolver))

   Not collective

   Input Parameters:
+  sname - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  cname - name of routine to Create method context
-  func - routine to Create method context

   Notes:
   TaoSolverRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (func)
   is ignored.

   Environmental variables such as ${TAO_DIR}, ${PETSC_ARCH}, ${PETSC_DIR}, 
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with the appropriate values used when PETSc and TAO were compiled.

   Sample usage:
.vb
   TaoSolverRegister("my_solver","/home/username/mylibraries/${PETSC_ARCH}/mylib.a",
                "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TaoSetType(tao,"my_solver")
   or at runtime via the option
$     -tao_method my_solver

   Level: advanced

.seealso: TaoSolverRegisterAll(), TaoSolverRegisterDestroy()
M*/
PetscErrorCode TaoSolverRegister(const char sname[], const char path[], const char cname[], PetscErrorCode (*func)(TaoSolver))
{
    char fullname[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscFListConcat(path,cname,fullname); CHKERRQ(ierr);
    ierr = PetscFListAdd(&TaoSolverList,sname, fullname,(void (*)(void))func); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverRegisterDestroy"
/*@C
   TaoSolverRegisterDestroy - Frees the list of minimization solvers that were
   registered by TaoSolverRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: TaoSolverRegisterAll(), TaoSolverRegister()
@*/
PetscErrorCode TaoSolverRegisterDestroy(void)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscFListDestroy(&TaoSolverList); CHKERRQ(ierr);
    TaoSolverRegisterAllCalled = PETSC_FALSE;
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoSetTerminationReason"
/*@
  TaoSetTerminationReason - Sets the termination flag on a TaoSolver object

  Collective on TaoSolver

  Input Parameters:
+ tao - the TaoSolver context
- reason - one of
$     TAO_CONVERGED_ATOL (2),
$     TAO_CONVERGED_RTOL (3),
$     TAO_CONVERGED_STEPTOL (4),
$     TAO_CONVERGED_MINF (5),
$     TAO_CONVERGED_USER (6),
$     TAO_DIVERGED_MAXITS (-2),
$     TAO_DIVERGED_NAN (-4),
$     TAO_DIVERGED_MAXFCN (-5),
$     TAO_DIVERGED_LS_FAILURE (-6),
$     TAO_DIVERGED_TR_REDUCTION (-7),
$     TAO_DIVERGED_USER (-8),
$     TAO_CONTINUE_ITERATING (0)

   Level: intermediate

@*/
PetscErrorCode TaoSetTerminationReason(TaoSolver tao, TaoSolverTerminationReason reason)
{
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  PetscFunctionBegin;
  tao->reason = reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGetTerminationReason"
/*@ 
   TaoGetTerminationReason - Gets the reason the TaoSolver iteration was stopped.

   Not Collective

   Input Parameter:
.  tao - the TaoSolver solver context

   Output Parameter:
.  reason - one of


$  TAO_CONVERGED_FATOL (1)           f(X)-f(X*) <= fatol
$  TAO_CONVERGED_FRTOL (2)           |f(X) - f(X*)|/|f(X)| < frtol 
$  TAO_CONVERGED_GATOL (3)           ||g(X)|| < gatol 
$  TAO_CONVERGED_GRTOL (4)           ||g(X)|| / f(X)  < grtol
$  TAO_CONVERGED_GTTOL (5)           ||g(X)|| / ||g(X0)|| < gttol
$  TAO_CONVERGED_STEPTOL (6)         step size small 
$  TAO_CONVERGED_MINF (7)            F < F_min
$  TAO_CONVERGED_USER (8)            User defined

$  TAO_DIVERGED_MAXITS (-2)          its > maxits
$  TAO_DIVERGED_NAN (-4)             Numerical problems
$  TAO_DIVERGED_MAXFCN (-5)          fevals > max_funcsals
$  TAO_DIVERGED_LS_FAILURE (-6)      line search failure
$  TAO_DIVERGED_TR_REDUCTION (-7)    trust region failure
$  TAO_DIVERGED_USER(-8)             (user defined)

$  TAO_CONTINUE_ITERATING (0)

   where
+  X - current solution
.  X0 - initial guess
.  f(X) - current function value
.  f(X*) - true solution (estimated)
.  g(X) - current gradient
.  its - current iterate number
.  maxits - maximum number of iterates
.  fevals - number of function evaluations
-  max_funcsals - maximum number of function evaluations

   Level: intermediate

.keywords: convergence, View

.seealso: TaoSetConvergenceTest(), TaoSetTolerances()

@*/
PetscErrorCode TaoGetTerminationReason(TaoSolver tao, TaoSolverTerminationReason *reason) 
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidPointer(reason,2);
    *reason = tao->reason;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoGetSolutionStatus"
/*@
  TaoGetSolutionStatus - Get the current iterate, objective value,
  residual, infeasibility, and termination 

   Input Parameters:
.  tao - the TaoSolver context

   Output Parameters:
+  iterate - the current iterate number (>=0)
.  f - the current function value
.  gnorm - the square of the gradient norm, duality gap, or other measure
indicating distance from optimality.
.  cnorm - the infeasibility of the current solution with regard to the constraints.
.  xdiff - the step length or trust region radius of the most recent iterate.
-  reason - The termination reason, which can equal TAO_CONTINUE_ITERATING

   Level: intermediate

   Note:
   TAO returns the values set by the solvers in the routine TaoMonitor().

   Note:
   If any of the output arguments are set to PETSC_NULL, no value will be 
   returned.


.seealso: TaoMonitor(), TaoGetTerminationReason()

.keywords: convergence, monitor
@*/
PetscErrorCode TaoGetSolutionStatus(TaoSolver tao, PetscInt *its, PetscReal *f, PetscReal *gnorm, PetscReal *cnorm, PetscReal *xdiff, TaoSolverTerminationReason *reason)
{
  PetscFunctionBegin;
  if (its) *its=tao->niter;
  if (f) *f=tao->fc;
  if (gnorm) *gnorm=tao->residual;
  if (cnorm) *cnorm=tao->cnorm;
  if (reason) *reason=tao->reason;
  if (xdiff) *xdiff=tao->step;

  PetscFunctionReturn(0);
  
}


#undef __FUNCT__ 
#define __FUNCT__ "TaoGetType"
/*@C
   TaoGetType - Gets the current TaoSolver algorithm.

   Not Collective

   Input Parameter:
.  tao - the TaoSolver solver context

   Output Parameter:
.  type - TaoSolver method 

   Level: intermediate

.keywords: method
@*/
PetscErrorCode TaoGetType(TaoSolver tao, const TaoSolverType *type)
{
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    PetscValidPointer(type,2); 
    *type=((PetscObject)tao)->type_name;
    PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoMonitor"
/*@C
  TaoMonitor - Monitor the solver and the current solution.  This
  routine will record the iteration number and residual statistics,
  call any monitors specified by the user, and calls the convergence-check routine.

   Input Parameters:
+  tao - the TaoSolver context
.  its - the current iterate number (>=0)
.  f - the current objective function value
.  res - the gradient norm, square root of the duality gap, or other measure
indicating distince from optimality.  This measure will be recorded and
used for some termination tests.
.  cnorm - the infeasibility of the current solution with regard to the constraints.
-  steplength - multiple of the step direction added to the previous iterate.

   Output Parameters:
.  reason - The termination reason, which can equal TAO_CONTINUE_ITERATING

   Options Database Key:
.  -tao_monitor - Use the default monitor, which prints statistics to standard output

.seealso TaoGetTerminationReason(), TaoDefaultMonitor(), TaoSetMonitor()

   Level: developer

@*/
PetscErrorCode TaoMonitor(TaoSolver tao, PetscInt its, PetscReal f, PetscReal res, PetscReal cnorm, PetscReal steplength, TaoSolverTerminationReason *reason) 
{
    PetscErrorCode ierr;
    int i;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
    tao->fc = f;
    tao->residual = res;
    tao->cnorm = cnorm;
    tao->step = steplength;
    tao->niter=its;
    TaoLogHistory(tao,f,res,cnorm);
    if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(res)) {
      SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
    }
    if (tao->ops->convergencetest) {
      ierr = (*tao->ops->convergencetest)(tao,tao->cnvP); CHKERRQ(ierr);
    }
    for (i=0;i<tao->numbermonitors;i++) {
      ierr = (*tao->monitor[i])(tao,tao->monitorcontext[i]); CHKERRQ(ierr);
    }
    *reason = tao->reason;
    
    PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "TaoSetHistory"
/*@
   TaoSetHistory - Sets the array used to hold the convergence history.

   Collective on TaoSolver

   Input Parameters:
+  tao - the TaoSolver solver context
.  obj   - array to hold objective value history
.  resid - array to hold residual history
.  cnorm - array to hold constraint violation history
.  na  - size of obj, resid, and cnorm
-  reset - PetscTrue indicates each new minimization resets the history counter to zero,
           else it continues storing new values for new minimizations after the old ones

   Notes:
   If set, TAO will fill the given arrays with the indicated
   information at each iteration.  If no information is desired
   for a given array, then PETSC_NULL may be used.

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: options, view, monitor, convergence, history

.seealso: TaoGetHistory()

@*/
PetscErrorCode TaoSetHistory(TaoSolver tao, PetscReal *obj, PetscReal *resid, PetscReal *cnorm, PetscInt na,PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->hist_obj = obj;
  tao->hist_resid = resid;
  tao->hist_cnorm = cnorm;
  tao->hist_max   = na;
  tao->hist_reset = reset;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoGetHistory"
/*@C
   TaoGetHistory - Gets the array used to hold the convergence history.

   Collective on TaoSolver

   Input Parameter:
+  tao - the TaoSolver solver context

   Output Parameters:
+  obj   - array used to hold objective value history
.  resid - array used to hold residual history
.  cnorm - array used to hold constraint violation history
-  nhist  - size of obj, resid, and cnorm (will be less than or equal to na given in TaoSetHistory)


   Notes:
    The calling sequence for this routine in Fortran is
$   call TaoGetHistory(TaoSolver tao, integer nhist, integer info)

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: advanced

.keywords: convergence, history, monitor, View

.seealso: TaoSetHistory()

@*/
PetscErrorCode TaoGetHistory(TaoSolver tao, PetscReal **obj, PetscReal **resid, PetscReal **cnorm, PetscInt *nhist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (obj)   *obj   = tao->hist_obj;
  if (cnorm) *cnorm = tao->hist_cnorm;
  if (resid) *resid = tao->hist_resid;
  if (nhist) *nhist   = tao->hist_len;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TaoSetApplicationContext"
/*@
   TaoSetApplicationContext - Sets the optional user-defined context for 
   a solver.  

   Logically Collective on TaoSolver

   Input Parameters:
+  tao  - the TaoSolver context
-  usrP - optional user context

   Level: intermediate

.seealso: TaoGetApplicationContext(), TaoSetApplicationContext()
@*/
PetscErrorCode  TaoSetApplicationContext(TaoSolver tao,void *usrP)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  tao->user = usrP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoGetApplicationContext"
/*@
   TaoGetApplicationContext - Gets the user-defined context for a
   TAO solvers.  

   Not Collective

   Input Parameter:
.  tao  - TaoSolver context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.seealso: TaoSetApplicationContext()
@*/
PetscErrorCode  TaoGetApplicationContext(TaoSolver tao,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  *(void**)usrP = tao->user;
  PetscFunctionReturn(0);
}
