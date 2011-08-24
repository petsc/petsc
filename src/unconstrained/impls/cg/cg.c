#include "taolinesearch.h"
#include "cg.h"

#define CG_FletcherReeves       0
#define CG_PolakRibiere         1
#define CG_PolakRibierePlus     2
#define CG_HestenesStiefel      3
#define CG_DaiYuan              4
#define CG_Types                5

static const char *CG_Table[64] = {
  "fr", "pr", "prp", "hs", "dy"
};


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_CG"
static PetscErrorCode TaoSolverSolve_CG(TaoSolver tao)
{
    TAO_CG *cgP = (TAO_CG*)tao->data;
    PetscErrorCode ierr;
    TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
    TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
    PetscReal step=1.0,f,gnorm,gnorm2,delta,gd,ginner,beta;
    PetscReal gd_old,gnorm2_old,f_old;
    PetscInt iter=0;

    
    PetscFunctionBegin;

    if (tao->XL || tao->XU || tao->ops->computebounds) {
      ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by cg algorithm\n"); CHKERRQ(ierr);
    }

    // Check convergence criteria
    ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient); CHKERRQ(ierr);
    ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
    if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
	SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
    }
    
    ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
    if (reason != TAO_CONTINUE_ITERATING) {
	PetscFunctionReturn(0);
    }

    
    // Set initial direction to -gradient
    ierr = VecCopy(tao->gradient, tao->stepdirection); CHKERRQ(ierr);
    ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);
    gnorm2 = gnorm*gnorm;
    
    // Set initial scaling for the function
    if (f != 0.0) {
	delta = 2.0*PetscAbsScalar(f) / gnorm2;
	delta = PetscMax(delta,cgP->delta_min);
	delta = PetscMin(delta,cgP->delta_max);
    } else {
	delta = 2.0 / gnorm2;
	delta = PetscMax(delta,cgP->delta_min);
	delta = PetscMin(delta,cgP->delta_max);
    }
    // Set counter for gradient and reset steps
    cgP->ngradsteps = 0;
    cgP->nresetsteps = 0;
    
    while (1) {
	// Save the current gradient information
	f_old = f;
	gnorm2_old = gnorm2;
	ierr = VecCopy(tao->solution, cgP->X_old); CHKERRQ(ierr);
	ierr = VecCopy(tao->gradient, cgP->G_old); CHKERRQ(ierr);
	ierr = VecDot(tao->gradient, tao->stepdirection, &gd); CHKERRQ(ierr);
	if ((gd >= 0) || TaoInfOrNaN(gd)) {
	    ++cgP->ngradsteps;
	    if (f != 0.0) {
		delta = 2.0*PetscAbsScalar(f) / gnorm2;
		delta = PetscMax(delta,cgP->delta_min);
		delta = PetscMin(delta,cgP->delta_max);
	    } else {
		delta = 2.0 / gnorm2;
		delta = PetscMax(delta,cgP->delta_min);
		delta = PetscMin(delta,cgP->delta_max);
	    }
	    
	    ierr = VecCopy(tao->gradient, tao->stepdirection); CHKERRQ(ierr);
	    ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);
	}
	
	// Search direction for improving point
	ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,delta);
	ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status); CHKERRQ(ierr);
	if (ls_status < 0) {
	    // Linesearch failed
	    // Reset factors and use scaled gradient step
	    ++cgP->nresetsteps;
	    f = f_old;
	    gnorm2 = gnorm2_old;
	    ierr = VecCopy(cgP->X_old, tao->solution); CHKERRQ(ierr);
	    ierr = VecCopy(cgP->G_old, tao->gradient); CHKERRQ(ierr);
	    
	    if (f != 0.0) {
		delta = 2.0*PetscAbsScalar(f) / gnorm2;
		delta = PetscMax(delta,cgP->delta_min);
		delta = PetscMin(delta,cgP->delta_max);
	    } else {
		delta = 2.0 / gnorm2;
		delta = PetscMax(delta,cgP->delta_min);
		delta = PetscMin(delta,cgP->delta_max);
	    }
	    
	    ierr = VecCopy(tao->gradient, tao->stepdirection); CHKERRQ(ierr);
	    ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);

	    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,delta);
	    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status); CHKERRQ(ierr);
	    
	    if (ls_status < 0) {
		// Linesearch failed again
		// switch to unscaled gradient
		f = f_old;
		gnorm2 = gnorm2_old;
		ierr = VecCopy(cgP->X_old, tao->solution); CHKERRQ(ierr);
		ierr = VecCopy(cgP->G_old, tao->gradient); CHKERRQ(ierr);
		delta = 1.0;
		ierr = VecCopy(tao->solution, tao->stepdirection); CHKERRQ(ierr);
		ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);

		ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,delta);
		ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status); CHKERRQ(ierr);
		if (ls_status < 0) {
		    // Line search failed for last time -- give up
		    f = f_old;
		    gnorm2 = gnorm2_old;
		    ierr = VecCopy(cgP->X_old, tao->solution); CHKERRQ(ierr);
		    ierr = VecCopy(cgP->G_old, tao->gradient); CHKERRQ(ierr);
		    step = 0.0;
		}
	    }
	}
	
	// Check for bad value
	ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
	if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
	    SETERRQ(PETSC_COMM_SELF,1,"User-provided compute function generated Inf or NaN");
	}
	
	// Check for termination
	gnorm2 =gnorm * gnorm;
	iter++;
	ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
	if (reason != TAO_CONTINUE_ITERATING) {
	    break;
	}

	// Check for restart condition
	ierr = VecDot(tao->gradient, cgP->G_old, &ginner); CHKERRQ(ierr);
	if (PetscAbsScalar(ginner) >= cgP->eta * gnorm2) {
	    // Gradients far from orthognal; use steepest descent direction
	    beta = 0.0;
	} else {
	    // Gradients close to orthogonal; use conjugate gradient formula
	    switch (cgP->cg_type) {
		case CG_FletcherReeves:
		    beta = gnorm2 / gnorm2_old;
		    break;
		    
		case CG_PolakRibiere:
		    beta = (gnorm2 - ginner) / gnorm2_old;
		    break;
		
		case CG_PolakRibierePlus:
		    beta = PetscMax((gnorm2-ginner)/gnorm2_old, 0.0);
		    break;

		case CG_HestenesStiefel:
		    ierr = VecDot(tao->gradient, tao->stepdirection, &gd); CHKERRQ(ierr);
		    ierr = VecDot(cgP->G_old, tao->stepdirection, &gd_old); CHKERRQ(ierr);
		    beta = (gnorm2 - ginner) / (gd - gd_old);
		    break;

		case CG_DaiYuan:
		    ierr = VecDot(tao->gradient, tao->stepdirection, &gd); CHKERRQ(ierr);
		    ierr = VecDot(cgP->G_old, tao->stepdirection, &gd_old); CHKERRQ(ierr);
		    beta = gnorm2 / (gd - gd_old);
		    break;

		default:
		    beta = 0.0;
		    break;
	    }
	}
	
	// Compute the direction d=-g + beta*d
	ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient); CHKERRQ(ierr);
	
	// update initial steplength choice
	delta = 1.0;
	delta = PetscMax(delta, cgP->delta_min);
	delta = PetscMin(delta, cgP->delta_max);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetUp_CG"
static PetscErrorCode TaoSolverSetUp_CG(TaoSolver tao)
{
    TAO_CG *cgP = (TAO_CG*)tao->data;
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);}
    if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr); }
    if (!cgP->X_old) {ierr = VecDuplicate(tao->solution,&cgP->X_old); CHKERRQ(ierr);}
    if (!cgP->G_old) {ierr = VecDuplicate(tao->gradient,&cgP->G_old); CHKERRQ(ierr); }

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_CG"
static PetscErrorCode TaoSolverDestroy_CG(TaoSolver tao)
{
    TAO_CG *cgP = (TAO_CG*) tao->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    if (tao->setupcalled) {
      ierr = VecDestroy(&cgP->X_old); CHKERRQ(ierr);
      ierr = VecDestroy(&cgP->G_old); CHKERRQ(ierr);
    }
    ierr = TaoLineSearchDestroy(&tao->linesearch); CHKERRQ(ierr);

    ierr = PetscFree(tao->data); CHKERRQ(ierr);
    tao->data = PETSC_NULL;
	
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_CG"
static PetscErrorCode TaoSolverSetFromOptions_CG(TaoSolver tao)
{
    TAO_CG *cgP = (TAO_CG*)tao->data;
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
    ierr = PetscOptionsHead("Nonlinear Conjugate Gradient method for unconstrained optimization"); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_cg_eta","restart tolerance", "", cgP->eta,&cgP->eta,0);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_cg_type","cg formula", "", CG_Table, CG_Types, CG_Table[cgP->cg_type], &cgP->cg_type, 0); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_cg_delta_min","minimum delta value", "", cgP->
delta_min,&cgP->delta_min,0); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_cg_delta_max","maximum delta value", "", cgP->
delta_max,&cgP->delta_max,0); CHKERRQ(ierr);
    ierr = PetscOptionsTail(); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}       

#undef __FUNCT__
#define __FUNCT__ "TaoSolverView_CG"
static PetscErrorCode TaoSolverView_CG(TaoSolver tao, PetscViewer viewer)
{
    PetscBool isascii;
    TAO_CG *cgP = (TAO_CG*)tao->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii); CHKERRQ(ierr);
    if (isascii) {
        ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "CG Type: %s\n", CG_Table[cgP->cg_type]); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %d\n", cgP->ngradsteps); CHKERRQ(ierr);
	ierr= PetscViewerASCIIPrintf(viewer, "Reset steps: %d\n", cgP->nresetsteps); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    } else {
      SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO CG",((PetscObject)viewer)->type_name);
    }
    PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoSolverCreate_CG"
PetscErrorCode TaoSolverCreate_CG(TaoSolver tao)
{
    TAO_CG *cgP;
    const char *morethuente_type = TAOLINESEARCH_MT;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    tao->ops->setup = TaoSolverSetUp_CG;
    tao->ops->solve = TaoSolverSolve_CG;
    tao->ops->view = TaoSolverView_CG;
    tao->ops->setfromoptions = TaoSolverSetFromOptions_CG;
    tao->ops->destroy = TaoSolverDestroy_CG;
    
    tao->max_its = 2000;
    tao->max_funcs = 4000;
    tao->fatol = 1e-4;
    tao->frtol = 1e-4;

    // Note: nondefault values should be used for nonlinear conjugate gradient 
    // method.  In particular, gtol should be less that 0.5; the value used in 
    // Nocedal and Wright is 0.10.  We use the default values for the 
    // linesearch because it seems to work better.
    ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
    ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);
    ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch, tao); CHKERRQ(ierr);
    
    
    ierr = PetscNewLog(tao,TAO_CG,&cgP); CHKERRQ(ierr);
    tao->data = (void*)cgP;
    cgP->eta = 0.1;
    cgP->delta_min = 1e-7;
    cgP->delta_max = 100;
    cgP->cg_type = CG_PolakRibierePlus;

    PetscFunctionReturn(0);
}

EXTERN_C_END

	    

		    
	    
