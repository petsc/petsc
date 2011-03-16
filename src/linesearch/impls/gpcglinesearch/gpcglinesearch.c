#include "private/taolinesearch_impl.h"
#include "gpcglinesearch.h" 

/* ---------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchDestroy_GPCG"
static PetscErrorCode TaoLineSearchDestroy_GPCG(TaoLineSearch ls) 
{
  PetscErrorCode  ierr;
  TAOLINESEARCH_GPCG_CTX *ctx = (TAOLINESEARCH_GPCG_CTX *)ls->data;

  PetscFunctionBegin;
  if (ctx->W1) { 
    ierr = VecDestroy(ctx->W1);CHKERRQ(ierr);
  }
  if (ctx->W2) {
    ierr = VecDestroy(ctx->W2);CHKERRQ(ierr);
  }
  if (ctx->Gold) {
    ierr = VecDestroy(ctx->Gold);CHKERRQ(ierr);
  }
  if (ctx->x) {
    ierr = PetscObjectDereference((PetscObject)ctx->x); CHKERRQ(ierr);
  }

  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ls->data = 0;
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchView_GPCG"
static PetscErrorCode TaoLineSearchView_GPCG(TaoLineSearch ls, PetscViewer viewer)
{
  PetscBool                 isascii;
  PetscErrorCode            ierr;
  PetscFunctionBegin;

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);

  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer," GPCG Line search"); CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ls)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for GPCG LineSearch",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchApply_GPCG"
static PetscErrorCode TaoLineSearchApply_GPCG(TaoLineSearch ls, Vec x, 
					      PetscReal *f, Vec g, Vec s)

{
  TAOLINESEARCH_GPCG_CTX *neP = (TAOLINESEARCH_GPCG_CTX *)ls->data;
  PetscErrorCode  ierr;
  PetscInt i;
  PetscReal d1,finit,actred,prered,rho, gdx;

  PetscFunctionBegin;
  /* ls->stepmin - lower bound for step */
  /* ls->stepmax - upper bound for step */
  /* ls->rtol 	  - relative tolerance for an acceptable step */
  /* ls->ftol 	  - tolerance for sufficient decrease condition */
  /* ls->gtol 	  - tolerance for curvature condition */
  /* ls->nfev 	  - number of function evaluations */
  /* ls->maxfev  - maximum number of function evaluations */

  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  ls->step = ls->initstep;
  if (!neP->W2) {
      ierr = VecDuplicate(x,&neP->W2); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&neP->W1); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&neP->Gold); CHKERRQ(ierr);
      neP->x = x;
      ierr = PetscObjectReference((PetscObject)neP->x); CHKERRQ(ierr);
  }

  /* If X has changed, remake work vectors */
  else if (x != neP->x) {
      ierr = VecDestroy(neP->x); CHKERRQ(ierr);
      ierr = VecDestroy(neP->W1); CHKERRQ(ierr);
      ierr = VecDestroy(neP->W2); CHKERRQ(ierr);
      ierr = VecDestroy(neP->Gold); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&neP->W1); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&neP->W2); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&neP->Gold); CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)neP->x); CHKERRQ(ierr);
      neP->x = x;
      ierr = PetscObjectReference((PetscObject)neP->x); CHKERRQ(ierr);
  }

  ierr = VecDot(g,s,&gdx); CHKERRQ(ierr);
  ierr = VecCopy(x,neP->W2); CHKERRQ(ierr);
  ierr = VecCopy(g,neP->Gold); CHKERRQ(ierr);
  if (ls->bounded) {
	/* Compute the smallest steplength that will make one nonbinding variable
	   equal the bound */
      ierr = VecStepBoundInfo(x,ls->lower,ls->upper,s,&rho,&actred,&d1); CHKERRQ(ierr);
      ls->step = PetscMin(ls->step,d1);
  }
  rho=0; actred=0;

  if (ls->step < 0) {
    ierr = PetscInfo1(ls,"Line search error: initial step parameter %g < 0\n",ls->step); CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_OTHER;
    PetscFunctionReturn(0);
  }

  /* Initialization */
  ls->nfev = 0;
  finit = *f;
  for (i=0; i< ls->maxfev; i++) {
    
    /* Force the step to be within the bounds */
    ls->step = PetscMax(ls->step,ls->stepmin);
    ls->step = PetscMin(ls->step,ls->stepmax);

    ierr = VecCopy(neP->W2,x); CHKERRQ(ierr);
    ierr = VecAXPY(x,ls->step,s); CHKERRQ(ierr);
    if (ls->bounded) {
	ierr = VecMedian(ls->lower,x,ls->upper,x); CHKERRQ(ierr);
    }
    ierr = TaoLineSearchComputeObjectiveAndGradient(ls,x,f,g); CHKERRQ(ierr);

    if (0 == i) {
	ls->f_fullstep = *f;
    }

    actred = *f - finit;
    ierr = VecCopy(x,neP->W1); CHKERRQ(ierr);
    ierr = VecAXPY(neP->W1,-1.0,neP->W2); CHKERRQ(ierr);    /* W1 = X - W2 */
    ierr = VecDot(neP->W1,neP->Gold,&prered); CHKERRQ(ierr);
    
    if (fabs(prered)<1.0e-100) prered=1.0e-12;
    rho = actred/prered;
    /* 
       If sufficient progress has been obtained, accept the
       point.  Otherwise, backtrack. 
    */

    if (rho > ls->ftol){
      break;
    } else{
      ls->step = (ls->step)/2;
    }

    /* Convergence testing */
  
    if (ls->step <= ls->stepmin || ls->step >= ls->stepmax) {
      ls->reason = TAOLINESEARCH_FAILED_OTHER;
      ierr = PetscInfo(ls,"Rounding errors may prevent further progress.  May not be a step satisfying\n"); CHKERRQ(ierr);
     ierr = PetscInfo(ls,"sufficient decrease and curvature conditions. Tolerances may be too small.\n"); CHKERRQ(ierr);
     break;
    }
    if (ls->step == ls->stepmax) {
      ierr = PetscInfo1(ls,"Step is at the upper bound, stepmax (%g)\n",ls->stepmax); CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_FAILED_UPPERBOUND;
      break;
    }
    if (ls->step == ls->stepmin) {
      ierr = PetscInfo1(ls,"Step is at the lower bound, stepmin (%g)\n",ls->stepmin); CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_FAILED_LOWERBOUND;
      break;
    }
    if (ls->nfev >= ls->maxfev) {
      ierr = PetscInfo2(ls,"Number of line search function evals (%d) > maximum (%d)\n",ls->nfev,ls->maxfev); CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_FAILED_MAXFCN;
      break;
    }
    if ((neP->bracket) && (ls->stepmax - ls->stepmin <= ls->rtol*ls->stepmax)){
        ierr = PetscInfo1(ls,"Relative width of interval of uncertainty is at most rtol (%g)\n",ls->rtol); CHKERRQ(ierr);
        ls->reason = TAOLINESEARCH_FAILED_RTOL;
	break;
    }
  }
  ierr = PetscInfo2(ls,"%d function evals in line search, step = %10.4f\n",ls->nfev,ls->step); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchCreate_GPCG"
PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch ls)
{
  PetscErrorCode ierr;
  TAOLINESEARCH_GPCG_CTX *neP;

  PetscFunctionBegin;



  ls->ftol		  = 0.05;
  ls->rtol		  = 0.0;
  ls->gtol		  = 0.0;
  ls->stepmin		  = 1.0e-20;
  ls->stepmax		  = 1.0e+20;
  ls->nfev		  = 0; 
  ls->maxfev		  = 30;
  ls->step                = 1.0;

  ierr = PetscNewLog(ls,TAOLINESEARCH_GPCG_CTX,&neP);CHKERRQ(ierr);
  neP->bracket		  = 0; 
  neP->infoc              = 1;
  ls->data = (void*)neP;

  ls->ops->setup = 0;
  ls->ops->apply=TaoLineSearchApply_GPCG;
  ls->ops->view =TaoLineSearchView_GPCG;
  ls->ops->destroy=TaoLineSearchDestroy_GPCG;
  ls->ops->setfromoptions=0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
