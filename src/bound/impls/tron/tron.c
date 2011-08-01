#include "tron.h"
#include "private/kspimpl.h"
#include "private/matimpl.h"
#include "src/matrix/submatfree.h"
static const char *TRON_SUBSET[64] = {
  "submat","mask","matrixfree"
};

#define TRON_SUBSET_SUBMAT 0
#define TRON_SUBSET_MASK 1
#define TRON_SUBSET_MATRIXFREE 2
#define TRON_SUBSET_TYPES 3



/* TRON Routines */
static PetscErrorCode TronGradientProjections(TaoSolver,TAO_TRON*);
static PetscErrorCode TronSetupKSP(TaoSolver, TAO_TRON*);
static PetscErrorCode TronApplyMask(TAO_TRON *tron, Mat M, Vec mask);

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_TRON"
static PetscErrorCode TaoSolverDestroy_TRON(TaoSolver tao)
{
  TAO_TRON *tron = (TAO_TRON *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (tao->setupcalled) {
    ierr = VecDestroy(&tron->X_New);CHKERRQ(ierr);
    ierr = VecDestroy(&tron->G_New);CHKERRQ(ierr);
    ierr = VecDestroy(&tron->Work);CHKERRQ(ierr);
  }
  if (tron->DXFree) {
    ierr = VecDestroy(&tron->DXFree);CHKERRQ(ierr);
  }
  if (tron->R) {
    ierr = VecDestroy(&tron->R); CHKERRQ(ierr);
  }
  if (tron->diag) {
    ierr = VecDestroy(&tron->diag); CHKERRQ(ierr);
  }
  if (tron->rmask) {
    ierr = VecDestroy(&tron->rmask); CHKERRQ(ierr);
  }
  if (tron->scatter) {
    ierr = VecScatterDestroy(&tron->scatter); CHKERRQ(ierr);
  }
  if (tron->Free_Local) {
    ierr = ISDestroy(&tron->Free_Local); CHKERRQ(ierr);
  }
  if (tron->H_sub) {
    ierr = MatDestroy(&tron->H_sub); CHKERRQ(ierr);
  }
  if (tron->Hpre_sub) {
    ierr = MatDestroy(&tron->Hpre_sub); CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_TRON"
static PetscErrorCode TaoSolverSetFromOptions_TRON(TaoSolver tao)
{
  TAO_TRON  *tron = (TAO_TRON *)tao->data;
  PetscErrorCode        ierr;
  PetscBool flg;

  PetscFunctionBegin;

  ierr = PetscOptionsHead("Newton Trust Region Method for bound constrained optimization");CHKERRQ(ierr);
  
  ierr = PetscOptionsInt("-tron_maxgpits","maximum number of gradient projections per TRON iterate","TaoSetMaxGPIts",tron->maxgpits,&tron->maxgpits,&flg);
  CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tron_subset_type","subset type", "", TRON_SUBSET, TRON_SUBSET_TYPES,TRON_SUBSET[tron->subset_type], &tron->subset_type, 0); CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_TRON"
static PetscErrorCode TaoSolverView_TRON(TaoSolver tao, PetscViewer viewer)
{
  TAO_TRON  *tron = (TAO_TRON *)tao->data;
  PetscBool isascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer," Total PG its: %d,",tron->total_gp_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," PG tolerance: %4.3f \n",tron->pg_ftol);CHKERRQ(ierr);
    ierr = TaoLineSearchView(tao->linesearch,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO TRON",((PetscObject)viewer)->type_name);
  }    
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetup_TRON"
static PetscErrorCode TaoSolverSetup_TRON(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_TRON *tron = (TAO_TRON *)tao->data;

  PetscFunctionBegin;

  /* Allocate some arrays */
  ierr = VecDuplicate(tao->solution, &tron->X_New); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->G_New); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->Work); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
  if (!tao->XL) {
      ierr = VecDuplicate(tao->solution, &tao->XL); CHKERRQ(ierr);
      ierr = VecSet(tao->XL, TAO_NINFINITY); CHKERRQ(ierr);
  }
  if (!tao->XU) {
      ierr = VecDuplicate(tao->solution, &tao->XU); CHKERRQ(ierr);
      ierr = VecSet(tao->XU, TAO_INFINITY); CHKERRQ(ierr);
  }
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_TRON"
static PetscErrorCode TaoSolverSolve_TRON(TaoSolver tao){

  TAO_TRON *tron = (TAO_TRON *)tao->data;;
  PetscErrorCode ierr;
  PetscInt iter=0;

  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal prered,actred,delta,f,f_new,rhok,gnorm,gdx,xdiff,stepsize;
  PetscFunctionBegin;

  tron->pgstepsize=1.0;


  /*   Project the current point onto the feasible set */
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution); CHKERRQ(ierr);

  
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&tron->f,tao->gradient);CHKERRQ(ierr);
  if (tron->Free_Local) {
    ierr = ISDestroy(&tron->Free_Local); CHKERRQ(ierr);
    tron->Free_Local = PETSC_NULL;
  }
  ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&tron->Free_Local); CHKERRQ(ierr);
  
  /* Project the gradient and calculate the norm */
  ierr = VecBoundGradientProjection(tao->gradient,tao->solution, tao->XL, tao->XU, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&tron->gnorm); CHKERRQ(ierr);

  if (TaoInfOrNaN(tron->f) || TaoInfOrNaN(tron->gnorm)) {
    SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf pr NaN");
  }

  if (tao->trust <= 0) {
    tao->trust=PetscMax(tron->gnorm*tron->gnorm,1.0);
  }

  tron->stepsize=tao->trust;
  ierr = TaoSolverMonitor(tao, iter, tron->f, tron->gnorm, 0.0, tron->stepsize, &reason); CHKERRQ(ierr);
  if (tron->R) {
    ierr = VecDestroy(&tron->R); CHKERRQ(ierr);
    tron->R = PETSC_NULL;
  }
  while (reason==TAO_CONTINUE_ITERATING){

    ierr = TronGradientProjections(tao,tron); CHKERRQ(ierr);
    f=tron->f; delta=tao->trust; gnorm=tron->gnorm; 
    
    tron->n_free_last = tron->n_free;
    ierr = ISGetSize(tron->Free_Local, &tron->n_free);  CHKERRQ(ierr);
    ierr = TaoSolverComputeHessian(tao,tao->solution,&tao->hessian, &tao->hessian_pre, &tron->matflag);CHKERRQ(ierr);

    /* Create a reduced linear system using free variables */
    ierr = ISGetSize(tron->Free_Local, &tron->n_free);  CHKERRQ(ierr);

    /* If no free variables */
    if (tron->n_free == 0) {
      actred=0;
      PetscInfo(tao,"No free variables in tron iteration.");
      break;

    }
    /* use free_local to mask/submat gradient, hessian, stepdirection */
    ierr = TronSetupKSP(tao,tron); CHKERRQ(ierr);
    while (1) {

      /* Approximately solve the reduced linear system */
      ierr = KSPSolve(tao->ksp, tron->R, tron->DXFree); CHKERRQ(ierr);
      ierr = VecSet(tao->stepdirection,0.0); CHKERRQ(ierr);
      
      /* Add dxfree matrix to compute step direction vector */
      if (tron->subset_type == TRON_SUBSET_SUBMAT) {
	ierr = VecScatterBegin(tron->scatter,tron->DXFree,tao->stepdirection,ADD_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
	ierr = VecScatterEnd(tron->scatter,tron->DXFree,tao->stepdirection,ADD_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
      } else if (tron->subset_type == TRON_SUBSET_MASK || tron->subset_type==TRON_SUBSET_MATRIXFREE) {
	ierr = VecAXPY(tao->stepdirection, 1.0, tron->DXFree); CHKERRQ(ierr);
      }
      
      
      ierr = VecDot(tao->gradient, tao->stepdirection, &gdx); CHKERRQ(ierr);
      
      ierr = PetscInfo1(tao,"Expected decrease in function value: %14.12e\n",gdx); CHKERRQ(ierr);
      
      ierr = VecCopy(tao->solution, tron->X_New); CHKERRQ(ierr);
      ierr = VecCopy(tao->gradient, tron->G_New); CHKERRQ(ierr);
      
      stepsize=1.0;f_new=f;
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tron->X_New, &f_new, tron->G_New, tao->stepdirection,
				&stepsize,&ls_reason); CHKERRQ(ierr); CHKERRQ(ierr);
      
      ierr = MatMult(tao->hessian, tao->stepdirection, tron->Work); CHKERRQ(ierr);
      ierr = VecAYPX(tron->Work, 0.5, tao->gradient); CHKERRQ(ierr);
      ierr = VecDot(tao->stepdirection, tron->Work, &prered); CHKERRQ(ierr);
      actred = f_new - f;
      if (actred<0) {
	rhok=PetscAbs(-actred/prered);
      } else {
	rhok=0.0;
      }
      
      /* Compare actual improvement to the quadratic model */
      if (rhok > tron->eta1) { /* Accept the point */
	/* d = x_new - x */
	ierr = VecCopy(tron->X_New, tao->stepdirection); CHKERRQ(ierr);
	ierr = VecAXPY(tao->stepdirection, -1.0, tao->solution); CHKERRQ(ierr);
	
	ierr = VecNorm(tao->stepdirection, NORM_2, &xdiff); CHKERRQ(ierr);
	xdiff *= stepsize;

	/* Adjust trust region size */
	if (rhok < tron->eta2 ){
	  delta = PetscMin(xdiff,delta)*tron->sigma1;
	} else if (rhok > tron->eta4 ){
	  delta= PetscMin(xdiff,delta)*tron->sigma3;
	} else if (rhok > tron->eta3 ){
	  delta=PetscMin(xdiff,delta)*tron->sigma2;
	}
	ierr = VecBoundGradientProjection(tron->G_New,tron->X_New, tao->XL, tao->XU, tao->gradient); CHKERRQ(ierr);
	if (tron->Free_Local) {
	  ierr = ISDestroy(&tron->Free_Local); CHKERRQ(ierr);
	  tron->Free_Local=PETSC_NULL;
	}
	ierr = VecWhichBetween(tao->XL, tron->X_New, tao->XU, &tron->Free_Local); CHKERRQ(ierr);
	f=f_new;
	ierr = VecNorm(tao->gradient,NORM_2,&tron->gnorm); CHKERRQ(ierr);
	ierr = VecCopy(tron->X_New, tao->solution); CHKERRQ(ierr);
	ierr = VecCopy(tron->G_New, tao->gradient); CHKERRQ(ierr);
	break;
      } 
      else if (delta <= 1e-30) {
	break;
      }
      else {
	delta /= 4.0;
      }
    } /* end linear solve loop */


    tron->f=f; tron->actred=actred; tao->trust=delta;
    iter++;
    ierr = TaoSolverMonitor(tao, iter, tron->f, tron->gnorm, 0.0, delta, &reason); CHKERRQ(ierr);
  }  /* END MAIN LOOP  */

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TronGradientProjections"
static PetscErrorCode TronGradientProjections(TaoSolver tao,TAO_TRON *tron)
{
  PetscErrorCode ierr;
  PetscInt i;
  TaoLineSearchTerminationReason ls_reason;
  PetscReal actred=-1.0,actred_max=0.0;
  PetscReal f_new;
  /*
     The gradient and function value passed into and out of this
     routine should be current and correct.
     
     The free, active, and binding variables should be already identified
  */
  
  PetscFunctionBegin;
  if (tron->Free_Local) {
    ierr = ISDestroy(&tron->Free_Local); CHKERRQ(ierr);
    tron->Free_Local = PETSC_NULL;
  }
  ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&tron->Free_Local); CHKERRQ(ierr);

  for (i=0;i<tron->maxgpits;i++){

    if ( -actred <= (tron->pg_ftol)*actred_max) break;
  
    tron->gp_iterates++; tron->total_gp_its++;      
    f_new=tron->f;

    ierr = VecCopy(tao->gradient, tao->stepdirection); CHKERRQ(ierr);
    ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,tron->pgstepsize); CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f_new, tao->gradient, tao->stepdirection,
			      &tron->pgstepsize, &ls_reason); CHKERRQ(ierr);


    /* Update the iterate */
    actred = f_new - tron->f;
    actred_max = PetscMax(actred_max,-(f_new - tron->f));
    tron->f = f_new;
    if (tron->Free_Local) {
      ierr = ISDestroy(&tron->Free_Local); CHKERRQ(ierr);
      tron->Free_Local = PETSC_NULL;
    }
    ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&tron->Free_Local); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TronSetupKSP"
PetscErrorCode TronSetupKSP(TaoSolver tao, TAO_TRON*tron)
{
  PetscErrorCode ierr;
  const PetscInt *s;
  PetscInt nlocal, low, high, i;
  PetscReal *v;
  PetscBool flg;
  KSP newksp;
  PC pc;
  IS fullis;
  PetscFunctionBegin;

  if (tron->subset_type == TRON_SUBSET_SUBMAT) {
    if (tron->R) {
      ierr = VecDestroy(&tron->R); CHKERRQ(ierr);
      tron->R = PETSC_NULL;
    }
    if (tron->DXFree) {
      ierr = VecDestroy(&tron->DXFree);
      tron->DXFree = PETSC_NULL;
    }
    ierr = ISGetLocalSize(tron->Free_Local, &nlocal); CHKERRQ(ierr);
    ierr = VecCreate(((PetscObject)tao)->comm,&tron->R); CHKERRQ(ierr);
    ierr = VecCreate(((PetscObject)tao)->comm,&tron->DXFree); CHKERRQ(ierr);
    ierr = VecSetSizes(tron->R, nlocal, tron->n_free); CHKERRQ(ierr);
    ierr = VecSetSizes(tron->DXFree, nlocal, tron->n_free); CHKERRQ(ierr);
    ierr = VecSetType(tron->R,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
    ierr = VecSetType(tron->DXFree,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tron->R); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tron->DXFree); CHKERRQ(ierr);
    ierr = VecSet(tron->DXFree,0.0);CHKERRQ(ierr);
    if (tron->scatter) {
      ierr = VecScatterDestroy(&tron->scatter); CHKERRQ(ierr);
      tron->scatter = PETSC_NULL;
    }
    ierr = VecGetOwnershipRange(tron->R,&low,&high); CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)tron->R)->comm,high-low,low,1,&fullis); CHKERRQ(ierr);

    ierr = VecScatterCreate(tao->gradient,tron->Free_Local,tron->R,fullis,&tron->scatter); CHKERRQ(ierr);

    ierr = ISDestroy(&fullis); CHKERRQ(ierr);

    ierr = VecScatterBegin(tron->scatter, tao->gradient, tron->R, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(tron->scatter, tao->gradient, tron->R, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScale(tron->R, -1.0); CHKERRQ(ierr);
    if (tron->H_sub) {
      ierr = MatDestroy(&tron->H_sub); CHKERRQ(ierr); tron->H_sub=PETSC_NULL;
    }
    ierr = MatGetSubMatrix(tao->hessian, tron->Free_Local, tron->Free_Local, MAT_INITIAL_MATRIX, &tron->H_sub); CHKERRQ(ierr);
    if (tron->Hpre_sub) {
      ierr = MatDestroy(&tron->Hpre_sub); CHKERRQ(ierr); tron->Hpre_sub=PETSC_NULL;
    }
    if (tao->hessian != tao->hessian_pre) {
      ierr = MatGetSubMatrix(tao->hessian_pre, tron->Free_Local, tron->Free_Local, MAT_INITIAL_MATRIX, &tron->Hpre_sub); CHKERRQ(ierr);
    } else {
      tron->Hpre_sub = tron->H_sub;
      ierr = PetscObjectReference((PetscObject)tron->H_sub); CHKERRQ(ierr);
    }
    /* Create New KSP if size changed */
    if (tron->n_free_last != tron->n_free) {
      ierr = KSPCreate(((PetscObject)tao)->comm, &newksp); CHKERRQ(ierr);
      newksp->pc_side = tao->ksp->pc_side;
      newksp->rtol = tao->ksp->rtol;
      newksp->max_it = tao->ksp->max_it;
      ierr = KSPSetType(newksp,((PetscObject)(tao->ksp))->type_name); CHKERRQ(ierr);
      ierr = KSPGetPC(tao->ksp, &pc); CHKERRQ(ierr);
      if (pc != PETSC_NULL && ((PetscObject)pc)->type_name) {
	/* Force newksp->pc to be created */
	PC newpc;
	ierr = KSPGetPC(newksp,&newpc); CHKERRQ(ierr);
	ierr = PCSetType(newpc, ((PetscObject)pc)->type_name); CHKERRQ(ierr);
      }
      ierr = KSPDestroy(&tao->ksp); CHKERRQ(ierr);
      tao->ksp = newksp;
      ierr = PetscLogObjectParent(tao,tao->ksp); CHKERRQ(ierr);
      ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
    }

    ierr = KSPSetOperators(tao->ksp, tron->H_sub, tron->Hpre_sub, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  } else if (tron->subset_type==TRON_SUBSET_MASK) {
    /* Create mask */
    if (tron->rmask == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->rmask); CHKERRQ(ierr);
    }
    if (tron->diag == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->diag); CHKERRQ(ierr);
    }
    if (tron->R == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->R); CHKERRQ(ierr);
    }
    if (tron->DXFree == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->DXFree); CHKERRQ(ierr);
    }
    ierr = VecSet(tron->rmask, 0.0); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(tron->rmask, &low, &high); CHKERRQ(ierr);
    ierr = ISGetLocalSize(tron->Free_Local, &nlocal); CHKERRQ(ierr);
    ierr = ISGetIndices(tron->Free_Local, &s); CHKERRQ(ierr);
    ierr = VecGetArray(tron->rmask, &v); CHKERRQ(ierr);
    for (i=0;i<nlocal; i++) {
      v[s[i]-low] = 1;
    }
    ierr = ISRestoreIndices(tron->Free_Local, &s); CHKERRQ(ierr);
    ierr = VecRestoreArray(tron->rmask, &v); CHKERRQ(ierr);
    
    /* Reduce vectors (r[i]=g[i], if i in Free_Local, 
                       r[i]=0     if i not in Free_Local) */

    ierr = VecPointwiseMult(tron->R, tao->gradient, tron->rmask); CHKERRQ(ierr);
    ierr = VecPointwiseMult(tron->DXFree, tao->stepdirection, tron->rmask); CHKERRQ(ierr);
    ierr = VecScale(tron->R, -1.0); CHKERRQ(ierr);

    /* Get Reduced Hessian 
       Hsub[i,j] = H[i,j] if i,j in Free_Local or i==j
       Hsub[i,j] = 0      if i!=j and i or j not in Free_Local
     */
    ierr = PetscOptionsHasName(0,"-different_submatrix",&flg);
    if (flg == PETSC_TRUE) {
      if (tron->H_sub) {
	ierr = MatDestroy(&tron->H_sub); CHKERRQ(ierr); tron->H_sub=PETSC_NULL;
      }
      ierr = MatDuplicate(tao->hessian, MAT_COPY_VALUES, &tron->H_sub); CHKERRQ(ierr);
      if (tao->hessian != tao->hessian_pre) {
	if (tron->Hpre_sub) {
	  ierr = MatDestroy(&tron->Hpre_sub); CHKERRQ(ierr); tron->Hpre_sub=PETSC_NULL;
	}
	ierr = MatDuplicate(tao->hessian_pre, MAT_COPY_VALUES, &tron->Hpre_sub); CHKERRQ(ierr);
      }
    } else {
      /* Act on hessian directly (default) */
      ierr = PetscObjectReference((PetscObject)tron->H_sub); CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)tron->Hpre_sub); CHKERRQ(ierr);
      tron->H_sub = tao->hessian;
      tron->Hpre_sub = tao->hessian_pre;
    }


    ierr = TronApplyMask(tron,tron->H_sub,tron->rmask); CHKERRQ(ierr);
    if (tron->H_sub != tron->Hpre_sub) {
      ierr = TronApplyMask(tron, tron->Hpre_sub, tron->rmask); CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(tao->ksp, tron->H_sub, tron->Hpre_sub, tron->matflag); CHKERRQ(ierr);
  } else if (tron->subset_type == TRON_SUBSET_MATRIXFREE) {

    if (tron->rmask == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->rmask); CHKERRQ(ierr);
    }
    if (tron->R == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->R); CHKERRQ(ierr);
    }
    if (tron->DXFree == PETSC_NULL) {
      ierr = VecDuplicate(tao->solution, &tron->DXFree); CHKERRQ(ierr);
    }
    ierr = VecSet(tron->rmask, 0.0); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(tron->rmask, &low, &high); CHKERRQ(ierr);
    ierr = ISGetLocalSize(tron->Free_Local, &nlocal); CHKERRQ(ierr);
    ierr = ISGetIndices(tron->Free_Local, &s); CHKERRQ(ierr);
    ierr = VecGetArray(tron->rmask, &v); CHKERRQ(ierr);
    for (i=0;i<nlocal; i++) {
      v[s[i]-low] = 1;
    }
    ierr = ISRestoreIndices(tron->Free_Local, &s); CHKERRQ(ierr);
    ierr = VecRestoreArray(tron->rmask, &v); CHKERRQ(ierr);
    
    /* Reduce vectors (r[i]=g[i], if i in Free_Local, 
                       r[i]=0     if i not in Free_Local) */

    ierr = VecPointwiseMult(tron->R, tao->gradient, tron->rmask); CHKERRQ(ierr);
    ierr = VecPointwiseMult(tron->DXFree, tao->stepdirection, tron->rmask); CHKERRQ(ierr);
    ierr = VecScale(tron->R, -1.0); CHKERRQ(ierr);


    /* create submat wrapper */


    if (tron->H_sub == PETSC_NULL) {
      ierr = MatCreateSubMatrixFree(tao->hessian,tron->Free_Local,tron->Free_Local,&tron->H_sub);
      if (tao->hessian == tao->hessian_pre) {
	tron->Hpre_sub = tron->H_sub;
	ierr = PetscObjectReference((PetscObject)tron->Hpre_sub); CHKERRQ(ierr);
      } else {
	ierr = MatCreateSubMatrixFree(tao->hessian_pre,tron->Free_Local,tron->Free_Local,&tron->Hpre_sub);
      }
      
    }

    ierr = KSPSetOperators(tao->ksp, tron->H_sub, tron->Hpre_sub, tron->matflag); CHKERRQ(ierr);
       

  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoGetDualVariables_TRON" 
static PetscErrorCode TaoSolverComputeDual_TRON(TaoSolver tao, Vec DXL, Vec DXU) {

  TAO_TRON *tron = (TAO_TRON *)tao->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);

  if (!tron->Work || !tao->gradient) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.\n");
  }

  ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,tron->Work); CHKERRQ(ierr);
  ierr = VecCopy(tron->Work,DXL); CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,tao->gradient); CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU); CHKERRQ(ierr);

  ierr = VecCopy(tao->gradient,DXU); CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,tron->Work); CHKERRQ(ierr);
  ierr = VecSet(tron->Work,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseMin(DXU,tron->Work,DXU); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TronApplyMask"
static PetscErrorCode TronApplyMask(TAO_TRON *tron, Mat M, Vec mask)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatGetDiagonal(M, tron->diag); CHKERRQ(ierr);
  ierr = MatDiagonalScale(M, mask, mask); CHKERRQ(ierr);
  if  (M->ops->diagonalset) {
    ierr = MatDiagonalSet(M, tron->diag, INSERT_VALUES); CHKERRQ(ierr);
  } else {
    PetscReal *d,*m;
    PetscInt lo,hi,lo2,hi2,i;
    ierr = VecGetOwnershipRange(mask, &lo, &hi); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(tron->diag, &lo2, &hi2); CHKERRQ(ierr);
    if (lo!=lo2 || hi != hi2) {
      SETERRQ(PETSC_COMM_SELF,1,"mask and diag vecs have different allocation");
    }
    ierr = VecGetArray(mask,&m); CHKERRQ(ierr);
    ierr = VecGetArray(tron->diag,&d); CHKERRQ(ierr);
    for (i=0;i<hi-lo;i++) {
      if (m[i] == 0) {
	ierr = MatSetValue(M,lo+i,lo+i,d[i],INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_TRON"
PetscErrorCode TaoSolverCreate_TRON(TaoSolver tao)
{
  TAO_TRON *tron;
  PetscErrorCode   ierr;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscFunctionBegin;

  tao->ops->setup = TaoSolverSetup_TRON;
  tao->ops->solve = TaoSolverSolve_TRON;
  tao->ops->view = TaoSolverView_TRON;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_TRON;
  tao->ops->destroy = TaoSolverDestroy_TRON;
  tao->ops->computedual = TaoSolverComputeDual_TRON;

  ierr = PetscNewLog(tao,TAO_TRON,&tron); CHKERRQ(ierr);

  tao->max_its = 50;
  tao->fatol = 1e-10;
  tao->frtol = 1e-10;
  tao->data = (void*)tron;
  tao->steptol = 1e-12;
  tao->trust        = 1.0;

  /* Initialize pointers and variables */
  tron->n            = 0;
  tron->maxgpits     = 3;
  tron->pg_ftol      = 0.001;

  tron->eta1         = 1.0e-4;
  tron->eta2         = 0.25;
  tron->eta3         = 0.50;
  tron->eta4         = 0.90;

  tron->sigma1       = 0.5;
  tron->sigma2       = 2.0;
  tron->sigma3       = 4.0;

  tron->gp_iterates  = 0; /* Cumulative number */
  tron->total_gp_its = 0;
 
  tron->n_free       = 0;

  tron->DXFree=PETSC_NULL;
  tron->R=PETSC_NULL;
  tron->X_New=PETSC_NULL;
  tron->G_New=PETSC_NULL;
  tron->Work=PETSC_NULL;
  tron->Free_Local=PETSC_NULL;
  tron->H_sub=PETSC_NULL;
  tron->Hpre_sub=PETSC_NULL;
  tron->subset_type = TRON_SUBSET_SUBMAT;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type); CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}
EXTERN_C_END
