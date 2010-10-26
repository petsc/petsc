#include "tron.h"


static const char *TAOSUBSET[64] = {
    "singleprocessor", "noredistribute", "redistribute", "mask", "matrixfree"
};


/* TRON Routines */
static PetscErrorCode TronGradientProjections(TaoSolver,TAO_TRON*);
static PetscErrorCode SolveTrustRegion();

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_TRON"
static int TaoSolverDestroy_TRON(TaoSolver tao)
{
  TAO_TRON *tron = (TAO_TRON *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecDestroy(tron->X_New);CHKERRQ(ierr);
  ierr = VecDestroy(tron->G_New);CHKERRQ(ierr);
  ierr = VecDestroy(tron->Work);CHKERRQ(ierr);
  ierr = VecDestroy(tron->DXFree);CHKERRQ(ierr);
  ierr = VecDestroy(tron->R);CHKERRQ(ierr);
  ierr = VecDestroy(tron->PG);CHKERRQ(ierr);
  

 
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_TRON"
static int TaoSolverSetFromOptions_TRON(TaoSolver tao);
{
  TAO_TRON  *tron = (TAO_TRON *)tao->data;
  PetscErrorCode        ierr;
  PetscInt     ival;
  PetscBool flg;

  PetscFunctionBegin;

  ierr = PetscOptionsHead("Newton Trust Region Method for bound constrained optimization");CHKERRQ(ierr);
  
  ierr = PetscOptionsInt("-tron_maxgpits","maximum number of gradient projections per TRON iterate","TaoSetMaxGPIts",tron->maxgpits,&tron->maxgpits,&flg);
  CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_subset_type","subset type", "", TAOSUBSET, TAOSUBSET_TYPES,TAOSUBSET[tron->subset_type], &tron->subset_type, 0); CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_TRON"
static PetscErrorCode TaoSolverView_TRON(TaoSolver tao, PetscViewer pv)
{
  TAO_TRON  *tron = (TAO_TRON *)tao->data;
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  

  PetscFunctionBegin;
  comm = ((PetscObject)tao)->comm;
  ierr = PetscPrintf(comm," Total PG its: %d,",tron->total_gp_its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm," PG tolerance: %4.3f \n",tron->pg_ftol);CHKERRQ(ierr);
  ierr = TaoLineSearchView(tao->linesearch,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetup_TRON"
static PetscErrorCode TaoSolverSetup_TRON(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_TRON *tron = (TAO_TRON *)tao->data;
  Vec X;
  PetscInt low,high;
  Mat H,Hpre;

  PetscFunctionBegin;

  /* Allocate some arrays */
  ierr = VecDuplicate(tao->solution, &tron->X_New); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->G_New); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->Work); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->DXFree); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->R); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tron->PG); CHKERRQ(ierr);
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
  PetscInt lsflag,iter=0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscScalar prered,actred,delta,f,f_new,f_full,rhok,gnorm,gdx,xdiff,stepsize;
  VecScatter scatter;
  PetscFunctionBegin;

  tron->pgstepsize=1.0;

  /*   Project the current point onto the feasible set */
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution); CHKERRQ(ierr);

  
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&tron->f,tao->gradient);CHKERRQ(ierr);
  ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&tron->Free_Local); CHKERRQ(ierr);
  
  /* Project the gradient and calculate the norm */
  ierr = VecBoundGradientProjection(tao->gradient,tao->solution, tao->XL, tao->XU, tron->PG); CHKERRQ(ierr);
  ierr = VecNorm(tron->PG,NORM_2,&tron->gnorm); CHKERRQ(ierr);

  if (TaoInfOrNaN(tron->f) || TaoInfOrNaN(tron->gnorm)) {
    SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf pr NaN");
  }

  tron->stepsize=tron->delta;
  ierr = TaoSolverMonitor(tao, iter, tron->f, tron->gnorm, 0.0, tron->stepsize, &reason); CHKERRQ(ierr);

  while (reason==TAO_CONTINUE_ITERATING){
     
    ierr = TronGradientProjections(tao,tron); CHKERRQ(ierr);
    f=tron->f; delta=tron->delta; gnorm=tron->gnorm; 
    
    ierr = ISGetSize(tron->Free_Local, &tron->n_free);  CHKERRQ(ierr);
    if (tron->n_free > 0){
      
      ierr = TaoSolverComputeHessian(tao,tao->solution,&tao->hessian, &tao->hessian_pre, &matflag);CHKERRQ(ierr);
      // LEFT OFF HERE
      /* Create a reduced linear system using free variables */
      ierr = ISGetSize(tron->Free_Local, &tron->n_free);  CHKERRQ(ierr);
      ierr = VecScatterCreate(
      ierr = R->SetReducedVec(G,Free_Local);CHKERRQ(ierr);
      ierr = R->Negate(); CHKERRQ(ierr);
      ierr = DXFree->SetReducedVec(DX,Free_Local);CHKERRQ(ierr);
      ierr = DXFree->SetToZero(); CHKERRQ(ierr);

      ierr = Hsub->SetReducedMatrix(H,Free_Local,Free_Local);CHKERRQ(ierr);

      ierr = TaoPreLinearSolve(tao,Hsub);CHKERRQ(ierr);

      while (1) {

 	/* Approximately solve the reduced linear system */
        ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);
	ierr = KSPSetOptionsPrefix(tao->ksp, "tao_"); CHKERRQ(ierr);
	ierr = KSPSetType(tao->ksp,KSPSTCG); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);

	ierr = SolveTrustRegion(tao, Hsub, R, DXFree, delta, &success); CHKERRQ(ierr);

	ierr=DX->SetToZero(); CHKERRQ(ierr);
	ierr=DX->ReducedXPY(DXFree,Free_Local);CHKERRQ(ierr);

	ierr = G->Dot(DX,&gdx); CHKERRQ(ierr);
	ierr = PetscInfo1(tao,"Expected decrease in function value: %14.12e\n",gdx); CHKERRQ(ierr);

	stepsize=1.0; f_new=f;
	ierr = X_New->CopyFrom(X); CHKERRQ(ierr);
	ierr = G_New->CopyFrom(G); CHKERRQ(ierr);
	
	ierr = TaoLineSearchApply(tao,X_New,G_New,DX,Work,
				  &f_new,&f_full,&stepsize,&lsflag);
	CHKERRQ(ierr);
	ierr = H->Multiply(DX,Work); CHKERRQ(ierr);
	ierr = Work->Aypx(0.5,G); CHKERRQ(ierr);
	ierr = Work->Dot(DX,&prered); CHKERRQ(ierr);
	actred = f_new - f;
	
	if (actred<0) rhok=TaoAbsScalar(-actred/prered);
	else rhok=0.0;

	/* Compare actual improvement to the quadratic model */
	if (rhok > tron->eta1) { /* Accept the point */

	  ierr = DX->Waxpby(1.0,X_New,-1.0, X); CHKERRQ(ierr);
	  ierr = DX->Norm2(&xdiff); CHKERRQ(ierr);
	  xdiff*=stepsize;

	  /* Adjust trust region size */
	  if (rhok < tron->eta2 ){
	    delta = TaoMin(xdiff,delta)*tron->sigma1;
	  } else if (rhok > tron->eta4 ){
	    delta= TaoMin(xdiff,delta)*tron->sigma3;
	  } else if (rhok > tron->eta3 ){
	    delta=TaoMin(xdiff,delta)*tron->sigma2;
	  }

	  ierr =  PG->BoundGradientProjection(G_New,XL,X_New,XU); CHKERRQ(ierr);
	  ierr = PG->Norm2(&gnorm);  CHKERRQ(ierr);
	  ierr = Free_Local->WhichBetween(XL,X_New,XU); CHKERRQ(ierr);
	  f=f_new;
	  ierr = X->CopyFrom(X_New); CHKERRQ(ierr);
	  ierr = G->CopyFrom(G_New); CHKERRQ(ierr);
	  break;
	} 
	else if (delta <= 1e-30) {
	  break;
	}
        else {
	  delta /= 4.0;
	}
      } /* end linear solve loop */
      
    } else {
      
      actred=0;
      ierr =  Work->BoundGradientProjection(G,XL,X,XU);
      CHKERRQ(ierr);
      ierr = Work->Norm2(&gnorm);  CHKERRQ(ierr);
      /* if there were no free variables, no cg method */

    }

    tron->f=f;tron->gnorm=gnorm; tron->actred=actred; tron->delta=delta;
    ierr = TaoMonitor(tao,iter,f,gnorm,0.0,delta,&reason); CHKERRQ(ierr);
    if (reason!=TAO_CONTINUE_ITERATING) break;
    iter++;
    
  }  /* END MAIN LOOP  */

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TronGradientProjections"
static PetscErrorCode TronGradientProjections(TAO_SOLVER tao,TAO_TRON *tron)
{
  PetscErrorCode ierr;
  PetscInt lsflag=0,i;
  PetscBool sameface=PETSC_FALSE;
  PetscScalar actred=-1.0,actred_max=0.0;
  PetscScalar f_new, stepsize;
  /*
     The gradient and function value passed into and out of this
     routine should be current and correct.
     
     The free, active, and binding variables should be already identified
  */
  
  PetscFunctionBegin;
  ierr = VecWhichBetween(XL,tao->solution,XU,&tron->Free_Local); CHKERRQ(ierr);

  for (i=0;i<tron->maxgpits;i++){

    if ( -actred <= (tron->pg_ftol)*actred_max) break;
  
    tron->gp_iterates++; tron->total_gp_its++;      
    f_new=tron->f;

    ierr = VecCopy(tao->gradient, tao->stepdirection); CHKERRQ(ierr);
    ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f_new, tao->gradient, tao->stepdirection,
			      &stepsize, &ls_status); CHKERRQ(ierr);


    /* Update the iterate */
    actred = f_new - tron->f;
    actred_max = PetscMax(actred_max,-(f_new - tron->f));
    tron->f = f_new;
    ierr = VecWhichBetween(XL,tao->solution,XU,&tron->Free_Local); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


/*
#undef __FUNCT__  
#define __FUNCT__ "TaoDefaultMonitor_TRON" 
int TaoDefaultMonitor_TRON(TAO_SOLVER tao,void *dummy)
{
  int ierr;
  TaoInt its,nfree,nbind;
  double fct,gnorm;
  TAO_TRON *tron;

  PetscFunctionBegin;
  ierr = TaoGetSolutionStatus(tao,&its,&fct,&gnorm,0,0,0);CHKERRQ(ierr);
  ierr = TaoGetSolverContext(tao,"tao_tron",(void**)&tron); CHKERRQ(ierr);
  if (tron){
    nfree=tron->n_free;
    nbind=tron->n_bind;
    ierr=TaoPrintInt(tao,"iter = %d,",its); CHKERRQ(ierr);
    ierr=TaoPrintDouble(tao," Function value: %g,",fct); CHKERRQ(ierr);
    ierr=TaoPrintDouble(tao,"  Residual: %g \n",gnorm);CHKERRQ(ierr);
    
    ierr=TaoPrintInt(tao," free vars = %d,",nfree); CHKERRQ(ierr);
    ierr=TaoPrintInt(tao," binding vars = %d\n",nbind); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

*/


#undef __FUNCT__  
#define __FUNCT__ "TaoGetDualVariables_TRON" 
static PetscErrorCode TaoSolverComputeDual_TRON(TaoSolver tao, Vec DXL, Vec DXU) {

  TAO_TRON *tron = (TAO_TRON *)tao->data;
  Vec  X,XL,XU;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);

  if (!blm->Work || !tao->gradient) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.\n");
  }

  ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->xl,tao->xu,tron->Work); CHKERRQ(ierr);
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

/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoCreate_TRON"
int TaoCreate_TRON(TAO_SOLVER tao)
{
  TAO_TRON *tron;
  int      ierr;

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
  tao->trtol = 1e-12;

  /* Initialize pointers and variables */
  tron->n            = 0;
  tron->delta        = -1.0;
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
  tron->cgits        = 0; /* Current iteration */
  tron->total_gp_its = 0;
  tron->cg_iterates  = 0;
  tron->total_cgits  = 0;
 
  tron->n_bind       = 0;
  tron->n_free       = 0;
  tron->n_upper      = 0;
  tron->n_lower      = 0;

  tron->DX=0;
  tron->DXFree=0;
  tron->R=0;
  tron->X_New=0;
  tron->G_New=0;
  tron->Work=0;
  tron->Free_Local=0;
  tron->TT=0;
  tron->Hsub=0;
  tron->subset_type = TRON_SUBSET_NOREDISTRIBUTE;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuentebound_type); CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
