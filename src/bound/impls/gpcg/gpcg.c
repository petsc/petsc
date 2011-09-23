/*$Id$*/
#include "private/kspimpl.h"
#include "gpcg.h"        /*I "tao_solver.h" I*/


#define GPCG_KSP_NASH  0
#define GPCG_KSP_STCG  1
#define GPCG_KSP_GLTR  2
#define GPCG_KSP_NTYPES 3


static const char *GPCG_KSP[64] = {
    "nash","stcg","gltr"
};
static const char *TAOSUBSET[64] = {
    "singleprocessor", "noredistribute", "redistribute", "mask", "matrixfree"
};

//static PetscErrorCode TaoGradProjections(TaoSolver);
//static PetscErrorCode GPCGCheckOptimalFace(Vec, Vec, Vec, Vec, Vec, IS, IS, PetscBool *);
static PetscErrorCode GPCGGradProjections(TaoSolver tao);
static PetscErrorCode GPCGObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_GPCG"
static PetscErrorCode TaoSolverDestroy_GPCG(TaoSolver tao)
{
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode      ierr;
  /* Free allocated memory in GPCG structure */
  PetscFunctionBegin;
  
  ierr = VecDestroy(&gpcg->B);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->Work);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->X_New);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->G_New);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->DXFree);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->R);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->PG);CHKERRQ(ierr);
  ierr = ISDestroy(&gpcg->Free_Local);CHKERRQ(ierr);
  ierr = PetscFree(tao->data); CHKERRQ(ierr);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_GPCG"
static PetscErrorCode TaoSolverSetFromOptions_GPCG(TaoSolver tao)
{
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode      ierr;
  PetscBool flg;
  MPI_Comm   comm;
  PetscMPIInt size;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Gradient Projection, Conjugate Gradient method for bound constrained optimization");CHKERRQ(ierr);

  ierr=PetscOptionsInt("-gpcg_maxpgits","maximum number of gradient projections per GPCG iterate",0,gpcg->maxgpits,&gpcg->maxgpits,&flg);
  CHKERRQ(ierr);

  comm = ((PetscObject)tao)->comm;
  gpcg->subset_type = TAOSUBSET_MASK;
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_subset_type","subset type", "", TAOSUBSET, TAOSUBSET_TYPES,TAOSUBSET[gpcg->subset_type], &gpcg->subset_type, 0); CHKERRQ(ierr);


  ierr = PetscOptionsEList("-tao_gpcg_ksp_type", "ksp type", "", GPCG_KSP, GPCG_KSP_NTYPES,
			 GPCG_KSP[gpcg->ksp_type], &gpcg->ksp_type,0); CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_GPCG"
static PetscErrorCode TaoSolverView_GPCG(TaoSolver tao, PetscViewer viewer)
{
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscBool           isascii;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Total PG its: %D,",gpcg->total_gp_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"PG tolerance: %G \n",gpcg->pg_ftol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"KSP type: %s\n",GPCG_KSP[gpcg->ksp_type]); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Subset type: %s\n", TAOSUBSET[gpcg->subset_type]); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO GPCG",((PetscObject)viewer)->type_name);
  }

  ierr = TaoLineSearchView(tao->linesearch,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* GPCGObjectiveAndGradient()
   Compute f=0.5 * x'Hx + b'x + c
           g=Hx + b
*/
#undef __FUNCT__  
#define __FUNCT__ "GPCGObjectiveAndGradient"
static PetscErrorCode GPCGObjectiveAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void*tptr){
  TaoSolver tao = (TaoSolver)tptr;
  TAO_GPCG *gpcg = (TAO_GPCG*)tao->data;
  PetscErrorCode ierr;
  PetscReal f1,f2;

  PetscFunctionBegin;
/*  ierr = MatMult(tao->hessian,tao->solution,tao->gradient); CHKERRQ(ierr);
  ierr = VecDot(tao->gradient,tao->solution,&f1); CHKERRQ(ierr);
  ierr = VecDot(gpcg->B,tao->solution,&f2); CHKERRQ(ierr);
  ierr = VecAXPY(tao->gradient,1.0,gpcg->B); CHKERRQ(ierr);*/

  ierr = MatMult(tao->hessian,X,G); CHKERRQ(ierr);
  ierr = VecDot(G,X,&f1); CHKERRQ(ierr);
  ierr = VecDot(gpcg->B,X,&f2); CHKERRQ(ierr);
  ierr = VecAXPY(G,1.0,gpcg->B); CHKERRQ(ierr);
  *f=f1/2.0 + f2 + gpcg->c;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetup_GPCG"
static PetscErrorCode TaoSolverSetup_GPCG(TaoSolver tao) {

  PetscErrorCode      ierr;
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;

  PetscFunctionBegin;

  /* Allocate some arrays */
  if (!tao->gradient) {
      ierr = VecDuplicate(tao->solution, &tao->gradient);
      CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
      ierr = VecDuplicate(tao->solution, &tao->stepdirection);
      CHKERRQ(ierr);
  }
  if (!tao->XL) {
      ierr = VecDuplicate(tao->solution,&tao->XL); CHKERRQ(ierr);
      ierr = VecSet(tao->XL,TAO_NINFINITY); CHKERRQ(ierr);
  }
  if (!tao->XU) {
      ierr = VecDuplicate(tao->solution,&tao->XU); CHKERRQ(ierr);
      ierr = VecSet(tao->XU,TAO_INFINITY); CHKERRQ(ierr);
  }

  ierr=VecDuplicate(tao->solution,&gpcg->B); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->Work); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->X_New); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->G_New); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->DXFree); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->R); CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->PG); CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU); CHKERRQ(ierr);

  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_GPCG"
static PetscErrorCode TaoSolverSolve_GPCG(TaoSolver tao)
{
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode ierr;
  PetscInt iter=0;
  PetscReal actred,f,f_new,gnorm,gdx,stepsize,xtb;
  PetscReal xtHx;
  MatStructure structure;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  gpcg->Hsub=PETSC_NULL;
  gpcg->Hsub_pre=PETSC_NULL;
  tao->ksp = PETSC_NULL;
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution); CHKERRQ(ierr);
  
  /* Using f = .5*x'Hx + x'b + c and g=Hx + b,  compute b,c */
  ierr = TaoSolverComputeHessian(tao,tao->solution,&tao->hessian, &tao->hessian_pre,&structure); CHKERRQ(ierr);
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);  CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, gpcg->B); CHKERRQ(ierr);
  ierr = MatMult(tao->hessian,tao->solution,gpcg->Work); CHKERRQ(ierr);
  ierr = VecDot(gpcg->Work, tao->solution, &xtHx); CHKERRQ(ierr);
  ierr = VecAXPY(gpcg->B,-1.0,gpcg->Work); CHKERRQ(ierr);
  ierr = VecDot(gpcg->B,tao->solution,&xtb); CHKERRQ(ierr);
  gpcg->c=f-xtHx/2.0-xtb;
  if (gpcg->Free_Local) {
      ierr = ISDestroy(&gpcg->Free_Local); CHKERRQ(ierr);
  }
  ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&gpcg->Free_Local); CHKERRQ(ierr);
  
  /* Project the gradient and calculate the norm */
  ierr = VecCopy(tao->gradient,gpcg->G_New); CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,gpcg->PG); CHKERRQ(ierr);
  ierr = VecNorm(gpcg->PG,NORM_2,&gpcg->gnorm);
  tao->step=1.0;
  gpcg->f = f;

    /* Check Stopping Condition      */
  ierr=TaoSolverMonitor(tao,iter,f,gpcg->gnorm,0.0,tao->step,&reason); CHKERRQ(ierr);

  while (reason == TAO_CONTINUE_ITERATING){

    ierr = GPCGGradProjections(tao); CHKERRQ(ierr);
    ierr = ISGetSize(gpcg->Free_Local,&gpcg->n_free); CHKERRQ(ierr);

    f=gpcg->f; gnorm=gpcg->gnorm; 

    if (gpcg->subset_type != TAOSUBSET_REDISTRIBUTE) {
      if (tao->ksp) {
	ierr = KSPDestroy(&tao->ksp); CHKERRQ(ierr);
      }
      ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);

      if (gpcg->ksp_type == GPCG_KSP_NASH) {
	ierr = KSPSetType(tao->ksp,KSPNASH); CHKERRQ(ierr);
      }	else if (gpcg->ksp_type == GPCG_KSP_STCG) {
	ierr = KSPSetType(tao->ksp,KSPSTCG); CHKERRQ(ierr);
      } else {
	ierr = KSPSetType(tao->ksp,KSPGLTR); CHKERRQ(ierr);
      }	  
      if (tao->ksp->ops->setfromoptions) {
	(*tao->ksp->ops->setfromoptions)(tao->ksp);
      }

    }      

     //    if (gpcg->n_free == gpcg->n)

    if (gpcg->n_free > 0){
      
      /* Create a reduced linear system */
      ierr = VecGetSubVec(tao->gradient,gpcg->Free_Local, &gpcg->R); CHKERRQ(ierr);
      ierr = VecScale(gpcg->R, -1.0); CHKERRQ(ierr);
      ierr = VecGetSubVec(tao->stepdirection,gpcg->Free_Local, &gpcg->DXFree); CHKERRQ(ierr);
      ierr = VecSet(gpcg->DXFree,0.0); CHKERRQ(ierr);

      
      ierr = MatGetSubMatrix(tao->hessian, gpcg->Free_Local, gpcg->Free_Local, MAT_INITIAL_MATRIX,&gpcg->Hsub); CHKERRQ(ierr);

      if (tao->hessian_pre == tao->hessian) {
	  gpcg->Hsub_pre = gpcg->Hsub;
	  ierr = PetscObjectReference((PetscObject)gpcg->Hsub); CHKERRQ(ierr);
      }	 else {
	  ierr = MatGetSubMatrix(tao->hessian_pre,  gpcg->Free_Local, gpcg->Free_Local, MAT_INITIAL_MATRIX,&gpcg->Hsub_pre); CHKERRQ(ierr);
      }

      if (gpcg->subset_type == TAOSUBSET_REDISTRIBUTE) {
	  // Need to create ksp each time  (really only if size changes...)
	  if (tao->ksp) {
	      ierr = KSPDestroy(&tao->ksp); CHKERRQ(ierr);
	  }
	  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);

	  if (gpcg->ksp_type == GPCG_KSP_NASH) {
	      ierr = KSPSetType(tao->ksp,KSPNASH); CHKERRQ(ierr);
	  }	else if (gpcg->ksp_type == GPCG_KSP_STCG) {
	      ierr = KSPSetType(tao->ksp,KSPSTCG); CHKERRQ(ierr);
	  } else {
	      ierr = KSPSetType(tao->ksp,KSPGLTR); CHKERRQ(ierr);
	  }	  
	  if (tao->ksp->ops->setfromoptions) {
	      (*tao->ksp->ops->setfromoptions)(tao->ksp);
	  }

	  //ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
      }      
      
      ierr = KSPSetOperators(tao->ksp,gpcg->Hsub,gpcg->Hsub_pre,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr); //give options for this...
      ierr = PetscObjectDereference((PetscObject)gpcg->Hsub); CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)gpcg->Hsub_pre); CHKERRQ(ierr);

      ierr = KSPSolve(tao->ksp,gpcg->R,gpcg->DXFree); CHKERRQ(ierr);

      ierr = KSPDestroy(&tao->ksp); CHKERRQ(ierr);
      tao->ksp = PETSC_NULL;

      ierr = VecSet(tao->stepdirection,0.0); CHKERRQ(ierr);
      ierr = VecReducedXPY(tao->stepdirection,gpcg->DXFree,gpcg->Free_Local);CHKERRQ(ierr);

      ierr = VecDot(tao->stepdirection,tao->gradient,&gdx); CHKERRQ(ierr);
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
      f_new=f;
      ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f_new,tao->gradient,tao->stepdirection,&stepsize,
				&ls_status);   CHKERRQ(ierr);
      
      actred = f_new - f;
      
      /* Evaluate the function and gradient at the new point */      
      ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU, gpcg->PG); CHKERRQ(ierr);
      ierr = VecNorm(gpcg->PG, NORM_2, &gnorm); CHKERRQ(ierr);
      f=f_new;
      ierr = ISDestroy(&gpcg->Free_Local); CHKERRQ(ierr);
      ierr = VecWhichBetween(tao->XL,tao->solution,tao->XU,&gpcg->Free_Local); CHKERRQ(ierr);
      
    } else {

      actred = 0; gpcg->step=1.0;
      /* if there were no free variables, no cg method */
    }

    iter++;
    ierr = TaoSolverMonitor(tao,iter,f,gnorm,0.0,gpcg->step,&reason); CHKERRQ(ierr);
    gpcg->f=f;gpcg->gnorm=gnorm; gpcg->actred=actred;
    if (reason!=TAO_CONTINUE_ITERATING) break;


  }  /* END MAIN LOOP  */

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "GPCGGradProjections"
static PetscErrorCode GPCGGradProjections(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscInt i;
  PetscReal actred=-1.0,actred_max=0.0, gAg,gtg=gpcg->gnorm,alpha;
  PetscReal f_new,gdx,stepsize;
  Vec DX=tao->stepdirection,XL=tao->XL,XU=tao->XU,Work=gpcg->Work;
  Vec X=tao->solution,G=tao->gradient;
  TaoLineSearchTerminationReason lsflag=TAOLINESEARCH_CONTINUE_ITERATING;

  /*
     The free, active, and binding variables should be already identified
  */
  
  PetscFunctionBegin;

  for (i=0;i<gpcg->maxgpits;i++){
    if ( -actred <= (gpcg->pg_ftol)*actred_max) break;
    ierr = VecBoundGradientProjection(G,X,XL,XU,DX); CHKERRQ(ierr);
    ierr = VecScale(DX,-1.0); CHKERRQ(ierr);
    ierr = VecDot(DX,G,&gdx); CHKERRQ(ierr);

    ierr = MatMult(tao->hessian,DX,Work); CHKERRQ(ierr);
    ierr = VecDot(DX,Work,&gAg); CHKERRQ(ierr);

    gpcg->gp_iterates++; 
    gpcg->total_gp_its++;    
  
    gtg=-gdx;
    alpha = PetscAbsReal(gtg/gAg);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,alpha); CHKERRQ(ierr);
    f_new=gpcg->f;
    ierr = TaoLineSearchApply(tao->linesearch,X,&f_new,G,DX,&stepsize,&lsflag);
    CHKERRQ(ierr);

    /* Update the iterate */
    actred = f_new - gpcg->f;
    actred_max = PetscMax(actred_max,-(f_new - gpcg->f));
    gpcg->f = f_new;
    ierr = ISDestroy(&gpcg->Free_Local); CHKERRQ(ierr);
    ierr = VecWhichBetween(XL,X,XU,&gpcg->Free_Local); CHKERRQ(ierr);
  }
  
  gpcg->gnorm=gtg;
  PetscFunctionReturn(0);

} /* End gradient projections */




#undef __FUNCT__  
#define __FUNCT__ "TaoSolverComputeDual_GPCG" 
static PetscErrorCode TaoSolverComputeDual_GPCG(TaoSolver tao, Vec DXL, Vec DXU)
{

  TAO_GPCG *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecBoundGradientProjection(tao->gradient, tao->solution, tao->XL, tao->XU, gpcg->Work); CHKERRQ(ierr);

  ierr = VecCopy(gpcg->Work, DXL); CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,tao->gradient); CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU); CHKERRQ(ierr);

  ierr = VecCopy(tao->gradient,DXU); CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,gpcg->Work); CHKERRQ(ierr);
  ierr = VecSet(gpcg->Work,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseMin(DXU,gpcg->Work,DXU); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_GPCG"
PetscErrorCode TaoSolverCreate_GPCG(TaoSolver tao)
{
  TAO_GPCG *gpcg;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSolverSetup_GPCG;
  tao->ops->solve = TaoSolverSolve_GPCG;
  tao->ops->view  = TaoSolverView_GPCG;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_GPCG;
  tao->ops->destroy = TaoSolverDestroy_GPCG;
  tao->ops->computedual = TaoSolverComputeDual_GPCG;

  ierr = PetscNewLog(tao, TAO_GPCG, &gpcg); CHKERRQ(ierr);
  tao->data = (void*)gpcg;

  tao->max_its = 500;
  tao->max_funcs = 100000;
  tao->fatol = 1e-12;
  tao->frtol = 1e-12;

  /* Initialize pointers and variables */
  gpcg->n=0;
  gpcg->maxgpits = 8;
  gpcg->pg_ftol = 0.1;

  gpcg->gp_iterates=0; /* Cumulative number */
  gpcg->total_gp_its = 0;
 
  /* Initialize pointers and variables */
  gpcg->n_bind=0;
  gpcg->n_free = 0;
  gpcg->n_upper=0;
  gpcg->n_lower=0;
  gpcg->subset_type = TAOSUBSET_MASK;
  gpcg->ksp_type = GPCG_KSP_STCG;


      
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, TAOLINESEARCH_GPCG); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, GPCGObjectiveAndGradient, tao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END




