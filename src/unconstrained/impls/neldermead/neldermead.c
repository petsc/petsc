#include "neldermead.h"
#include "petscvec.h"

static PetscErrorCode NelderMeadSort(TAO_NelderMead *nm);
static PetscErrorCode NelderMeadReplace(TAO_NelderMead *nm, PetscInt index, Vec Xmu, PetscReal f);
/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetUp_NM"
static PetscErrorCode TaoSolverSetUp_NM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscInt size;

  PetscFunctionBegin;
  ierr = VecGetSize(tao->solution,&size); CHKERRQ(ierr);
  nm->N = size;
  nm->oneOverN = 1.0/size;
  ierr = VecDuplicateVecs(tao->solution,nm->N+1,&nm->simplex); CHKERRQ(ierr);
  ierr = PetscMalloc((nm->N+1)*sizeof(PetscReal),&nm->f_values); CHKERRQ(ierr);
  ierr = PetscMalloc((nm->N+1)*sizeof(PetscInt),&nm->indices); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xbar); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmur); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmue); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmuc); CHKERRQ(ierr);

  tao->gradient=0;
  tao->step=0;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_NM"
PetscErrorCode TaoSolverDestroy_NM(TaoSolver tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroyVecs(nm->N+1,&nm->simplex); CHKERRQ(ierr);
    ierr = VecDestroy(nm->Xmuc); CHKERRQ(ierr);
    ierr = VecDestroy(nm->Xmue); CHKERRQ(ierr);
    ierr = VecDestroy(nm->Xmur); CHKERRQ(ierr);
    ierr = VecDestroy(nm->Xbar); CHKERRQ(ierr);
  }
  ierr = PetscFree(nm->indices); CHKERRQ(ierr);
  ierr = PetscFree(nm->f_values); CHKERRQ(ierr);
  ierr = PetscFree(tao->data);
  tao->data = 0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_NM"
PetscErrorCode TaoSolverSetFromOptions_NM(TaoSolver tao)
{
  
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscBool flg;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Nelder-Mead options"); CHKERRQ(ierr);

  ierr = PetscOptionsReal("-tao_nm_lamda","initial step length","",nm->lamda,&nm->lamda,&flg);  CHKERRQ(ierr);

  ierr = PetscOptionsReal("-tao_nm_mu","mu","",nm->mu_oc,&nm->mu_oc,&flg); CHKERRQ(ierr);
  nm->mu_ic = -nm->mu_oc;
  nm->mu_r = nm->mu_oc*2.0;
  nm->mu_e = nm->mu_oc*4.0;

  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_NM"
PetscErrorCode TaoSolverView_NM(TaoSolver tao,PetscViewer viewer)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscBool isascii;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  expansions: %d\n",nm->nexpand); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  reflections: %d\n",nm->nreflect); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  inside contractions: %d\n",nm->nincontract); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  outside contractionss: %d\n",nm->noutcontract); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Shrink steps: %d\n",nm->nshrink); CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO NelderMead",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_NM"
PetscErrorCode TaoSolverSolve_NM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  TaoSolverTerminationReason reason;
  PetscReal *x;
  PetscInt iter=0,i;
  Vec Xmur=nm->Xmur, Xmue=nm->Xmue, Xmuc=nm->Xmuc, Xbar=nm->Xbar;
  PetscReal fr,fe,fc;
  PetscInt shrink;
  PetscInt low,high;
  
  
  PetscFunctionBegin;
  nm->nshrink =      0;
  nm->nreflect =     0;
  nm->nincontract =  0;
  nm->noutcontract = 0;
  nm->nexpand =      0;
  
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by NelderMead algorithm\n"); CHKERRQ(ierr);
  }

  ierr = VecCopy(tao->solution,nm->simplex[0]); CHKERRQ(ierr);
  ierr = TaoSolverComputeObjective(tao,nm->simplex[0],&nm->f_values[0]); CHKERRQ(ierr);
  nm->indices[0]=0;
  for (i=1;i<nm->N+1;i++){
    ierr = VecCopy(tao->solution,nm->simplex[i]); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(nm->simplex[i],&low,&high); CHKERRQ(ierr);
    if (i-1 >= low && i-1 < high) {
      ierr = VecGetArray(nm->simplex[i],&x); CHKERRQ(ierr);
      x[i-1-low] += nm->lamda;
      ierr = VecRestoreArray(nm->simplex[i],&x); CHKERRQ(ierr);
    }

    ierr = TaoSolverComputeObjective(tao,nm->simplex[i],&nm->f_values[i]); CHKERRQ(ierr);
    nm->indices[i] = i;
  }
  
  // Xbar  = (Sum of all simplex vectors - worst vector)/N
  ierr = NelderMeadSort(nm); CHKERRQ(ierr);
  ierr = VecSet(Xbar,0.0); CHKERRQ(ierr);
  for (i=0;i<nm->N;i++) {
    ierr = VecAXPY(Xbar,1.0,nm->simplex[nm->indices[i]]);
  }
  ierr = VecScale(Xbar,nm->oneOverN);
  reason = TAO_CONTINUE_ITERATING;
  while (1) {
    shrink = 0;
    ierr = VecCopy(nm->simplex[nm->indices[0]],tao->solution); CHKERRQ(ierr);
    ierr = TaoSolverMonitor(tao,iter++,nm->f_values[nm->indices[0]],nm->f_values[nm->indices[nm->N]]-nm->f_values[nm->indices[0]],0.0,1.0,&reason); CHKERRQ(ierr);
    if (reason != TAO_CONTINUE_ITERATING) break;

    
    
    //x(mu) = (1 + mu)Xbar - mu*X_N+1
    ierr = VecAXPBYPCZ(Xmur,1+nm->mu_r,-nm->mu_r,0,Xbar,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
    ierr = TaoSolverComputeObjective(tao,Xmur,&fr); CHKERRQ(ierr);


    if (nm->f_values[nm->indices[0]] <= fr && fr < nm->f_values[nm->indices[nm->N-1]]) {
      // reflect
      nm->nreflect++;
      ierr = PetscInfo(0,"Reflect\n"); CHKERRQ(ierr);
      ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr); CHKERRQ(ierr);
    }

    else if (fr < nm->f_values[nm->indices[0]]) {
      // expand
      nm->nexpand++;
      ierr = PetscInfo(0,"Expand\n"); CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmue,1+nm->mu_e,-nm->mu_e,0,Xbar,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
      ierr = TaoSolverComputeObjective(tao,Xmue,&fe); CHKERRQ(ierr);
      if (fe < fr) {
	ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmue,fe); CHKERRQ(ierr);
      } else {
	ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr); CHKERRQ(ierr);
      }

    } else if (nm->f_values[nm->indices[nm->N-1]] <= fr && fr < nm->f_values[nm->indices[nm->N]]) {
      //outside contraction
      nm->noutcontract++;
      ierr = PetscInfo(0,"Outside Contraction\n"); CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmuc,1+nm->mu_oc,-nm->mu_oc,0,Xbar,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);

      ierr = TaoSolverComputeObjective(tao,Xmuc,&fc); CHKERRQ(ierr);
      if (fc <= fr) {
	ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc); CHKERRQ(ierr);
      }	else 
	shrink=1;
    } else {
      //inside contraction
      nm->nincontract++;
      ierr = PetscInfo(0,"Inside Contraction\n"); CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmuc,1+nm->mu_ic,-nm->mu_ic,0,Xbar,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
      ierr = TaoSolverComputeObjective(tao,Xmuc,&fc); CHKERRQ(ierr);
      if (fc < nm->f_values[nm->indices[nm->N]]) {
	ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc); CHKERRQ(ierr);
      } else
	shrink = 1;
    }

    if (shrink) {
      nm->nshrink++;
      ierr = PetscInfo(0,"Shrink\n"); CHKERRQ(ierr);
      
      for (i=1;i<nm->N+1;i++) {
	  ierr = VecAXPBY(nm->simplex[nm->indices[i]],1.5,-0.5,nm->simplex[nm->indices[0]]);
	ierr = TaoSolverComputeObjective(tao,nm->simplex[nm->indices[i]],
				 &nm->f_values[nm->indices[i]]); CHKERRQ(ierr);
      }
      ierr = VecAXPBY(Xbar,1.5*nm->oneOverN,-0.5,nm->simplex[nm->indices[0]]); CHKERRQ(ierr);

      // Add last vector's fraction of average
      ierr = VecAXPY(Xbar,nm->oneOverN,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
      ierr = NelderMeadSort(nm);
      // Subtract new last vector from average
      ierr = VecAXPY(Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
    }
    
    
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_NM"
PetscErrorCode TaoSolverCreate_NM(TaoSolver tao)
{
  TAO_NelderMead *nm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,TAO_NelderMead,&nm); CHKERRQ(ierr);
  tao->data = (void*)nm;

  tao->ops->setup = TaoSolverSetUp_NM;
  tao->ops->solve = TaoSolverSolve_NM;
  tao->ops->view = TaoSolverView_NM;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_NM;
  tao->ops->destroy = TaoSolverDestroy_NM;

  tao->max_its = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-8;
  tao->frtol = 1e-8;

  nm->simplex = 0;
  nm->lamda = 1;

  nm->mu_ic = -0.5;
  nm->mu_oc = 0.5;
  nm->mu_r = 1.0;
  nm->mu_e = 2.0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
    

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "NelderMeadSort"
PetscErrorCode NelderMeadSort(TAO_NelderMead *nm) {
  PetscReal *values = nm->f_values;
  PetscInt *indices = nm->indices;
  PetscInt dim = nm->N+1;

  PetscInt i,j,index;
  PetscReal val;
  PetscFunctionBegin;  
  for (i=1;i<dim;i++) {
    index = indices[i];
    val = values[index];
    for (j=i-1; j>=0 && values[indices[j]] > val; j--) {
      indices[j+1] = indices[j];
    }
    indices[j+1] = index;
  }  
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "NelderMeadReplace"
PetscErrorCode NelderMeadReplace(TAO_NelderMead *nm, PetscInt index, Vec Xmu, PetscReal f)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // Add new vector's fraction of average
  ierr = VecAXPY(nm->Xbar,nm->oneOverN,Xmu); CHKERRQ(ierr);
  ierr = VecCopy(Xmu,nm->simplex[index]); CHKERRQ(ierr);
  nm->f_values[index] = f;

  ierr = NelderMeadSort(nm);

  // Subtract last vector from average
  ierr = VecAXPY(nm->Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
