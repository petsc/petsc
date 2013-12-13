#include "private/taosolver_impl.h"
#include "poundersb.h"
#undef __FUNCT__
#define __FUNCT__ "TaoPounders_formquad"
/* Calls Matlab version of formquad for error checking in development */
/* function [Mdir,np,valid,G,H,Mind] = formquad(X,F,delta,xkin,npmax,Pars,vf)*/
/*
  Given:
    mfqP->delta
    mfqP->Xhist (nHist-array of n-Vec)
    mfqP->Fhist (nHist-array of m-Vec)
    mfqP->minindex

  Computes:
    mfqP->Mdir (array of n-Vec)
    mfqP->nmodelpoints
    mfqP->valid
    mfqP->model_indices (array of integers)
    if (!checkonly):
       mfqP->Gdel (m-array of n-Vec)
       mfqp->Hdel (m-array of nxn Mat)
*/
PetscErrorCode TaoPounders_formquad(TAO_POUNDERS *mfqP,PetscBool checkonly)
{
  PetscErrorCode ierr;
  PetscScalar *h,tempx[1];
  PetscInt i,j,*ind;
  const char *machine="localhost";
  char name[12];
  Vec Mind_real;

#ifdef PETSC_HAVE_MATLAB_ENGINE  
  if (!mfqP->me) {
    ierr = PetscMatlabEngineCreate(PETSC_COMM_SELF,machine,&mfqP->me); CHKERRQ(ierr);
  }
#endif
  ierr = PetscMatlabEngineEvaluate(me,"Xhist = zeros(%d,%d);",mfqP->nHist,mfqP->n); CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(me,"Fhist = zeros(%d,%d);",mfqP->nHist,mfqP->n); CHKERRQ(ierr);
  for (i=0;i<mfqP->nHist;i++) {
    snprintf(name,12,"x%010d",i);
    ierr = PetscObjectSetName((PetscObject)(mfqP->Xhist[i]),name);CHKERRQ(ierr);
    ierr = PetscMatlabEnginePut(me,mfqP->Xhist[i]);CHKERRQ(ierr);
    ierr = PetscMatlabEngineEvaluate(me,"Xhist(%d+1,:) = %s;",i,name); CHKERRQ(ierr);
    snprintf(name,12,"f%010d",i);
    ierr = PetscObjectSetName((PetscObject)(mfqP->Fhist[i]),name);CHKERRQ(ierr);
    ierr = PetscMatlabEnginePut(me,mfqP->Fhist[i]);CHKERRQ(ierr);
    ierr = PetscMatlabEngineEvaluate(me,"Fhist(%d,:) = %s;",i,name); CHKERRQ(ierr);
  }
  ierr = PetscMatlabEngineEvaluate(me,"delta=%f;",mfqP->delta); CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(me,"xkin=%d;",mfqP->xkin+1); CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(me,"npmax=%d;",mfqP->npmax); CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(me,"Pars = [%f,%f,%f,%f];",mfqP->par1,mfqP->par2,mfqP->par3,mfqP->par4); CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(me,"vf=%d;",checkonly?1:0); CHKERRQ(ierr);

  ierr = PetscMatlabEngineEvaluate(me,"[Mdir,np,valid,G,H,Mind] = formquad(X,F,delta,xkin,npmax,Pars,vf);"); CHKERRQ(ierr);

  /* Get Mdir */
  ierr = PetscObjectSetName((PetscObject)(mfqP->Mdir),"Mdir"); CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(me,mfqP->Mdir); CHKERRQ(ierr);

  /* Get np */
  ierr = PetscMatlabEngineGetArray(me,1,1,tempx,"np"); CHKERRQ(ierr);
  mfqP->nmodelpoints = floor(tempx[0]+0.5);

  /* Get valid */
  ierr = PetscMatlabEngineGetArray(me,1,1,tempx,"valid"); CHKERRQ(ierr);
  mfqP->valid = floor(tempx[0]+0.5);

  /* Get Mind */
  ierr = VecCreate(PETSC_COMM_SELF,&Mind_real);
  ierr = PetscObjectSetName((PetscObject)(mfqP->Mind_real),"Mind"); CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(me,mfqP->Mind_real); CHKERRQ(ierr);
  ierr = VecGetArray(mfqP->Mind_real,&v); CHKERRQ(ierr);
  for (i=0;i<mfqP->n;i++) {
    mfqP->model_indices[i] = floor(v[i]+0.5);
  }
  ierr = VecRestoreArray(mfqP->Mind_real,&v); CHKERRQ(ierr);
  ierr = VecDestroy(&Mind_real); CHKERRQ(ierr);

  if (!checkonly) {
    /* Get Gdel */
    for (i=0;i<mfqP->m;i++) {
      snprintf(name,12,"g%010d",i);
      ierr = PetscObjectSetName((PetscObject)(mfqP->Gdel[i]),name);CHKERRQ(ierr);
      ierr = PetscMatlabEngineEvaluate(me,"%s = G(%d+1,:);",name,i); CHKERRQ(ierr);
      ierr = PetscMatlabEngineGet(me,mfqP->Gdel[i]);CHKERRQ(ierr);
    }  
    /* Get Hdel */
    for (i=0;i<mfqP->m;i++) {
      snprintf(name,12,"h%010d",i);
      ierr = PetscMatlabeEngineEvaluate(me,"%s = H(:,:,%d+1);",name,i); CHKERRQ(ierr);
      ierr = PetscMatlabEngineGetArray(me,);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "pounders_h"
static PetscErrorCode pounders_h(TaoSolver subtao, Vec v, Mat *H, Mat *Hpre, MatStructure *flag, void *ctx)
{
  PetscFunctionBegin;
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
#undef __FUNCT__ 
#define __FUNCT__ "pounders_fg"
static PetscErrorCode  pounders_fg(TaoSolver subtao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)ctx;
  PetscReal d1,d2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* g = A*x  (add b later)*/
  ierr = MatMult(mfqP->Hs,x,g); CHKERRQ(ierr);


  /* f = 1/2 * x'*(Ax) + b'*x  */
  ierr = VecDot(x,g,&d1); CHKERRQ(ierr);
  ierr = VecDot(mfqP->b,x,&d2); CHKERRQ(ierr);
  *f = 0.5 *d1 + d2;

  /* now  g = g + b */
  ierr = VecAXPY(g, 1.0, mfqP->b); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoPounders_solvequadratic"
PetscErrorCode TaoPounders_solvequadratic(TaoSolver tao,PetscReal *gnorm, PetscReal *qmin) 
{
    PetscErrorCode ierr;
    PetscReal atol=1.0e-10;
    PetscInt info,its;
    TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
    PetscReal maxval;
    PetscInt i,j;
								    
    PetscFunctionBegin;

    ierr = VecCopy(mfqP->Gres, mfqP->subb); CHKERRQ(ierr);

    ierr = VecSet(mfqP->subx,0.0); CHKERRQ(ierr);

    ierr = VecSet(mfqP->subndel,-mfqP->delta); CHKERRQ(ierr);
    ierr = VecSet(mfqP->subpdel,mfqP->delta); CHKERRQ(ierr);

    ierr = MatCopy(mfqP->Hres,mfqP->subH,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
      
    ierr = TaoResetStatistics(mfqP->subtao); CHKERRQ(ierr);
    ierr = TaoSetTolerances(mfqP->subtao,PETSC_NULL,PETSC_NULL,*gnorm,*gnorm,PETSC_NULL); CHKERRQ(ierr);
    /* enforce bound constraints -- experimental */
    if (tao->XU && tao->XL) {
      ierr = VecCopy(tao->XU,mfqP->subxu); CHKERRQ(ierr);
      ierr = VecAXPY(mfqP->subxu,-1.0,tao->solution); CHKERRQ(ierr);
      ierr = VecScale(mfqP->subxu,1.0/mfqP->delta); CHKERRQ(ierr);
      ierr = VecCopy(tao->XL,mfqP->subxl); CHKERRQ(ierr);
      ierr = VecAXPY(mfqP->subxl,-1.0,tao->solution); CHKERRQ(ierr);
      ierr = VecScale(mfqP->subxl,1.0/mfqP->delta); CHKERRQ(ierr);
	
      ierr = VecPointwiseMin(mfqP->subxu,mfqP->subxu,mfqP->subpdel); CHKERRQ(ierr);
      ierr = VecPointwiseMax(mfqP->subxl,mfqP->subxl,mfqP->subndel); CHKERRQ(ierr);
    } else {
      ierr = VecCopy(mfqP->subpdel,mfqP->subxu); CHKERRQ(ierr);
      ierr = VecCopy(mfqP->subndel,mfqP->subxl); CHKERRQ(ierr);
    }
    /* Make sure xu > xl */
    ierr = VecCopy(mfqP->subxl,mfqP->subpdel); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu);  CHKERRQ(ierr);
    ierr = VecMax(mfqP->subpdel,PETSC_NULL,&maxval); CHKERRQ(ierr);
    if (maxval > 1e-10) {
      SETERRQ(PETSC_COMM_WORLD,1,"upper bound < lower bound in subproblem");
    }
    /* Make sure xu > tao->solution > xl */
    ierr = VecCopy(mfqP->subxl,mfqP->subpdel); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->subpdel,-1.0,mfqP->subx);  CHKERRQ(ierr);
    ierr = VecMax(mfqP->subpdel,PETSC_NULL,&maxval); CHKERRQ(ierr);
    if (maxval > 1e-10) {
      SETERRQ(PETSC_COMM_WORLD,1,"initial guess < lower bound in subproblem");
    }

    ierr = VecCopy(mfqP->subx,mfqP->subpdel); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu);  CHKERRQ(ierr);
    ierr = VecMax(mfqP->subpdel,PETSC_NULL,&maxval); CHKERRQ(ierr);
    if (maxval > 1e-10) {
      SETERRQ(PETSC_COMM_WORLD,1,"initial guess > upper bound in subproblem");
    }


    ierr = TaoSolve(mfqP->subtao); CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(mfqP->subtao,PETSC_NULL,qmin,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);

    /* test bounds post-solution*/
    ierr = VecCopy(mfqP->subxl,mfqP->subpdel); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->subpdel,-1.0,mfqP->subx);  CHKERRQ(ierr);
    ierr = VecMax(mfqP->subpdel,PETSC_NULL,&maxval); CHKERRQ(ierr);
    if (maxval > 1e-5) {
      ierr = PetscInfo(tao,"subproblem solution < lower bound"); CHKERRQ(ierr);
      tao->reason = TAO_DIVERGED_TR_REDUCTION;
    }

    ierr = VecCopy(mfqP->subx,mfqP->subpdel); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu);  CHKERRQ(ierr);
    ierr = VecMax(mfqP->subpdel,PETSC_NULL,&maxval); CHKERRQ(ierr);
    if (maxval > 1e-5) {
      ierr = PetscInfo(tao,"subproblem solution > upper bound");
      tao->reason = TAO_DIVERGED_TR_REDUCTION;
    }
      

    *qmin *= -1;
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoPounders_bpmts"
PetscErrorCode TaoPounders_bmpts(TaoSolver tao)
{
  /* TODO: set t1,t2 as data members of TAO_POUNDERS */
  PetscErrorCode ierr;
  PetscInt i,low,high;
  PetscReal minnorm,*t1,*t2;
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data; 
  PetscFunctionBegin;
  
  ierr = PetscMalloc(sizeof(PetscReal)*mfqP->nmodelpoints,&t1); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*mfqP->nmodelpoints,&t2); CHKERRQ(ierr);
  /* For each ray, find largest t to remain feasible */
  mint = TAO_INFINITY;
  maxt = TAO_NINFINITY;
  for (i=1;i<=mfqP->nmodelpoints;i++) {
    ierr = VecStepMaxBounded(mfqP->Xhist[mfqP->modelindices[i]],mfqP->Xhist[mfqP->minindex],tao->XL,tao->XU,&t[i]); CHKERRQ(ierr);
    ierr = VecCopy(mfqP->Xhist[mfqP->modelindices[i]],mfqP->workxvec); CHKERRQ(ierr);
    ierr = VecScale(mfqP->workxvec,-1.0); CHKERRQ(ierr);
    ierr = VecStepMaxBounded(mfqP->workxvec,mfqP->Xhist[mfqP->minindex],tao->XL,tao->XU,&t[i]); CHKERRQ(ierr);
    mint = PetscMin(mint,t1);
    mint = PetscMin(mint,t2);
    maxt = PetscMax(maxt,t1);
    maxt = PetscMax(maxt,t2);
  }
  
  /* Compute objective at x+delta*e_i, i=1..n*/
  ierr = VecGetOwnershipRange(mfqP->Xhist[0],&low,&high); CHKERRQ(ierr);
  for (i=1;i<=mfqP->n;i++) {
      ierr = VecCopy(tao->solution,mfqP->Xhist[i]); CHKERRQ(ierr);
      if (i-1 >= low && i-1 < high) {
	  ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	  x[i-1-low] += mfqP->delta;
	  ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
      }
      ierr = TaoComputeSeparableObjective(tao,mfqP->Xhist[i],mfqP->Fhist[i]); CHKERRQ(ierr);
      ierr = VecNorm(mfqP->Fhist[i],NORM_2,&mfqP->Fres[i]); CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(mfqP->Fres[i])) {
	SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
      }
      mfqP->Fres[i]*=mfqP->Fres[i];
      if (mfqP->Fres[i] < minnorm) {
	  mfqP->minindex = i;
	  minnorm = mfqP->Fres[i];
      }
  }
  ierr = PetscFree(t1); CHKERRQ(ierr);
  ierr = PetscFree(t2); CHKERRQ(ierr);
  

  PetscFunctionReturn(0);
}
/*
#undef __FUNCT__
#define __FUNCT__ "phi2eval"
PetscErrorCode phi2eval(Vec *X, Vec *Phi, PetscInt n) {
  PetscInt i,j,k,lo,hi;
  PetscErrorCode ierr;
  PetscReal sqrt2;

  PetscFunctionBegin;  
  sqrt = PetscSqrtReal(2.0);
  ierr = VecGetOwnershipRange(X,&lo,&hi); CHKERRQ(ierr);
PetscErrorCode phi2eval(PetscReal *x, PetscInt n, PetscReal *phi) {
// Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2] 
    PetscInt i,j,k;
    PetscReal sqrt2 = PetscSqrtReal(2.0);
    PetscFunctionBegin;
    j=0;

    for (i=0;i<n;i++) {
	phi[j] = 0.5 * x[i]*x[i];
	j++;
	for (k=i+1;k<n;k++) {
	    phi[j]  = x[i]*x[k]/sqrt2;
	    j++;
	}
	
    }

    PetscFunctionReturn(0);
    }*/
PetscErrorCode PoundersGramSchmidtReset(TAO_POUNDERS *mfqP, Vec *Q, PetscInt n)
{
  PetscInt i;
  for (i=0;i<n;i++) {
    ierr = VecSet(Q[i],0.0); CHKERRQ(ierr);
    ierr = VecSetValue(Q[i],i,1.0,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(Q[i]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Q[i]); CHKERRQ(ierr);
  }
}


#undef __FUNCT__
#define __FUNCT__ "PoundersGramSchmidtInsert"
static PetscErrorCode PoundersGramSchmidtInsert(TAO_POUNDERS *mfqP, Mat,Vec)
{
  return 0;
}
