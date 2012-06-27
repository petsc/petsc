#include "private/taosolver_impl.h"
#undef __FUNCT__
#define __FUNCT__ "TaoPounders_formquad"
/* Calls Matlab version of formquad for error checking in development */
/* function [Mdir,np,valid,G,H,Mind] = formquad(X,F,delta,xkin,npmax,Pars,vf)*/
/*
  Given:
    mfqP->delta
    mfqP->Xhist (array of n-Vec)
    mfqP->Fhist (array of m-Vec)
    mfqP->minindex

  Computes:
    mfqP->model_indices (array of integers)
    mfqP->nmodelpoints
    mfqP->valid
    mfqP->Mdir (array of n-Vec)
    if (!checkonly):
       mfqP->Gdel (m-array of n-Vec)
       mfqp->Hdel (m-array of nxn Mat)
*/
PetscErrorCode TaoPounders_formquad(TAO_POUNDERS *mfqP,PetscBool checkonly)
{
  PetscErrorCode ierr;
  PetscScalar *h;
  PetscInt i,j;
  
  for (i=0;i<mfqP->nHist;i++) {
    

  
  /* Write X */
  fprintf(outfile,"X = [");
  for (i=0;i<mfqP->nHist;i++) {
    ierr = VecGetArray(mfqP->Xhist[i],&v); CHKERRQ(ierr);
    for (j=0;j<mfqP->n;j++) {
      fprintf(outfile,"%20.12f ",v[j]); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mfqP->Xhist[i],&v); CHKERRQ(ierr);
    fprintf(outfile,";\n");
  }
  fprintf(outfile,"];\n\n");


  /* Write F */
  fprintf(outfile,"F = [");
  for (i=0;i<mfqP->nHist;i++) {
    ierr = VecGetArray(mfqP->Fhist[i],&v); CHKERRQ(ierr);
    for (j=0;j<mfqP->m;j++) {
      fprintf(outfile,"%20.12f ",v[j]); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mfqP->Fhist[i],&v); CHKERRQ(ierr);
    fprintf(outfile,";\n");
  }
  fprintf(outfile,"];\n\n");

  fprintf(outfile,"delta=%20.12f;\n",mfqP->delta);
  fprintf(outfile,"xkin=%d;\n",mfqP->minindex+1);
  fprintf(outfile,"npmax=%d;\n",mfqP->npmax);
  fprintf(outfile,"Pars=[%f,%f,%f,%f];\n",mfqP->par1,mfqP->par2,
		mfqP->par3,mfqP->par4);
  fprintf(outfile,"vf=%d;\n",checkonly?1:0);
  fprintf(outfile,"[Mdir,np,valid,Gres,Hresdel,Mind] = formquad(X,F,delta,xkin,npmax,Pars,vf)\n");
  fprintf(outfile,"fid = fopen('formquad.out','wt');\n");
  fprintf(outfile,"fprintf(fid,'%d\\n',np);\n");
  fprintf(outfile,"fprintf(fid,'%d\\n',Mind);\n");
  fprintf(outfile,"fprintf(fid,'%d\\n',valid);\n");
  fprintf(outfile,"fprintf(fid,'%20.12f',Gres');\n");
  fprintf(outfile,"fprintf(fid,'\\n');\n");
  fprintf(outfile,"fprintf(fid,'%20.12f',Hresdel');\n");
  fprintf(outfile,"fprintf(fid,'\\n');\n");
  fclose(outfile);

  system("octave < formquad.m");

  /* Read nmodelpoints,model_indices,valid,G,H,Mdir */
  infile = fopen("formquad.out","r");
  fscanf(infile,"%d\n",&mfqP->nmodelpoints);
  if (mfqP->nmodelpoints > mfqP->npmax) {
    SETERRQ(PETSC_COMM_SELF,"Cowardly bailing out, too many model points\n");
  }
  for (i=0;i<mfqP->nmodelpoints;i++) {
    fscanf(infile,"%d",mfqP->model_indices[i]);
    mfqP->model_indices[i]--;
  }
  fscanf(infile,"\n");
  fscanf(infile,"%d\n",&mfqP->valid);

  for (i=0;i<mfqP->m;i++) {
    for (j=0;j<mfqP->n;j++) {
      fscanf(infile,"%d",mfqP->Gdel_array[j]);
    }
    ierr = VecSetValues(mfqP->Gdel[i],mfqP->n,indices,mfqP->Gdel_array,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mfqP->Gdel[i]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mfqP->Gdel[i]); CHKERRQ(ierr);
    fscanf(infile,"\n");
  }
  fscanf(infile,"\n");
  for (i=0;i<mfqP->m;i++) {
    for (j=0;j<mfqP->n;j++) {
      for (k=0;k<mfqP->n;k++) {
	fscanf(infile,"%d",mfqP->Hdel_array[j*mfqP->n + k]);
      }
      fscanf(infile,"\n");
    }
    ierr = MatSetValues(mfqP->Hdel[i],mfqP->n, indices, mfqP->n, indices,
			mfqP->Hdel_array,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(mfqP->Hdel[i],MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mfqP->Hdel[i],MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  ierr = PetscFree(&indices); CHKERRQ(ierr);
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
