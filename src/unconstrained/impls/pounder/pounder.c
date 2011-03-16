#include "src/leastsquares/impls/pounders/pounders.h"

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_POUNDER"
static PetscErrorCode TaoSolverSolve_POUNDER(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS *)tao->data;

  PetscInt i,ii,j,k,l,iter=0;
  PetscReal step=1.0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;

  PetscInt low,high;
  PetscReal minnorm;
  PetscReal *x,*f,*fmin,*xmint;
  PetscReal cres,deltaold;
  PetscReal gnorm;
  PetscBLASInt info,ione=1,iblas;
  PetscBool valid;
  PetscReal mdec, rho, normxsp;
  PetscReal one=1.0,zero=0.0,ratio;
  PetscReal objvalue;
  PetscBLASInt blasm,blasn,blasnpmax;
  PetscErrorCode ierr;
  
  
  /* n = # of parameters 
     m = dimension (components) of function  */
  
  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by pounder algorithm\n"); CHKERRQ(ierr);
  }

  CHKMEMQ;
  blasm = mfqP->m; blasn=mfqP->n; blasnpmax = mfqP->npmax;
  for (i=0;i<mfqP->n*mfqP->n*mfqP->m;i++) mfqP->H[i]=0;

  ierr = VecCopy(tao->solution,mfqP->Xhist[0]); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = TaoSolverComputeObjective(tao,tao->solution,&objvalue); CHKERRQ(ierr);
  ierr = VecSetValue(mfqP->Fhist[0],0,objvalue,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(mfqP->Fhist[0]); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(mfqP->Fhist[0]); CHKERRQ(ierr);

  ierr = VecNorm(mfqP->Fhist[0],NORM_2,&mfqP->Fres[0]); CHKERRQ(ierr);
  mfqP->Fres[0]*=mfqP->Fres[0];
  mfqP->minindex = 0;
  minnorm = mfqP->Fres[0];
  ierr = VecGetOwnershipRange(mfqP->Xhist[0],&low,&high); CHKERRQ(ierr);
  for (i=1;i<mfqP->n+1;i++) {
      ierr = VecCopy(tao->solution,mfqP->Xhist[i]); CHKERRQ(ierr);
      if (i-1 >= low && i-1 < high) {
	  ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	  x[i-1-low] += mfqP->delta;
	  ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
      }
      CHKMEMQ;
      ierr = TaoSolverComputeObjective(tao,mfqP->Xhist[i],&objvalue); CHKERRQ(ierr);
      ierr = VecSetValue(mfqP->Fhist[i],0,objvalue,INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(mfqP->Fhist[i]); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(mfqP->Fhist[i]); CHKERRQ(ierr);

      ierr = VecNorm(mfqP->Fhist[i],NORM_2,&mfqP->Fres[i]); CHKERRQ(ierr);
      mfqP->Fres[i]*=mfqP->Fres[i];
      if (mfqP->Fres[i] < minnorm) {
	  mfqP->minindex = i;
	  minnorm = mfqP->Fres[i];
      }
  }

  ierr = VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution); CHKERRQ(ierr);
  /* Gather mpi vecs to one big local vec */

  

  /* Begin serial code */

  /* Disp[i] = Xi-xmin, i=1,..,mfqP->minindex-1,mfqP->minindex+1,..,n */
  /* Fdiff[i] = (Fi-Fmin)', i=1,..,mfqP->minindex-1,mfqP->minindex+1,..,n */
  /* (Column oriented for blas calls) */
  ii=0;

  if (mfqP->mpisize == 1) {
      ierr = VecGetArray(mfqP->Xhist[mfqP->minindex],&xmint); CHKERRQ(ierr);
      for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i]; 
      ierr = VecRestoreArray(mfqP->Xhist[mfqP->minindex],&xmint); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->Fhist[mfqP->minindex],&fmin); CHKERRQ(ierr);
      for (i=0;i<mfqP->n+1;i++) {
	  if (i == mfqP->minindex) continue;

	  ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	  for (j=0;j<mfqP->n;j++) {
	      mfqP->Disp[ii+mfqP->npmax*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
	  }
	  ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);

	  ierr = VecGetArray(mfqP->Fhist[i],&f); CHKERRQ(ierr);
	  for (j=0;j<mfqP->m;j++) {
	      mfqP->Fdiff[ii+mfqP->n*j] = f[j] - fmin[j];
	  }
	  ierr = VecRestoreArray(mfqP->Fhist[i],&f); CHKERRQ(ierr);
	  mfqP->model_indices[ii++] = i;

      }
      for (j=0;j<mfqP->m;j++) {
	  mfqP->C[j] = fmin[j];
      }
      ierr = VecRestoreArray(mfqP->Fhist[mfqP->minindex],&fmin); CHKERRQ(ierr);

  } else {
      ierr = VecScatterBegin(mfqP->scatterx,mfqP->Xhist[mfqP->minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecScatterEnd(mfqP->scatterx,mfqP->Xhist[mfqP->minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->localxmin,&xmint); CHKERRQ(ierr);
      for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i];
      ierr = VecRestoreArray(mfqP->localxmin,&mfqP->xmin); CHKERRQ(ierr);



      ierr = VecScatterBegin(mfqP->scatterf,mfqP->Fhist[mfqP->minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecScatterEnd(mfqP->scatterf,mfqP->Fhist[mfqP->minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->localfmin,&fmin); CHKERRQ(ierr);
      for (i=0;i<mfqP->n+1;i++) {
	  if (i == mfqP->minindex) continue;
				 
	  mfqP->model_indices[ii++] = i;
	  ierr = VecScatterBegin(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecScatterEnd(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecGetArray(mfqP->localx,&x); CHKERRQ(ierr);
	  for (j=0;j<mfqP->n;j++) {
	      mfqP->Disp[i+mfqP->npmax*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
	  }
	  ierr = VecRestoreArray(mfqP->localx,&x); CHKERRQ(ierr);

	  
	  ierr = VecScatterBegin(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecScatterEnd(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecGetArray(mfqP->localf,&f); CHKERRQ(ierr);
	  for (j=0;j<mfqP->m;j++) {
	      mfqP->Fdiff[i*mfqP->n+j] = f[j] - fmin[j];

	  }
	  ierr = VecRestoreArray(mfqP->localf,&f); CHKERRQ(ierr);
      }
      for (j=0;j<mfqP->m;j++) {
	  mfqP->C[j] = fmin[j];
      }
	  
      ierr = VecRestoreArray(mfqP->localfmin,&fmin); CHKERRQ(ierr);
  
  }

	  
  /* Determine the initial quadratic models */
  
  /* G = D(ModelIn,:) \ (F(ModelIn,1:m)-repmat(F(xkin,1:m),n,1)); */
  /* D (nxn) Fdiff (nxm)  => G (nxm) */
  LAPACKgesv_(&blasn,&blasm,mfqP->Disp,&blasnpmax,mfqP->iwork,mfqP->Fdiff,&blasn,&info);
  ierr = PetscInfo1(tao,"gesv returned %d\n",info); CHKERRQ(ierr);

  cres = minnorm;
  /* Gres = G*F(xkin,1:m)' //  G (nxm)   Fk (m)   */
  BLASgemv_("N",&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->C,&ione,&zero,mfqP->Gres,&ione);

  /*  Hres = G*G'  */
  BLASgemm_("N","T",&blasn,&blasn,&blasm,&one,mfqP->Fdiff, &blasn,mfqP->Fdiff,&blasn,&zero,mfqP->Hres,&blasn);

  valid = PETSC_TRUE;

  ierr = VecSetValues(tao->gradient,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(tao->gradient);
  ierr = VecAssemblyEnd(tao->gradient);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  gnorm *= mfqP->delta;
  ierr = VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution); CHKERRQ(ierr);
  ierr = TaoSolverMonitor(tao, iter, minnorm, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
  mfqP->nHist = mfqP->n+1;
  mfqP->nmodelpoints = mfqP->n+1;

  while (reason == TAO_CONTINUE_ITERATING) {

    iter++;

    /* Solve the subproblem min{Q(s): ||s|| <= delta} */
    ierr = gqtwrap(mfqP,&gnorm,&mdec); CHKERRQ(ierr);
    /* Evaluate the function at the new point */

    for (i=0;i<mfqP->n;i++) {
	mfqP->work[i] = mfqP->Xsubproblem[i]*mfqP->delta + mfqP->xmin[i];
    }
    ierr = VecDuplicate(tao->solution,&mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,1,&mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecSetValues(mfqP->Xhist[mfqP->nHist],mfqP->n,mfqP->indices,mfqP->work,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
    CHKMEMQ;
    ierr = TaoSolverComputeObjective(tao,mfqP->Xhist[mfqP->nHist],&objvalue); CHKERRQ(ierr);
    ierr = VecSetValue(mfqP->Fhist[mfqP->nHist],0,objvalue,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecNorm(mfqP->Fhist[mfqP->nHist],NORM_2,&mfqP->Fres[mfqP->nHist]); CHKERRQ(ierr);
    mfqP->Fres[mfqP->nHist]*=mfqP->Fres[mfqP->nHist];
    rho = (mfqP->Fres[mfqP->minindex] - mfqP->Fres[mfqP->nHist]) / mdec;
    mfqP->nHist++;

    /* Update the center */
    if ((rho >= mfqP->eta1) || (rho > mfqP->eta0 && valid==PETSC_TRUE)) {
	/* Update model to reflect new base point */
	for (i=0;i<mfqP->n;i++) {
	    mfqP->work[i] = (mfqP->work[i] - mfqP->xmin[i])/mfqP->delta;
	}
	for (j=0;j<mfqP->m;j++) {
	    /* C(j) = C(j) + work*G(:,j) + .5*work*H(:,:,j)*work';
	       G(:,j) = G(:,j) + H(:,:,j)*work' */
	    for (k=0;k<mfqP->n;k++) {
		mfqP->work2[k]=0.0;
		for (l=0;l<mfqP->n;l++) {
		    mfqP->work2[k]+=mfqP->H[j + mfqP->m*(k + l*mfqP->n)]*mfqP->work[l];
		}
	    }
	    for (i=0;i<mfqP->n;i++) {
		mfqP->C[j]+=mfqP->work[i]*(mfqP->Fdiff[i + mfqP->n* j] + 0.5*mfqP->work2[i]);
		mfqP->Fdiff[i+mfqP->n*j] +=mfqP-> work2[i];
	    }
	}
	/* Cres += work*Gres + .5*work*Hres*work';
	   Gres += Hres*work'; */

	BLASgemv_("N",&blasn,&blasn,&one,mfqP->Hres,&blasn,mfqP->work,&ione,
		  &zero,mfqP->work2,&ione);
	for (i=0;j<mfqP->n;j++) {
	    cres += mfqP->work[i]*(mfqP->Gres[i]  + 0.5*mfqP->work2[i]);
	    mfqP->Gres[i] += mfqP->work2[i];
	}
	mfqP->minindex = mfqP->nHist-1;
	minnorm = mfqP->Fres[mfqP->minindex];
	/* Change current center */
	ierr = VecGetArray(mfqP->Xhist[mfqP->minindex],&xmint); CHKERRQ(ierr);
	for (i=0;i<mfqP->n;i++) {
	    mfqP->xmin[i] = xmint[i];
	}
	ierr = VecRestoreArray(mfqP->Xhist[mfqP->minindex],&xmint); CHKERRQ(ierr);


    }

    /* Evaluate at a model-improving point if necessary */
    if (valid == PETSC_FALSE) {
	mfqP->q_is_I = 1;
	ierr = affpoints(mfqP,mfqP->xmin,mfqP->c1); CHKERRQ(ierr);
	if (mfqP->nmodelpoints < mfqP->n) {
	  ierr = PetscInfo(tao,"Model not valid -- model-improving");
	  ierr = modelimprove(tao,mfqP,1); CHKERRQ(ierr);
	}
    }
    

    /* Update the trust region radius */
    deltaold = mfqP->delta;
    normxsp = 0;
    for (i=0;i<mfqP->n;i++) {
	normxsp += mfqP->Xsubproblem[i]*mfqP->Xsubproblem[i];
    }
    normxsp = sqrt(normxsp);
    if (rho >= mfqP->eta1 && normxsp > 0.5*mfqP->delta) {
	mfqP->delta = PetscMin(mfqP->delta*mfqP->gamma1,mfqP->deltamax); 
    } else if (valid == PETSC_TRUE) {
	mfqP->delta = PetscMax(mfqP->delta*mfqP->gamma0,mfqP->deltamin);
    }

    /* Compute the next interpolation set */
    mfqP->q_is_I = 1;
    mfqP->nmodelpoints=0;
    ierr = affpoints(mfqP,mfqP->xmin,mfqP->c1); CHKERRQ(ierr);
    if (mfqP->nmodelpoints == mfqP->n) {
      valid = PETSC_TRUE;
    } else {
      valid = PETSC_FALSE;
      ierr = affpoints(mfqP,mfqP->xmin,mfqP->c2); CHKERRQ(ierr);
      if (mfqP->n > mfqP->nmodelpoints) {
	ierr = PetscInfo(tao,"Model not valid -- adding geometry points");
	ierr = modelimprove(tao,mfqP,mfqP->n - mfqP->nmodelpoints); CHKERRQ(ierr);
      }
    }
    for (i=mfqP->nmodelpoints;i>0;i--) {
	mfqP->model_indices[i] = mfqP->model_indices[i-1];
    }
    mfqP->nmodelpoints++;
    mfqP->model_indices[0] = mfqP->minindex;
    ierr = morepoints(mfqP); CHKERRQ(ierr);
    if (mfqP->nmodelpoints - mfqP->n - 1 == 0) {
      reason = TAO_DIVERGED_USER;
      tao->reason = TAO_DIVERGED_USER;
      continue;
    }
    for (i=0;i<mfqP->nmodelpoints;i++) {
	ierr = VecGetArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	for (j=0;j<mfqP->n;j++) {
	    mfqP->Disp[i + mfqP->npmax*j] = (x[j]  - mfqP->xmin[j]) / deltaold;
	}
	ierr = VecRestoreArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	ierr = VecGetArray(mfqP->Fhist[mfqP->model_indices[i]],&f); CHKERRQ(ierr);
	for (j=0;j<mfqP->m;j++) {
	    for (k=0;k<mfqP->n;k++)  {
		mfqP->work[k]=0.0;
		for (l=0;l<mfqP->n;l++) {
		    mfqP->work[k] += mfqP->H[j + mfqP->m*(k + mfqP->n*l)] * mfqP->Disp[i + mfqP->npmax*l];
		}
	    }
	    mfqP->RES[j*mfqP->npmax + i] = -mfqP->C[j] - BLASdot_(&blasn,&mfqP->Fdiff[j*mfqP->n],&ione,&mfqP->Disp[i],&blasnpmax) - 0.5*BLASdot_(&blasn,mfqP->work,&ione,&mfqP->Disp[i],&blasnpmax) + f[j];
	}
	ierr = VecRestoreArray(mfqP->Fhist[mfqP->model_indices[i]],&f); CHKERRQ(ierr);
    }


    /* Update the quadratic model */
    ierr = getquadpounders(mfqP); CHKERRQ(ierr);
    ierr = VecGetArray(mfqP->Fhist[mfqP->minindex],&fmin); CHKERRQ(ierr);
    BLAScopy_(&blasm,fmin,&ione,mfqP->C,&ione);
    /* G = G*(delta/deltaold) + Gdel */
    ratio = mfqP->delta/deltaold;
    iblas = blasm*blasn;
    BLASscal_(&iblas,&ratio,mfqP->Fdiff,&ione);
    BLASaxpy_(&iblas,&one,mfqP->Gdel,&ione,mfqP->Fdiff,&ione);
    /* H = H*(delta/deltaold) + Hdel */
    iblas = blasm*blasn*blasn;
    ratio *= ratio;
    BLASscal_(&iblas,&ratio,mfqP->H,&ione);
    BLASaxpy_(&iblas,&one,mfqP->Hdel,&ione,mfqP->H,&ione);


    /* Get residuals */
    cres = mfqP->Fres[mfqP->minindex];
    /* Gres = G*F(xkin,1:m)' */
    BLASgemv_("N",&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->C,&ione,&zero,mfqP->Gres,&ione);
    /* Hres = sum i=1..m {F(xkin,i)*H(:,:,i)}   + G*G' */
    BLASgemm_("N","T",&blasn,&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->Fdiff,&blasn,
	      &zero,mfqP->Hres,&blasn);

    iblas = mfqP->n*mfqP->n;

    for (j=0;j<mfqP->m;j++) { 
	BLASaxpy_(&iblas,&fmin[j],&mfqP->H[j],&blasm,mfqP->Hres,&ione);
    }
    
    /* Export solution and gradient residual to TAO */
    ierr = VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution); CHKERRQ(ierr);
    ierr = VecSetValues(tao->gradient,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tao->gradient);
    ierr = VecAssemblyEnd(tao->gradient);
    ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
    gnorm *= mfqP->delta;
    /*  final criticality test */
    ierr = TaoSolverMonitor(tao, iter, minnorm, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetUp_POUNDER"
static PetscErrorCode TaoSolverSetUp_POUNDER(TaoSolver tao)
{
    TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
    PetscInt i;
    IS isfloc,isfglob,isxloc,isxglob;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
    if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
    ierr = VecGetSize(tao->solution,&mfqP->n); CHKERRQ(ierr);
    mfqP->c1 = sqrt(mfqP->n);
    mfqP->npmax = (mfqP->n+1)*(mfqP->n+2)/2; /* TODO check if manually set */

    ierr = PetscMalloc((tao->max_funcs+10)*sizeof(Vec),&mfqP->Xhist); CHKERRQ(ierr);
    ierr = PetscMalloc((tao->max_funcs+10)*sizeof(Vec),&mfqP->Fhist); CHKERRQ(ierr);
    for (i=0;i<mfqP->n +1;i++) {
	ierr = VecDuplicate(tao->solution,&mfqP->Xhist[i]); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF,1,&mfqP->Fhist[i]); CHKERRQ(ierr);
    }
    ierr = VecDuplicate(tao->solution,&mfqP->workxvec); CHKERRQ(ierr);
    mfqP->nHist = 0;

    ierr = PetscMalloc((tao->max_funcs+10)*sizeof(PetscReal),&mfqP->Fres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*mfqP->m*sizeof(PetscReal),&mfqP->RES); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work2); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work3); CHKERRQ(ierr);
    ierr = PetscMalloc(PetscMax(mfqP->m,mfqP->n+1)*sizeof(PetscReal),&mfqP->mwork); CHKERRQ(ierr);
    ierr = PetscMalloc((mfqP->npmax - mfqP->n - 1)*sizeof(PetscReal),&mfqP->omega); CHKERRQ(ierr);
    ierr = PetscMalloc((mfqP->n * (mfqP->n+1) / 2)*sizeof(PetscReal),&mfqP->beta); CHKERRQ(ierr);
    ierr = PetscMalloc((mfqP->n + 1) *sizeof(PetscReal),&mfqP->alpha); CHKERRQ(ierr);

    ierr = PetscMalloc(mfqP->n*mfqP->n*mfqP->m*sizeof(PetscReal),&mfqP->H); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*mfqP->npmax*sizeof(PetscReal),&mfqP->Q); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*mfqP->npmax*sizeof(PetscReal),&mfqP->Q_tmp); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax)*sizeof(PetscReal),&mfqP->L); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax)*sizeof(PetscReal),&mfqP->L_tmp); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax)*sizeof(PetscReal),&mfqP->L_save); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax)*sizeof(PetscReal),&mfqP->N); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax * (mfqP->n+1) * sizeof(PetscReal),&mfqP->M); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax * (mfqP->npmax - mfqP->n - 1) * sizeof(PetscReal), &mfqP->Z); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscReal),&mfqP->tau); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscReal),&mfqP->tau_tmp); CHKERRQ(ierr);
    ierr = PetscMalloc(5*mfqP->npmax*sizeof(PetscReal),&mfqP->npmaxwork); CHKERRQ(ierr);
    ierr = PetscMalloc(5*mfqP->npmax*sizeof(PetscBLASInt),&mfqP->npmaxiwork); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->xmin); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*sizeof(PetscReal),&mfqP->C); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*mfqP->n*sizeof(PetscReal),&mfqP->Fdiff); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*mfqP->n*sizeof(PetscReal),&mfqP->Disp); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->Gres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Hres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Gpoints); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscInt),&mfqP->model_indices); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->Xsubproblem); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*mfqP->n*sizeof(PetscReal),&mfqP->Gdel); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*mfqP->m*sizeof(PetscReal), &mfqP->Hdel); CHKERRQ(ierr);
    ierr = PetscMalloc(PetscMax(mfqP->m,mfqP->n)*sizeof(PetscInt),&mfqP->indices); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscBLASInt),&mfqP->iwork); CHKERRQ(ierr);
    for (i=0;i<PetscMax(mfqP->m,mfqP->n);i++) {
	mfqP->indices[i] = i;
    }
    ierr = MPI_Comm_size(((PetscObject)tao)->comm,&mfqP->mpisize); CHKERRQ(ierr);
    if (mfqP->mpisize > 1) {
      VecCreateSeq(PETSC_COMM_SELF,mfqP->n,&mfqP->localx); CHKERRQ(ierr);
      VecCreateSeq(PETSC_COMM_SELF,mfqP->n,&mfqP->localxmin); CHKERRQ(ierr);
      VecCreateSeq(PETSC_COMM_SELF,mfqP->m,&mfqP->localf); CHKERRQ(ierr);
      VecCreateSeq(PETSC_COMM_SELF,mfqP->m,&mfqP->localfmin); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,mfqP->n,0,1,&isxloc); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,mfqP->n,0,1,&isxglob); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,mfqP->m,0,1,&isfloc); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,mfqP->m,0,1,&isfglob); CHKERRQ(ierr);
      

      ierr = VecScatterCreate(tao->solution,isxglob,mfqP->localx,isxloc,&mfqP->scatterx); CHKERRQ(ierr);
      ierr = VecScatterCreate(mfqP->Fhist[0],isfglob,mfqP->localf,isfloc,&mfqP->scatterf); CHKERRQ(ierr);

      ierr = ISDestroy(isxloc); CHKERRQ(ierr);
      ierr = ISDestroy(isxglob); CHKERRQ(ierr);
      ierr = ISDestroy(isfloc); CHKERRQ(ierr);
      ierr = ISDestroy(isfglob); CHKERRQ(ierr);

    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_POUNDER"
static PetscErrorCode TaoSolverDestroy_POUNDER(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscInt i;
  PetscErrorCode ierr;
  

  PetscFunctionBegin;
  ierr = PetscFree(mfqP->Fres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->RES); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work2); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work3); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->mwork); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->omega); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->beta); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->alpha); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->H); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Q); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Q_tmp); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->L); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->L_tmp); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->L_save); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->N); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->M); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Z); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->tau); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->tau_tmp); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->npmaxwork); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->npmaxiwork); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->xmin); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->C); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Fdiff); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Disp); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Gres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Hres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Gpoints); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->model_indices); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Xsubproblem); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Gdel); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Hdel); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->indices); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->iwork); CHKERRQ(ierr);
  
  for (i=0;i<mfqP->nHist;i++) {
      ierr = VecDestroy(mfqP->Xhist[i]); CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->Fhist[i]); CHKERRQ(ierr);
  }
  if (mfqP->workxvec) {
    ierr = VecDestroy(mfqP->workxvec); CHKERRQ(ierr);
  }
  ierr = PetscFree(mfqP->Xhist); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Fhist); CHKERRQ(ierr);

  if (mfqP->mpisize > 1) {
      ierr = VecDestroy(mfqP->localx);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localxmin);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localf);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localfmin);  CHKERRQ(ierr);
  }



  ierr = PetscFree(tao->data); CHKERRQ(ierr);
  tao->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_POUNDER"
static PetscErrorCode TaoSolverSetFromOptions_POUNDER(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("POUNDER method for derivative-free optimization"); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pounders_delta","initial delta","",mfqP->delta,&mfqP->delta,0); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverView_POUNDER"
static PetscErrorCode TaoSolverView_POUNDER(TaoSolver tao, PetscViewer viewer)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscBool isascii;
  PetscErrorCode ierr;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  points in model: %d\n",mfqP->nmodelpoints); 
  } else {
    SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO POUNDER",((PetscObject)viewer)->type_name);
  }    
  PetscFunctionReturn(0);

}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoSolverCreate_POUNDER"
PetscErrorCode TaoSolverCreate_POUNDER(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  tao->ops->setup = TaoSolverSetUp_POUNDER;
  tao->ops->solve = TaoSolverSolve_POUNDER;
  tao->ops->view = TaoSolverView_POUNDER;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_POUNDER;
  tao->ops->destroy = TaoSolverDestroy_POUNDER;


  ierr = PetscNewLog(tao, TAO_POUNDERS, &mfqP); CHKERRQ(ierr);
  tao->data = (void*)mfqP;
  tao->max_its = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;
  mfqP->delta = 0.1;
  mfqP->deltamax=1e3;
  mfqP->deltamin=1e-6;
  mfqP->c2 = 100.0;
  mfqP->theta1=1e-5;
  mfqP->theta2=1e-4;
  mfqP->gamma0=0.5;
  mfqP->gamma1=2.0;
  mfqP->eta0 = 0.0;
  mfqP->eta1 = 0.1;
  mfqP->gqt_rtol = 0.001;
  mfqP->gqt_maxits = 50;
  mfqP->workxvec = 0;
  mfqP->m = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

