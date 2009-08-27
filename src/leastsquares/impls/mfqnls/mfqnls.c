#include "mfqnls.h"

#undef __FUNCT__
#define __FUNCT__ "gqt"
static PetscErrorCode gqt(TAO_MFQNLS *mfqP,PetscReal gnorm, PetscReal *qmin) {
    PetscReal one=1.0,atol=1.0e-10;
    int info,its;
    PetscFunctionBegin;
//      subroutine dgqt(n,a,lda,b,delta,rtol,atol,itmax,par,f,x,info,iter,z,wa1,wa2)
    dgqt_(&mfqP->n,mfqP->Hres,mfqP->n,mfqP->Gres,&one,&mfqP->gqt_rtol,&atol,
	  &mfqP->gqt_maxits,&gnorm,qmin,mfqP->Xsubproblem,&info,&its,mfqP->work,mfqP->work2,
	  mfqP->work3);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "phi2eval"
static PetscErrorCode phi2eval(TAO_MFQNLS *mfqP, PetscReal *x) {
/* Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2] */
    int i,j,k;
    j=0;
    for (i=0;i<mfqP->n;i++) {
	mfqP->phi[j] = 0.5 * x[i]*x[i];
	j++;
	for (k=i+1;k<n;k++) {
	    mfqP->phi[j]  = x[i]*x[k]/sqrt(2.0);
	    j++;
	}

    }
}

#undef __FUNCT__
#define __FUNCT__ "getquadmfqnls"
static PetscErrorCode getquadmfqnls(TAO_MFQNLS *mfqP) {
/* Computes the parameters of the quadratic Q(x) = c + g'*x + 0.5*x*G*x'
   that satisfies the interpolation conditions Q(X[:,j]) = f(j)
   for j=1,...,m and with a Hessian matrix of least Frobenius norm */
    PetscFunctionBegin;
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "morepoints"
static PetscErrorCode morepoints(TAO_MFQNLS *mfqP,PetscInt index, PetscInt *np,
				 PetscInt *indices, PetscReal c) {
    PetscFunctionBegin;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "affpoints"
static PetscErrorCode affpoints(TAO_MFQNLS *mfqP, PetscReal *xmin, PetscInt nf, 
				PetscReal c, PetscInt *np, PetscInt *indices, 
				PetscTruth *flag) {
    PetscInt i,j;
    PetscBLASInt blasm=mfqP->m,blask,blasn=mfqP->n,ione=1,info;
    PetscReal proj,normd;
    PetscReal *x;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    *np=0;
    if (flag != PETSC_NULL)  *flag = PETSC_FALSE;
    for (i=nf-1;i>=0;i--) {
	ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	for (j=0;j<mfqP->n;j++) {
	    mfqP->work[j] = (x[j] - xmin[j])/mfqP->delta;
	}
	ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	BLAScopy_(&blasn,mfqP->work,&ione,mfqP->work2,&ione);
	normd = BLASnrm2_(&blasn,mfqP->work,&ione);
	if (normd <= c*c) {
	    if (!mfqP->q_is_I) {
		// project D onto null
		blask=mfqP->n-(*np);
		LAPACKormqr_("R","N",&ione,&blasn,&blask,mfqP->Q,&blasn,mfqP->tau,
			     mfqP->work2,&ione,mfqP->mwork,&blasm,&info);
		if (info < 0) {
		    SETERRQ1(1,"ormqr returned value %d\n",info);
		}
	    }
	    proj = BLASnrm2_(&blasn,mfqP->work2,&ione);
	    if (proj >= mfqP->theta1) { /* add this index to model */
		(*np)++;
		indices[*np]=i;
		LAPACKgeqrf_(&blasn,&blasn,mfqP->Q,&blasn,mfqP->tau,mfqP->mwork,
			     &blasm,&info);
		mfqP->q_is_I = 0;
		if (info < 0) {
		    SETERRQ1(1,"geqrf returned value %d\n",info);
		}
		    
	    }
	    if (*np == mfqP->n)  {
		if (flag != PETSC_NULL) *flag = PETSC_TRUE;
		break;
	    }
	}		
    }
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_MFQNLS"
static PetscErrorCode TaoSolverSolve_MFQNLS(TaoSolver tao)
{
  TAO_MFQNLS *mfqP = (TAO_MFQNLS *)tao->data;

  PetscInt i,ii,j,iter=0;
  PetscReal step=1.0;
  TaoSolverConvergedReason reason = TAO_CONTINUE_ITERATING;

  PetscInt low,high,minindex;
  PetscReal minnorm;
  PetscInt *indices;
  PetscReal *x,*f,*fmin,*xmint;
  PetscReal cres,deltaold;
  PetscReal gnorm,temp;
  PetscInt index=0;
  PetscBLASInt info,ione=1,iblas;
  PetscTruth valid;
  PetscInt np;
  PetscReal mdec, rho, normxsp;
  PetscReal one=1.0,zero=0.0,ratio;
  PetscBLASInt blasm,blasn;
  PetscErrorCode ierr;
  
  
  /* n = # of parameters 
     m = dimension (components) of function  */
  
  PetscFunctionBegin;
  blasm = mfqP->m; blasn=mfqP->n;
  for (i=0;i<mfqP->n*mfqP->n*mfqP->m;i++) mfqP->H[i]=0;

  ierr = VecCopy(tao->solution,mfqP->Xhist[0]); CHKERRQ(ierr);

  ierr = TaoSolverComputeSeparableObjective(tao,tao->solution,mfqP->Fhist[0]); CHKERRQ(ierr);
  ierr = VecNorm(mfqP->Fhist[0],NORM_2,&mfqP->Fres[0]); CHKERRQb(ierr);
  mfqP->Fres[0]*=mfqP->Fres[0];
  minindex = 0;
  minnorm = mfqP->Fres[0];
  ierr = VecGetOwnershipRange(mfqP->Xhist[0],&low,&high); CHKERRQ(ierr);
  for (i=1;i<mfqP->n+1;i++) {
      ierr = VecCopy(tao->solution,mfqP->Xhist[i]); CHKERRQ(ierr);
      if (i-1 >= low && i-1 < high) {
	  ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	  x[i-1-low] += mfqP->delta;
	  ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
      }
      ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[i],mfqP->Fhist[i]); CHKERRQ(ierr);
      ierr = VecNorm(mfqP->Fhist[i],NORM_2,&mfqP->Fres[i]); CHKERRQ(ierr);
      mfqP->Fres[i]*=mfqP->Fres[i];
      if (mfqP->Fres[i] < minnorm) {
	  minindex = i;
	  minnorm = mfqP->Fres[i];
      }
  }

  ierr = VecCopy(mfqP->Xhist[minindex],tao->solution); CHKERRQ(ierr);
  ierr = VecCopy(mfqP->Fhist[minindex],tao->sep_objective); CHKERRQ(ierr);
  /* Gather mpi vecs to one big local vec */

  

  /* Begin serial code */

  /* Disp[i] = Xi-xmin, i=1,..,minindex-1,minindex+1,..,n */
  /* Fdiff[i] = (Fi-Fmin)', i=1,..,minindex-1,minindex+1,..,n */
  /* (Column oriented for blas calls) */
  ii=0;

  if (mfqP->mpisize == 1) {
      ierr = VecGetArray(mfqP->Xhist[minindex],&xmint); CHKERRQ(ierr);
      for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i]; 
      ierr = VecRestoreArray(mfqP->Xhist[minindex],&xmint); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->Fhist[minindex],&fmin); CHKERRQ(ierr);
      for (i=0;i<mfqP->n+1;i++) {
	  if (i == minindex) continue;

	  ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	  for (j=0;j<mfqP->n;j++) {
	      mfqP->Disp[ii+mfqP->n*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
	  }
	  ierr = VecRestoreArray(mfqP->Xhist[i],&f); CHKERRQ(ierr);

	  ierr = VecGetArray(mfqP->Fhist[i],&f); CHKERRQ(ierr);
	  for (j=0;j<mfqP->m;j++) {
	      mfqP->Fdiff[ii+mfqP->m*j] = f[j] - fmin[j];
	  }
	  ierr = VecRestoreArray(mfqP->Fhist[i],&f); CHKERRQ(ierr);
	  mfqP->model_indices[ii++] = i;

      }
      for (j=0;j<mfqP->m;j++) {
	  mfqP->C[j] = fmin[j];
      }
      ierr = VecRestoreArray(mfqP->Fhist[minindex],&fmin); CHKERRQ(ierr);

  } else {
      ierr = VecScatterBegin(mfqP->scatterx,mfqP->Xhist[minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecScatterEnd(mfqP->scatterx,mfqP->Xhist[minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->localxmin,&xmint); CHKERRQ(ierr);
      for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i];
      ierr = VecRestoreArray(mfqP->localxmin,&mfqP->xmin); CHKERRQ(ierr);



      ierr = VecScatterBegin(mfqP->scatterf,mfqP->Fhist[minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecScatterEnd(mfqP->scatterf,mfqP->Fhist[minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGetArray(mfqP->localfmin,&fmin); CHKERRQ(ierr);
      for (i=0;i<mfqP->n+1;i++) {
	  if (i == minindex) continue;
				 
	  mfqP->model_indices[ii++] = i;
	  ierr = VecScatterBegin(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecScatterBegin(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecGetArray(mfqP->localx,&x); CHKERRQ(ierr);
	  for (j=0;j<mfqP->n;j++) {
	      mfqP->Disp[i+mfqP->n*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
	  }
	  ierr = VecRestoreArray(mfqP->localx,&x); CHKERRQ(ierr);

	  
	  ierr = VecScatterBegin(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,
				 INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	  ierr = VecScatterBegin(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,
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
  
  //G = D(ModelIn,:) \ (F(ModelIn,1:m)-repmat(F(xkin,1:m),n,1));
  // D (nxn) Fdiff (nxm)  => G (nxm)
  LAPACKgesv_(&blasn,&blasm,mfqP->Disp,&blasn,mfqP->iwork,mfqP->Fdiff,&blasn,&info);
  ierr = PetscInfo1(tao,"gesv returned %d\n",info); CHKERRQ(ierr);

  cres = minnorm;
  //Gres = G*F(xkin,1:m)' //  G (nxm)   Fk (m)  
  BLASgemv_("N",&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->C,&ione,&zero,mfqP->Gres,&ione);

  //  Hres = G*G' 
  BLASgemm_("N","T",&blasn,&blasn,&blasm,&one,mfqP->Fdiff, &blasn,mfqP->Fdiff,&blasn,&zero,mfqP->Hres,&blasn);

  valid = PETSC_TRUE;

  ierr = VecSetValues(tao->gradient,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(tao->gradient);
  ierr = VecAssemblyEnd(tao->gradient);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  gnorm *= mfqP->delta;
  ierr = TaoSolverMonitor(tao, iter, minnorm, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
  index = mfqP->n;
  while (reason == TAO_CONTINUE_ITERATING) {

    iter++;

    /* Solve the subproblem min{Q(s): ||s|| <= delta} */
    ierr = gqt(mfqP,gnorm,&mdec); CHKERRQ(ierr);
    
    /* Evaluate the function at the new point */
    for (i=0;i<mfqP->n;i++) {
	mfqP->Xsubproblem[i] *= mfqP->delta;
	mfqP->Xsubproblem[i] += mfqP->xmin[i];
    }
    index++;
    ierr = VecSetValues(mfqP->Xhist[index],mfqP->n,mfqP->indices,mfqP->Xsubproblem,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mfqP->Xhist[index]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mfqP->Xhist[index]); CHKERRQ(ierr);
    ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[index],mfqP->Fhist[index]); CHKERRQ(ierr);
    ierr = VecNorm(mfqP->Fhist[index],NORM_2,&mfqP->Fres[index]); CHKERRQ(ierr);
    mfqP->Fres[index]*=mfqP->Fres[index];
    rho = (mfqP->Fres[minindex] - mfqP->Fres[index]) / mdec;

    /* Update the center */
    if ((rho >= mfqP->eta1) || (rho > mfqP->eta0 && valid==PETSC_TRUE)) {
	/* Update model to reflect new base point */
	for (i=0;i<mfqP->n;i++) {
	    mfqP->work[i] = (x[i] - mfqP->xmin[i])/mfqP->delta;
	}
	for (j=0;j<mfqP->m;j++) {
	    // C(j) = C(j) + work*G(:,j) + .5*work*H(:,:,j)*work';
	    // G(:,j) = G(:,j) + H(:,:,j)*work'
	    BLASgemv_("N",&blasn,&blasn,&one,&mfqP->H[j*mfqP->n*mfqP->n],&blasn,mfqP->work,&ione,
		      &zero,mfqP->work2,&ione);
	    for (i=0;i<mfqP->n;i++) {
		mfqP->C[j]+=mfqP->work[i]*(mfqP->Fdiff[i + mfqP->n* j] + 0.5*mfqP->work2[i]);
		mfqP->Fdiff[i+mfqP->n*j] +=mfqP-> work2[i];
	    }
	}
	//Cres += work*Gres + .5*work*Hres*work';
	//Gres += Hres*work';

	BLASgemv_("N",&blasn,&blasn,&one,mfqP->Hres,&blasn,mfqP->work,&ione,
		  &zero,mfqP->work2,&ione);
	for (i=0;j<mfqP->n;j++) {
	    cres += mfqP->work[i]*(mfqP->Gres[i]  + 0.5*mfqP->work2[i]);
	    mfqP->Gres[i] += mfqP->work2[i];
	}
	minindex = index;
	minnorm = mfqP->Fres[minindex];
	ierr = VecGetArray(mfqP->Xhist[minindex],&xmint); CHKERRQ(ierr);
	for (i=0;i<mfqP->n;i++) {
	    mfqP->xmin[i] = xmint[i];
	ierr = VecRestoreArray(mfqP->Xhist[minindex],&xmint); CHKERRQ(ierr);
    }

    /* Evaluate at a model-improving point if necessary */
    if (valid == PETSC_FALSE) {
	mfqP->q_is_I = 1;
	ierr = affpoints(mfqP,mfqP->xmin,minindex,mfqP->c1,&np,mfqP->interp_indices,&valid); CHKERRQ(ierr);
	if (valid == PETSC_FALSE) {
	    SETERRQ(1,"Model not valid -- model-improving not implemented yet");
	}
    }

    
    
    /* Update the trust region radius */
    deltaold = mfqP->delta;
    normxsp = 0;
    for (i=0;i<mfqP->n;i++) {
	normxsp += mfqP->Xsubproblem[i];
    }
    normxsp = sqrt(normxsp);
    if (rho >= mfqP->eta1 && normxsp > 0.5*mfqP->delta) {
	mfqP->delta = PetscMin(mfqP->delta*mfqP->gamma1,mfqP->deltamax); 
    } else {
	mfqP->delta = PetscMax(mfqP->delta*mfqP->gamma0,mfqP->deltamin);
    }

    /* Compute the next interpolation set */
    mfqP->q_is_I = 1;
    ierr = affpoints(mfqP,mfqP->xmin,minindex,mfqP->c1,&np,mfqP->interp_indices,&valid); CHKERRQ(ierr);
    if (valid == PETSC_FALSE) {
	ierr = affpoints(mfqP,mfqP->xmin,minindex,mfqP->c2,&np,mfqP->interp_indices,PETSC_NULL); CHKERRQ(ierr);
	for (i=0;i<mfqP->n - np; i++) {
	    temp=0.0;
	    for (j=0;j<mfqP->n;j++) {
		temp += mfqP->Gpoints[i+mfqP->n*j]*mfqP->Gres[j];
	    }
	    if (temp > 0) {
		for (j=0;j<mfqP->n;j++) {
		    mfqP->Gpoints[i+mfqP->n*j]*=-1;
		}
	    }
	    index++;
	    mfqP->interp_indices[np+i] = index;
	    for (j=0;j<mfqP->n;j++) {
		mfqP->Xsubproblem[j] = mfqP->xmin[j] + mfqP->delta*mfqP->Gpoints[i+mfqP->n*j];
	    }
	    ierr = VecSetValues(mfqP->Xhist[index],mfqP->n,mfqP->indices,mfqP->Xsubproblem,INSERT_VALUES); CHKERRQ(ierr);
	    ierr = VecAssemblyBegin(mfqP->Xhist[index]); CHKERRQ(ierr);
	    ierr = VecAssemblyEnd(mfqP->Xhist[index]); CHKERRQ(ierr);
	    ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[index],mfqP->Fhist[index]); CHKERRQ(ierr);
	    ierr = VecNorm(mfqP->Fhist[index],NORM_2,&mfqP->Fres[index]); CHKERRQ(ierr);
	    mfqP->Fres[index]*=mfqP->Fres[index];
	    ierr = PetscInfo1(tao,"value of Geometry point: %g\n",mfqP->Fres[index]); CHKERRQ(ierr);
	}
    }
    for (i=0;i<np;i++) {
	mfqP->model_indices[i+1] = mfqP->interp_indices[i];
    }
    mfqP->model_indices[0] = minindex;

    ierr = morepoints(mfqP,minindex,&np,mfqP->model_indices,mfqP->c2); CHKERRQ(ierr);

    for (i=0;i<np;i++) {
	ierr = VecGetArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	for (j=0;j<mfqP->n;j++) {
	    mfqP->Disp[i + mfqP->n*j] = (x[j]  - mfqP->xmin[j]) / deltaold;
	}
	ierr = VecRestoreArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	ierr = VecGetArray(mfqP->Fhist[mfqP->model_indices[i]],&f); CHKERRQ(ierr);
	// RES(i,j) = -C(j) - D(i,:)*(G(:,j) + .5*H(:,:,j)*D(i,:)') + F(model[i],j)
	for (j=0;j<mfqP->m;j++) {
	    BLASgemv_("N",&blasn,&blasn,&one,&mfqP->H[j*mfqP->n*mfqP->n],&blasn,&mfqP->Disp[i*mfqP->n],&ione,&zero,mfqP->work,&ione); 
	    mfqP->RES[i + mfqP->n*j] = -mfqP->C[j] - BLASdot_(&blasn,&mfqP->Fdiff[i*mfqP->n],&ione,&mfqP->Disp[i*mfqP->n],&ione) - 0.5*BLASdot_(&blasn,mfqP->work,&ione,&mfqP->Disp[i*mfqP->n],&ione) + f[j];
	}
	ierr = VecRestoreArray(mfqP->Fhist[mfqP->model_indices[i]],&f); CHKERRQ(ierr);
    }

    /* Update the quadratic model */
    ierr = getquadmfqnls(mfqP); CHKERRQ(ierr);
    ierr = VecGetArray(mfqP->Fhist[minindex],&fmin); CHKERRQ(ierr);
    BLAScopy_(&blasm,fmin,&ione,mfqP->C,&ione);
    // G = G*(delta/deltaold) + Gdel
    ratio = mfqP->delta/deltaold;
    iblas = blasm*blasn;
    BLASscal_(&iblas,&ratio,mfqP->Fdiff,&ione);
    BLASaxpy_(&iblas,&one,mfqP->Gdel,&ione,mfqP->Fdiff,&ione);
    // H = H*(delta/deltaold) + Hdel
    iblas = blasm*blasm*blasn;
    ratio *= ratio;
    BLASscal_(&iblas,&ratio,mfqP->H,&ione);
    BLASaxpy_(&iblas,&one,mfqP->Hdel,&ione,mfqP->H,&ione);
    
    /* Get residuals */
    cres = mfqP->Fres[minindex];
    //Gres = G*F(xkin,1:m)'
    BLASgemv_("N",&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->C,&ione,&zero,mfqP->Gres,&ione);
    // Hres = sum i=1..m {F(xkin,i)*H(:,:,i)}   + G*G'
    BLASgemm_("N","T",&blasn,&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->Fdiff,&blasn,
	      &zero,mfqP->Hres,&blasn);
    iblas = mfqP->n*mfqP->n;
    for (j=0;j<mfqP->m;j++) { //TODO rewrite as gemv
	BLASaxpy_(&iblas,&fmin[j],&mfqP->H[j*mfqP->n*mfqP->n],&ione,mfqP->Hres,&ione);
    }

    /* Export gradient residual to TAO */
    ierr = VecSetValues(tao->gradient,mfqP->n,indices,mfqP->Gres,INSERT_VALUES); CHKERRQ(ierr);
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
#define __FUNCT__ "TaoSetUp_MFQNLS"
static PetscErrorCode TaoSolverSetUp_MFQNLS(TaoSolver tao)
{
    TAO_MFQNLS *mfqP = (TAO_MFQNLS*)tao->data;
    int i;
    IS isfloc,isfglob,isxloc,isxglob;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
    if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
    ierr = VecGetSize(tao->solution,&mfqP->n); CHKERRQ(ierr);
    ierr = VecGetSize(tao->sep_objective,&mfqP->m); CHKERRQ(ierr);
    mfqP->c1 = sqrt(mfqP->n);
    mfqP->npmax = 2*mfqP->n+1; // TODO check if manually set


    ierr = VecDuplicateVecs(tao->solution,mfqP->npmax,&mfqP->Xhist); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(tao->sep_objective,mfqP->npmax,&mfqP->Fhist); CHKERRQ(ierr);

    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscReal),&mfqP->Fres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*mfqP->m*sizeof(PetscReal),&mfqP->RES); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work2); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->work3); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*sizeof(PetscReal),&mfqP->mwork); CHKERRQ(ierr);

    ierr = PetscMalloc(mfqP->n*mfqP->n*mfqP->m*sizeof(PetscReal),&mfqP->H); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Q); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->tau); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*sizeof(PetscReal),&mfqP->C); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*mfqP->n*sizeof(PetscReal),&mfqP->Fdiff); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Disp); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->Gres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Hres); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*sizeof(PetscReal),&mfqP->Gpoints); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscInt),&mfqP->model_indices); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->npmax*sizeof(PetscInt),&mfqP->interp_indices); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->Xsubproblem); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscReal),&mfqP->Gdel); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*mfqP->n*mfqP->m*sizeof(PetscReal), &mfqP->Hdel); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*(mfqP->n+1)/2*sizeof(PetscReal), &mfqP->phi); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->m*sizeof(PetscInt),&mfqP->indices); CHKERRQ(ierr);
    ierr = PetscMalloc(mfqP->n*sizeof(PetscInt),&mfqP->iwork); CHKERRQ(ierr);
    for (i=0;i<mfqP->m;i++) {
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
      ierr = VecScatterCreate(tao->sep_objective,isfglob,mfqP->localf,isfloc,&mfqP->scatterf); CHKERRQ(ierr);

      ierr = ISDestroy(isxloc); CHKERRQ(ierr);
      ierr = ISDestroy(isxglob); CHKERRQ(ierr);
      ierr = ISDestroy(isfloc); CHKERRQ(ierr);
      ierr = ISDestroy(isfglob); CHKERRQ(ierr);
      
  }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_MFQNLS"
static PetscErrorCode TaoSolverDestroy_MFQNLS(TaoSolver tao)
{
  TAO_MFQNLS *mfqP = (TAO_MFQNLS*)tao->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscFree(mfqP->Fres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->RES); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work2); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->work3); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->mwork); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Q); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->tau); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->H); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->C); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Fdiff); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Disp); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Gres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Hres); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Gpoints); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Xsubproblem); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->model_indices); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->interp_indices); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->indices); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->iwork); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->phi); CHKERRQ(ierr);

  ierr = VecDestroyVecs(mfqP->Xhist,mfqP->npmax); CHKERRQ(ierr);
  ierr = VecDestroyVecs(mfqP->Fhist,mfqP->npmax); CHKERRQ(ierr);

  if (mfqP->mpisize > 1) {
      ierr = VecDestroy(mfqP->localx);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localxmin);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localf);  CHKERRQ(ierr);
      ierr = VecDestroy(mfqP->localfmin);  CHKERRQ(ierr);
  }




  tao->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_MFQNLS"
static PetscErrorCode TaoSolverSetFromOptions_MFQNLS(TaoSolver tao)
{
  TAO_MFQNLS *mfqP = (TAO_MFQNLS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("MFQNLS method for least-squares optimization"); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mfqnls_delta","initial delta","",mfqP->delta,&mfqP->delta,0); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverView_MFQNLS"
static PetscErrorCode TaoSolverView_MFQNLS(TaoSolver tao, PetscViewer viewer)
{
  return 0;
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoCreate_MFQNLS"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverCreate_MFQNLS(TaoSolver tao)
{
  TAO_MFQNLS *mfqP = (TAO_MFQNLS*)tao->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  tao->ops->setup = TaoSolverSetUp_MFQNLS;
  tao->ops->solve = TaoSolverSolve_MFQNLS;
  tao->ops->view = TaoSolverView_MFQNLS;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_MFQNLS;
  tao->ops->destroy = TaoSolverDestroy_MFQNLS;


  ierr = PetscNewLog(tao, TAO_MFQNLS, &mfqP); CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}
EXTERN_C_END


