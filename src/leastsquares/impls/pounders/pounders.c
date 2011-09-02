#include "pounders.h"

#undef __FUNCT__
#define __FUNCT__ "gqtwrap"
PetscErrorCode gqtwrap(TAO_POUNDERS *mfqP,PetscReal *gnorm, PetscReal *qmin) {
    PetscReal atol=1.0e-10;
    PetscInt info,its;
    PetscFunctionBegin;

    gqt(mfqP->n,mfqP->Hres,mfqP->n,mfqP->Gres,1.0,mfqP->gqt_rtol,atol,
	mfqP->gqt_maxits,gnorm,qmin,mfqP->Xsubproblem,&info,&its,
	mfqP->work,mfqP->work2, mfqP->work3);
    /*dgqt_(&mfqP->n,mfqP->Hres,&mfqP->n,mfqP->Gres,&one,&mfqP->gqt_rtol,&atol,
	&mfqP->gqt_maxits,gnorm,qmin,mfqP->Xsubproblem,&info,&its,
	mfqP->work,mfqP->work2, mfqP->work3);*/
    *qmin *= -1;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "phi2eval"
PetscErrorCode phi2eval(PetscReal *x, PetscInt n, PetscReal *phi) {
/* Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2] */
    PetscInt i,j,k;
    PetscReal sqrt2 = sqrt(2);
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
}

#undef __FUNCT__
#define __FUNCT__ "getquadpounders"
PetscErrorCode getquadpounders(TAO_POUNDERS *mfqP) {
/* Computes the parameters of the quadratic Q(x) = c + g'*x + 0.5*x*G*x'
   that satisfies the interpolation conditions Q(X[:,j]) = f(j)
   for j=1,...,m and with a Hessian matrix of least Frobenius norm */

    /* NB --we are ignoring c */
    PetscInt i,j,k,num,np = mfqP->nmodelpoints;
    PetscReal one = 1.0,zero=0.0,negone=-1.0;
    PetscBLASInt blasnpmax = mfqP->npmax;
    PetscBLASInt blasnplus1 = mfqP->n+1;
    PetscBLASInt blasnp = np;
    PetscBLASInt blasint = mfqP->n*(mfqP->n+1) / 2;
    PetscBLASInt blasint2 = np - mfqP->n-1;
    PetscBLASInt blasinfo,ione=1;
    PetscReal sqrt2 = sqrt(2);
	
    PetscFunctionBegin;

    for (i=0;i<mfqP->n*mfqP->m;i++) {
	mfqP->Gdel[i] = 0;
    }
    for (i=0;i<mfqP->n*mfqP->n*mfqP->m;i++) {
	mfqP->Hdel[i] = 0;
    }

    /* Let Ltmp = (L'*L) */
    BLASgemm_("T","N",&blasint2,&blasint2,&blasint,&one,&mfqP->L[(mfqP->n+1)*blasint],&blasint,&mfqP->L[(mfqP->n+1)*blasint],&blasint,&zero,mfqP->L_tmp,&blasint);
    
    /* factor Ltmp */
    LAPACKpotrf_("L",&blasint2,mfqP->L_tmp,&blasint,&blasinfo);
    if (blasinfo != 0) {
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine potrf returned with value %d\n",blasinfo);
    }

    /* factor M */
    LAPACKgetrf_(&blasnplus1,&blasnpmax,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&blasinfo);
    if (blasinfo != 0) {
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine getrf returned with value %d\n",blasinfo);
    }
    
    for (k=0;k<mfqP->m;k++) {
	/* Solve L'*L*Omega = Z' * RESk*/
	BLASgemv_("T",&blasnp,&blasint2,&one,mfqP->Z,&blasnpmax,&mfqP->RES[mfqP->npmax*k],&ione,&zero,mfqP->omega,&ione);

	LAPACKpotrs_("L",&blasint2,&ione,mfqP->L_tmp,&blasint,mfqP->omega,&blasint2,&blasinfo);
	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine potrs returned with value %d\n",blasinfo);
	}
	
	
	/* Beta = L*Omega */
	BLASgemv_("N",&blasint,&blasint2,&one,&mfqP->L[(mfqP->n+1)*blasint],&blasint,mfqP->omega,&ione,&zero,mfqP->beta,&ione);
	
	/* solve M'*Alpha = RESk - N'*Beta */
	BLASgemv_("T",&blasint,&blasnp,&negone,mfqP->N,&blasint,mfqP->beta,&ione,&one,&mfqP->RES[mfqP->npmax*k],&ione);
	LAPACKgetrs_("T",&blasnplus1,&ione,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&mfqP->RES[mfqP->npmax*k],&blasnplus1,&blasinfo);
	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine getrs returned with value %d\n",blasinfo);
	}

	/* Gdel(:,k) = Alpha(2:n+1) */
	for (i=0;i<mfqP->n;i++) {
	    mfqP->Gdel[i + mfqP->n*k] = mfqP->RES[mfqP->npmax*k + i+1];
	}
	
	/* Set Hdels */
	num=0;
	for (i=0;i<mfqP->n;i++) {
	    /* H[i,i,k] = Beta(num) */
	    mfqP->Hdel[(i*mfqP->n + i)*mfqP->m + k] = mfqP->beta[num];
	    num++;
	    for (j=i+1;j<mfqP->n;j++) {
		/* H[i,j,k] = H[j,i,k] = Beta(num)/sqrt(2) */
		mfqP->Hdel[(j*mfqP->n + i)*mfqP->m + k] = mfqP->beta[num]/sqrt2;
		mfqP->Hdel[(i*mfqP->n + j)*mfqP->m + k] = mfqP->beta[num]/sqrt2;
		num++;
	    }
	}
    }
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "morepoints"
PetscErrorCode morepoints(TAO_POUNDERS *mfqP) {
    /* Assumes mfqP->model_indices[0]  is minimum index
       Finishes adding points to mfqP->model_indices (up to npmax)
       Computes L,Z,M,N
       np is actual number of points in model (should equal npmax?) */
    PetscInt point,i,j,offset;
    PetscInt reject;
    PetscBLASInt blasn=mfqP->n,blasnpmax=mfqP->npmax,blasnplus1=mfqP->n+1,blasinfo,blasnpmax_x_5=mfqP->npmax*5,blasint,blasint2,blasnp,blasmaxmn;
    PetscReal *x,normd;
    PetscErrorCode ierr;
    PetscFunctionBegin;

    for (i=0;i<mfqP->n+1;i++) {
	ierr = VecGetArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	mfqP->M[(mfqP->n+1)*i] = 1.0;
	for (j=0;j<mfqP->n;j++) {
	    mfqP->M[j+1+((mfqP->n+1)*i)] = (x[j]  - mfqP->xmin[j]) / mfqP->delta;
	}
	ierr = VecRestoreArray(mfqP->Xhist[mfqP->model_indices[i]],&x); CHKERRQ(ierr);
	ierr = phi2eval(&mfqP->M[1+((mfqP->n+1)*i)],mfqP->n,&mfqP->N[mfqP->n*(mfqP->n+1)/2 * i]); CHKERRQ(ierr);
	


    }

    /* Now we add points until we have npmax starting with the most recent ones */
    point = mfqP->nHist-1;
    mfqP->nmodelpoints = mfqP->n+1;
    
    while (mfqP->nmodelpoints < mfqP->npmax && point>=0) {
	/* Reject any points already in the model */
	reject = 0;
	for (j=0;j<mfqP->n+1;j++) {
	    if (point == mfqP->model_indices[j]) {
		reject = 1;
		break;
	    }
	}

	/* Reject if norm(d) >c2 */
	if (!reject) {
	    ierr = VecCopy(mfqP->Xhist[point],mfqP->workxvec); CHKERRQ(ierr);
	    ierr = VecAXPY(mfqP->workxvec,-1.0,mfqP->Xhist[mfqP->minindex]); CHKERRQ(ierr);
	    ierr = VecNorm(mfqP->workxvec,NORM_2,&normd); CHKERRQ(ierr);
	    normd /= mfqP->delta;
	    if (normd > mfqP->c2) {
		reject =1;
	    }
	}
	if (reject)
	{
	    point--;
	    continue;
	}

	
	ierr = VecGetArray(mfqP->Xhist[point],&x); CHKERRQ(ierr);
	mfqP->M[(mfqP->n+1)*mfqP->nmodelpoints] = 1.0;
	for (j=0;j<mfqP->n;j++) {
	    mfqP->M[j+1+((mfqP->n+1)*mfqP->nmodelpoints)] = (x[j]  - mfqP->xmin[j]) / mfqP->delta;
	}

	ierr = VecRestoreArray(mfqP->Xhist[point],&x); CHKERRQ(ierr);
	ierr = phi2eval(&mfqP->M[1+(mfqP->n+1)*mfqP->nmodelpoints],mfqP->n,&mfqP->N[mfqP->n*(mfqP->n+1)/2 * (mfqP->nmodelpoints)]); CHKERRQ(ierr);

	/* Update QR factorization */

	/* Copy M' to Q_tmp */
	for (i=0;i<mfqP->n+1;i++) {
	    for (j=0;j<mfqP->npmax;j++) {
		mfqP->Q_tmp[j+mfqP->npmax*i] = mfqP->M[i+(mfqP->n+1)*j];
	    }
	}
	blasnp = mfqP->nmodelpoints+1;
	/* Q_tmp,R = qr(M') */
	blasmaxmn=PetscMax(mfqP->m,mfqP->n+1);
	LAPACKgeqrf_(&blasnp,&blasnplus1,mfqP->Q_tmp,&blasnpmax,mfqP->tau_tmp,mfqP->mwork,&blasmaxmn,&blasinfo);

	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine geqrf returned with value %d\n",blasinfo);
	}
	
	/* Reject if min(svd(N*Q(:,n+2:np+1)) <= theta2 */
	/* L = N*Qtmp */
	blasint2 = mfqP->n * (mfqP->n+1) / 2;
	/* Copy N to L_tmp */
	for (i=0;i<mfqP->n*(mfqP->n+1)/2 * mfqP->npmax;i++) {
	    mfqP->L_tmp[i]= mfqP->N[i];
	}
	
	/* Copy L_save to L_tmp */

	/* L_tmp = N*Qtmp' */
	LAPACKormqr_("R","N",&blasint2,&blasnp,&blasnplus1,mfqP->Q_tmp,&blasnpmax,mfqP->tau_tmp,
		     mfqP->L_tmp,&blasint2,mfqP->npmaxwork,&blasnpmax_x_5,&blasinfo);

	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine ormqr returned with value %d\n",blasinfo);
	}

	/* Copy L_tmp to L_save */
	for (i=0;i<mfqP->npmax * mfqP->n*(mfqP->n+1)/2;i++) {
	    mfqP->L_save[i] = mfqP->L_tmp[i];
	}
	
	/* Get svd for L_tmp(:,n+2:np+1) (L_tmp is modified in process) */
	blasint = mfqP->nmodelpoints - mfqP->n;
	LAPACKgesvd_("N","N",&blasint2,&blasint,&mfqP->L_tmp[(mfqP->n+1)*blasint2],&blasint2,
		     mfqP->beta,mfqP->work,&blasn,mfqP->work,&blasn,mfqP->npmaxwork,&blasnpmax_x_5,
		     &blasinfo);
	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine gesvd returned with value %d\n",blasinfo);
	}

	if (mfqP->beta[PetscMin(blasint,blasint2)-1] > mfqP->theta2) {
	    /* accept point */
	    mfqP->model_indices[mfqP->nmodelpoints] = point;
	    /* Copy Q_tmp to Q */
	    for (i=0;i<mfqP->npmax* mfqP->npmax;i++) {
		mfqP->Q[i] = mfqP->Q_tmp[i];
	    }
	    for (i=0;i<mfqP->npmax;i++){
		mfqP->tau[i] = mfqP->tau_tmp[i]; 
	    }
	    mfqP->nmodelpoints++;
	    blasnp = mfqP->nmodelpoints+1;

	    /* Copy L_save to L */
	    for (i=0;i<mfqP->npmax * mfqP->n*(mfqP->n+1)/2;i++) {
		mfqP->L[i] = mfqP->L_save[i];
	    }
	}
	point--;

    }    


    blasnp = mfqP->nmodelpoints;
    /* Copy Q(:,n+2:np) to Z */
    /* First set Q_tmp to I */
    for (i=0;i<mfqP->npmax*mfqP->npmax;i++) {
	mfqP->Q_tmp[i] = 0.0;
    }
    for (i=0;i<mfqP->npmax;i++) {
	mfqP->Q_tmp[i + mfqP->npmax*i] = 1.0;
    }

    /* Q_tmp = I * Q */
    LAPACKormqr_("R","N",&blasnp,&blasnp,&blasnplus1,mfqP->Q,&blasnpmax,mfqP->tau,mfqP->Q_tmp,&blasnpmax,mfqP->npmaxwork,&blasnpmax_x_5,&blasinfo);

    if (blasinfo != 0) {
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine ormqr returned with value %d\n",blasinfo);
    }

    /* Copy Q_tmp(:,n+2:np) to Z) */
    offset = mfqP->npmax * (mfqP->n+1);
    for (i=offset;i<mfqP->npmax*mfqP->npmax;i++) {
	mfqP->Z[i-offset] = mfqP->Q_tmp[i];
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "addpoint"
/* Only call from modelimprove, addpoint() needs ->Q_tmp and ->work to be set */
PetscErrorCode addpoint(TaoSolver tao, TAO_POUNDERS *mfqP, PetscInt index) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Create new vector in history: X[newidx] = X[mfqP->index] + delta*X[index]*/
    
  ierr = VecDuplicate(mfqP->Xhist[0],&mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
  ierr = VecSetValues(mfqP->Xhist[mfqP->nHist],mfqP->n,mfqP->indices,&mfqP->Q_tmp[index*mfqP->npmax],INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
  ierr = VecAYPX(mfqP->Xhist[mfqP->nHist],mfqP->delta,mfqP->Xhist[mfqP->minindex]); CHKERRQ(ierr);

  /* Compute value of new vector */
  ierr = VecDuplicate(mfqP->Fhist[0],&mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[mfqP->nHist],mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
  ierr = VecNorm(mfqP->Fhist[mfqP->nHist],NORM_2,&mfqP->Fres[mfqP->nHist]); CHKERRQ(ierr);
  mfqP->Fres[mfqP->nHist]*=mfqP->Fres[mfqP->nHist];

  /* Add new vector to model */
  mfqP->model_indices[mfqP->nmodelpoints] = mfqP->nHist;
  mfqP->nmodelpoints++;
  mfqP->nHist++;

  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "modelimprove"
PetscErrorCode modelimprove(TaoSolver tao, TAO_POUNDERS *mfqP, PetscInt addallpoints) {
  /* modeld = Q(:,np+1:n)' */
  PetscErrorCode ierr;
  PetscInt i,j,minindex;
  PetscReal dp,half=0.5,one=1.0,minvalue;
  PetscBLASInt blasn=mfqP->n,  blasnpmax = mfqP->npmax, blask,info;
  PetscBLASInt blas1=1,blasnpmax_x_5 = mfqP->npmax*5;
  blask = mfqP->nmodelpoints;

   
  /* Qtmp = I(n x n) */
  for (i=0;i<mfqP->n;i++) {
    for (j=0;j<mfqP->n;j++) {
      mfqP->Q_tmp[i + mfqP->npmax*j] = 0.0;
    }
  }
  for (j=0;j<mfqP->n;j++) {
    mfqP->Q_tmp[j + mfqP->npmax*j] = 1.0;
  }
  

  /* Qtmp = Q * I */
  LAPACKormqr_("R","N",&blasn,&blasn,&blask,mfqP->Q,&blasnpmax,
	       mfqP->tau, mfqP->Q_tmp, &blasnpmax, mfqP->npmaxwork,&blasnpmax_x_5, &info);
  
  for (i=mfqP->nmodelpoints;i<mfqP->n;i++) {
    dp = BLASdot_(&blasn,&mfqP->Q_tmp[i*mfqP->npmax],&blas1,mfqP->Gres,&blas1);
    if (dp>0.0) { /* Model says use the other direction! */
      for (j=0;j<mfqP->n;j++) {
	mfqP->Q_tmp[i*mfqP->npmax+j] *= -1;
      }
    }
    /* mfqP->work[i] = Cres+Modeld(i,:)*(Gres+.5*Hres*Modeld(i,:)') */
    for (j=0;j<mfqP->n;j++) {
      mfqP->work2[j] = mfqP->Gres[j];
    }
    BLASgemv_("N",&blasn,&blasn,&half,mfqP->Hres,&blasn,
	      &mfqP->Q_tmp[i*mfqP->npmax], &blas1, &one, mfqP->work2,&blas1);
    mfqP->work[i] = BLASdot_(&blasn,&mfqP->Q_tmp[i*mfqP->npmax],&blas1,
			     mfqP->work2,&blas1);
    if (i==mfqP->nmodelpoints || mfqP->work[i] < minvalue) {
      minindex=i;
      minvalue = mfqP->work[i];
    }
    
    if (addallpoints != 0) {
	ierr = addpoint(tao,mfqP,i); CHKERRQ(ierr);
    }

  }
  
  if (!addallpoints) {
      ierr = addpoint(tao,mfqP,minindex); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "affpoints"
PetscErrorCode affpoints(TAO_POUNDERS *mfqP, PetscReal *xmin, 
				PetscReal c) {
    PetscInt i,j;
    PetscBLASInt blasm=mfqP->m,blasj,blask,blasn=mfqP->n,ione=1,info;
    PetscBLASInt blasnpmax = mfqP->npmax,blasmaxmn;
    PetscReal proj,normd;
    PetscReal *x;
    PetscErrorCode ierr;
    PetscFunctionBegin;

    for (i=mfqP->nHist-1;i>=0;i--) {
	ierr = VecGetArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	for (j=0;j<mfqP->n;j++) {
	    mfqP->work[j] = (x[j] - xmin[j])/mfqP->delta;
	}
	ierr = VecRestoreArray(mfqP->Xhist[i],&x); CHKERRQ(ierr);
	BLAScopy_(&blasn,mfqP->work,&ione,mfqP->work2,&ione);
	normd = BLASnrm2_(&blasn,mfqP->work,&ione);
	if (normd <= c*c) {
	  blasj=(mfqP->n - mfqP->nmodelpoints);
	    if (!mfqP->q_is_I) {
		/* project D onto null */
		blask=(mfqP->nmodelpoints);
		LAPACKormqr_("R","N",&ione,&blasn,&blask,mfqP->Q,&blasnpmax,mfqP->tau,
			     mfqP->work2,&ione,mfqP->mwork,&blasm,&info);
		
		if (info < 0) {
		    SETERRQ1(PETSC_COMM_SELF,1,"ormqr returned value %d\n",info);
		}
	    }
	    proj = BLASnrm2_(&blasj,&mfqP->work2[mfqP->nmodelpoints],&ione);

	    if (proj >= mfqP->theta1) { /* add this index to model */
		mfqP->model_indices[mfqP->nmodelpoints]=i;
		mfqP->nmodelpoints++;
		BLAScopy_(&blasn,mfqP->work,&ione,&mfqP->Q_tmp[mfqP->npmax*(mfqP->nmodelpoints-1)],&ione);
		blask=mfqP->npmax*(mfqP->nmodelpoints);
		BLAScopy_(&blask,mfqP->Q_tmp,&ione,mfqP->Q,&ione);
		blask = mfqP->nmodelpoints;
		blasmaxmn = PetscMax(mfqP->m,mfqP->n);
		LAPACKgeqrf_(&blasn,&blask,mfqP->Q,&blasnpmax,mfqP->tau,mfqP->mwork,
			     &blasmaxmn,&info);
		mfqP->q_is_I = 0;
		if (info < 0) {
		    SETERRQ1(PETSC_COMM_SELF,1,"geqrf returned value %d\n",info);
		}

		    
	    }
	    if (mfqP->nmodelpoints == mfqP->n)  {
		break;
	    }
	}		
    }
    
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_POUNDERS"
static PetscErrorCode TaoSolverSolve_POUNDERS(TaoSolver tao)
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
  PetscBLASInt blasm,blasn,blasnpmax;
  PetscErrorCode ierr;
  
  
  /* n = # of parameters 
     m = dimension (components) of function  */
  
  PetscFunctionBegin;
  CHKMEMQ;
  blasm = mfqP->m; blasn=mfqP->n; blasnpmax = mfqP->npmax;
  for (i=0;i<mfqP->n*mfqP->n*mfqP->m;i++) mfqP->H[i]=0;

  ierr = VecCopy(tao->solution,mfqP->Xhist[0]); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = TaoSolverComputeSeparableObjective(tao,tao->solution,mfqP->Fhist[0]); CHKERRQ(ierr);
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
      ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[i],mfqP->Fhist[i]); CHKERRQ(ierr);
      ierr = VecNorm(mfqP->Fhist[i],NORM_2,&mfqP->Fres[i]); CHKERRQ(ierr);
      mfqP->Fres[i]*=mfqP->Fres[i];
      if (mfqP->Fres[i] < minnorm) {
	  mfqP->minindex = i;
	  minnorm = mfqP->Fres[i];
      }
  }

  ierr = VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution); CHKERRQ(ierr);
  ierr = VecCopy(mfqP->Fhist[mfqP->minindex],tao->sep_objective); CHKERRQ(ierr);
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
  PetscBLASInt blasn2 = blasn;
  LAPACKgesv_(&blasn,&blasm,mfqP->Disp,&blasnpmax,mfqP->iwork,mfqP->Fdiff,&blasn2,&info);
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
    ierr = VecDuplicate(tao->sep_objective,&mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecSetValues(mfqP->Xhist[mfqP->nHist],mfqP->n,mfqP->indices,mfqP->work,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mfqP->Xhist[mfqP->nHist]); CHKERRQ(ierr);
    CHKMEMQ;
    ierr = TaoSolverComputeSeparableObjective(tao,mfqP->Xhist[mfqP->nHist],mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
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
#define __FUNCT__ "TaoSolverSetUp_POUNDERS"
static PetscErrorCode TaoSolverSetUp_POUNDERS(TaoSolver tao)
{
    TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
    PetscInt i;
    IS isfloc,isfglob,isxloc,isxglob;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
    if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
    ierr = VecGetSize(tao->solution,&mfqP->n); CHKERRQ(ierr);
    ierr = VecGetSize(tao->sep_objective,&mfqP->m); CHKERRQ(ierr);
    mfqP->c1 = sqrt(mfqP->n);
    mfqP->npmax = (mfqP->n+1)*(mfqP->n+2)/2; /* TODO check if manually set */

    ierr = PetscMalloc((tao->max_funcs+10)*sizeof(Vec),&mfqP->Xhist); CHKERRQ(ierr);
    ierr = PetscMalloc((tao->max_funcs+10)*sizeof(Vec),&mfqP->Fhist); CHKERRQ(ierr);
    for (i=0;i<mfqP->n +1;i++) {
	ierr = VecDuplicate(tao->solution,&mfqP->Xhist[i]); CHKERRQ(ierr);
	ierr = VecDuplicate(tao->sep_objective,&mfqP->Fhist[i]); CHKERRQ(ierr);
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
      ierr = VecScatterCreate(tao->sep_objective,isfglob,mfqP->localf,isfloc,&mfqP->scatterf); CHKERRQ(ierr);

      ierr = ISDestroy(&isxloc); CHKERRQ(ierr);
      ierr = ISDestroy(&isxglob); CHKERRQ(ierr);
      ierr = ISDestroy(&isfloc); CHKERRQ(ierr);
      ierr = ISDestroy(&isfglob); CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_POUNDERS"
static PetscErrorCode TaoSolverDestroy_POUNDERS(TaoSolver tao)
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
      ierr = VecDestroy(&mfqP->Xhist[i]); CHKERRQ(ierr);
      ierr = VecDestroy(&mfqP->Fhist[i]); CHKERRQ(ierr);
  }
  if (mfqP->workxvec) {
    ierr = VecDestroy(&mfqP->workxvec); CHKERRQ(ierr);
  }
  ierr = PetscFree(mfqP->Xhist); CHKERRQ(ierr);
  ierr = PetscFree(mfqP->Fhist); CHKERRQ(ierr);

  if (mfqP->mpisize > 1) {
      ierr = VecDestroy(&mfqP->localx);  CHKERRQ(ierr);
      ierr = VecDestroy(&mfqP->localxmin);  CHKERRQ(ierr);
      ierr = VecDestroy(&mfqP->localf);  CHKERRQ(ierr);
      ierr = VecDestroy(&mfqP->localfmin);  CHKERRQ(ierr);
  }



  if (tao->data) {
    ierr = PetscFree(tao->data); CHKERRQ(ierr);
  }
  tao->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_POUNDERS"
static PetscErrorCode TaoSolverSetFromOptions_POUNDERS(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("POUNDERS method for least-squares optimization"); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pounders_delta","initial delta","",mfqP->delta,&mfqP->delta,0); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoSolverView_POUNDERS"
static PetscErrorCode TaoSolverView_POUNDERS(TaoSolver tao, PetscViewer viewer)
{
  return 0;
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoSolverCreate_POUNDERS"
PetscErrorCode TaoSolverCreate_POUNDERS(TaoSolver tao)
{
  TAO_POUNDERS *mfqP = (TAO_POUNDERS*)tao->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  tao->ops->setup = TaoSolverSetUp_POUNDERS;
  tao->ops->solve = TaoSolverSolve_POUNDERS;
  tao->ops->view = TaoSolverView_POUNDERS;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_POUNDERS;
  tao->ops->destroy = TaoSolverDestroy_POUNDERS;


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
  PetscFunctionReturn(0);
}
EXTERN_C_END

