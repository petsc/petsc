#include "private/taosolver_impl.h"
#undef __FUNCT__
#define __FUNCT__ "formquad"
static PetscErrorCode formquad(TAO_POUNDERS *mfqP,PetscBool checkonly)
{
  PetscErrorCode ierr;
  PetscReal *d,dqi;
  PetscInt i,j,np;
  PetscBLASInt blasm =mfqP->m;
  PetscBLASInt blasn = mfqP->n;
  PetscBLASInt blasnpmax = mfqP->npmax;
  PetscBLASInt blasj, blask, blasmaxmn, ione=1, info;
  PetscFunctionBegin;
  /* Di = X(i,:) - X(xkin,:)/delta */
  for (i=0;i<mfqP->nHist;i++) {
    ierr = VecCopy(mfqP->Xhist[i],mfqP->D[i]); CHKERRQ(ierr);
    ierr = VecAXPY(mfqP->D[i], -1.0, mfqP->Xhist[mfqP->minindex]); CHKERRQ(ierr);
    ierr = VecScale(mfqP->D[i],1.0/mfqP->delta); CHKERRQ(ierr);
    ierr = VecNorm(mfqP->D[i],NORM_2,&mfqP->Nd[i]); CHKERRQ(ierr);
  }
  /* Initialize Q to I */
  ierr = PoundersGramSchmidtReset(mfqP,mfqP->QD,mfqP->n); CHKERRQ(ierr);

  /* Get n+1 sufficiently affinely independent points */
  mfqP->valid = PETSC_FALSE;
  mfqP->nmodel_improving_points = 0;
  mfqP->sizemdir=0;
  np = 0;
  /* Add points based on delta multiplier */
  for (i=mfqP->nHist-1,i>=0;i--) {
    if (mfqP->Nd[i] <= mfqP->par1) {
      proj = 0.0;
      /* project D onto null */
      for (j=np;j<mfqP->n;j++) {
	ierr = VecDot(mfqP->D[i],mfqP->QD[i],&gdi); CHKERRQ(ierr);
	proj += gdi*gdi;
      }
      proj = PetscSqrtReal(proj);

      if (proj >= mfqP->par3) { 
	/* Add this index to model */
	mfqP->model_indices[mfqP->nmodelpoints++]=i;
	
	/* Reconstruct Q */
	ierr = PoundersGramSchmidtInsert(mfqP,mfqP->QD,mfqP->D[i]); CHKERRQ(ierr);

	if (mfqP->np < mfqP->n) {
	  /* Output model-improving directions */
	  mfqP->nmodel_improving_points = mfqp->np - mfqP->n;
	  for (i=0;i<mfqP->n_model_improving_points; i++) {
	    mfqP->Mdir[i] = mfqP->QD[mfqP->n + i];
	  }
	} else {
	  break;
	}
      }
    }
  }
  
  if (mfqP->np < mfqP->n) {
    /* Add points based on pivot threshold */
    for (i=mfqP->nHist-1,i>=0;i--) {
      if (mfqP->Nd[i] <= mfqP->par2) {
	proj = 0.0;
	/* project D onto null */
	for (j=np;j<mfqP->n;j++) {
	  ierr = VecDot(mfqP->D[i],mfqP->QD[i],&gdi); CHKERRQ(ierr);
	  proj += gdi*gdi;
	}
	proj = PetscSqrtReal(proj);

	if (proj >= mfqP->par4) { 
	  /* Add this index to model */
	  mfqP->model_indices[mfqP->nmodelpoints++]=i;
	
	  /* Reconstruct Q */
	  ierr = PoundersGramSchmidtInsert(mfqP,mfqP->QD,mfqP->D[i]); CHKERRQ(ierr);

	  if (mfqP->np == n)  break;
	}
      }
    }
  }

  if (mfqP->np == mfqP->n) {
    mfqP->valid = true;
  }
 if (checkonly) {
    PetscFunctionReturn(0);
  }
  
  if (mfqP->np < mfqP->n) {
    /* Need to evaluate more points, then recall */
    mfqP->nmodel_improving_points = mfqp->np - mfqP->n;
    for (i=0;i<mfqP->n_model_improving_points; i++) {
      mfqP->Mdir[i] = mfqP->Q[mfqP->n + i];
    }
    ierr =MatDestroy(&mfqP->G); CHKERRQ(ierr);
    ierr =MatDestroy(&mfqP->H); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Collect additional points */
  for (i=1; i<mfqP->n+1; i++) {
    ierr = phi2eval(mfqP->D[mfqP->model_indices[i]],mfqP->N[i]);
  }
  
  /* Create M = [ones(n+1,1) D(Mind,:)]' */
  ierr = PoundersGramSchmidtReset(mfqP,mfqP->n+1);
  for (i=1; i<mfqP->n+1; i++) {
    ierr = PoundersCopyDtoM(mfqP,mfqP->Mdir[i]);
    ierr = PoundersGramSchmidtInsert(mfqP,mfqP->MQ,mfqP->M[i]);
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
#define __FUNCT__ "solvequadratic"
PetscErrorCode solvequadratic(TAO_POUNDERS *mfqP,PetscReal *gnorm, PetscReal *qmin) 
{
    PetscErrorCode ierr;
    PetscReal atol=1.0e-10;
    PetscInt info,its;
    TaoSolver subtao;
    Vec       x,xl,xu;
    PetscInt i,j;
    PetscFunctionBegin;

    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,mfqP->n,mfqP->Xsubproblem,&x); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,mfqP->n,mfqP->Gres,&mfqP->b); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&xl); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&xu); CHKERRQ(ierr);
    ierr = VecSet(x,0.0); CHKERRQ(ierr);
    ierr = VecSet(xl,-mfqP->delta); CHKERRQ(ierr);
    ierr = VecSet(xu,mfqP->delta); CHKERRQ(ierr);
    for (i=0;i<mfqP->n;i++) {
      for (j=i;j<mfqP->n;j++) {
	mfqP->Hres[j+mfqP->n*i] = mfqP->Hres[mfqP->n*j+i];
      }
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,mfqP->n,mfqP->n,mfqP->Hres,&mfqP->Hs); CHKERRQ(ierr);
    ierr = TaoCreate(PETSC_COMM_SELF,&subtao); CHKERRQ(ierr);
    ierr = TaoSetType(subtao,"tao_bqpip"); CHKERRQ(ierr);
    ierr = TaoSetOptionsPrefix(subtao,"pounders_subsolver_"); CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(subtao,pounders_fg,(void*)mfqP); CHKERRQ(ierr);
    ierr = TaoSetInitialVector(subtao,x); CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(subtao,mfqP->Hs,mfqP->Hs,pounders_h,(void*)mfqP); CHKERRQ(ierr);
    ierr = TaoSetTolerances(subtao,PETSC_NULL,PETSC_NULL,*gnorm,*gnorm,PETSC_NULL); CHKERRQ(ierr);
    ierr = TaoSetMaximumIterations(subtao,mfqP->gqt_maxits); CHKERRQ(ierr);
    ierr = TaoSetVariableBounds(subtao,xl,xu); CHKERRQ(ierr);
    ierr = TaoSetFromOptions(subtao); CHKERRQ(ierr);
    ierr = TaoSolve(subtao); CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(subtao,PETSC_NULL,qmin,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&xl); CHKERRQ(ierr);
    ierr = VecDestroy(&xu); CHKERRQ(ierr);
    ierr = VecDestroy(&mfqP->b); CHKERRQ(ierr);
    ierr = MatDestroy(&mfqP->Hs); CHKERRQ(ierr);
    ierr = TaoDestroy(&subtao); CHKERRQ(ierr);
    *qmin *= -1;
    PetscFunctionReturn(0);
}


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
/* Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2] */
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
    PetscReal sqrt2 = PetscSqrtReal(2.0);
	
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
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine potrf returned with value %D\n",blasinfo);
    }

    /* factor M */
    LAPACKgetrf_(&blasnplus1,&blasnpmax,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&blasinfo);
    if (blasinfo != 0) {
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine getrf returned with value %D\n",blasinfo);
    }
    
    for (k=0;k<mfqP->m;k++) {
	/* Solve L'*L*Omega = Z' * RESk*/
	BLASgemv_("T",&blasnp,&blasint2,&one,mfqP->Z,&blasnpmax,&mfqP->RES[mfqP->npmax*k],&ione,&zero,mfqP->omega,&ione);

	LAPACKpotrs_("L",&blasint2,&ione,mfqP->L_tmp,&blasint,mfqP->omega,&blasint2,&blasinfo);
	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine potrs returned with value %D\n",blasinfo);
	}
	
	
	/* Beta = L*Omega */
	BLASgemv_("N",&blasint,&blasint2,&one,&mfqP->L[(mfqP->n+1)*blasint],&blasint,mfqP->omega,&ione,&zero,mfqP->beta,&ione);
	
	/* solve M'*Alpha = RESk - N'*Beta */
	BLASgemv_("T",&blasint,&blasnp,&negone,mfqP->N,&blasint,mfqP->beta,&ione,&one,&mfqP->RES[mfqP->npmax*k],&ione);
	LAPACKgetrs_("T",&blasnplus1,&ione,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&mfqP->RES[mfqP->npmax*k],&blasnplus1,&blasinfo);
	if (blasinfo != 0) {
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine getrs returned with value %D\n",blasinfo);
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
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine geqrf returned with value %D\n",blasinfo);
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
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine ormqr returned with value %D\n",blasinfo);
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
	    SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine gesvd returned with value %D\n",blasinfo);
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
	SETERRQ1(PETSC_COMM_SELF,1,"LAPACK routine ormqr returned with value %D\n",blasinfo);
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
  ierr = TaoComputeSeparableObjective(tao,mfqP->Xhist[mfqP->nHist],mfqP->Fhist[mfqP->nHist]); CHKERRQ(ierr);
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
  PetscInt i,j,minindex=0;
  PetscReal dp,half=0.5,one=1.0,minvalue=TAO_INFINITY;
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
		    SETERRQ1(PETSC_COMM_SELF,1,"ormqr returned value %D\n",info);
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
		    SETERRQ1(PETSC_COMM_SELF,1,"geqrf returned value %D\n",info);
		}

		    
	    }
	    if (mfqP->nmodelpoints == mfqP->n)  {
		break;
	    }
	}		
    }
    
    PetscFunctionReturn(0);
}
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
