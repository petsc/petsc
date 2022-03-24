#include <../src/tao/leastsquares/impls/pounders/pounders.h>

static PetscErrorCode pounders_h(Tao subtao, Vec v, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode  pounders_fg(Tao subtao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)ctx;
  PetscReal      d1,d2;

  PetscFunctionBegin;
  /* g = A*x  (add b later)*/
  CHKERRQ(MatMult(mfqP->subH,x,g));

  /* f = 1/2 * x'*(Ax) + b'*x  */
  CHKERRQ(VecDot(x,g,&d1));
  CHKERRQ(VecDot(mfqP->subb,x,&d2));
  *f = 0.5 *d1 + d2;

  /* now  g = g + b */
  CHKERRQ(VecAXPY(g, 1.0, mfqP->subb));
  PetscFunctionReturn(0);
}

static PetscErrorCode pounders_feval(Tao tao, Vec x, Vec F, PetscReal *fsum)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;
  PetscInt       i,row,col;
  PetscReal      fr,fc;

  PetscFunctionBegin;
  CHKERRQ(TaoComputeResidual(tao,x,F));
  if (tao->res_weights_v) {
    CHKERRQ(VecPointwiseMult(mfqP->workfvec,tao->res_weights_v,F));
    CHKERRQ(VecDot(mfqP->workfvec,mfqP->workfvec,fsum));
  } else if (tao->res_weights_w) {
    *fsum=0;
    for (i=0;i<tao->res_weights_n;i++) {
      row=tao->res_weights_rows[i];
      col=tao->res_weights_cols[i];
      CHKERRQ(VecGetValues(F,1,&row,&fr));
      CHKERRQ(VecGetValues(F,1,&col,&fc));
      *fsum += tao->res_weights_w[i]*fc*fr;
    }
  } else {
    CHKERRQ(VecDot(F,F,fsum));
  }
  CHKERRQ(PetscInfo(tao,"Least-squares residual norm: %20.19e\n",(double)*fsum));
  PetscCheck(!PetscIsInfOrNanReal(*fsum),PETSC_COMM_SELF,PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
  PetscFunctionReturn(0);
}

static PetscErrorCode gqtwrap(Tao tao,PetscReal *gnorm, PetscReal *qmin)
{
#if defined(PETSC_USE_REAL_SINGLE)
  PetscReal      atol=1.0e-5;
#else
  PetscReal      atol=1.0e-10;
#endif
  PetscInt       info,its;
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;

  PetscFunctionBegin;
  if (!mfqP->usegqt) {
    PetscReal maxval;
    PetscInt  i,j;

    CHKERRQ(VecSetValues(mfqP->subb,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(mfqP->subb));
    CHKERRQ(VecAssemblyEnd(mfqP->subb));

    CHKERRQ(VecSet(mfqP->subx,0.0));

    CHKERRQ(VecSet(mfqP->subndel,-1.0));
    CHKERRQ(VecSet(mfqP->subpdel,+1.0));

    /* Complete the lower triangle of the Hessian matrix */
    for (i=0;i<mfqP->n;i++) {
      for (j=i+1;j<mfqP->n;j++) {
        mfqP->Hres[j+mfqP->n*i] = mfqP->Hres[mfqP->n*j+i];
      }
    }
    CHKERRQ(MatSetValues(mfqP->subH,mfqP->n,mfqP->indices,mfqP->n,mfqP->indices,mfqP->Hres,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(mfqP->subH,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(mfqP->subH,MAT_FINAL_ASSEMBLY));

    CHKERRQ(TaoResetStatistics(mfqP->subtao));
    /* CHKERRQ(TaoSetTolerances(mfqP->subtao,*gnorm,*gnorm,PETSC_DEFAULT)); */
    /* enforce bound constraints -- experimental */
    if (tao->XU && tao->XL) {
      CHKERRQ(VecCopy(tao->XU,mfqP->subxu));
      CHKERRQ(VecAXPY(mfqP->subxu,-1.0,tao->solution));
      CHKERRQ(VecScale(mfqP->subxu,1.0/mfqP->delta));
      CHKERRQ(VecCopy(tao->XL,mfqP->subxl));
      CHKERRQ(VecAXPY(mfqP->subxl,-1.0,tao->solution));
      CHKERRQ(VecScale(mfqP->subxl,1.0/mfqP->delta));

      CHKERRQ(VecPointwiseMin(mfqP->subxu,mfqP->subxu,mfqP->subpdel));
      CHKERRQ(VecPointwiseMax(mfqP->subxl,mfqP->subxl,mfqP->subndel));
    } else {
      CHKERRQ(VecCopy(mfqP->subpdel,mfqP->subxu));
      CHKERRQ(VecCopy(mfqP->subndel,mfqP->subxl));
    }
    /* Make sure xu > xl */
    CHKERRQ(VecCopy(mfqP->subxl,mfqP->subpdel));
    CHKERRQ(VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu));
    CHKERRQ(VecMax(mfqP->subpdel,NULL,&maxval));
    PetscCheck(maxval <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"upper bound < lower bound in subproblem");
    /* Make sure xu > tao->solution > xl */
    CHKERRQ(VecCopy(mfqP->subxl,mfqP->subpdel));
    CHKERRQ(VecAXPY(mfqP->subpdel,-1.0,mfqP->subx));
    CHKERRQ(VecMax(mfqP->subpdel,NULL,&maxval));
    PetscCheck(maxval <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"initial guess < lower bound in subproblem");

    CHKERRQ(VecCopy(mfqP->subx,mfqP->subpdel));
    CHKERRQ(VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu));
    CHKERRQ(VecMax(mfqP->subpdel,NULL,&maxval));
    PetscCheck(maxval <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"initial guess > upper bound in subproblem");

    CHKERRQ(TaoSolve(mfqP->subtao));
    CHKERRQ(TaoGetSolutionStatus(mfqP->subtao,NULL,qmin,NULL,NULL,NULL,NULL));

    /* test bounds post-solution*/
    CHKERRQ(VecCopy(mfqP->subxl,mfqP->subpdel));
    CHKERRQ(VecAXPY(mfqP->subpdel,-1.0,mfqP->subx));
    CHKERRQ(VecMax(mfqP->subpdel,NULL,&maxval));
    if (maxval > 1e-5) {
      CHKERRQ(PetscInfo(tao,"subproblem solution < lower bound\n"));
      tao->reason = TAO_DIVERGED_TR_REDUCTION;
    }

    CHKERRQ(VecCopy(mfqP->subx,mfqP->subpdel));
    CHKERRQ(VecAXPY(mfqP->subpdel,-1.0,mfqP->subxu));
    CHKERRQ(VecMax(mfqP->subpdel,NULL,&maxval));
    if (maxval > 1e-5) {
      CHKERRQ(PetscInfo(tao,"subproblem solution > upper bound\n"));
      tao->reason = TAO_DIVERGED_TR_REDUCTION;
    }
  } else {
    gqt(mfqP->n,mfqP->Hres,mfqP->n,mfqP->Gres,1.0,mfqP->gqt_rtol,atol,mfqP->gqt_maxits,gnorm,qmin,mfqP->Xsubproblem,&info,&its,mfqP->work,mfqP->work2, mfqP->work3);
  }
  *qmin *= -1;
  PetscFunctionReturn(0);
}

static PetscErrorCode pounders_update_res(Tao tao)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;
  PetscInt       i,row,col;
  PetscBLASInt   blasn=mfqP->n,blasn2=blasn*blasn,blasm=mfqP->m,ione=1;
  PetscReal      zero=0.0,one=1.0,wii,factor;

  PetscFunctionBegin;
  for (i=0;i<mfqP->n;i++) {
    mfqP->Gres[i]=0;
  }
  for (i=0;i<mfqP->n*mfqP->n;i++) {
    mfqP->Hres[i]=0;
  }

  /* Compute Gres= sum_ij[wij * (cjgi + cigj)] */
  if (tao->res_weights_v) {
    /* Vector(diagonal) weights: gres = sum_i(wii*ci*gi) */
    for (i=0;i<mfqP->m;i++) {
      CHKERRQ(VecGetValues(tao->res_weights_v,1,&i,&factor));
      factor=factor*mfqP->C[i];
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn,&factor,&mfqP->Fdiff[blasn*i],&ione,mfqP->Gres,&ione));
    }

    /* compute Hres = sum_ij [wij * (*ci*Hj + cj*Hi + gi gj' + gj gi') ] */
    /* vector(diagonal weights) Hres = sum_i(wii*(ci*Hi + gi * gi')*/
    for (i=0;i<mfqP->m;i++) {
      CHKERRQ(VecGetValues(tao->res_weights_v,1,&i,&wii));
      if (tao->niter>1) {
        factor=wii*mfqP->C[i];
        /* add wii * ci * Hi */
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn2,&factor,&mfqP->H[i],&blasm,mfqP->Hres,&ione));
      }
      /* add wii * gi * gi' */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&blasn,&blasn,&ione,&wii,&mfqP->Fdiff[blasn*i],&blasn,&mfqP->Fdiff[blasn*i],&blasn,&one,mfqP->Hres,&blasn));
    }
  } else if (tao->res_weights_w) {
    /* General case: .5 * Gres= sum_ij[wij * (cjgi + cigj)] */
    for (i=0;i<tao->res_weights_n;i++) {
      row=tao->res_weights_rows[i];
      col=tao->res_weights_cols[i];

      factor = tao->res_weights_w[i]*mfqP->C[col]/2.0;
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn,&factor,&mfqP->Fdiff[blasn*row],&ione,mfqP->Gres,&ione));
      factor = tao->res_weights_w[i]*mfqP->C[row]/2.0;
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn,&factor,&mfqP->Fdiff[blasn*col],&ione,mfqP->Gres,&ione));
    }

    /* compute Hres = sum_ij [wij * (*ci*Hj + cj*Hi + gi gj' + gj gi') ] */
    /* .5 * sum_ij [wij * (*ci*Hj + cj*Hi + gi gj' + gj gi') ] */
    for (i=0;i<tao->res_weights_n;i++) {
      row=tao->res_weights_rows[i];
      col=tao->res_weights_cols[i];
      factor=tao->res_weights_w[i]/2.0;
      /* add wij * gi gj' + wij * gj gi' */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&blasn,&blasn,&ione,&factor,&mfqP->Fdiff[blasn*row],&blasn,&mfqP->Fdiff[blasn*col],&blasn,&one,mfqP->Hres,&blasn));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&blasn,&blasn,&ione,&factor,&mfqP->Fdiff[blasn*col],&blasn,&mfqP->Fdiff[blasn*row],&blasn,&one,mfqP->Hres,&blasn));
    }
    if (tao->niter > 1) {
      for (i=0;i<tao->res_weights_n;i++) {
        row=tao->res_weights_rows[i];
        col=tao->res_weights_cols[i];

        /* add  wij*cj*Hi */
        factor = tao->res_weights_w[i]*mfqP->C[col]/2.0;
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn2,&factor,&mfqP->H[row],&blasm,mfqP->Hres,&ione));

        /* add wij*ci*Hj */
        factor = tao->res_weights_w[i]*mfqP->C[row]/2.0;
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn2,&factor,&mfqP->H[col],&blasm,mfqP->Hres,&ione));
      }
    }
  } else {
    /* Default: Gres= sum_i[cigi] = G*c' */
    CHKERRQ(PetscInfo(tao,"Identity weights\n"));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&blasn,&blasm,&one,mfqP->Fdiff,&blasn,mfqP->C,&ione,&zero,mfqP->Gres,&ione));

    /* compute Hres = sum_ij [wij * (*ci*Hj + cj*Hi + gi gj' + gj gi') ] */
    /*  Hres = G*G' + 0.5 sum {F(xkin,i)*H(:,:,i)}  */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&blasn,&blasn,&blasm,&one,mfqP->Fdiff, &blasn,mfqP->Fdiff, &blasn,&zero,mfqP->Hres,&blasn));

    /* sum(F(xkin,i)*H(:,:,i)) */
    if (tao->niter>1) {
      for (i=0;i<mfqP->m;i++) {
        factor = mfqP->C[i];
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn2,&factor,&mfqP->H[i],&blasm,mfqP->Hres,&ione));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode phi2eval(PetscReal *x, PetscInt n, PetscReal *phi)
{
/* Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2] */
  PetscInt  i,j,k;
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

static PetscErrorCode getquadpounders(TAO_POUNDERS *mfqP)
{
/* Computes the parameters of the quadratic Q(x) = c + g'*x + 0.5*x*G*x'
   that satisfies the interpolation conditions Q(X[:,j]) = f(j)
   for j=1,...,m and with a Hessian matrix of least Frobenius norm */

    /* NB --we are ignoring c */
  PetscInt     i,j,k,num,np = mfqP->nmodelpoints;
  PetscReal    one = 1.0,zero=0.0,negone=-1.0;
  PetscBLASInt blasnpmax = mfqP->npmax;
  PetscBLASInt blasnplus1 = mfqP->n+1;
  PetscBLASInt blasnp = np;
  PetscBLASInt blasint = mfqP->n*(mfqP->n+1) / 2;
  PetscBLASInt blasint2 = np - mfqP->n-1;
  PetscBLASInt info,ione=1;
  PetscReal    sqrt2 = PetscSqrtReal(2.0);

  PetscFunctionBegin;
  for (i=0;i<mfqP->n*mfqP->m;i++) {
    mfqP->Gdel[i] = 0;
  }
  for (i=0;i<mfqP->n*mfqP->n*mfqP->m;i++) {
    mfqP->Hdel[i] = 0;
  }

    /* factor M */
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&blasnplus1,&blasnp,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&info));
  PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine getrf returned with value %d",info);

  if (np == mfqP->n+1) {
    for (i=0;i<mfqP->npmax-mfqP->n-1;i++) {
      mfqP->omega[i]=0.0;
    }
    for (i=0;i<mfqP->n*(mfqP->n+1)/2;i++) {
      mfqP->beta[i]=0.0;
    }
  } else {
    /* Let Ltmp = (L'*L) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&blasint2,&blasint2,&blasint,&one,&mfqP->L[(mfqP->n+1)*blasint],&blasint,&mfqP->L[(mfqP->n+1)*blasint],&blasint,&zero,mfqP->L_tmp,&blasint));

    /* factor Ltmp */
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&blasint2,mfqP->L_tmp,&blasint,&info));
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine potrf returned with value %d",info);
  }

  for (k=0;k<mfqP->m;k++) {
    if (np != mfqP->n+1) {
      /* Solve L'*L*Omega = Z' * RESk*/
      PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&blasnp,&blasint2,&one,mfqP->Z,&blasnpmax,&mfqP->RES[mfqP->npmax*k],&ione,&zero,mfqP->omega,&ione));
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&blasint2,&ione,mfqP->L_tmp,&blasint,mfqP->omega,&blasint2,&info));
      PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine potrs returned with value %d",info);

      /* Beta = L*Omega */
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&blasint,&blasint2,&one,&mfqP->L[(mfqP->n+1)*blasint],&blasint,mfqP->omega,&ione,&zero,mfqP->beta,&ione));
    }

    /* solve M'*Alpha = RESk - N'*Beta */
    PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&blasint,&blasnp,&negone,mfqP->N,&blasint,mfqP->beta,&ione,&one,&mfqP->RES[mfqP->npmax*k],&ione));
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("T",&blasnplus1,&ione,mfqP->M,&blasnplus1,mfqP->npmaxiwork,&mfqP->RES[mfqP->npmax*k],&blasnplus1,&info));
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine getrs returned with value %d",info);

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

static PetscErrorCode morepoints(TAO_POUNDERS *mfqP)
{
  /* Assumes mfqP->model_indices[0]  is minimum index
   Finishes adding points to mfqP->model_indices (up to npmax)
   Computes L,Z,M,N
   np is actual number of points in model (should equal npmax?) */
  PetscInt        point,i,j,offset;
  PetscInt        reject;
  PetscBLASInt    blasn=mfqP->n,blasnpmax=mfqP->npmax,blasnplus1=mfqP->n+1,info,blasnmax=mfqP->nmax,blasint,blasint2,blasnp,blasmaxmn;
  const PetscReal *x;
  PetscReal       normd;

  PetscFunctionBegin;
  /* Initialize M,N */
  for (i=0;i<mfqP->n+1;i++) {
    CHKERRQ(VecGetArrayRead(mfqP->Xhist[mfqP->model_indices[i]],&x));
    mfqP->M[(mfqP->n+1)*i] = 1.0;
    for (j=0;j<mfqP->n;j++) {
      mfqP->M[j+1+((mfqP->n+1)*i)] = (x[j]  - mfqP->xmin[j]) / mfqP->delta;
    }
    CHKERRQ(VecRestoreArrayRead(mfqP->Xhist[mfqP->model_indices[i]],&x));
    CHKERRQ(phi2eval(&mfqP->M[1+((mfqP->n+1)*i)],mfqP->n,&mfqP->N[mfqP->n*(mfqP->n+1)/2 * i]));
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
      CHKERRQ(VecCopy(mfqP->Xhist[point],mfqP->workxvec));
      CHKERRQ(VecAXPY(mfqP->workxvec,-1.0,mfqP->Xhist[mfqP->minindex]));
      CHKERRQ(VecNorm(mfqP->workxvec,NORM_2,&normd));
      normd /= mfqP->delta;
      if (normd > mfqP->c2) {
        reject =1;
      }
    }
    if (reject) {
      point--;
      continue;
    }

    CHKERRQ(VecGetArrayRead(mfqP->Xhist[point],&x));
    mfqP->M[(mfqP->n+1)*mfqP->nmodelpoints] = 1.0;
    for (j=0;j<mfqP->n;j++) {
      mfqP->M[j+1+((mfqP->n+1)*mfqP->nmodelpoints)] = (x[j]  - mfqP->xmin[j]) / mfqP->delta;
    }
    CHKERRQ(VecRestoreArrayRead(mfqP->Xhist[point],&x));
    CHKERRQ(phi2eval(&mfqP->M[1+(mfqP->n+1)*mfqP->nmodelpoints],mfqP->n,&mfqP->N[mfqP->n*(mfqP->n+1)/2 * (mfqP->nmodelpoints)]));

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
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&blasnp,&blasnplus1,mfqP->Q_tmp,&blasnpmax,mfqP->tau_tmp,mfqP->mwork,&blasmaxmn,&info));
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine geqrf returned with value %d",info);

    /* Reject if min(svd(N*Q(:,n+2:np+1)) <= theta2 */
    /* L = N*Qtmp */
    blasint2 = mfqP->n * (mfqP->n+1) / 2;
    /* Copy N to L_tmp */
    for (i=0;i<mfqP->n*(mfqP->n+1)/2 * mfqP->npmax;i++) {
      mfqP->L_tmp[i]= mfqP->N[i];
    }
    /* Copy L_save to L_tmp */

    /* L_tmp = N*Qtmp' */
    PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("R","N",&blasint2,&blasnp,&blasnplus1,mfqP->Q_tmp,&blasnpmax,mfqP->tau_tmp,mfqP->L_tmp,&blasint2,mfqP->npmaxwork,&blasnmax,&info));
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine ormqr returned with value %d",info);

    /* Copy L_tmp to L_save */
    for (i=0;i<mfqP->npmax * mfqP->n*(mfqP->n+1)/2;i++) {
      mfqP->L_save[i] = mfqP->L_tmp[i];
    }

    /* Get svd for L_tmp(:,n+2:np+1) (L_tmp is modified in process) */
    blasint = mfqP->nmodelpoints - mfqP->n;
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&blasint2,&blasint,&mfqP->L_tmp[(mfqP->n+1)*blasint2],&blasint2,mfqP->beta,mfqP->work,&blasn,mfqP->work,&blasn,mfqP->npmaxwork,&blasnmax,&info));
    CHKERRQ(PetscFPTrapPop());
    PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine gesvd returned with value %d",info);

    if (mfqP->beta[PetscMin(blasint,blasint2)-1] > mfqP->theta2) {
      /* accept point */
      mfqP->model_indices[mfqP->nmodelpoints] = point;
      /* Copy Q_tmp to Q */
      for (i=0;i<mfqP->npmax* mfqP->npmax;i++) {
        mfqP->Q[i] = mfqP->Q_tmp[i];
      }
      for (i=0;i<mfqP->npmax;i++) {
        mfqP->tau[i] = mfqP->tau_tmp[i];
      }
      mfqP->nmodelpoints++;
      blasnp = mfqP->nmodelpoints;

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
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("R","N",&blasnp,&blasnp,&blasnplus1,mfqP->Q,&blasnpmax,mfqP->tau,mfqP->Q_tmp,&blasnpmax,mfqP->npmaxwork,&blasnmax,&info));
  PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACK routine ormqr returned with value %d",info);

  /* Copy Q_tmp(:,n+2:np) to Z) */
  offset = mfqP->npmax * (mfqP->n+1);
  for (i=offset;i<mfqP->npmax*mfqP->npmax;i++) {
    mfqP->Z[i-offset] = mfqP->Q_tmp[i];
  }

  if (mfqP->nmodelpoints == mfqP->n + 1) {
    /* Set L to I_{n+1} */
    for (i=0;i<mfqP->npmax * mfqP->n*(mfqP->n+1)/2;i++) {
      mfqP->L[i] = 0.0;
    }
    for (i=0;i<mfqP->n;i++) {
      mfqP->L[(mfqP->n*(mfqP->n+1)/2)*i + i] = 1.0;
    }
  }
  PetscFunctionReturn(0);
}

/* Only call from modelimprove, addpoint() needs ->Q_tmp and ->work to be set */
static PetscErrorCode addpoint(Tao tao, TAO_POUNDERS *mfqP, PetscInt index)
{
  PetscFunctionBegin;
  /* Create new vector in history: X[newidx] = X[mfqP->index] + delta*X[index]*/
  CHKERRQ(VecDuplicate(mfqP->Xhist[0],&mfqP->Xhist[mfqP->nHist]));
  CHKERRQ(VecSetValues(mfqP->Xhist[mfqP->nHist],mfqP->n,mfqP->indices,&mfqP->Q_tmp[index*mfqP->npmax],INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(mfqP->Xhist[mfqP->nHist]));
  CHKERRQ(VecAssemblyEnd(mfqP->Xhist[mfqP->nHist]));
  CHKERRQ(VecAYPX(mfqP->Xhist[mfqP->nHist],mfqP->delta,mfqP->Xhist[mfqP->minindex]));

  /* Project into feasible region */
  if (tao->XU && tao->XL) {
    CHKERRQ(VecMedian(mfqP->Xhist[mfqP->nHist], tao->XL, tao->XU, mfqP->Xhist[mfqP->nHist]));
  }

  /* Compute value of new vector */
  CHKERRQ(VecDuplicate(mfqP->Fhist[0],&mfqP->Fhist[mfqP->nHist]));
  CHKMEMQ;
  CHKERRQ(pounders_feval(tao,mfqP->Xhist[mfqP->nHist],mfqP->Fhist[mfqP->nHist],&mfqP->Fres[mfqP->nHist]));

  /* Add new vector to model */
  mfqP->model_indices[mfqP->nmodelpoints] = mfqP->nHist;
  mfqP->nmodelpoints++;
  mfqP->nHist++;
  PetscFunctionReturn(0);
}

static PetscErrorCode modelimprove(Tao tao, TAO_POUNDERS *mfqP, PetscInt addallpoints)
{
  /* modeld = Q(:,np+1:n)' */
  PetscInt       i,j,minindex=0;
  PetscReal      dp,half=0.5,one=1.0,minvalue=PETSC_INFINITY;
  PetscBLASInt   blasn=mfqP->n,  blasnpmax = mfqP->npmax, blask,info;
  PetscBLASInt   blas1=1,blasnmax = mfqP->nmax;

  PetscFunctionBegin;
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
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("R","N",&blasn,&blasn,&blask,mfqP->Q,&blasnpmax,mfqP->tau, mfqP->Q_tmp, &blasnpmax, mfqP->npmaxwork,&blasnmax, &info));

  for (i=mfqP->nmodelpoints;i<mfqP->n;i++) {
    PetscStackCallBLAS("BLASdot",dp = BLASdot_(&blasn,&mfqP->Q_tmp[i*mfqP->npmax],&blas1,mfqP->Gres,&blas1));
    if (dp>0.0) { /* Model says use the other direction! */
      for (j=0;j<mfqP->n;j++) {
        mfqP->Q_tmp[i*mfqP->npmax+j] *= -1;
      }
    }
    /* mfqP->work[i] = Cres+Modeld(i,:)*(Gres+.5*Hres*Modeld(i,:)') */
    for (j=0;j<mfqP->n;j++) {
      mfqP->work2[j] = mfqP->Gres[j];
    }
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&blasn,&blasn,&half,mfqP->Hres,&blasn,&mfqP->Q_tmp[i*mfqP->npmax], &blas1, &one, mfqP->work2,&blas1));
    PetscStackCallBLAS("BLASdot",mfqP->work[i] = BLASdot_(&blasn,&mfqP->Q_tmp[i*mfqP->npmax],&blas1,mfqP->work2,&blas1));
    if (i==mfqP->nmodelpoints || mfqP->work[i] < minvalue) {
      minindex=i;
      minvalue = mfqP->work[i];
    }
    if (addallpoints != 0) {
      CHKERRQ(addpoint(tao,mfqP,i));
    }
  }
  if (!addallpoints) {
    CHKERRQ(addpoint(tao,mfqP,minindex));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode affpoints(TAO_POUNDERS *mfqP, PetscReal *xmin,PetscReal c)
{
  PetscInt        i,j;
  PetscBLASInt    blasm=mfqP->m,blasj,blask,blasn=mfqP->n,ione=1,info;
  PetscBLASInt    blasnpmax = mfqP->npmax,blasmaxmn;
  PetscReal       proj,normd;
  const PetscReal *x;

  PetscFunctionBegin;
  for (i=mfqP->nHist-1;i>=0;i--) {
    CHKERRQ(VecGetArrayRead(mfqP->Xhist[i],&x));
    for (j=0;j<mfqP->n;j++) {
      mfqP->work[j] = (x[j] - xmin[j])/mfqP->delta;
    }
    CHKERRQ(VecRestoreArrayRead(mfqP->Xhist[i],&x));
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn,mfqP->work,&ione,mfqP->work2,&ione));
    PetscStackCallBLAS("BLASnrm2",normd = BLASnrm2_(&blasn,mfqP->work,&ione));
    if (normd <= c) {
      blasj=PetscMax((mfqP->n - mfqP->nmodelpoints),0);
      if (!mfqP->q_is_I) {
        /* project D onto null */
        blask=(mfqP->nmodelpoints);
        PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("R","N",&ione,&blasn,&blask,mfqP->Q,&blasnpmax,mfqP->tau,mfqP->work2,&ione,mfqP->mwork,&blasm,&info));
        PetscCheck(info >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"ormqr returned value %d",info);
      }
      PetscStackCallBLAS("BLASnrm2",proj = BLASnrm2_(&blasj,&mfqP->work2[mfqP->nmodelpoints],&ione));

      if (proj >= mfqP->theta1) { /* add this index to model */
        mfqP->model_indices[mfqP->nmodelpoints]=i;
        mfqP->nmodelpoints++;
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn,mfqP->work,&ione,&mfqP->Q_tmp[mfqP->npmax*(mfqP->nmodelpoints-1)],&ione));
        blask=mfqP->npmax*(mfqP->nmodelpoints);
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&blask,mfqP->Q_tmp,&ione,mfqP->Q,&ione));
        blask = mfqP->nmodelpoints;
        blasmaxmn = PetscMax(mfqP->m,mfqP->n);
        PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&blasn,&blask,mfqP->Q,&blasnpmax,mfqP->tau,mfqP->mwork,&blasmaxmn,&info));
        PetscCheck(info >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"geqrf returned value %d",info);
        mfqP->q_is_I = 0;
      }
      if (mfqP->nmodelpoints == mfqP->n)  {
        break;
      }
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_POUNDERS(Tao tao)
{
  TAO_POUNDERS       *mfqP = (TAO_POUNDERS *)tao->data;
  PetscInt           i,ii,j,k,l;
  PetscReal          step=1.0;
  PetscInt           low,high;
  PetscReal          minnorm;
  PetscReal          *x,*f;
  const PetscReal    *xmint,*fmin;
  PetscReal          deltaold;
  PetscReal          gnorm;
  PetscBLASInt       info,ione=1,iblas;
  PetscBool          valid,same;
  PetscReal          mdec, rho, normxsp;
  PetscReal          one=1.0,zero=0.0,ratio;
  PetscBLASInt       blasm,blasn,blasncopy,blasnpmax;
  static PetscBool   set = PETSC_FALSE;

  /* n = # of parameters
     m = dimension (components) of function  */
  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister("@article{UNEDF0,\n"
                                 "title = {Nuclear energy density optimization},\n"
                                "author = {Kortelainen, M.  and Lesinski, T.  and Mor\'e, J.  and Nazarewicz, W.\n"
                                "          and Sarich, J.  and Schunck, N.  and Stoitsov, M. V. and Wild, S. },\n"
                                "journal = {Phys. Rev. C},\n"
                                "volume = {82},\n"
                                "number = {2},\n"
                                "pages = {024313},\n"
                                "numpages = {18},\n"
                                "year = {2010},\n"
                                "month = {Aug},\n"
                                 "doi = {10.1103/PhysRevC.82.024313}\n}\n",&set));
  tao->niter=0;
  if (tao->XL && tao->XU) {
    /* Check x0 <= XU */
    PetscReal val;

    CHKERRQ(VecCopy(tao->solution,mfqP->Xhist[0]));
    CHKERRQ(VecAXPY(mfqP->Xhist[0],-1.0,tao->XU));
    CHKERRQ(VecMax(mfqP->Xhist[0],NULL,&val));
    PetscCheck(val <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"X0 > upper bound");

    /* Check x0 >= xl */
    CHKERRQ(VecCopy(tao->XL,mfqP->Xhist[0]));
    CHKERRQ(VecAXPY(mfqP->Xhist[0],-1.0,tao->solution));
    CHKERRQ(VecMax(mfqP->Xhist[0],NULL,&val));
    PetscCheck(val <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"X0 < lower bound");

    /* Check x0 + delta < XU  -- should be able to get around this eventually */

    CHKERRQ(VecSet(mfqP->Xhist[0],mfqP->delta));
    CHKERRQ(VecAXPY(mfqP->Xhist[0],1.0,tao->solution));
    CHKERRQ(VecAXPY(mfqP->Xhist[0],-1.0,tao->XU));
    CHKERRQ(VecMax(mfqP->Xhist[0],NULL,&val));
    PetscCheck(val <= 1e-10,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"X0 + delta > upper bound");
  }

  blasm = mfqP->m; blasn=mfqP->n; blasnpmax = mfqP->npmax;
  for (i=0;i<mfqP->n*mfqP->n*mfqP->m;++i) mfqP->H[i]=0;

  CHKERRQ(VecCopy(tao->solution,mfqP->Xhist[0]));

  /* This provides enough information to approximate the gradient of the objective */
  /* using a forward difference scheme. */

  CHKERRQ(PetscInfo(tao,"Initialize simplex; delta = %10.9e\n",(double)mfqP->delta));
  CHKERRQ(pounders_feval(tao,mfqP->Xhist[0],mfqP->Fhist[0],&mfqP->Fres[0]));
  mfqP->minindex = 0;
  minnorm = mfqP->Fres[0];

  CHKERRQ(VecGetOwnershipRange(mfqP->Xhist[0],&low,&high));
  for (i=1;i<mfqP->n+1;++i) {
    CHKERRQ(VecCopy(mfqP->Xhist[0],mfqP->Xhist[i]));

    if (i-1 >= low && i-1 < high) {
      CHKERRQ(VecGetArray(mfqP->Xhist[i],&x));
      x[i-1-low] += mfqP->delta;
      CHKERRQ(VecRestoreArray(mfqP->Xhist[i],&x));
    }
    CHKMEMQ;
    CHKERRQ(pounders_feval(tao,mfqP->Xhist[i],mfqP->Fhist[i],&mfqP->Fres[i]));
    if (mfqP->Fres[i] < minnorm) {
      mfqP->minindex = i;
      minnorm = mfqP->Fres[i];
    }
  }
  CHKERRQ(VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution));
  CHKERRQ(VecCopy(mfqP->Fhist[mfqP->minindex],tao->ls_res));
  CHKERRQ(PetscInfo(tao,"Finalize simplex; minnorm = %10.9e\n",(double)minnorm));

  /* Gather mpi vecs to one big local vec */

  /* Begin serial code */

  /* Disp[i] = Xi-xmin, i=1,..,mfqP->minindex-1,mfqP->minindex+1,..,n */
  /* Fdiff[i] = (Fi-Fmin)', i=1,..,mfqP->minindex-1,mfqP->minindex+1,..,n */
  /* (Column oriented for blas calls) */
  ii=0;

  CHKERRQ(PetscInfo(tao,"Build matrix: %D\n",(PetscInt)mfqP->size));
  if (1 == mfqP->size) {
    CHKERRQ(VecGetArrayRead(mfqP->Xhist[mfqP->minindex],&xmint));
    for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i];
    CHKERRQ(VecRestoreArrayRead(mfqP->Xhist[mfqP->minindex],&xmint));
    CHKERRQ(VecGetArrayRead(mfqP->Fhist[mfqP->minindex],&fmin));
    for (i=0;i<mfqP->n+1;i++) {
      if (i == mfqP->minindex) continue;

      CHKERRQ(VecGetArray(mfqP->Xhist[i],&x));
      for (j=0;j<mfqP->n;j++) {
        mfqP->Disp[ii+mfqP->npmax*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
      }
      CHKERRQ(VecRestoreArray(mfqP->Xhist[i],&x));

      CHKERRQ(VecGetArray(mfqP->Fhist[i],&f));
      for (j=0;j<mfqP->m;j++) {
        mfqP->Fdiff[ii+mfqP->n*j] = f[j] - fmin[j];
      }
      CHKERRQ(VecRestoreArray(mfqP->Fhist[i],&f));

      mfqP->model_indices[ii++] = i;
    }
    for (j=0;j<mfqP->m;j++) {
      mfqP->C[j] = fmin[j];
    }
    CHKERRQ(VecRestoreArrayRead(mfqP->Fhist[mfqP->minindex],&fmin));
  } else {
    CHKERRQ(VecSet(mfqP->localxmin,0));
    CHKERRQ(VecScatterBegin(mfqP->scatterx,mfqP->Xhist[mfqP->minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mfqP->scatterx,mfqP->Xhist[mfqP->minindex],mfqP->localxmin,INSERT_VALUES,SCATTER_FORWARD));

    CHKERRQ(VecGetArrayRead(mfqP->localxmin,&xmint));
    for (i=0;i<mfqP->n;i++) mfqP->xmin[i] = xmint[i];
    CHKERRQ(VecRestoreArrayRead(mfqP->localxmin,&xmint));

    CHKERRQ(VecScatterBegin(mfqP->scatterf,mfqP->Fhist[mfqP->minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mfqP->scatterf,mfqP->Fhist[mfqP->minindex],mfqP->localfmin,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetArrayRead(mfqP->localfmin,&fmin));
    for (i=0;i<mfqP->n+1;i++) {
      if (i == mfqP->minindex) continue;

      CHKERRQ(VecScatterBegin(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,INSERT_VALUES, SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(mfqP->scatterx,mfqP->Xhist[ii],mfqP->localx,INSERT_VALUES, SCATTER_FORWARD));
      CHKERRQ(VecGetArray(mfqP->localx,&x));
      for (j=0;j<mfqP->n;j++) {
        mfqP->Disp[ii+mfqP->npmax*j] = (x[j] - mfqP->xmin[j])/mfqP->delta;
      }
      CHKERRQ(VecRestoreArray(mfqP->localx,&x));

      CHKERRQ(VecScatterBegin(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,INSERT_VALUES, SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(mfqP->scatterf,mfqP->Fhist[ii],mfqP->localf,INSERT_VALUES, SCATTER_FORWARD));
      CHKERRQ(VecGetArray(mfqP->localf,&f));
      for (j=0;j<mfqP->m;j++) {
        mfqP->Fdiff[ii+mfqP->n*j] = f[j] - fmin[j];
      }
      CHKERRQ(VecRestoreArray(mfqP->localf,&f));

      mfqP->model_indices[ii++] = i;
    }
    for (j=0;j<mfqP->m;j++) {
      mfqP->C[j] = fmin[j];
    }
    CHKERRQ(VecRestoreArrayRead(mfqP->localfmin,&fmin));
  }

  /* Determine the initial quadratic models */
  /* G = D(ModelIn,:) \ (F(ModelIn,1:m)-repmat(F(xkin,1:m),n,1)); */
  /* D (nxn) Fdiff (nxm)  => G (nxm) */
  blasncopy = blasn;
  PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&blasn,&blasm,mfqP->Disp,&blasnpmax,mfqP->iwork,mfqP->Fdiff,&blasncopy,&info));
  CHKERRQ(PetscInfo(tao,"Linear solve return: %D\n",(PetscInt)info));

  CHKERRQ(pounders_update_res(tao));

  valid = PETSC_TRUE;

  CHKERRQ(VecSetValues(tao->gradient,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(tao->gradient));
  CHKERRQ(VecAssemblyEnd(tao->gradient));
  CHKERRQ(VecNorm(tao->gradient,NORM_2,&gnorm));
  gnorm *= mfqP->delta;
  CHKERRQ(VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution));

  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,minnorm,gnorm,0.0,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,minnorm,gnorm,0.0,step));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));

  mfqP->nHist = mfqP->n+1;
  mfqP->nmodelpoints = mfqP->n+1;
  CHKERRQ(PetscInfo(tao,"Initial gradient: %20.19e\n",(double)gnorm));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscReal gnm = 1e-4;
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->niter++;
    /* Solve the subproblem min{Q(s): ||s|| <= 1.0} */
    CHKERRQ(gqtwrap(tao,&gnm,&mdec));
    /* Evaluate the function at the new point */

    for (i=0;i<mfqP->n;i++) {
        mfqP->work[i] = mfqP->Xsubproblem[i]*mfqP->delta + mfqP->xmin[i];
    }
    CHKERRQ(VecDuplicate(tao->solution,&mfqP->Xhist[mfqP->nHist]));
    CHKERRQ(VecDuplicate(tao->ls_res,&mfqP->Fhist[mfqP->nHist]));
    CHKERRQ(VecSetValues(mfqP->Xhist[mfqP->nHist],mfqP->n,mfqP->indices,mfqP->work,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(mfqP->Xhist[mfqP->nHist]));
    CHKERRQ(VecAssemblyEnd(mfqP->Xhist[mfqP->nHist]));

    CHKERRQ(pounders_feval(tao,mfqP->Xhist[mfqP->nHist],mfqP->Fhist[mfqP->nHist],&mfqP->Fres[mfqP->nHist]));

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

      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&blasn,&blasn,&one,mfqP->Hres,&blasn,mfqP->work,&ione,&zero,mfqP->work2,&ione));
      for (i=0;i<mfqP->n;i++) {
        mfqP->Gres[i] += mfqP->work2[i];
      }
      mfqP->minindex = mfqP->nHist-1;
      minnorm = mfqP->Fres[mfqP->minindex];
      CHKERRQ(VecCopy(mfqP->Fhist[mfqP->minindex],tao->ls_res));
      /* Change current center */
      CHKERRQ(VecGetArrayRead(mfqP->Xhist[mfqP->minindex],&xmint));
      for (i=0;i<mfqP->n;i++) {
        mfqP->xmin[i] = xmint[i];
      }
      CHKERRQ(VecRestoreArrayRead(mfqP->Xhist[mfqP->minindex],&xmint));
    }

    /* Evaluate at a model-improving point if necessary */
    if (valid == PETSC_FALSE) {
      mfqP->q_is_I = 1;
      mfqP->nmodelpoints = 0;
      CHKERRQ(affpoints(mfqP,mfqP->xmin,mfqP->c1));
      if (mfqP->nmodelpoints < mfqP->n) {
        CHKERRQ(PetscInfo(tao,"Model not valid -- model-improving\n"));
        CHKERRQ(modelimprove(tao,mfqP,1));
      }
    }

    /* Update the trust region radius */
    deltaold = mfqP->delta;
    normxsp = 0;
    for (i=0;i<mfqP->n;i++) {
      normxsp += mfqP->Xsubproblem[i]*mfqP->Xsubproblem[i];
    }
    normxsp = PetscSqrtReal(normxsp);
    if (rho >= mfqP->eta1 && normxsp > 0.5*mfqP->delta) {
      mfqP->delta = PetscMin(mfqP->delta*mfqP->gamma1,mfqP->deltamax);
    } else if (valid == PETSC_TRUE) {
      mfqP->delta = PetscMax(mfqP->delta*mfqP->gamma0,mfqP->deltamin);
    }

    /* Compute the next interpolation set */
    mfqP->q_is_I = 1;
    mfqP->nmodelpoints=0;
    CHKERRQ(PetscInfo(tao,"Affine Points: xmin = %20.19e, c1 = %20.19e\n",(double)*mfqP->xmin,(double)mfqP->c1));
    CHKERRQ(affpoints(mfqP,mfqP->xmin,mfqP->c1));
    if (mfqP->nmodelpoints == mfqP->n) {
      valid = PETSC_TRUE;
    } else {
      valid = PETSC_FALSE;
      CHKERRQ(PetscInfo(tao,"Affine Points: xmin = %20.19e, c2 = %20.19e\n",(double)*mfqP->xmin,(double)mfqP->c2));
      CHKERRQ(affpoints(mfqP,mfqP->xmin,mfqP->c2));
      if (mfqP->n > mfqP->nmodelpoints) {
        CHKERRQ(PetscInfo(tao,"Model not valid -- adding geometry points\n"));
        CHKERRQ(modelimprove(tao,mfqP,mfqP->n - mfqP->nmodelpoints));
      }
    }
    for (i=mfqP->nmodelpoints;i>0;i--) {
      mfqP->model_indices[i] = mfqP->model_indices[i-1];
    }
    mfqP->nmodelpoints++;
    mfqP->model_indices[0] = mfqP->minindex;
    CHKERRQ(morepoints(mfqP));
    for (i=0;i<mfqP->nmodelpoints;i++) {
      CHKERRQ(VecGetArray(mfqP->Xhist[mfqP->model_indices[i]],&x));
      for (j=0;j<mfqP->n;j++) {
        mfqP->Disp[i + mfqP->npmax*j] = (x[j]  - mfqP->xmin[j]) / deltaold;
      }
      CHKERRQ(VecRestoreArray(mfqP->Xhist[mfqP->model_indices[i]],&x));
      CHKERRQ(VecGetArray(mfqP->Fhist[mfqP->model_indices[i]],&f));
      for (j=0;j<mfqP->m;j++) {
        for (k=0;k<mfqP->n;k++)  {
          mfqP->work[k]=0.0;
          for (l=0;l<mfqP->n;l++) {
            mfqP->work[k] += mfqP->H[j + mfqP->m*(k + mfqP->n*l)] * mfqP->Disp[i + mfqP->npmax*l];
          }
        }
        PetscStackCallBLAS("BLASdot",mfqP->RES[j*mfqP->npmax + i] = -mfqP->C[j] - BLASdot_(&blasn,&mfqP->Fdiff[j*mfqP->n],&ione,&mfqP->Disp[i],&blasnpmax) - 0.5*BLASdot_(&blasn,mfqP->work,&ione,&mfqP->Disp[i],&blasnpmax) + f[j]);
      }
      CHKERRQ(VecRestoreArray(mfqP->Fhist[mfqP->model_indices[i]],&f));
    }

    /* Update the quadratic model */
    CHKERRQ(PetscInfo(tao,"Get Quad, size: %D, points: %D\n",mfqP->n,mfqP->nmodelpoints));
    CHKERRQ(getquadpounders(mfqP));
    CHKERRQ(VecGetArrayRead(mfqP->Fhist[mfqP->minindex],&fmin));
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasm,fmin,&ione,mfqP->C,&ione));
    /* G = G*(delta/deltaold) + Gdel */
    ratio = mfqP->delta/deltaold;
    iblas = blasm*blasn;
    PetscStackCallBLAS("BLASscal",BLASscal_(&iblas,&ratio,mfqP->Fdiff,&ione));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&iblas,&one,mfqP->Gdel,&ione,mfqP->Fdiff,&ione));
    /* H = H*(delta/deltaold)^2 + Hdel */
    iblas = blasm*blasn*blasn;
    ratio *= ratio;
    PetscStackCallBLAS("BLASscal",BLASscal_(&iblas,&ratio,mfqP->H,&ione));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&iblas,&one,mfqP->Hdel,&ione,mfqP->H,&ione));

    /* Get residuals */
    CHKERRQ(pounders_update_res(tao));

    /* Export solution and gradient residual to TAO */
    CHKERRQ(VecCopy(mfqP->Xhist[mfqP->minindex],tao->solution));
    CHKERRQ(VecSetValues(tao->gradient,mfqP->n,mfqP->indices,mfqP->Gres,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(tao->gradient));
    CHKERRQ(VecAssemblyEnd(tao->gradient));
    CHKERRQ(VecNorm(tao->gradient,NORM_2,&gnorm));
    gnorm *= mfqP->delta;
    /*  final criticality test */
    CHKERRQ(TaoLogConvergenceHistory(tao,minnorm,gnorm,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,minnorm,gnorm,0.0,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    /* test for repeated model */
    if (mfqP->nmodelpoints==mfqP->last_nmodelpoints) {
      same = PETSC_TRUE;
    } else {
      same = PETSC_FALSE;
    }
    for (i=0;i<mfqP->nmodelpoints;i++) {
      if (same) {
        if (mfqP->model_indices[i] == mfqP->last_model_indices[i]) {
          same = PETSC_TRUE;
        } else {
          same = PETSC_FALSE;
        }
      }
      mfqP->last_model_indices[i] = mfqP->model_indices[i];
    }
    mfqP->last_nmodelpoints = mfqP->nmodelpoints;
    if (same && mfqP->delta == deltaold) {
      CHKERRQ(PetscInfo(tao,"Identical model used in successive iterations\n"));
      tao->reason = TAO_CONVERGED_STEPTOL;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_POUNDERS(Tao tao)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;
  PetscInt       i,j;
  IS             isfloc,isfglob,isxloc,isxglob;

  PetscFunctionBegin;
  if (!tao->gradient) CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  if (!tao->stepdirection) CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  CHKERRQ(VecGetSize(tao->solution,&mfqP->n));
  CHKERRQ(VecGetSize(tao->ls_res,&mfqP->m));
  mfqP->c1 = PetscSqrtReal((PetscReal)mfqP->n);
  if (mfqP->npmax == PETSC_DEFAULT) {
    mfqP->npmax = 2*mfqP->n + 1;
  }
  mfqP->npmax = PetscMin((mfqP->n+1)*(mfqP->n+2)/2,mfqP->npmax);
  mfqP->npmax = PetscMax(mfqP->npmax, mfqP->n+2);

  CHKERRQ(PetscMalloc1(tao->max_funcs+100,&mfqP->Xhist));
  CHKERRQ(PetscMalloc1(tao->max_funcs+100,&mfqP->Fhist));
  for (i=0;i<mfqP->n+1;i++) {
    CHKERRQ(VecDuplicate(tao->solution,&mfqP->Xhist[i]));
    CHKERRQ(VecDuplicate(tao->ls_res,&mfqP->Fhist[i]));
  }
  CHKERRQ(VecDuplicate(tao->solution,&mfqP->workxvec));
  CHKERRQ(VecDuplicate(tao->ls_res,&mfqP->workfvec));
  mfqP->nHist = 0;

  CHKERRQ(PetscMalloc1(tao->max_funcs+100,&mfqP->Fres));
  CHKERRQ(PetscMalloc1(mfqP->npmax*mfqP->m,&mfqP->RES));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->work));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->work2));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->work3));
  CHKERRQ(PetscMalloc1(PetscMax(mfqP->m,mfqP->n+1),&mfqP->mwork));
  CHKERRQ(PetscMalloc1(mfqP->npmax - mfqP->n - 1,&mfqP->omega));
  CHKERRQ(PetscMalloc1(mfqP->n * (mfqP->n+1) / 2,&mfqP->beta));
  CHKERRQ(PetscMalloc1(mfqP->n + 1 ,&mfqP->alpha));

  CHKERRQ(PetscMalloc1(mfqP->n*mfqP->n*mfqP->m,&mfqP->H));
  CHKERRQ(PetscMalloc1(mfqP->npmax*mfqP->npmax,&mfqP->Q));
  CHKERRQ(PetscMalloc1(mfqP->npmax*mfqP->npmax,&mfqP->Q_tmp));
  CHKERRQ(PetscMalloc1(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax),&mfqP->L));
  CHKERRQ(PetscMalloc1(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax),&mfqP->L_tmp));
  CHKERRQ(PetscMalloc1(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax),&mfqP->L_save));
  CHKERRQ(PetscMalloc1(mfqP->n*(mfqP->n+1)/2*(mfqP->npmax),&mfqP->N));
  CHKERRQ(PetscMalloc1(mfqP->npmax * (mfqP->n+1) ,&mfqP->M));
  CHKERRQ(PetscMalloc1(mfqP->npmax * (mfqP->npmax - mfqP->n - 1) , &mfqP->Z));
  CHKERRQ(PetscMalloc1(mfqP->npmax,&mfqP->tau));
  CHKERRQ(PetscMalloc1(mfqP->npmax,&mfqP->tau_tmp));
  mfqP->nmax = PetscMax(5*mfqP->npmax,mfqP->n*(mfqP->n+1)/2);
  CHKERRQ(PetscMalloc1(mfqP->nmax,&mfqP->npmaxwork));
  CHKERRQ(PetscMalloc1(mfqP->nmax,&mfqP->npmaxiwork));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->xmin));
  CHKERRQ(PetscMalloc1(mfqP->m,&mfqP->C));
  CHKERRQ(PetscMalloc1(mfqP->m*mfqP->n,&mfqP->Fdiff));
  CHKERRQ(PetscMalloc1(mfqP->npmax*mfqP->n,&mfqP->Disp));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->Gres));
  CHKERRQ(PetscMalloc1(mfqP->n*mfqP->n,&mfqP->Hres));
  CHKERRQ(PetscMalloc1(mfqP->n*mfqP->n,&mfqP->Gpoints));
  CHKERRQ(PetscMalloc1(mfqP->npmax,&mfqP->model_indices));
  CHKERRQ(PetscMalloc1(mfqP->npmax,&mfqP->last_model_indices));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->Xsubproblem));
  CHKERRQ(PetscMalloc1(mfqP->m*mfqP->n,&mfqP->Gdel));
  CHKERRQ(PetscMalloc1(mfqP->n*mfqP->n*mfqP->m, &mfqP->Hdel));
  CHKERRQ(PetscMalloc1(PetscMax(mfqP->m,mfqP->n),&mfqP->indices));
  CHKERRQ(PetscMalloc1(mfqP->n,&mfqP->iwork));
  CHKERRQ(PetscMalloc1(mfqP->m*mfqP->m,&mfqP->w));
  for (i=0;i<mfqP->m;i++) {
    for (j=0;j<mfqP->m;j++) {
      if (i==j) {
        mfqP->w[i+mfqP->m*j]=1.0;
      } else {
        mfqP->w[i+mfqP->m*j]=0.0;
      }
    }
  }
  for (i=0;i<PetscMax(mfqP->m,mfqP->n);i++) {
    mfqP->indices[i] = i;
  }
  CHKERRMPI(MPI_Comm_size(((PetscObject)tao)->comm,&mfqP->size));
  if (mfqP->size > 1) {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mfqP->n,&mfqP->localx));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mfqP->n,&mfqP->localxmin));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mfqP->m,&mfqP->localf));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mfqP->m,&mfqP->localfmin));
    CHKERRQ(ISCreateStride(MPI_COMM_SELF,mfqP->n,0,1,&isxloc));
    CHKERRQ(ISCreateStride(MPI_COMM_SELF,mfqP->n,0,1,&isxglob));
    CHKERRQ(ISCreateStride(MPI_COMM_SELF,mfqP->m,0,1,&isfloc));
    CHKERRQ(ISCreateStride(MPI_COMM_SELF,mfqP->m,0,1,&isfglob));

    CHKERRQ(VecScatterCreate(tao->solution,isxglob,mfqP->localx,isxloc,&mfqP->scatterx));
    CHKERRQ(VecScatterCreate(tao->ls_res,isfglob,mfqP->localf,isfloc,&mfqP->scatterf));

    CHKERRQ(ISDestroy(&isxloc));
    CHKERRQ(ISDestroy(&isxglob));
    CHKERRQ(ISDestroy(&isfloc));
    CHKERRQ(ISDestroy(&isfglob));
  }

  if (!mfqP->usegqt) {
    KSP       ksp;
    PC        pc;
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,mfqP->n,mfqP->n,mfqP->Xsubproblem,&mfqP->subx));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mfqP->n,&mfqP->subxl));
    CHKERRQ(VecDuplicate(mfqP->subxl,&mfqP->subb));
    CHKERRQ(VecDuplicate(mfqP->subxl,&mfqP->subxu));
    CHKERRQ(VecDuplicate(mfqP->subxl,&mfqP->subpdel));
    CHKERRQ(VecDuplicate(mfqP->subxl,&mfqP->subndel));
    CHKERRQ(TaoCreate(PETSC_COMM_SELF,&mfqP->subtao));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)mfqP->subtao, (PetscObject)tao, 1));
    CHKERRQ(TaoSetType(mfqP->subtao,TAOBNTR));
    CHKERRQ(TaoSetOptionsPrefix(mfqP->subtao,"pounders_subsolver_"));
    CHKERRQ(TaoSetSolution(mfqP->subtao,mfqP->subx));
    CHKERRQ(TaoSetObjectiveAndGradient(mfqP->subtao,NULL,pounders_fg,(void*)mfqP));
    CHKERRQ(TaoSetMaximumIterations(mfqP->subtao,mfqP->gqt_maxits));
    CHKERRQ(TaoSetFromOptions(mfqP->subtao));
    CHKERRQ(TaoGetKSP(mfqP->subtao,&ksp));
    if (ksp) {
      CHKERRQ(KSPGetPC(ksp,&pc));
      CHKERRQ(PCSetType(pc,PCNONE));
    }
    CHKERRQ(TaoSetVariableBounds(mfqP->subtao,mfqP->subxl,mfqP->subxu));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,mfqP->n,mfqP->n,mfqP->Hres,&mfqP->subH));
    CHKERRQ(TaoSetHessian(mfqP->subtao,mfqP->subH,mfqP->subH,pounders_h,(void*)mfqP));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_POUNDERS(Tao tao)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (!mfqP->usegqt) {
    CHKERRQ(TaoDestroy(&mfqP->subtao));
    CHKERRQ(VecDestroy(&mfqP->subx));
    CHKERRQ(VecDestroy(&mfqP->subxl));
    CHKERRQ(VecDestroy(&mfqP->subxu));
    CHKERRQ(VecDestroy(&mfqP->subb));
    CHKERRQ(VecDestroy(&mfqP->subpdel));
    CHKERRQ(VecDestroy(&mfqP->subndel));
    CHKERRQ(MatDestroy(&mfqP->subH));
  }
  CHKERRQ(PetscFree(mfqP->Fres));
  CHKERRQ(PetscFree(mfqP->RES));
  CHKERRQ(PetscFree(mfqP->work));
  CHKERRQ(PetscFree(mfqP->work2));
  CHKERRQ(PetscFree(mfqP->work3));
  CHKERRQ(PetscFree(mfqP->mwork));
  CHKERRQ(PetscFree(mfqP->omega));
  CHKERRQ(PetscFree(mfqP->beta));
  CHKERRQ(PetscFree(mfqP->alpha));
  CHKERRQ(PetscFree(mfqP->H));
  CHKERRQ(PetscFree(mfqP->Q));
  CHKERRQ(PetscFree(mfqP->Q_tmp));
  CHKERRQ(PetscFree(mfqP->L));
  CHKERRQ(PetscFree(mfqP->L_tmp));
  CHKERRQ(PetscFree(mfqP->L_save));
  CHKERRQ(PetscFree(mfqP->N));
  CHKERRQ(PetscFree(mfqP->M));
  CHKERRQ(PetscFree(mfqP->Z));
  CHKERRQ(PetscFree(mfqP->tau));
  CHKERRQ(PetscFree(mfqP->tau_tmp));
  CHKERRQ(PetscFree(mfqP->npmaxwork));
  CHKERRQ(PetscFree(mfqP->npmaxiwork));
  CHKERRQ(PetscFree(mfqP->xmin));
  CHKERRQ(PetscFree(mfqP->C));
  CHKERRQ(PetscFree(mfqP->Fdiff));
  CHKERRQ(PetscFree(mfqP->Disp));
  CHKERRQ(PetscFree(mfqP->Gres));
  CHKERRQ(PetscFree(mfqP->Hres));
  CHKERRQ(PetscFree(mfqP->Gpoints));
  CHKERRQ(PetscFree(mfqP->model_indices));
  CHKERRQ(PetscFree(mfqP->last_model_indices));
  CHKERRQ(PetscFree(mfqP->Xsubproblem));
  CHKERRQ(PetscFree(mfqP->Gdel));
  CHKERRQ(PetscFree(mfqP->Hdel));
  CHKERRQ(PetscFree(mfqP->indices));
  CHKERRQ(PetscFree(mfqP->iwork));
  CHKERRQ(PetscFree(mfqP->w));
  for (i=0;i<mfqP->nHist;i++) {
    CHKERRQ(VecDestroy(&mfqP->Xhist[i]));
    CHKERRQ(VecDestroy(&mfqP->Fhist[i]));
  }
  CHKERRQ(VecDestroy(&mfqP->workxvec));
  CHKERRQ(VecDestroy(&mfqP->workfvec));
  CHKERRQ(PetscFree(mfqP->Xhist));
  CHKERRQ(PetscFree(mfqP->Fhist));

  if (mfqP->size > 1) {
    CHKERRQ(VecDestroy(&mfqP->localx));
    CHKERRQ(VecDestroy(&mfqP->localxmin));
    CHKERRQ(VecDestroy(&mfqP->localf));
    CHKERRQ(VecDestroy(&mfqP->localfmin));
  }
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_POUNDERS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"POUNDERS method for least-squares optimization"));
  CHKERRQ(PetscOptionsReal("-tao_pounders_delta","initial delta","",mfqP->delta,&mfqP->delta0,NULL));
  mfqP->delta = mfqP->delta0;
  CHKERRQ(PetscOptionsInt("-tao_pounders_npmax","max number of points in model","",mfqP->npmax,&mfqP->npmax,NULL));
  CHKERRQ(PetscOptionsBool("-tao_pounders_gqt","use gqt algorithm for subproblem","",mfqP->usegqt,&mfqP->usegqt,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_POUNDERS(Tao tao, PetscViewer viewer)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS *)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "initial delta: %g\n",(double)mfqP->delta0));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "final delta: %g\n",(double)mfqP->delta));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "model points: %D\n",mfqP->nmodelpoints));
    if (mfqP->usegqt) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "subproblem solver: gqt\n"));
    } else {
      CHKERRQ(TaoView(mfqP->subtao, viewer));
    }
  }
  PetscFunctionReturn(0);
}
/*MC
  TAOPOUNDERS - POUNDERS derivate-free model-based algorithm for nonlinear least squares

  Options Database Keys:
+ -tao_pounders_delta - initial step length
. -tao_pounders_npmax - maximum number of points in model
- -tao_pounders_gqt - use gqt algorithm for subproblem instead of TRON

  Level: beginner

M*/

PETSC_EXTERN PetscErrorCode TaoCreate_POUNDERS(Tao tao)
{
  TAO_POUNDERS   *mfqP = (TAO_POUNDERS*)tao->data;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_POUNDERS;
  tao->ops->solve = TaoSolve_POUNDERS;
  tao->ops->view = TaoView_POUNDERS;
  tao->ops->setfromoptions = TaoSetFromOptions_POUNDERS;
  tao->ops->destroy = TaoDestroy_POUNDERS;

  CHKERRQ(PetscNewLog(tao,&mfqP));
  tao->data = (void*)mfqP;
  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;
  mfqP->npmax = PETSC_DEFAULT;
  mfqP->delta0 = 0.1;
  mfqP->delta = 0.1;
  mfqP->deltamax=1e3;
  mfqP->deltamin=1e-6;
  mfqP->c2 = 10.0;
  mfqP->theta1=1e-5;
  mfqP->theta2=1e-4;
  mfqP->gamma0=0.5;
  mfqP->gamma1=2.0;
  mfqP->eta0 = 0.0;
  mfqP->eta1 = 0.1;
  mfqP->usegqt = PETSC_FALSE;
  mfqP->gqt_rtol = 0.001;
  mfqP->gqt_maxits = 50;
  mfqP->workxvec = NULL;
  PetscFunctionReturn(0);
}
