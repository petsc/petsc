#include <../src/tao/constrained/impls/ipm/pdipm.h>

/*
   TaoPDIPMEvaluateFunctionsAndJacobians - Evaluate the objective function f, gradient fx, constraints, and all the Jacobians at current vector

   Collective on tao

   Input Parameter:
+  tao - solver context
-  x - vector at which all objects to be evaluated

   Level: beginner

.seealso: TaoPDIPMUpdateConstraints(), TaoPDIPMSetUpBounds()
*/
static PetscErrorCode TaoPDIPMEvaluateFunctionsAndJacobians(Tao tao,Vec x)
{
  TAO_PDIPM      *pdipm=(TAO_PDIPM*)tao->data;

  PetscFunctionBegin;
  /* Compute user objective function and gradient */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao,x,&pdipm->obj,tao->gradient));

  /* Equality constraints and Jacobian */
  if (pdipm->Ng) {
    CHKERRQ(TaoComputeEqualityConstraints(tao,x,tao->constraints_equality));
    CHKERRQ(TaoComputeJacobianEquality(tao,x,tao->jacobian_equality,tao->jacobian_equality_pre));
  }

  /* Inequality constraints and Jacobian */
  if (pdipm->Nh) {
    CHKERRQ(TaoComputeInequalityConstraints(tao,x,tao->constraints_inequality));
    CHKERRQ(TaoComputeJacobianInequality(tao,x,tao->jacobian_inequality,tao->jacobian_inequality_pre));
  }
  PetscFunctionReturn(0);
}

/*
  TaoPDIPMUpdateConstraints - Update the vectors ce and ci at x

  Collective

  Input Parameter:
+ tao - Tao context
- x - vector at which constraints to be evaluated

   Level: beginner

.seealso: TaoPDIPMEvaluateFunctionsAndJacobians()
*/
static PetscErrorCode TaoPDIPMUpdateConstraints(Tao tao,Vec x)
{
  TAO_PDIPM         *pdipm=(TAO_PDIPM*)tao->data;
  PetscInt          i,offset,offset1,k,xstart;
  PetscScalar       *carr;
  const PetscInt    *ubptr,*lbptr,*bxptr,*fxptr;
  const PetscScalar *xarr,*xuarr,*xlarr,*garr,*harr;

  PetscFunctionBegin;
  CHKERRQ(VecGetOwnershipRange(x,&xstart,NULL));

  CHKERRQ(VecGetArrayRead(x,&xarr));
  CHKERRQ(VecGetArrayRead(tao->XU,&xuarr));
  CHKERRQ(VecGetArrayRead(tao->XL,&xlarr));

  /* (1) Update ce vector */
  CHKERRQ(VecGetArrayWrite(pdipm->ce,&carr));

  if (pdipm->Ng) {
    /* (1.a) Inserting updated g(x) */
    CHKERRQ(VecGetArrayRead(tao->constraints_equality,&garr));
    CHKERRQ(PetscMemcpy(carr,garr,pdipm->ng*sizeof(PetscScalar)));
    CHKERRQ(VecRestoreArrayRead(tao->constraints_equality,&garr));
  }

  /* (1.b) Update xfixed */
  if (pdipm->Nxfixed) {
    offset = pdipm->ng;
    CHKERRQ(ISGetIndices(pdipm->isxfixed,&fxptr)); /* global indices in x */
    for (k=0;k < pdipm->nxfixed;k++) {
      i = fxptr[k]-xstart;
      carr[offset + k] = xarr[i] - xuarr[i];
    }
  }
  CHKERRQ(VecRestoreArrayWrite(pdipm->ce,&carr));

  /* (2) Update ci vector */
  CHKERRQ(VecGetArrayWrite(pdipm->ci,&carr));

  if (pdipm->Nh) {
    /* (2.a) Inserting updated h(x) */
    CHKERRQ(VecGetArrayRead(tao->constraints_inequality,&harr));
    CHKERRQ(PetscMemcpy(carr,harr,pdipm->nh*sizeof(PetscScalar)));
    CHKERRQ(VecRestoreArrayRead(tao->constraints_inequality,&harr));
  }

  /* (2.b) Update xub */
  offset = pdipm->nh;
  if (pdipm->Nxub) {
    CHKERRQ(ISGetIndices(pdipm->isxub,&ubptr));
    for (k=0; k<pdipm->nxub; k++) {
      i = ubptr[k]-xstart;
      carr[offset + k] = xuarr[i] - xarr[i];
    }
  }

  if (pdipm->Nxlb) {
    /* (2.c) Update xlb */
    offset += pdipm->nxub;
    CHKERRQ(ISGetIndices(pdipm->isxlb,&lbptr)); /* global indices in x */
    for (k=0; k<pdipm->nxlb; k++) {
      i = lbptr[k]-xstart;
      carr[offset + k] = xarr[i] - xlarr[i];
    }
  }

  if (pdipm->Nxbox) {
    /* (2.d) Update xbox */
    offset += pdipm->nxlb;
    offset1 = offset + pdipm->nxbox;
    CHKERRQ(ISGetIndices(pdipm->isxbox,&bxptr)); /* global indices in x */
    for (k=0; k<pdipm->nxbox; k++) {
      i = bxptr[k]-xstart; /* local indices in x */
      carr[offset+k]  = xuarr[i] - xarr[i];
      carr[offset1+k] = xarr[i]  - xlarr[i];
    }
  }
  CHKERRQ(VecRestoreArrayWrite(pdipm->ci,&carr));

  /* Restoring Vectors */
  CHKERRQ(VecRestoreArrayRead(x,&xarr));
  CHKERRQ(VecRestoreArrayRead(tao->XU,&xuarr));
  CHKERRQ(VecRestoreArrayRead(tao->XL,&xlarr));
  PetscFunctionReturn(0);
}

/*
   TaoPDIPMSetUpBounds - Create upper and lower bound vectors of x

   Collective

   Input Parameter:
.  tao - holds pdipm and XL & XU

   Level: beginner

.seealso: TaoPDIPMUpdateConstraints
*/
static PetscErrorCode TaoPDIPMSetUpBounds(Tao tao)
{
  TAO_PDIPM         *pdipm=(TAO_PDIPM*)tao->data;
  const PetscScalar *xl,*xu;
  PetscInt          n,*ixlb,*ixub,*ixfixed,*ixfree,*ixbox,i,low,high,idx;
  MPI_Comm          comm;
  PetscInt          sendbuf[5],recvbuf[5];

  PetscFunctionBegin;
  /* Creates upper and lower bounds vectors on x, if not created already */
  CHKERRQ(TaoComputeVariableBounds(tao));

  CHKERRQ(VecGetLocalSize(tao->XL,&n));
  CHKERRQ(PetscMalloc5(n,&ixlb,n,&ixub,n,&ixfree,n,&ixfixed,n,&ixbox));

  CHKERRQ(VecGetOwnershipRange(tao->XL,&low,&high));
  CHKERRQ(VecGetArrayRead(tao->XL,&xl));
  CHKERRQ(VecGetArrayRead(tao->XU,&xu));
  for (i=0; i<n; i++) {
    idx = low + i;
    if ((PetscRealPart(xl[i]) > PETSC_NINFINITY) && (PetscRealPart(xu[i]) < PETSC_INFINITY)) {
      if (PetscRealPart(xl[i]) == PetscRealPart(xu[i])) {
        ixfixed[pdipm->nxfixed++]  = idx;
      } else ixbox[pdipm->nxbox++] = idx;
    } else {
      if ((PetscRealPart(xl[i]) > PETSC_NINFINITY) && (PetscRealPart(xu[i]) >= PETSC_INFINITY)) {
        ixlb[pdipm->nxlb++] = idx;
      } else if ((PetscRealPart(xl[i]) <= PETSC_NINFINITY) && (PetscRealPart(xu[i]) < PETSC_INFINITY)) {
        ixub[pdipm->nxlb++] = idx;
      } else  ixfree[pdipm->nxfree++] = idx;
    }
  }
  CHKERRQ(VecRestoreArrayRead(tao->XL,&xl));
  CHKERRQ(VecRestoreArrayRead(tao->XU,&xu));

  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  sendbuf[0] = pdipm->nxlb;
  sendbuf[1] = pdipm->nxub;
  sendbuf[2] = pdipm->nxfixed;
  sendbuf[3] = pdipm->nxbox;
  sendbuf[4] = pdipm->nxfree;

  CHKERRMPI(MPI_Allreduce(sendbuf,recvbuf,5,MPIU_INT,MPI_SUM,comm));
  pdipm->Nxlb    = recvbuf[0];
  pdipm->Nxub    = recvbuf[1];
  pdipm->Nxfixed = recvbuf[2];
  pdipm->Nxbox   = recvbuf[3];
  pdipm->Nxfree  = recvbuf[4];

  if (pdipm->Nxlb) {
    CHKERRQ(ISCreateGeneral(comm,pdipm->nxlb,ixlb,PETSC_COPY_VALUES,&pdipm->isxlb));
  }
  if (pdipm->Nxub) {
    CHKERRQ(ISCreateGeneral(comm,pdipm->nxub,ixub,PETSC_COPY_VALUES,&pdipm->isxub));
  }
  if (pdipm->Nxfixed) {
    CHKERRQ(ISCreateGeneral(comm,pdipm->nxfixed,ixfixed,PETSC_COPY_VALUES,&pdipm->isxfixed));
  }
  if (pdipm->Nxbox) {
    CHKERRQ(ISCreateGeneral(comm,pdipm->nxbox,ixbox,PETSC_COPY_VALUES,&pdipm->isxbox));
  }
  if (pdipm->Nxfree) {
    CHKERRQ(ISCreateGeneral(comm,pdipm->nxfree,ixfree,PETSC_COPY_VALUES,&pdipm->isxfree));
  }
  CHKERRQ(PetscFree5(ixlb,ixub,ixfixed,ixbox,ixfree));
  PetscFunctionReturn(0);
}

/*
   TaoPDIPMInitializeSolution - Initialize PDIPM solution X = [x; lambdae; lambdai; z].
   X consists of four subvectors in the order [x; lambdae; lambdai; z]. These
     four subvectors need to be initialized and its values copied over to X. Instead
     of copying, we use VecPlace/ResetArray functions to share the memory locations for
     X and the subvectors

   Collective

   Input Parameter:
.  tao - Tao context

   Level: beginner
*/
static PetscErrorCode TaoPDIPMInitializeSolution(Tao tao)
{
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscScalar       *Xarr,*z,*lambdai;
  PetscInt          i;
  const PetscScalar *xarr,*h;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayWrite(pdipm->X,&Xarr));

  /* Set Initialize X.x = tao->solution */
  CHKERRQ(VecGetArrayRead(tao->solution,&xarr));
  CHKERRQ(PetscMemcpy(Xarr,xarr,pdipm->nx*sizeof(PetscScalar)));
  CHKERRQ(VecRestoreArrayRead(tao->solution,&xarr));

  /* Initialize X.lambdae = 0.0 */
  if (pdipm->lambdae) {
    CHKERRQ(VecSet(pdipm->lambdae,0.0));
  }

  /* Initialize X.lambdai = push_init_lambdai, X.z = push_init_slack */
  if (pdipm->Nci) {
    CHKERRQ(VecSet(pdipm->lambdai,pdipm->push_init_lambdai));
    CHKERRQ(VecSet(pdipm->z,pdipm->push_init_slack));

    /* Additional modification for X.lambdai and X.z */
    CHKERRQ(VecGetArrayWrite(pdipm->lambdai,&lambdai));
    CHKERRQ(VecGetArrayWrite(pdipm->z,&z));
    if (pdipm->Nh) {
      CHKERRQ(VecGetArrayRead(tao->constraints_inequality,&h));
      for (i=0; i < pdipm->nh; i++) {
        if (h[i] < -pdipm->push_init_slack) z[i] = -h[i];
        if (pdipm->mu/z[i] > pdipm->push_init_lambdai) lambdai[i] = pdipm->mu/z[i];
      }
      CHKERRQ(VecRestoreArrayRead(tao->constraints_inequality,&h));
    }
    CHKERRQ(VecRestoreArrayWrite(pdipm->lambdai,&lambdai));
    CHKERRQ(VecRestoreArrayWrite(pdipm->z,&z));
  }

  CHKERRQ(VecRestoreArrayWrite(pdipm->X,&Xarr));
  PetscFunctionReturn(0);
}

/*
   TaoSNESJacobian_PDIPM - Evaluate the Hessian matrix at X

   Input Parameter:
   snes - SNES context
   X - KKT Vector
   *ctx - pdipm context

   Output Parameter:
   J - Hessian matrix
   Jpre - Preconditioner
*/
static PetscErrorCode TaoSNESJacobian_PDIPM(SNES snes,Vec X, Mat J, Mat Jpre, void *ctx)
{
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscInt          i,row,cols[2],Jrstart,rjstart,nc,j;
  const PetscInt    *aj,*ranges,*Jranges,*rranges,*cranges;
  const PetscScalar *Xarr,*aa;
  PetscScalar       vals[2];
  PetscInt          proc,nx_all,*nce_all=pdipm->nce_all;
  MPI_Comm          comm;
  PetscMPIInt       rank,size;
  Mat               jac_equality_trans=pdipm->jac_equality_trans,jac_inequality_trans=pdipm->jac_inequality_trans;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)snes,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_rank(comm,&size));

  CHKERRQ(MatGetOwnershipRanges(Jpre,&Jranges));
  CHKERRQ(MatGetOwnershipRange(Jpre,&Jrstart,NULL));
  CHKERRQ(MatGetOwnershipRangesColumn(tao->hessian,&rranges));
  CHKERRQ(MatGetOwnershipRangesColumn(tao->hessian,&cranges));

  CHKERRQ(VecGetArrayRead(X,&Xarr));

  /* (1) insert Z and Ci to the 4th block of Jpre -- overwrite existing values */
  if (pdipm->solve_symmetric_kkt) { /* 1 for eq 17 revised pdipm doc 0 for eq 18 (symmetric KKT) */
    vals[0] = 1.0;
    for (i=0; i < pdipm->nci; i++) {
        row     = Jrstart + pdipm->off_z + i;
        cols[0] = Jrstart + pdipm->off_lambdai + i;
        cols[1] = row;
        vals[1] = Xarr[pdipm->off_lambdai + i]/Xarr[pdipm->off_z + i];
        CHKERRQ(MatSetValues(Jpre,1,&row,2,cols,vals,INSERT_VALUES));
    }
  } else {
    for (i=0; i < pdipm->nci; i++) {
      row     = Jrstart + pdipm->off_z + i;
      cols[0] = Jrstart + pdipm->off_lambdai + i;
      cols[1] = row;
      vals[0] = Xarr[pdipm->off_z + i];
      vals[1] = Xarr[pdipm->off_lambdai + i];
      CHKERRQ(MatSetValues(Jpre,1,&row,2,cols,vals,INSERT_VALUES));
    }
  }

  /* (2) insert 2nd row block of Jpre: [ grad g, 0, 0, 0] */
  if (pdipm->Ng) {
    CHKERRQ(MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL));
    for (i=0; i<pdipm->ng; i++) {
      row = Jrstart + pdipm->off_lambdae + i;

      CHKERRQ(MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa));
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        CHKERRQ(MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES));
      }
      CHKERRQ(MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa));
      if (pdipm->kkt_pd) {
        /* add shift \delta_c */
        CHKERRQ(MatSetValue(Jpre,row,row,-pdipm->deltac,INSERT_VALUES));
      }
    }
  }

  /* (3) insert 3rd row block of Jpre: [ -grad h, 0, deltac, I] */
  if (pdipm->Nh) {
    CHKERRQ(MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL));
    for (i=0; i < pdipm->nh; i++) {
      row = Jrstart + pdipm->off_lambdai + i;
      CHKERRQ(MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa));
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        CHKERRQ(MatSetValue(Jpre,row,cols[0],-aa[j],INSERT_VALUES));
      }
      CHKERRQ(MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa));
      if (pdipm->kkt_pd) {
        /* add shift \delta_c */
        CHKERRQ(MatSetValue(Jpre,row,row,-pdipm->deltac,INSERT_VALUES));
      }
    }
  }

  /* (4) insert 1st row block of Jpre: [Wxx, grad g', -grad h', 0] */
  if (pdipm->Ng) { /* grad g' */
    CHKERRQ(MatTranspose(tao->jacobian_equality,MAT_REUSE_MATRIX,&jac_equality_trans));
  }
  if (pdipm->Nh) { /* grad h' */
    CHKERRQ(MatTranspose(tao->jacobian_inequality,MAT_REUSE_MATRIX,&jac_inequality_trans));
  }

  CHKERRQ(VecPlaceArray(pdipm->x,Xarr));
  CHKERRQ(TaoComputeHessian(tao,pdipm->x,tao->hessian,tao->hessian_pre));
  CHKERRQ(VecResetArray(pdipm->x));

  CHKERRQ(MatGetOwnershipRange(tao->hessian,&rjstart,NULL));
  for (i=0; i<pdipm->nx; i++) {
    row = Jrstart + i;

    /* insert Wxx = fxx + ... -- provided by user */
    CHKERRQ(MatGetRow(tao->hessian,i+rjstart,&nc,&aj,&aa));
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      cols[0] = aj[j] - cranges[proc] + Jranges[proc];
      if (row == cols[0] && pdipm->kkt_pd) {
        /* add shift deltaw to Wxx component */
        CHKERRQ(MatSetValue(Jpre,row,cols[0],aa[j]+pdipm->deltaw,INSERT_VALUES));
      } else {
        CHKERRQ(MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES));
      }
    }
    CHKERRQ(MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,&aa));

    /* insert grad g' */
    if (pdipm->ng) {
      CHKERRQ(MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa));
      CHKERRQ(MatGetOwnershipRanges(tao->jacobian_equality,&ranges));
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        CHKERRQ(MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES));
      }
      CHKERRQ(MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa));
    }

    /* insert -grad h' */
    if (pdipm->nh) {
      CHKERRQ(MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa));
      CHKERRQ(MatGetOwnershipRanges(tao->jacobian_inequality,&ranges));
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        CHKERRQ(MatSetValue(Jpre,row,cols[0],-aa[j],INSERT_VALUES));
      }
      CHKERRQ(MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa));
    }
  }
  CHKERRQ(VecRestoreArrayRead(X,&Xarr));

  /* (6) assemble Jpre and J */
  CHKERRQ(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));

  if (J != Jpre) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*
   TaoSnesFunction_PDIPM - Evaluate KKT function at X

   Input Parameter:
   snes - SNES context
   X - KKT Vector
   *ctx - pdipm

   Output Parameter:
   F - Updated Lagrangian vector
*/
static PetscErrorCode TaoSNESFunction_PDIPM(SNES snes,Vec X,Vec F,void *ctx)
{
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscScalar       *Farr;
  Vec               x,L1;
  PetscInt          i;
  const PetscScalar *Xarr,*carr,*zarr,*larr;

  PetscFunctionBegin;
  CHKERRQ(VecSet(F,0.0));

  CHKERRQ(VecGetArrayRead(X,&Xarr));
  CHKERRQ(VecGetArrayWrite(F,&Farr));

  /* (0) Evaluate f, fx, gradG, gradH at X.x Note: pdipm->x is not changed below */
  x = pdipm->x;
  CHKERRQ(VecPlaceArray(x,Xarr));
  CHKERRQ(TaoPDIPMEvaluateFunctionsAndJacobians(tao,x));

  /* Update ce, ci, and Jci at X.x */
  CHKERRQ(TaoPDIPMUpdateConstraints(tao,x));
  CHKERRQ(VecResetArray(x));

  /* (1) L1 = fx + (gradG'*DE + Jce_xfixed'*lambdae_xfixed) - (gradH'*DI + Jci_xb'*lambdai_xb) */
  L1 = pdipm->x;
  CHKERRQ(VecPlaceArray(L1,Farr)); /* L1 = 0.0 */
  if (pdipm->Nci) {
    if (pdipm->Nh) {
      /* L1 += gradH'*DI. Note: tao->DI is not changed below */
      CHKERRQ(VecPlaceArray(tao->DI,Xarr+pdipm->off_lambdai));
      CHKERRQ(MatMultTransposeAdd(tao->jacobian_inequality,tao->DI,L1,L1));
      CHKERRQ(VecResetArray(tao->DI));
    }

    /* L1 += Jci_xb'*lambdai_xb */
    CHKERRQ(VecPlaceArray(pdipm->lambdai_xb,Xarr+pdipm->off_lambdai+pdipm->nh));
    CHKERRQ(MatMultTransposeAdd(pdipm->Jci_xb,pdipm->lambdai_xb,L1,L1));
    CHKERRQ(VecResetArray(pdipm->lambdai_xb));

    /* L1 = - (gradH'*DI + Jci_xb'*lambdai_xb) */
    CHKERRQ(VecScale(L1,-1.0));
  }

  /* L1 += fx */
  CHKERRQ(VecAXPY(L1,1.0,tao->gradient));

  if (pdipm->Nce) {
    if (pdipm->Ng) {
      /* L1 += gradG'*DE. Note: tao->DE is not changed below */
      CHKERRQ(VecPlaceArray(tao->DE,Xarr+pdipm->off_lambdae));
      CHKERRQ(MatMultTransposeAdd(tao->jacobian_equality,tao->DE,L1,L1));
      CHKERRQ(VecResetArray(tao->DE));
    }
    if (pdipm->Nxfixed) {
      /* L1 += Jce_xfixed'*lambdae_xfixed */
      CHKERRQ(VecPlaceArray(pdipm->lambdae_xfixed,Xarr+pdipm->off_lambdae+pdipm->ng));
      CHKERRQ(MatMultTransposeAdd(pdipm->Jce_xfixed,pdipm->lambdae_xfixed,L1,L1));
      CHKERRQ(VecResetArray(pdipm->lambdae_xfixed));
    }
  }
  CHKERRQ(VecResetArray(L1));

  /* (2) L2 = ce(x) */
  if (pdipm->Nce) {
    CHKERRQ(VecGetArrayRead(pdipm->ce,&carr));
    for (i=0; i<pdipm->nce; i++) Farr[pdipm->off_lambdae + i] = carr[i];
    CHKERRQ(VecRestoreArrayRead(pdipm->ce,&carr));
  }

  if (pdipm->Nci) {
    if (pdipm->solve_symmetric_kkt) {
      /* (3) L3 = z - ci(x);
         (4) L4 = Lambdai * e - mu/z *e  */
      CHKERRQ(VecGetArrayRead(pdipm->ci,&carr));
      larr = Xarr+pdipm->off_lambdai;
      zarr = Xarr+pdipm->off_z;
      for (i=0; i<pdipm->nci; i++) {
        Farr[pdipm->off_lambdai + i] = zarr[i] - carr[i];
        Farr[pdipm->off_z       + i] = larr[i] - pdipm->mu/zarr[i];
      }
      CHKERRQ(VecRestoreArrayRead(pdipm->ci,&carr));
    } else {
      /* (3) L3 = z - ci(x);
         (4) L4 = Z * Lambdai * e - mu * e  */
      CHKERRQ(VecGetArrayRead(pdipm->ci,&carr));
      larr = Xarr+pdipm->off_lambdai;
      zarr = Xarr+pdipm->off_z;
      for (i=0; i<pdipm->nci; i++) {
        Farr[pdipm->off_lambdai + i] = zarr[i] - carr[i];
        Farr[pdipm->off_z       + i] = zarr[i]*larr[i] - pdipm->mu;
      }
      CHKERRQ(VecRestoreArrayRead(pdipm->ci,&carr));
    }
  }

  CHKERRQ(VecRestoreArrayRead(X,&Xarr));
  CHKERRQ(VecRestoreArrayWrite(F,&Farr));
  PetscFunctionReturn(0);
}

/*
  Evaluate F(X); then update update tao->gnorm0, tao->step = mu,
  tao->residual = norm2(F_x,F_z) and tao->cnorm = norm2(F_ce,F_ci).
*/
static PetscErrorCode TaoSNESFunction_PDIPM_residual(SNES snes,Vec X,Vec F,void *ctx)
{
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscScalar       *Farr,*tmparr;
  Vec               L1;
  PetscInt          i;
  PetscReal         res[2],cnorm[2];
  const PetscScalar *Xarr=NULL;

  PetscFunctionBegin;
  CHKERRQ(TaoSNESFunction_PDIPM(snes,X,F,(void*)tao));
  CHKERRQ(VecGetArrayWrite(F,&Farr));
  CHKERRQ(VecGetArrayRead(X,&Xarr));

  /* compute res[0] = norm2(F_x) */
  L1 = pdipm->x;
  CHKERRQ(VecPlaceArray(L1,Farr));
  CHKERRQ(VecNorm(L1,NORM_2,&res[0]));
  CHKERRQ(VecResetArray(L1));

  /* compute res[1] = norm2(F_z), cnorm[1] = norm2(F_ci) */
  if (pdipm->z) {
    if (pdipm->solve_symmetric_kkt) {
      CHKERRQ(VecPlaceArray(pdipm->z,Farr+pdipm->off_z));
      if (pdipm->Nci) {
        CHKERRQ(VecGetArrayWrite(pdipm->z,&tmparr));
        for (i=0; i<pdipm->nci; i++) tmparr[i] *= Xarr[pdipm->off_z + i];
        CHKERRQ(VecRestoreArrayWrite(pdipm->z,&tmparr));
      }

      CHKERRQ(VecNorm(pdipm->z,NORM_2,&res[1]));

      if (pdipm->Nci) {
        CHKERRQ(VecGetArrayWrite(pdipm->z,&tmparr));
        for (i=0; i<pdipm->nci; i++) {
          tmparr[i] /= Xarr[pdipm->off_z + i];
        }
        CHKERRQ(VecRestoreArrayWrite(pdipm->z,&tmparr));
      }
      CHKERRQ(VecResetArray(pdipm->z));
    } else { /* !solve_symmetric_kkt */
      CHKERRQ(VecPlaceArray(pdipm->z,Farr+pdipm->off_z));
      CHKERRQ(VecNorm(pdipm->z,NORM_2,&res[1]));
      CHKERRQ(VecResetArray(pdipm->z));
    }

    CHKERRQ(VecPlaceArray(pdipm->ci,Farr+pdipm->off_lambdai));
    CHKERRQ(VecNorm(pdipm->ci,NORM_2,&cnorm[1]));
    CHKERRQ(VecResetArray(pdipm->ci));
  } else {
    res[1] = 0.0; cnorm[1] = 0.0;
  }

  /* compute cnorm[0] = norm2(F_ce) */
  if (pdipm->Nce) {
    CHKERRQ(VecPlaceArray(pdipm->ce,Farr+pdipm->off_lambdae));
    CHKERRQ(VecNorm(pdipm->ce,NORM_2,&cnorm[0]));
    CHKERRQ(VecResetArray(pdipm->ce));
  } else cnorm[0] = 0.0;

  CHKERRQ(VecRestoreArrayWrite(F,&Farr));
  CHKERRQ(VecRestoreArrayRead(X,&Xarr));

  tao->gnorm0   = tao->residual;
  tao->residual = PetscSqrtReal(res[0]*res[0] + res[1]*res[1]);
  tao->cnorm    = PetscSqrtReal(cnorm[0]*cnorm[0] + cnorm[1]*cnorm[1]);
  tao->step     = pdipm->mu;
  PetscFunctionReturn(0);
}

/*
  KKTAddShifts - Check the inertia of Cholesky factor of KKT matrix.
  If it does not match the numbers of prime and dual variables, add shifts to the KKT matrix.
*/
static PetscErrorCode KKTAddShifts(Tao tao,SNES snes,Vec X)
{
  TAO_PDIPM      *pdipm = (TAO_PDIPM*)tao->data;
  KSP            ksp;
  PC             pc;
  Mat            Factor;
  PetscBool      isCHOL;
  PetscInt       nneg,nzero,npos;

  PetscFunctionBegin;
  /* Get the inertia of Cholesky factor */
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCCHOLESKY,&isCHOL));
  if (!isCHOL) PetscFunctionReturn(0);

  CHKERRQ(PCFactorGetMatrix(pc,&Factor));
  CHKERRQ(MatGetInertia(Factor,&nneg,&nzero,&npos));

  if (npos < pdipm->Nx+pdipm->Nci) {
    pdipm->deltaw = PetscMax(pdipm->lastdeltaw/3, 1.e-4*PETSC_MACHINE_EPSILON);
    CHKERRQ(PetscInfo(tao,"Test reduced deltaw=%g; previous MatInertia: nneg %D, nzero %D, npos %D(<%D)\n",(double)pdipm->deltaw,nneg,nzero,npos,pdipm->Nx+pdipm->Nci));
    CHKERRQ(TaoSNESJacobian_PDIPM(snes,X, pdipm->K, pdipm->K, tao));
    CHKERRQ(PCSetUp(pc));
    CHKERRQ(MatGetInertia(Factor,&nneg,&nzero,&npos));

    if (npos < pdipm->Nx+pdipm->Nci) {
      pdipm->deltaw = pdipm->lastdeltaw; /* in case reduction update does not help, this prevents that step from impacting increasing update */
      while (npos < pdipm->Nx+pdipm->Nci && pdipm->deltaw <= 1./PETSC_SMALL) { /* increase deltaw */
        CHKERRQ(PetscInfo(tao,"  deltaw=%g fails, MatInertia: nneg %D, nzero %D, npos %D(<%D)\n",(double)pdipm->deltaw,nneg,nzero,npos,pdipm->Nx+pdipm->Nci));
        pdipm->deltaw = PetscMin(8*pdipm->deltaw,PetscPowReal(10,20));
        CHKERRQ(TaoSNESJacobian_PDIPM(snes,X, pdipm->K, pdipm->K, tao));
        CHKERRQ(PCSetUp(pc));
        CHKERRQ(MatGetInertia(Factor,&nneg,&nzero,&npos));
      }

      PetscCheck(pdipm->deltaw < 1./PETSC_SMALL,PetscObjectComm((PetscObject)tao),PETSC_ERR_CONV_FAILED,"Reached maximum delta w will not converge, try different initial x0");

      CHKERRQ(PetscInfo(tao,"Updated deltaw %g\n",(double)pdipm->deltaw));
      pdipm->lastdeltaw = pdipm->deltaw;
      pdipm->deltaw     = 0.0;
    }
  }

  if (nzero) { /* Jacobian is singular */
    if (pdipm->deltac == 0.0) {
      pdipm->deltac = PETSC_SQRT_MACHINE_EPSILON;
    } else {
      pdipm->deltac = pdipm->deltac*PetscPowReal(pdipm->mu,.25);
    }
    CHKERRQ(PetscInfo(tao,"Updated deltac=%g, MatInertia: nneg %D, nzero %D(!=0), npos %D\n",(double)pdipm->deltac,nneg,nzero,npos));
    CHKERRQ(TaoSNESJacobian_PDIPM(snes,X, pdipm->K, pdipm->K, tao));
    CHKERRQ(PCSetUp(pc));
    CHKERRQ(MatGetInertia(Factor,&nneg,&nzero,&npos));
  }
  PetscFunctionReturn(0);
}

/*
  PCPreSolve_PDIPM -- called betwee MatFactorNumeric() and MatSolve()
*/
PetscErrorCode PCPreSolve_PDIPM(PC pc,KSP ksp)
{
  Tao            tao;
  TAO_PDIPM      *pdipm;

  PetscFunctionBegin;
  CHKERRQ(KSPGetApplicationContext(ksp,&tao));
  pdipm = (TAO_PDIPM*)tao->data;
  CHKERRQ(KKTAddShifts(tao,pdipm->snes,pdipm->X));
  PetscFunctionReturn(0);
}

/*
   SNESLineSearch_PDIPM - Custom line search used with PDIPM.

   Collective on TAO

   Notes:
   This routine employs a simple backtracking line-search to keep
   the slack variables (z) and inequality constraints Lagrange multipliers
   (lambdai) positive, i.e., z,lambdai >=0. It does this by calculating scalars
   alpha_p and alpha_d to keep z,lambdai non-negative. The decision (x), and the
   slack variables are updated as X = X - alpha_d*dx. The constraint multipliers
   are updated as Lambdai = Lambdai + alpha_p*dLambdai. The barrier parameter mu
   is also updated as mu = mu + z'lambdai/Nci
*/
static PetscErrorCode SNESLineSearch_PDIPM(SNESLineSearch linesearch,void *ctx)
{
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  SNES              snes;
  Vec               X,F,Y;
  PetscInt          i,iter;
  PetscReal         alpha_p=1.0,alpha_d=1.0,alpha[4];
  PetscScalar       *Xarr,*z,*lambdai,dot,*taosolarr;
  const PetscScalar *dXarr,*dz,*dlambdai;

  PetscFunctionBegin;
  CHKERRQ(SNESLineSearchGetSNES(linesearch,&snes));
  CHKERRQ(SNESGetIterationNumber(snes,&iter));

  CHKERRQ(SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED));
  CHKERRQ(SNESLineSearchGetVecs(linesearch,&X,&F,&Y,NULL,NULL));

  CHKERRQ(VecGetArrayWrite(X,&Xarr));
  CHKERRQ(VecGetArrayRead(Y,&dXarr));
  z  = Xarr + pdipm->off_z;
  dz = dXarr + pdipm->off_z;
  for (i=0; i < pdipm->nci; i++) {
    if (z[i] - dz[i] < 0.0) alpha_p = PetscMin(alpha_p, 0.9999*z[i]/dz[i]);
  }

  lambdai  = Xarr + pdipm->off_lambdai;
  dlambdai = dXarr + pdipm->off_lambdai;

  for (i=0; i<pdipm->nci; i++) {
    if (lambdai[i] - dlambdai[i] < 0.0) alpha_d = PetscMin(0.9999*lambdai[i]/dlambdai[i], alpha_d);
  }

  alpha[0] = alpha_p;
  alpha[1] = alpha_d;
  CHKERRQ(VecRestoreArrayRead(Y,&dXarr));
  CHKERRQ(VecRestoreArrayWrite(X,&Xarr));

  /* alpha = min(alpha) over all processes */
  CHKERRMPI(MPI_Allreduce(alpha,alpha+2,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)tao)));

  alpha_p = alpha[2];
  alpha_d = alpha[3];

  /* X = X - alpha * Y */
  CHKERRQ(VecGetArrayWrite(X,&Xarr));
  CHKERRQ(VecGetArrayRead(Y,&dXarr));
  for (i=0; i<pdipm->nx; i++) Xarr[i] -= alpha_p * dXarr[i];
  for (i=0; i<pdipm->nce; i++) Xarr[i+pdipm->off_lambdae] -= alpha_d * dXarr[i+pdipm->off_lambdae];

  for (i=0; i<pdipm->nci; i++) {
    Xarr[i+pdipm->off_lambdai] -= alpha_d * dXarr[i+pdipm->off_lambdai];
    Xarr[i+pdipm->off_z]       -= alpha_p * dXarr[i+pdipm->off_z];
  }
  CHKERRQ(VecGetArrayWrite(tao->solution,&taosolarr));
  CHKERRQ(PetscMemcpy(taosolarr,Xarr,pdipm->nx*sizeof(PetscScalar)));
  CHKERRQ(VecRestoreArrayWrite(tao->solution,&taosolarr));

  CHKERRQ(VecRestoreArrayWrite(X,&Xarr));
  CHKERRQ(VecRestoreArrayRead(Y,&dXarr));

  /* Update mu = mu_update_factor * dot(z,lambdai)/pdipm->nci at updated X */
  if (pdipm->z) {
    CHKERRQ(VecDot(pdipm->z,pdipm->lambdai,&dot));
  } else dot = 0.0;

  /* if (PetscAbsReal(pdipm->gradL) < 0.9*pdipm->mu)  */
  pdipm->mu = pdipm->mu_update_factor * dot/pdipm->Nci;

  /* Update F; get tao->residual and tao->cnorm */
  CHKERRQ(TaoSNESFunction_PDIPM_residual(snes,X,F,(void*)tao));

  tao->niter++;
  CHKERRQ(TaoLogConvergenceHistory(tao,pdipm->obj,tao->residual,tao->cnorm,tao->niter));
  CHKERRQ(TaoMonitor(tao,tao->niter,pdipm->obj,tao->residual,tao->cnorm,pdipm->mu));

  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason) {
    CHKERRQ(SNESSetConvergedReason(snes,SNES_CONVERGED_FNORM_ABS));
  }
  PetscFunctionReturn(0);
}

/*
   TaoSolve_PDIPM

   Input Parameter:
   tao - TAO context

   Output Parameter:
   tao - TAO context
*/
PetscErrorCode TaoSolve_PDIPM(Tao tao)
{
  TAO_PDIPM          *pdipm = (TAO_PDIPM*)tao->data;
  SNESLineSearch     linesearch; /* SNESLineSearch context */
  Vec                dummy;

  PetscFunctionBegin;
  PetscCheck(tao->constraints_equality || tao->constraints_inequality,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_NULL,"Equality and inequality constraints are not set. Either set them or switch to a different algorithm");

  /* Initialize all variables */
  CHKERRQ(TaoPDIPMInitializeSolution(tao));

  /* Set linesearch */
  CHKERRQ(SNESGetLineSearch(pdipm->snes,&linesearch));
  CHKERRQ(SNESLineSearchSetType(linesearch,SNESLINESEARCHSHELL));
  CHKERRQ(SNESLineSearchShellSetUserFunc(linesearch,SNESLineSearch_PDIPM,tao));
  CHKERRQ(SNESLineSearchSetFromOptions(linesearch));

  tao->reason = TAO_CONTINUE_ITERATING;

  /* -tao_monitor for iteration 0 and check convergence */
  CHKERRQ(VecDuplicate(pdipm->X,&dummy));
  CHKERRQ(TaoSNESFunction_PDIPM_residual(pdipm->snes,pdipm->X,dummy,(void*)tao));

  CHKERRQ(TaoLogConvergenceHistory(tao,pdipm->obj,tao->residual,tao->cnorm,tao->niter));
  CHKERRQ(TaoMonitor(tao,tao->niter,pdipm->obj,tao->residual,tao->cnorm,pdipm->mu));
  CHKERRQ(VecDestroy(&dummy));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason) {
    CHKERRQ(SNESSetConvergedReason(pdipm->snes,SNES_CONVERGED_FNORM_ABS));
  }

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    SNESConvergedReason reason;
    CHKERRQ(SNESSolve(pdipm->snes,NULL,pdipm->X));

    /* Check SNES convergence */
    CHKERRQ(SNESGetConvergedReason(pdipm->snes,&reason));
    if (reason < 0) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)pdipm->snes),"SNES solve did not converged due to reason %D\n",reason));
    }

    /* Check TAO convergence */
    PetscCheck(!PetscIsInfOrNanReal(pdipm->obj),PETSC_COMM_SELF,PETSC_ERR_SUP,"User-provided compute function generated Inf or NaN");
  }
  PetscFunctionReturn(0);
}

/*
  TaoView_PDIPM - View PDIPM

   Input Parameter:
    tao - TAO object
    viewer - PetscViewer

   Output:
*/
PetscErrorCode TaoView_PDIPM(Tao tao,PetscViewer viewer)
{
  TAO_PDIPM      *pdipm = (TAO_PDIPM *)tao->data;

  PetscFunctionBegin;
  tao->constrained = PETSC_TRUE;
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Number of prime=%D, Number of dual=%D\n",pdipm->Nx+pdipm->Nci,pdipm->Nce + pdipm->Nci));
  if (pdipm->kkt_pd) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"KKT shifts deltaw=%g, deltac=%g\n",(double)pdipm->deltaw,(double)pdipm->deltac));
  }
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

/*
   TaoSetup_PDIPM - Sets up tao and pdipm

   Input Parameter:
   tao - TAO object

   Output:   pdipm - initialized object
*/
PetscErrorCode TaoSetup_PDIPM(Tao tao)
{
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  PetscMPIInt       size;
  PetscInt          row,col,Jcrstart,Jcrend,k,tmp,nc,proc,*nh_all,*ng_all;
  PetscInt          offset,*xa,*xb,i,j,rstart,rend;
  PetscScalar       one=1.0,neg_one=-1.0;
  const PetscInt    *cols,*rranges,*cranges,*aj,*ranges;
  const PetscScalar *aa,*Xarr;
  Mat               J,jac_equality_trans,jac_inequality_trans;
  Mat               Jce_xfixed_trans,Jci_xb_trans;
  PetscInt          *dnz,*onz,rjstart,nx_all,*nce_all,*Jranges,cols1[2];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  /* (1) Setup Bounds and create Tao vectors */
  CHKERRQ(TaoPDIPMSetUpBounds(tao));

  if (!tao->gradient) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
    CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  }

  /* (2) Get sizes */
  /* Size of vector x - This is set by TaoSetSolution */
  CHKERRQ(VecGetSize(tao->solution,&pdipm->Nx));
  CHKERRQ(VecGetLocalSize(tao->solution,&pdipm->nx));

  /* Size of equality constraints and vectors */
  if (tao->constraints_equality) {
    CHKERRQ(VecGetSize(tao->constraints_equality,&pdipm->Ng));
    CHKERRQ(VecGetLocalSize(tao->constraints_equality,&pdipm->ng));
  } else {
    pdipm->ng = pdipm->Ng = 0;
  }

  pdipm->nce = pdipm->ng + pdipm->nxfixed;
  pdipm->Nce = pdipm->Ng + pdipm->Nxfixed;

  /* Size of inequality constraints and vectors */
  if (tao->constraints_inequality) {
    CHKERRQ(VecGetSize(tao->constraints_inequality,&pdipm->Nh));
    CHKERRQ(VecGetLocalSize(tao->constraints_inequality,&pdipm->nh));
  } else {
    pdipm->nh = pdipm->Nh = 0;
  }

  pdipm->nci = pdipm->nh + pdipm->nxlb + pdipm->nxub + 2*pdipm->nxbox;
  pdipm->Nci = pdipm->Nh + pdipm->Nxlb + pdipm->Nxub + 2*pdipm->Nxbox;

  /* Full size of the KKT system to be solved */
  pdipm->n = pdipm->nx + pdipm->nce + 2*pdipm->nci;
  pdipm->N = pdipm->Nx + pdipm->Nce + 2*pdipm->Nci;

  /* (3) Offsets for subvectors */
  pdipm->off_lambdae = pdipm->nx;
  pdipm->off_lambdai = pdipm->off_lambdae + pdipm->nce;
  pdipm->off_z       = pdipm->off_lambdai + pdipm->nci;

  /* (4) Create vectors and subvectors */
  /* Ce and Ci vectors */
  CHKERRQ(VecCreate(comm,&pdipm->ce));
  CHKERRQ(VecSetSizes(pdipm->ce,pdipm->nce,pdipm->Nce));
  CHKERRQ(VecSetFromOptions(pdipm->ce));

  CHKERRQ(VecCreate(comm,&pdipm->ci));
  CHKERRQ(VecSetSizes(pdipm->ci,pdipm->nci,pdipm->Nci));
  CHKERRQ(VecSetFromOptions(pdipm->ci));

  /* X=[x; lambdae; lambdai; z] for the big KKT system */
  CHKERRQ(VecCreate(comm,&pdipm->X));
  CHKERRQ(VecSetSizes(pdipm->X,pdipm->n,pdipm->N));
  CHKERRQ(VecSetFromOptions(pdipm->X));

  /* Subvectors; they share local arrays with X */
  CHKERRQ(VecGetArrayRead(pdipm->X,&Xarr));
  /* x shares local array with X.x */
  if (pdipm->Nx) {
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->nx,pdipm->Nx,Xarr,&pdipm->x));
  }

  /* lambdae shares local array with X.lambdae */
  if (pdipm->Nce) {
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->nce,pdipm->Nce,Xarr+pdipm->off_lambdae,&pdipm->lambdae));
  }

  /* tao->DE shares local array with X.lambdae_g */
  if (pdipm->Ng) {
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->ng,pdipm->Ng,Xarr+pdipm->off_lambdae,&tao->DE));

    CHKERRQ(VecCreate(comm,&pdipm->lambdae_xfixed));
    CHKERRQ(VecSetSizes(pdipm->lambdae_xfixed,pdipm->nxfixed,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(pdipm->lambdae_xfixed));
  }

  if (pdipm->Nci) {
    /* lambdai shares local array with X.lambdai */
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->nci,pdipm->Nci,Xarr+pdipm->off_lambdai,&pdipm->lambdai));

    /* z for slack variables; it shares local array with X.z */
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->nci,pdipm->Nci,Xarr+pdipm->off_z,&pdipm->z));
  }

  /* tao->DI which shares local array with X.lambdai_h */
  if (pdipm->Nh) {
    CHKERRQ(VecCreateMPIWithArray(comm,1,pdipm->nh,pdipm->Nh,Xarr+pdipm->off_lambdai,&tao->DI));
  }
  CHKERRQ(VecCreate(comm,&pdipm->lambdai_xb));
  CHKERRQ(VecSetSizes(pdipm->lambdai_xb,(pdipm->nci - pdipm->nh),PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(pdipm->lambdai_xb));

  CHKERRQ(VecRestoreArrayRead(pdipm->X,&Xarr));

  /* (5) Create Jacobians Jce_xfixed and Jci */
  /* (5.1) PDIPM Jacobian of equality bounds cebound(x) = J_nxfixed */
  if (pdipm->Nxfixed) {
    /* Create Jce_xfixed */
    CHKERRQ(MatCreate(comm,&pdipm->Jce_xfixed));
    CHKERRQ(MatSetSizes(pdipm->Jce_xfixed,pdipm->nxfixed,pdipm->nx,PETSC_DECIDE,pdipm->Nx));
    CHKERRQ(MatSetFromOptions(pdipm->Jce_xfixed));
    CHKERRQ(MatSeqAIJSetPreallocation(pdipm->Jce_xfixed,1,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(pdipm->Jce_xfixed,1,NULL,1,NULL));

    CHKERRQ(MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,&Jcrend));
    CHKERRQ(ISGetIndices(pdipm->isxfixed,&cols));
    k = 0;
    for (row = Jcrstart; row < Jcrend; row++) {
      CHKERRQ(MatSetValues(pdipm->Jce_xfixed,1,&row,1,cols+k,&one,INSERT_VALUES));
      k++;
    }
    CHKERRQ(ISRestoreIndices(pdipm->isxfixed, &cols));
    CHKERRQ(MatAssemblyBegin(pdipm->Jce_xfixed,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(pdipm->Jce_xfixed,MAT_FINAL_ASSEMBLY));
  }

  /* (5.2) PDIPM inequality Jacobian Jci = [tao->jacobian_inequality; ...] */
  CHKERRQ(MatCreate(comm,&pdipm->Jci_xb));
  CHKERRQ(MatSetSizes(pdipm->Jci_xb,pdipm->nci-pdipm->nh,pdipm->nx,PETSC_DECIDE,pdipm->Nx));
  CHKERRQ(MatSetFromOptions(pdipm->Jci_xb));
  CHKERRQ(MatSeqAIJSetPreallocation(pdipm->Jci_xb,1,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(pdipm->Jci_xb,1,NULL,1,NULL));

  CHKERRQ(MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,&Jcrend));
  offset = Jcrstart;
  if (pdipm->Nxub) {
    /* Add xub to Jci_xb */
    CHKERRQ(ISGetIndices(pdipm->isxub,&cols));
    k = 0;
    for (row = offset; row < offset + pdipm->nxub; row++) {
      CHKERRQ(MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES));
      k++;
    }
    CHKERRQ(ISRestoreIndices(pdipm->isxub, &cols));
  }

  if (pdipm->Nxlb) {
    /* Add xlb to Jci_xb */
    CHKERRQ(ISGetIndices(pdipm->isxlb,&cols));
    k = 0;
    offset += pdipm->nxub;
    for (row = offset; row < offset + pdipm->nxlb; row++) {
      CHKERRQ(MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&one,INSERT_VALUES));
      k++;
    }
    CHKERRQ(ISRestoreIndices(pdipm->isxlb, &cols));
  }

  /* Add xbox to Jci_xb */
  if (pdipm->Nxbox) {
    CHKERRQ(ISGetIndices(pdipm->isxbox,&cols));
    k = 0;
    offset += pdipm->nxlb;
    for (row = offset; row < offset + pdipm->nxbox; row++) {
      CHKERRQ(MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES));
      tmp = row + pdipm->nxbox;
      CHKERRQ(MatSetValues(pdipm->Jci_xb,1,&tmp,1,cols+k,&one,INSERT_VALUES));
      k++;
    }
    CHKERRQ(ISRestoreIndices(pdipm->isxbox, &cols));
  }

  CHKERRQ(MatAssemblyBegin(pdipm->Jci_xb,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(pdipm->Jci_xb,MAT_FINAL_ASSEMBLY));
  /* CHKERRQ(MatView(pdipm->Jci_xb,PETSC_VIEWER_STDOUT_WORLD)); */

  /* (6) Set up ISs for PC Fieldsplit */
  if (pdipm->solve_reduced_kkt) {
    CHKERRQ(PetscMalloc2(pdipm->nx+pdipm->nce,&xa,2*pdipm->nci,&xb));
    for (i=0; i < pdipm->nx + pdipm->nce; i++) xa[i] = i;
    for (i=0; i < 2*pdipm->nci; i++) xb[i] = pdipm->off_lambdai + i;

    CHKERRQ(ISCreateGeneral(comm,pdipm->nx+pdipm->nce,xa,PETSC_OWN_POINTER,&pdipm->is1));
    CHKERRQ(ISCreateGeneral(comm,2*pdipm->nci,xb,PETSC_OWN_POINTER,&pdipm->is2));
  }

  /* (7) Gather offsets from all processes */
  CHKERRQ(PetscMalloc1(size,&pdipm->nce_all));

  /* Get rstart of KKT matrix */
  CHKERRMPI(MPI_Scan(&pdipm->n,&rstart,1,MPIU_INT,MPI_SUM,comm));
  rstart -= pdipm->n;

  CHKERRMPI(MPI_Allgather(&pdipm->nce,1,MPIU_INT,pdipm->nce_all,1,MPIU_INT,comm));

  CHKERRQ(PetscMalloc3(size,&ng_all,size,&nh_all,size,&Jranges));
  CHKERRMPI(MPI_Allgather(&rstart,1,MPIU_INT,Jranges,1,MPIU_INT,comm));
  CHKERRMPI(MPI_Allgather(&pdipm->nh,1,MPIU_INT,nh_all,1,MPIU_INT,comm));
  CHKERRMPI(MPI_Allgather(&pdipm->ng,1,MPIU_INT,ng_all,1,MPIU_INT,comm));

  CHKERRQ(MatGetOwnershipRanges(tao->hessian,&rranges));
  CHKERRQ(MatGetOwnershipRangesColumn(tao->hessian,&cranges));

  if (pdipm->Ng) {
    CHKERRQ(TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre));
    CHKERRQ(MatTranspose(tao->jacobian_equality,MAT_INITIAL_MATRIX,&pdipm->jac_equality_trans));
  }
  if (pdipm->Nh) {
    CHKERRQ(TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre));
    CHKERRQ(MatTranspose(tao->jacobian_inequality,MAT_INITIAL_MATRIX,&pdipm->jac_inequality_trans));
  }

  /* Count dnz,onz for preallocation of KKT matrix */
  jac_equality_trans   = pdipm->jac_equality_trans;
  jac_inequality_trans = pdipm->jac_inequality_trans;
  nce_all = pdipm->nce_all;

  if (pdipm->Nxfixed) {
    CHKERRQ(MatTranspose(pdipm->Jce_xfixed,MAT_INITIAL_MATRIX,&Jce_xfixed_trans));
  }
  CHKERRQ(MatTranspose(pdipm->Jci_xb,MAT_INITIAL_MATRIX,&Jci_xb_trans));

  ierr = MatPreallocateInitialize(comm,pdipm->n,pdipm->n,dnz,onz);CHKERRQ(ierr);

  /* 1st row block of KKT matrix: [Wxx; gradCe'; -gradCi'; 0] */
  CHKERRQ(TaoPDIPMEvaluateFunctionsAndJacobians(tao,pdipm->x));
  CHKERRQ(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));

  /* Insert tao->hessian */
  CHKERRQ(MatGetOwnershipRange(tao->hessian,&rjstart,NULL));
  for (i=0; i<pdipm->nx; i++) {
    row = rstart + i;

    CHKERRQ(MatGetRow(tao->hessian,i+rjstart,&nc,&aj,NULL));
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      col = aj[j] - cranges[proc] + Jranges[proc];
      CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
    }
    CHKERRQ(MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,NULL));

    if (pdipm->ng) {
      /* Insert grad g' */
      CHKERRQ(MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL));
      CHKERRQ(MatGetOwnershipRanges(tao->jacobian_equality,&ranges));
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
      }
      CHKERRQ(MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL));
    }

    /* Insert Jce_xfixed^T' */
    if (pdipm->nxfixed) {
      CHKERRQ(MatGetRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL));
      CHKERRQ(MatGetOwnershipRanges(pdipm->Jce_xfixed,&ranges));
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + ng_all[proc];
        CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
      }
      CHKERRQ(MatRestoreRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL));
    }

    if (pdipm->nh) {
      /* Insert -grad h' */
      CHKERRQ(MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL));
      CHKERRQ(MatGetOwnershipRanges(tao->jacobian_inequality,&ranges));
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
      }
      CHKERRQ(MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL));
    }

    /* Insert Jci_xb^T' */
    CHKERRQ(MatGetRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL));
    CHKERRQ(MatGetOwnershipRanges(pdipm->Jci_xb,&ranges));
    proc = 0;
    for (j=0; j < nc; j++) {
      /* find row ownership of */
      while (aj[j] >= ranges[proc+1]) proc++;
      nx_all = rranges[proc+1] - rranges[proc];
      col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc] + nh_all[proc];
      CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
    }
    CHKERRQ(MatRestoreRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL));
  }

  /* 2nd Row block of KKT matrix: [grad Ce, deltac*I, 0, 0] */
  if (pdipm->Ng) {
    CHKERRQ(MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL));
    for (i=0; i < pdipm->ng; i++) {
      row = rstart + pdipm->off_lambdae + i;

      CHKERRQ(MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL));
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz)); /* grad g */
      }
      CHKERRQ(MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL));
    }
  }
  /* Jce_xfixed */
  if (pdipm->Nxfixed) {
    CHKERRQ(MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,NULL));
    for (i=0; i < (pdipm->nce - pdipm->ng); i++) {
      row = rstart + pdipm->off_lambdae + pdipm->ng + i;

      CHKERRQ(MatGetRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL));
      PetscCheck(nc == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");

      proc = 0;
      j    = 0;
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
      CHKERRQ(MatRestoreRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL));
    }
  }

  /* 3rd Row block of KKT matrix: [ gradCi, 0, deltac*I, -I] */
  if (pdipm->Nh) {
    CHKERRQ(MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL));
    for (i=0; i < pdipm->nh; i++) {
      row = rstart + pdipm->off_lambdai + i;

      CHKERRQ(MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL));
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz)); /* grad h */
      }
      CHKERRQ(MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL));
    }
    /* I */
    for (i=0; i < pdipm->nh; i++) {
      row = rstart + pdipm->off_lambdai + i;
      col = rstart + pdipm->off_z + i;
      CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
    }
  }

  /* Jci_xb */
  CHKERRQ(MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,NULL));
  for (i=0; i < (pdipm->nci - pdipm->nh); i++) {
    row = rstart + pdipm->off_lambdai + pdipm->nh + i;

    CHKERRQ(MatGetRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,NULL));
    PetscCheck(nc == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
    }
    CHKERRQ(MatRestoreRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,NULL));
    /* I */
    col = rstart + pdipm->off_z + pdipm->nh + i;
    CHKERRQ(MatPreallocateSet(row,1,&col,dnz,onz));
  }

  /* 4-th Row block of KKT matrix: Z and Ci */
  for (i=0; i < pdipm->nci; i++) {
    row     = rstart + pdipm->off_z + i;
    cols1[0] = rstart + pdipm->off_lambdai + i;
    cols1[1] = row;
    CHKERRQ(MatPreallocateSet(row,2,cols1,dnz,onz));
  }

  /* diagonal entry */
  for (i=0; i<pdipm->n; i++) dnz[i]++; /* diagonal entry */

  /* Create KKT matrix */
  CHKERRQ(MatCreate(comm,&J));
  CHKERRQ(MatSetSizes(J,pdipm->n,pdipm->n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSeqAIJSetPreallocation(J,0,dnz));
  CHKERRQ(MatMPIAIJSetPreallocation(J,0,dnz,0,onz));
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  pdipm->K = J;

  /* (8) Insert constant entries to  K */
  /* Set 0.0 to diagonal of K, so that the solver does not complain *about missing diagonal value */
  CHKERRQ(MatGetOwnershipRange(J,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatSetValue(J,i,i,0.0,INSERT_VALUES));
  }
  /* In case Wxx has no diagonal entries preset set diagonal to deltaw given */
  if (pdipm->kkt_pd) {
      for (i=0; i<pdipm->nh; i++) {
        row  = rstart + i;
        CHKERRQ(MatSetValue(J,row,row,pdipm->deltaw,INSERT_VALUES));
      }
  }

  /* Row block of K: [ grad Ce, 0, 0, 0] */
  if (pdipm->Nxfixed) {
    CHKERRQ(MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,NULL));
    for (i=0; i < (pdipm->nce - pdipm->ng); i++) {
      row = rstart + pdipm->off_lambdae + pdipm->ng + i;

      CHKERRQ(MatGetRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa));
      proc = 0;
      for (j=0; j < nc; j++) {
        while (cols[j] >= cranges[proc+1]) proc++;
        col = cols[j] - cranges[proc] + Jranges[proc];
        CHKERRQ(MatSetValue(J,row,col,aa[j],INSERT_VALUES)); /* grad Ce */
        CHKERRQ(MatSetValue(J,col,row,aa[j],INSERT_VALUES)); /* grad Ce' */
      }
      CHKERRQ(MatRestoreRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa));
    }
  }

  /* Row block of K: [ -grad Ci, 0, 0, I] */
  CHKERRQ(MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,NULL));
  for (i=0; i < pdipm->nci - pdipm->nh; i++) {
    row = rstart + pdipm->off_lambdai + pdipm->nh + i;

    CHKERRQ(MatGetRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,&aa));
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      CHKERRQ(MatSetValue(J,col,row,-aa[j],INSERT_VALUES));
      CHKERRQ(MatSetValue(J,row,col,-aa[j],INSERT_VALUES));
    }
    CHKERRQ(MatRestoreRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,&aa));

    col = rstart + pdipm->off_z + pdipm->nh + i;
    CHKERRQ(MatSetValue(J,row,col,1,INSERT_VALUES));
  }

  for (i=0; i < pdipm->nh; i++) {
    row = rstart + pdipm->off_lambdai + i;
    col = rstart + pdipm->off_z + i;
    CHKERRQ(MatSetValue(J,row,col,1,INSERT_VALUES));
  }

  /* Row block of K: [ 0, 0, I, ...] */
  for (i=0; i < pdipm->nci; i++) {
    row = rstart + pdipm->off_z + i;
    col = rstart + pdipm->off_lambdai + i;
    CHKERRQ(MatSetValue(J,row,col,1,INSERT_VALUES));
  }

  if (pdipm->Nxfixed) {
    CHKERRQ(MatDestroy(&Jce_xfixed_trans));
  }
  CHKERRQ(MatDestroy(&Jci_xb_trans));
  CHKERRQ(PetscFree3(ng_all,nh_all,Jranges));

  /* (9) Set up nonlinear solver SNES */
  CHKERRQ(SNESSetFunction(pdipm->snes,NULL,TaoSNESFunction_PDIPM,(void*)tao));
  CHKERRQ(SNESSetJacobian(pdipm->snes,J,J,TaoSNESJacobian_PDIPM,(void*)tao));

  if (pdipm->solve_reduced_kkt) {
    PC pc;
    CHKERRQ(KSPGetPC(tao->ksp,&pc));
    CHKERRQ(PCSetType(pc,PCFIELDSPLIT));
    CHKERRQ(PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR));
    CHKERRQ(PCFieldSplitSetIS(pc,"2",pdipm->is2));
    CHKERRQ(PCFieldSplitSetIS(pc,"1",pdipm->is1));
  }
  CHKERRQ(SNESSetFromOptions(pdipm->snes));

  /* (10) Setup PCPreSolve() for pdipm->solve_symmetric_kkt */
  if (pdipm->solve_symmetric_kkt) {
    KSP       ksp;
    PC        pc;
    PetscBool isCHOL;
    CHKERRQ(SNESGetKSP(pdipm->snes,&ksp));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetPreSolve(pc,PCPreSolve_PDIPM));

    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCCHOLESKY,&isCHOL));
    if (isCHOL) {
      Mat        Factor;
      PetscBool  isMUMPS;
      CHKERRQ(PCFactorGetMatrix(pc,&Factor));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)Factor,"mumps",&isMUMPS));
      if (isMUMPS) { /* must set mumps ICNTL(13)=1 and ICNTL(24)=1 to call MatGetInertia() */
#if defined(PETSC_HAVE_MUMPS)
        CHKERRQ(MatMumpsSetIcntl(Factor,24,1)); /* detection of null pivot rows */
        if (size > 1) {
          CHKERRQ(MatMumpsSetIcntl(Factor,13,1)); /* parallelism of the root node (enable ScaLAPACK) and its splitting */
        }
#else
        SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Requires external package MUMPS");
#endif
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
   TaoDestroy_PDIPM - Destroys the pdipm object

   Input:
   full pdipm

   Output:
   Destroyed pdipm
*/
PetscErrorCode TaoDestroy_PDIPM(Tao tao)
{
  TAO_PDIPM      *pdipm = (TAO_PDIPM*)tao->data;

  PetscFunctionBegin;
  /* Freeing Vectors assocaiated with KKT (X) */
  CHKERRQ(VecDestroy(&pdipm->x)); /* Solution x */
  CHKERRQ(VecDestroy(&pdipm->lambdae)); /* Equality constraints lagrangian multiplier*/
  CHKERRQ(VecDestroy(&pdipm->lambdai)); /* Inequality constraints lagrangian multiplier*/
  CHKERRQ(VecDestroy(&pdipm->z));       /* Slack variables */
  CHKERRQ(VecDestroy(&pdipm->X));       /* Big KKT system vector [x; lambdae; lambdai; z] */

  /* work vectors */
  CHKERRQ(VecDestroy(&pdipm->lambdae_xfixed));
  CHKERRQ(VecDestroy(&pdipm->lambdai_xb));

  /* Legrangian equality and inequality Vec */
  CHKERRQ(VecDestroy(&pdipm->ce)); /* Vec of equality constraints */
  CHKERRQ(VecDestroy(&pdipm->ci)); /* Vec of inequality constraints */

  /* Matrices */
  CHKERRQ(MatDestroy(&pdipm->Jce_xfixed));
  CHKERRQ(MatDestroy(&pdipm->Jci_xb)); /* Jacobian of inequality constraints Jci = [tao->jacobian_inequality ; J(nxub); J(nxlb); J(nxbx)] */
  CHKERRQ(MatDestroy(&pdipm->K));

  /* Index Sets */
  if (pdipm->Nxub) {
    CHKERRQ(ISDestroy(&pdipm->isxub));    /* Finite upper bound only -inf < x < ub */
  }

  if (pdipm->Nxlb) {
    CHKERRQ(ISDestroy(&pdipm->isxlb));    /* Finite lower bound only  lb <= x < inf */
  }

  if (pdipm->Nxfixed) {
    CHKERRQ(ISDestroy(&pdipm->isxfixed)); /* Fixed variables         lb =  x = ub */
  }

  if (pdipm->Nxbox) {
    CHKERRQ(ISDestroy(&pdipm->isxbox));   /* Boxed variables         lb <= x <= ub */
  }

  if (pdipm->Nxfree) {
    CHKERRQ(ISDestroy(&pdipm->isxfree));  /* Free variables        -inf <= x <= inf */
  }

  if (pdipm->solve_reduced_kkt) {
    CHKERRQ(ISDestroy(&pdipm->is1));
    CHKERRQ(ISDestroy(&pdipm->is2));
  }

  /* SNES */
  CHKERRQ(SNESDestroy(&pdipm->snes)); /* Nonlinear solver */
  CHKERRQ(PetscFree(pdipm->nce_all));
  CHKERRQ(MatDestroy(&pdipm->jac_equality_trans));
  CHKERRQ(MatDestroy(&pdipm->jac_inequality_trans));

  /* Destroy pdipm */
  CHKERRQ(PetscFree(tao->data)); /* Holding locations of pdipm */

  /* Destroy Dual */
  CHKERRQ(VecDestroy(&tao->DE)); /* equality dual */
  CHKERRQ(VecDestroy(&tao->DI)); /* dinequality dual */
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetFromOptions_PDIPM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_PDIPM      *pdipm = (TAO_PDIPM*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PDIPM method for constrained optimization"));
  CHKERRQ(PetscOptionsReal("-tao_pdipm_push_init_slack","parameter to push initial slack variables away from bounds",NULL,pdipm->push_init_slack,&pdipm->push_init_slack,NULL));
  CHKERRQ(PetscOptionsReal("-tao_pdipm_push_init_lambdai","parameter to push initial (inequality) dual variables away from bounds",NULL,pdipm->push_init_lambdai,&pdipm->push_init_lambdai,NULL));
  CHKERRQ(PetscOptionsBool("-tao_pdipm_solve_reduced_kkt","Solve reduced KKT system using Schur-complement",NULL,pdipm->solve_reduced_kkt,&pdipm->solve_reduced_kkt,NULL));
  CHKERRQ(PetscOptionsReal("-tao_pdipm_mu_update_factor","Update scalar for barrier parameter (mu) update",NULL,pdipm->mu_update_factor,&pdipm->mu_update_factor,NULL));
  CHKERRQ(PetscOptionsBool("-tao_pdipm_symmetric_kkt","Solve non reduced symmetric KKT system",NULL,pdipm->solve_symmetric_kkt,&pdipm->solve_symmetric_kkt,NULL));
  CHKERRQ(PetscOptionsBool("-tao_pdipm_kkt_shift_pd","Add shifts to make KKT matrix positive definite",NULL,pdipm->kkt_pd,&pdipm->kkt_pd,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*MC
  TAOPDIPM - Barrier-based primal-dual interior point algorithm for generally constrained optimization.

  Option Database Keys:
+   -tao_pdipm_push_init_lambdai - parameter to push initial dual variables away from bounds (> 0)
.   -tao_pdipm_push_init_slack - parameter to push initial slack variables away from bounds (> 0)
.   -tao_pdipm_mu_update_factor - update scalar for barrier parameter (mu) update (> 0)
.   -tao_pdipm_symmetric_kkt - Solve non-reduced symmetric KKT system
-   -tao_pdipm_kkt_shift_pd - Add shifts to make KKT matrix positive definite

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_PDIPM(Tao tao)
{
  TAO_PDIPM      *pdipm;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_PDIPM;
  tao->ops->solve          = TaoSolve_PDIPM;
  tao->ops->setfromoptions = TaoSetFromOptions_PDIPM;
  tao->ops->view           = TaoView_PDIPM;
  tao->ops->destroy        = TaoDestroy_PDIPM;

  CHKERRQ(PetscNewLog(tao,&pdipm));
  tao->data = (void*)pdipm;

  pdipm->nx      = pdipm->Nx      = 0;
  pdipm->nxfixed = pdipm->Nxfixed = 0;
  pdipm->nxlb    = pdipm->Nxlb    = 0;
  pdipm->nxub    = pdipm->Nxub    = 0;
  pdipm->nxbox   = pdipm->Nxbox   = 0;
  pdipm->nxfree  = pdipm->Nxfree  = 0;

  pdipm->ng = pdipm->Ng = pdipm->nce = pdipm->Nce = 0;
  pdipm->nh = pdipm->Nh = pdipm->nci = pdipm->Nci = 0;
  pdipm->n  = pdipm->N  = 0;
  pdipm->mu = 1.0;
  pdipm->mu_update_factor = 0.1;

  pdipm->deltaw     = 0.0;
  pdipm->lastdeltaw = 3*1.e-4;
  pdipm->deltac     = 0.0;
  pdipm->kkt_pd     = PETSC_FALSE;

  pdipm->push_init_slack     = 1.0;
  pdipm->push_init_lambdai   = 1.0;
  pdipm->solve_reduced_kkt   = PETSC_FALSE;
  pdipm->solve_symmetric_kkt = PETSC_TRUE;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;

  CHKERRQ(SNESCreate(((PetscObject)tao)->comm,&pdipm->snes));
  CHKERRQ(SNESSetOptionsPrefix(pdipm->snes,tao->hdr.prefix));
  CHKERRQ(SNESGetKSP(pdipm->snes,&tao->ksp));
  CHKERRQ(PetscObjectReference((PetscObject)tao->ksp));
  CHKERRQ(KSPSetApplicationContext(tao->ksp,(void *)tao));
  PetscFunctionReturn(0);
}
