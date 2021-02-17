#include <petsctaolinesearch.h>
#include <../src/tao/constrained/impls/ipm/pdipm.h>
#include <petscsnes.h>

/*
   TaoPDIPMEvaluateFunctionsAndJacobians - Evaluate the objective function f, gradient fx, constraints, and all the Jacobians at current vector

   Collective on tao

   Input Parameter:
+  tao - solver context
-  x - vector at which all objects to be evaluated

   Level: beginner

.seealso: TaoPDIPMUpdateConstraints(), TaoPDIPMSetUpBounds()
*/
PetscErrorCode TaoPDIPMEvaluateFunctionsAndJacobians(Tao tao,Vec x)
{
  PetscErrorCode ierr;
  TAO_PDIPM      *pdipm=(TAO_PDIPM*)tao->data;

  PetscFunctionBegin;
  /* Compute user objective function and gradient */
  ierr = TaoComputeObjectiveAndGradient(tao,x,&pdipm->obj,tao->gradient);CHKERRQ(ierr);

  /* Equality constraints and Jacobian */
  if (pdipm->Ng) {
    ierr = TaoComputeEqualityConstraints(tao,x,tao->constraints_equality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao,x,tao->jacobian_equality,tao->jacobian_equality_pre);CHKERRQ(ierr);
  }

  /* Inequality constraints and Jacobian */
  if (pdipm->Nh) {
    ierr = TaoComputeInequalityConstraints(tao,x,tao->constraints_inequality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao,x,tao->jacobian_inequality,tao->jacobian_inequality_pre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  TaoPDIPMUpdateConstraints - Update the vectors ce and ci at x

  Collective

  Input Parameter:
+ tao - Tao context
- x - vector at which constraints to be evaluted

   Level: beginner

.seealso: TaoPDIPMEvaluateFunctionsAndJacobians()
*/
PetscErrorCode TaoPDIPMUpdateConstraints(Tao tao,Vec x)
{
  PetscErrorCode    ierr;
  TAO_PDIPM         *pdipm=(TAO_PDIPM*)tao->data;
  PetscInt          i,offset,offset1,k,xstart;
  PetscScalar       *carr;
  const PetscInt    *ubptr,*lbptr,*bxptr,*fxptr;
  const PetscScalar *xarr,*xuarr,*xlarr,*garr,*harr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(x,&xstart,NULL);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x,&xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XU,&xuarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XL,&xlarr);CHKERRQ(ierr);

  /* (1) Update ce vector */
  ierr = VecGetArray(pdipm->ce,&carr);CHKERRQ(ierr);

  if (pdipm->Ng) {
    /* (1.a) Inserting updated g(x) */
    ierr = VecGetArrayRead(tao->constraints_equality,&garr);CHKERRQ(ierr);
    ierr = PetscMemcpy(carr,garr,pdipm->ng*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(tao->constraints_equality,&garr);CHKERRQ(ierr);
  }

  /* (1.b) Update xfixed */
  if (pdipm->Nxfixed) {
    offset = pdipm->ng;
    ierr = ISGetIndices(pdipm->isxfixed,&fxptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0;k < pdipm->nxfixed;k++){
      i = fxptr[k]-xstart;
      carr[offset + k] = xarr[i] - xuarr[i];
    }
  }
  ierr = VecRestoreArray(pdipm->ce,&carr);CHKERRQ(ierr);

  /* (2) Update ci vector */
  ierr = VecGetArray(pdipm->ci,&carr);CHKERRQ(ierr);

  if (pdipm->Nh) {
    /* (2.a) Inserting updated h(x) */
    ierr = VecGetArrayRead(tao->constraints_inequality,&harr);CHKERRQ(ierr);
    ierr = PetscMemcpy(carr,harr,pdipm->nh*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(tao->constraints_inequality,&harr);CHKERRQ(ierr);
  }

  /* (2.b) Update xub */
  offset = pdipm->nh;
  if (pdipm->Nxub) {
    ierr = ISGetIndices(pdipm->isxub,&ubptr);CHKERRQ(ierr);
    for (k=0; k<pdipm->nxub; k++){
      i = ubptr[k]-xstart;
      carr[offset + k] = xuarr[i] - xarr[i];
    }
  }

  if (pdipm->Nxlb) {
    /* (2.c) Update xlb */
    offset += pdipm->nxub;
    ierr = ISGetIndices(pdipm->isxlb,&lbptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0; k<pdipm->nxlb; k++){
      i = lbptr[k]-xstart;
      carr[offset + k] = xarr[i] - xlarr[i];
    }
  }

  if (pdipm->Nxbox) {
    /* (2.d) Update xbox */
    offset += pdipm->nxlb;
    offset1 = offset + pdipm->nxbox;
    ierr = ISGetIndices(pdipm->isxbox,&bxptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0; k<pdipm->nxbox; k++){
      i = bxptr[k]-xstart; /* local indices in x */
      carr[offset+k]  = xuarr[i] - xarr[i];
      carr[offset1+k] = xarr[i]  - xlarr[i];
    }
  }
  ierr = VecRestoreArray(pdipm->ci,&carr);CHKERRQ(ierr);

  /* Restoring Vectors */
  ierr = VecRestoreArrayRead(x,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XU,&xuarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XL,&xlarr);CHKERRQ(ierr);
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
PetscErrorCode TaoPDIPMSetUpBounds(Tao tao)
{
  PetscErrorCode    ierr;
  TAO_PDIPM         *pdipm=(TAO_PDIPM*)tao->data;
  const PetscScalar *xl,*xu;
  PetscInt          n,*ixlb,*ixub,*ixfixed,*ixfree,*ixbox,i,low,high,idx;
  MPI_Comm          comm;
  PetscInt          sendbuf[5],recvbuf[5];

  PetscFunctionBegin;
  /* Creates upper and lower bounds vectors on x, if not created already */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);

  ierr = VecGetLocalSize(tao->XL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc5(n,&ixlb,n,&ixub,n,&ixfree,n,&ixfixed,n,&ixbox);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->XL,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XL,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XU,&xu);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(tao->XL,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XU,&xu);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  sendbuf[0] = pdipm->nxlb;
  sendbuf[1] = pdipm->nxub;
  sendbuf[2] = pdipm->nxfixed;
  sendbuf[3] = pdipm->nxbox;
  sendbuf[4] = pdipm->nxfree;

  ierr = MPI_Allreduce(sendbuf,recvbuf,5,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  pdipm->Nxlb    = recvbuf[0];
  pdipm->Nxub    = recvbuf[1];
  pdipm->Nxfixed = recvbuf[2];
  pdipm->Nxbox   = recvbuf[3];
  pdipm->Nxfree  = recvbuf[4];

  if (pdipm->Nxlb) {
    ierr = ISCreateGeneral(comm,pdipm->nxlb,ixlb,PETSC_COPY_VALUES,&pdipm->isxlb);CHKERRQ(ierr);
  }
  if (pdipm->Nxub) {
    ierr = ISCreateGeneral(comm,pdipm->nxub,ixub,PETSC_COPY_VALUES,&pdipm->isxub);CHKERRQ(ierr);
  }
  if (pdipm->Nxfixed) {
    ierr = ISCreateGeneral(comm,pdipm->nxfixed,ixfixed,PETSC_COPY_VALUES,&pdipm->isxfixed);CHKERRQ(ierr);
  }
  if (pdipm->Nxbox) {
    ierr = ISCreateGeneral(comm,pdipm->nxbox,ixbox,PETSC_COPY_VALUES,&pdipm->isxbox);CHKERRQ(ierr);
  }
  if (pdipm->Nxfree) {
    ierr = ISCreateGeneral(comm,pdipm->nxfree,ixfree,PETSC_COPY_VALUES,&pdipm->isxfree);CHKERRQ(ierr);
  }
  ierr = PetscFree5(ixlb,ixub,ixfixed,ixbox,ixfree);CHKERRQ(ierr);
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
PetscErrorCode TaoPDIPMInitializeSolution(Tao tao)
{
  PetscErrorCode    ierr;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscScalar       *Xarr,*z,*lambdai;
  PetscInt          i;
  const PetscScalar *xarr,*h;

  PetscFunctionBegin;
  ierr = VecGetArray(pdipm->X,&Xarr);CHKERRQ(ierr);

  /* Set Initialize X.x = tao->solution */
  ierr = VecGetArrayRead(tao->solution,&xarr);CHKERRQ(ierr);
  ierr = PetscMemcpy(Xarr,xarr,pdipm->nx*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->solution,&xarr);CHKERRQ(ierr);

  /* Initialize X.lambdae = 0.0 */
  if (pdipm->lambdae) {
    ierr = VecSet(pdipm->lambdae,0.0);CHKERRQ(ierr);
  }
  /* Initialize X.lambdai = push_init_lambdai, X.z = push_init_slack */
  if (pdipm->lambdai) {
    ierr = VecSet(pdipm->lambdai,pdipm->push_init_lambdai);CHKERRQ(ierr);
  }
  if (pdipm->z) {
    ierr = VecSet(pdipm->z,pdipm->push_init_slack);CHKERRQ(ierr);
  }

  /* Additional modification for X.lambdai and X.z */
  if (pdipm->lambdai) {
    ierr = VecGetArray(pdipm->lambdai,&lambdai);CHKERRQ(ierr);
  }
  if (pdipm->z) {
    ierr = VecGetArray(pdipm->z,&z);CHKERRQ(ierr);
  }
  if (pdipm->Nh) {
    ierr = VecGetArrayRead(tao->constraints_inequality,&h);CHKERRQ(ierr);
    for (i=0; i < pdipm->nh; i++) {
      if (h[i] < -pdipm->push_init_slack) z[i] = -h[i];
      if (pdipm->mu/z[i] > pdipm->push_init_lambdai) lambdai[i] = pdipm->mu/z[i];
    }
    ierr = VecRestoreArrayRead(tao->constraints_inequality,&h);CHKERRQ(ierr);
  }
  if (pdipm->lambdai) {
    ierr = VecRestoreArray(pdipm->lambdai,&lambdai);CHKERRQ(ierr);
  }
  if (pdipm->z) {
    ierr = VecRestoreArray(pdipm->z,&z);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(pdipm->X,&Xarr);CHKERRQ(ierr);
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
PetscErrorCode TaoSNESJacobian_PDIPM(SNES snes,Vec X, Mat J, Mat Jpre, void *ctx)
{
  PetscErrorCode    ierr;
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
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&size);CHKERRMPI(ierr);

  ierr = MatGetOwnershipRanges(Jpre,&Jranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Jpre,&Jrstart,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&rranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&cranges);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X,&Xarr);CHKERRQ(ierr);

  if (pdipm->solve_symmetric_kkt) { /* 1 for eq 17 revised pdipm doc 0 for eq 18 (symmetric KKT) */
    vals[0] = 1.0;
    for (i=0; i < pdipm->nci; i++) {
        row     = Jrstart + pdipm->off_z + i;
        cols[0] = Jrstart + pdipm->off_lambdai + i;
        cols[1] = row;
        vals[1] = Xarr[pdipm->off_lambdai + i]/Xarr[pdipm->off_z + i];
        ierr = MatSetValues(Jpre,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  } else {
    /* (2) insert Z and Ci to Jpre -- overwrite existing values */
    for (i=0; i < pdipm->nci; i++) {
      row     = Jrstart + pdipm->off_z + i;
      cols[0] = Jrstart + pdipm->off_lambdai + i;
      cols[1] = row;
      vals[0] = Xarr[pdipm->off_z + i];
      vals[1] = Xarr[pdipm->off_lambdai + i];
      ierr = MatSetValues(Jpre,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* (3) insert 2nd row block of Jpre: [ grad g, 0, 0, 0] */
  if (pdipm->Ng) {
    ierr = MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i<pdipm->ng; i++){
      row = Jrstart + pdipm->off_lambdae + i;

      ierr = MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      if (pdipm->kkt_pd) {
        /* (3) insert 2nd row block of Jpre: [ grad g, \delta_c*I, 0, 0] */
        ierr = MatSetValue(Jpre,row,row,-pdipm->deltac,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  if (pdipm->Nh) {
    /* (4) insert 3nd row block of Jpre: [ -grad h, 0, deltac, I] */
    ierr = MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdipm->nh; i++){
      row = Jrstart + pdipm->off_lambdai + i;
      ierr = MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(Jpre,row,cols[0],-aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      if (pdipm->kkt_pd) {
        ierr = MatSetValue(Jpre,row,row,-pdipm->deltac,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* (5) insert Wxx, grad g' and -grad h' to Jpre */
  if (pdipm->Ng) {
    ierr = MatTranspose(tao->jacobian_equality,MAT_REUSE_MATRIX,&jac_equality_trans);CHKERRQ(ierr);
  }
  if (pdipm->Nh) {
    ierr = MatTranspose(tao->jacobian_inequality,MAT_REUSE_MATRIX,&jac_inequality_trans);CHKERRQ(ierr);
  }

  ierr = VecPlaceArray(pdipm->x,Xarr);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,pdipm->x,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  ierr = VecResetArray(pdipm->x);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(tao->hessian,&rjstart,NULL);CHKERRQ(ierr);
  for (i=0; i<pdipm->nx; i++) {
    row = Jrstart + i;

    /* insert Wxx */
    ierr = MatGetRow(tao->hessian,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      cols[0] = aj[j] - cranges[proc] + Jranges[proc];
      if (row == cols[0] && pdipm->kkt_pd) {
        /* add diagonal shift to Wxx component */
        ierr = MatSetValue(Jpre,row,cols[0],aa[j]+pdipm->deltaw,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);

    if (pdipm->ng) {
      /* insert grad g' */
      ierr = MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_equality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }

    if (pdipm->nh) {
      /* insert -grad h' */
      ierr = MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_inequality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        ierr = MatSetValue(Jpre,row,cols[0],-aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(X,&Xarr);CHKERRQ(ierr);

  /* (6) assemble Jpre and J */
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
PetscErrorCode TaoSNESFunction_PDIPM(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  PetscScalar       *Farr,*tmparr;
  Vec               x,L1;
  PetscInt          i;
  PetscReal         res[2],cnorm[2];
  const PetscScalar *Xarr,*carr,*zarr,*larr;

  PetscFunctionBegin;
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArray(F,&Farr);CHKERRQ(ierr);

  /* (0) Evaluate f, fx, Gx, Hx at X.x Note: pdipm->x is not changed below */
  x = pdipm->x;
  ierr = VecPlaceArray(x,Xarr);CHKERRQ(ierr);
  ierr = TaoPDIPMEvaluateFunctionsAndJacobians(tao,x);CHKERRQ(ierr);

  /* Update ce, ci, and Jci at X.x */
  ierr = TaoPDIPMUpdateConstraints(tao,x);CHKERRQ(ierr);
  ierr = VecResetArray(x);CHKERRQ(ierr);

  /* (1) L1 = fx + (gradG'*DE + Jce_xfixed'*lambdae_xfixed) - (gradH'*DI + Jci_xb'*lambdai_xb) */
  L1 = pdipm->x;
  ierr = VecPlaceArray(L1,Farr);CHKERRQ(ierr);
  if (pdipm->Nci) {
    if (pdipm->Nh) {
      /* L1 += gradH'*DI. Note: tao->DI is not changed below */
      ierr = VecPlaceArray(tao->DI,Xarr+pdipm->off_lambdai);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(tao->jacobian_inequality,tao->DI,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(tao->DI);CHKERRQ(ierr);
    }

    /* L1 += Jci_xb'*lambdai_xb */
    ierr = VecPlaceArray(pdipm->lambdai_xb,Xarr+pdipm->off_lambdai+pdipm->nh);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(pdipm->Jci_xb,pdipm->lambdai_xb,L1,L1);CHKERRQ(ierr);
    ierr = VecResetArray(pdipm->lambdai_xb);CHKERRQ(ierr);

    /* (1.4) L1 = - (gradH'*DI + Jci_xb'*lambdai_xb) */
    ierr = VecScale(L1,-1.0);CHKERRQ(ierr);
  }

  /* L1 += fx */
  ierr = VecAXPY(L1,1.0,tao->gradient);CHKERRQ(ierr);

  if (pdipm->Nce) {
    if (pdipm->Ng) {
      /* L1 += gradG'*DE. Note: tao->DE is not changed below */
      ierr = VecPlaceArray(tao->DE,Xarr+pdipm->off_lambdae);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(tao->jacobian_equality,tao->DE,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(tao->DE);CHKERRQ(ierr);
    }
    if (pdipm->Nxfixed) {
      /* L1 += Jce_xfixed'*lambdae_xfixed */
      ierr = VecPlaceArray(pdipm->lambdae_xfixed,Xarr+pdipm->off_lambdae+pdipm->ng);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pdipm->Jce_xfixed,pdipm->lambdae_xfixed,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(pdipm->lambdae_xfixed);CHKERRQ(ierr);
    }
  }
  ierr = VecNorm(L1,NORM_2,&res[0]);CHKERRQ(ierr);
  ierr = VecResetArray(L1);CHKERRQ(ierr);

  /* (2) L2 = ce(x) */
  if (pdipm->Nce) {
    ierr = VecGetArrayRead(pdipm->ce,&carr);CHKERRQ(ierr);
    for (i=0; i<pdipm->nce; i++) Farr[pdipm->off_lambdae + i] = carr[i];
    ierr = VecRestoreArrayRead(pdipm->ce,&carr);CHKERRQ(ierr);
  }
  ierr = VecNorm(pdipm->ce,NORM_2,&cnorm[0]);CHKERRQ(ierr);

  if (pdipm->Nci) {
    if (pdipm->solve_symmetric_kkt) {
      /* (3) L3 = ci(x) - z;
         (4) L4 = Lambdai * e - mu/z *e
      */
      ierr = VecGetArrayRead(pdipm->ci,&carr);CHKERRQ(ierr);
      larr = Xarr+pdipm->off_lambdai;
      zarr = Xarr+pdipm->off_z;
      for (i=0; i<pdipm->nci; i++) {
        Farr[pdipm->off_lambdai + i] = zarr[i] - carr[i];
        Farr[pdipm->off_z       + i] = larr[i] - pdipm->mu/zarr[i];
      }
      ierr = VecRestoreArrayRead(pdipm->ci,&carr);CHKERRQ(ierr);
    } else {
      /* (3) L3 = ci(x) - z;
         (4) L4 = Z * Lambdai * e - mu * e
      */
      ierr = VecGetArrayRead(pdipm->ci,&carr);CHKERRQ(ierr);
      larr = Xarr+pdipm->off_lambdai;
      zarr = Xarr+pdipm->off_z;
      for (i=0; i<pdipm->nci; i++) {
        Farr[pdipm->off_lambdai + i] = zarr[i] - carr[i];
        Farr[pdipm->off_z       + i] = zarr[i]*larr[i] - pdipm->mu;
      }
      ierr = VecRestoreArrayRead(pdipm->ci,&carr);CHKERRQ(ierr);
    }
  }

  ierr = VecPlaceArray(pdipm->ci,Farr+pdipm->off_lambdai);CHKERRQ(ierr);
  ierr = VecNorm(pdipm->ci,NORM_2,&cnorm[1]);CHKERRQ(ierr);
  ierr = VecResetArray(pdipm->ci);CHKERRQ(ierr);

  /* note: pdipm->z is not changed below */
  if (pdipm->z) {
    if (pdipm->solve_symmetric_kkt) {
      ierr = VecPlaceArray(pdipm->z,Farr+pdipm->off_z);CHKERRQ(ierr);
      if (pdipm->Nci) {
        zarr = Xarr+pdipm->off_z;
        ierr = VecGetArrayWrite(pdipm->z,&tmparr);CHKERRQ(ierr);
        for (i=0; i<pdipm->nci; i++) tmparr[i] *= Xarr[pdipm->off_z + i];
        ierr = VecRestoreArrayWrite(pdipm->z,&tmparr);CHKERRQ(ierr);
      }

      ierr = VecNorm(pdipm->z,NORM_2,&res[1]);CHKERRQ(ierr);
      if (pdipm->Nci) {
        zarr = Xarr+pdipm->off_z;CHKERRQ(ierr);
        ierr = VecGetArray(pdipm->z,&tmparr);
        for (i=0; i<pdipm->nci; i++) {
          tmparr[i] /= Xarr[pdipm->off_z + i];
        }
        ierr = VecRestoreArray(pdipm->z,&tmparr);CHKERRQ(ierr);
      }
      ierr = VecResetArray(pdipm->z);CHKERRQ(ierr);
    } else {
      ierr = VecPlaceArray(pdipm->z,Farr+pdipm->off_z);CHKERRQ(ierr);
      ierr = VecNorm(pdipm->z,NORM_2,&res[1]);CHKERRQ(ierr);
      ierr = VecResetArray(pdipm->z);CHKERRQ(ierr);
    }
  } else res[1] = 0.0;

  tao->residual = PetscSqrtReal(res[0]*res[0] + res[1]*res[1]);
  tao->cnorm    = PetscSqrtReal(cnorm[0]*cnorm[0] + cnorm[1]*cnorm[1]);

  ierr = VecRestoreArrayRead(X,&Xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&Farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
/*
   PDIPMLineSearch - Custom line search used with PDIPM.

   Collective on TAO

   Notes:
   PDIPMLineSearch employs a simple backtracking line-search to keep
   the slack variables (z) and inequality constraints lagrange multipliers
   (lambdai) positive, i.e., z,lambdai >=0. It does this by calculating scalars
   alpha_p and alpha_d to keep z,lambdai non-negative. The decision (x), and the
   slack variables are updated as X = X + alpha_d*dx. The constraint multipliers
   are updated as Lambdai = Lambdai + alpha_p*dLambdai. The barrier parameter mu
   is also updated as mu = mu + z'lambdai/Nci
*/
PetscErrorCode PDIPMLineSearch(SNESLineSearch linesearch,void *ctx)
{
  PetscErrorCode    ierr;
  Tao               tao=(Tao)ctx;
  TAO_PDIPM         *pdipm = (TAO_PDIPM*)tao->data;
  SNES              snes;
  KSP               ksp;
  PC                pc;
  PCType            ptype;
  Mat               Factor;
  Vec               X,F,Y,W,G;
  PetscInt          i,iter,nneg,nzero,npos;
  PetscReal         alpha_p=1.0,alpha_d=1.0,alpha[4];
  PetscScalar       *Xarr,*z,*lambdai,dot,*taosolarr;
  const PetscScalar *dXarr,*dz,*dlambdai;
  PetscBool         isCHOL;

  PetscFunctionBegin;
  ierr = SNESLineSearchGetSNES(linesearch,&snes);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);

  ierr = SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);
  ierr = SNESLineSearchGetVecs(linesearch,&X,&F,&Y,&W,&G);CHKERRQ(ierr);

  ierr = VecGetArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&dXarr);CHKERRQ(ierr);
  z  = Xarr + pdipm->off_z;
  dz = dXarr + pdipm->off_z;
  for (i=0; i < pdipm->nci; i++) {
    if (z[i] - dz[i] < 0.0) {
      alpha_p = PetscMin(alpha_p,0.9999*z[i]/dz[i]);
    }
  }

  lambdai  = Xarr + pdipm->off_lambdai;
  dlambdai = dXarr + pdipm->off_lambdai;

  for (i=0; i<pdipm->nci; i++) {
    if (lambdai[i] - dlambdai[i] < 0.0) {
      alpha_d = PetscMin(0.9999*lambdai[i]/dlambdai[i],alpha_d);
    }
  }

  alpha[0] = alpha_p;
  alpha[1] = alpha_d;
  ierr = VecRestoreArrayRead(Y,&dXarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&Xarr);CHKERRQ(ierr);

  /* alpha = min(alpha) over all processes */
  ierr = MPI_Allreduce(alpha,alpha+2,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)tao));CHKERRMPI(ierr);

  alpha_p = alpha[2];
  alpha_d = alpha[3];

  ierr = VecGetArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&dXarr);CHKERRQ(ierr);
  for (i=0; i<pdipm->nx; i++) {
    Xarr[i] = Xarr[i] - alpha_p * dXarr[i];
  }

  for (i=0; i<pdipm->nce; i++) {
    Xarr[i+pdipm->off_lambdae] = Xarr[i+pdipm->off_lambdae] - alpha_d * dXarr[i+pdipm->off_lambdae];
  }

  for (i=0; i<pdipm->nci; i++) {
    Xarr[i+pdipm->off_lambdai] = Xarr[i+pdipm->off_lambdai] - alpha_d * dXarr[i+pdipm->off_lambdai];
  }

  for (i=0; i<pdipm->nci; i++) {
    Xarr[i+pdipm->off_z] = Xarr[i+pdipm->off_z] - alpha_p * dXarr[i+pdipm->off_z];
  }

  ierr = VecGetArray(tao->solution,&taosolarr);CHKERRQ(ierr);
  ierr = PetscMemcpy(taosolarr,Xarr,pdipm->nx*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(tao->solution,&taosolarr);CHKERRQ(ierr);


  ierr = VecRestoreArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&dXarr);CHKERRQ(ierr);

  /* Evaluate F at X */
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr); /* must call this func, do not know why */

  /* update mu = mu_update_factor * dot(z,lambdai)/pdipm->nci at updated X */
  if (pdipm->z) {
    ierr = VecDot(pdipm->z,pdipm->lambdai,&dot);CHKERRQ(ierr);
  } else dot = 0.0;

  /* if (PetscAbsReal(pdipm->gradL) < 0.9*pdipm->mu)  */
  pdipm->mu = pdipm->mu_update_factor * dot/pdipm->Nci;

  /* Update F; get tao->residual and tao->cnorm */
  ierr = TaoSNESFunction_PDIPM(snes,X,F,(void*)tao);CHKERRQ(ierr);

  tao->niter++;
  ierr = TaoLogConvergenceHistory(tao,pdipm->obj,tao->residual,tao->cnorm,tao->niter);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,pdipm->obj,tao->residual,tao->cnorm,pdipm->mu);CHKERRQ(ierr);

  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason) {
    ierr = SNESSetConvergedReason(snes,SNES_CONVERGED_FNORM_ABS);CHKERRQ(ierr);
  }

  if (!pdipm->kkt_pd) PetscFunctionReturn(0);

  /* Get the inertia of Cholesky factor to set shifts for next SNES interation */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetType(pc,&ptype);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCCHOLESKY,&isCHOL);CHKERRQ(ierr);

  if (isCHOL) {
    PetscMPIInt       size;
    ierr = PCFactorGetMatrix(pc,&Factor);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)Factor),&size);CHKERRQ(ierr);
    if (Factor->ops->getinertia) {
#if defined(PETSC_HAVE_MUMPS)
      MatSolverType     stype;
      PetscBool         isMUMPS;
      ierr = PCFactorGetMatSolverType(pc,&stype);CHKERRQ(ierr);
      ierr = PetscStrcmp(stype, MATSOLVERMUMPS, &isMUMPS);CHKERRQ(ierr);
      if (isMUMPS) { /* must set mumps ICNTL(13)=1 and ICNTL(24)=1 to call MatGetInertia() */
        ierr = MatMumpsSetIcntl(Factor,24,1);CHKERRQ(ierr);
        if (size > 1) {
          ierr = MatMumpsSetIcntl(Factor,13,1);CHKERRQ(ierr);
        }
      }
#else
      if (size > 1) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Requires external package MUMPS");
#endif
      ierr = MatGetInertia(Factor,&nneg,&nzero,&npos);CHKERRQ(ierr);

      if (npos < pdipm->Nx+pdipm->Nci) {
        pdipm->deltaw = PetscMax(pdipm->lastdeltaw/3, 1.e-4*PETSC_MACHINE_EPSILON);
        ierr = PetscInfo5(tao,"Test reduced deltaw=%g; previous MatInertia: nneg %d, nzero %d, npos %d(<%d)\n",pdipm->deltaw,nneg,nzero,npos,pdipm->Nx+pdipm->Nci);CHKERRQ(ierr);
        ierr = TaoSNESJacobian_PDIPM(snes,X, pdipm->K, pdipm->K, tao);CHKERRQ(ierr);
        ierr = PCSetUp(pc);CHKERRQ(ierr);
        ierr = MatGetInertia(Factor,&nneg,&nzero,&npos);CHKERRQ(ierr);

        if (npos < pdipm->Nx+pdipm->Nci) {
          pdipm->deltaw = pdipm->lastdeltaw; /* in case reduction update does not help, this prevents that step from impacting increasing update */
          while (npos < pdipm->Nx+pdipm->Nci && pdipm->deltaw <= 1.e10) { /* increase deltaw */
            ierr = PetscInfo5(tao,"  deltaw=%g fails, MatInertia: nneg %d, nzero %d, npos %d(<%d)\n",pdipm->deltaw,nneg,nzero,npos,pdipm->Nx+pdipm->Nci);CHKERRQ(ierr);
            pdipm->deltaw = PetscMin(8*pdipm->deltaw,PetscPowReal(10,20));
            ierr = TaoSNESJacobian_PDIPM(snes,X, pdipm->K, pdipm->K, tao);CHKERRQ(ierr);
            ierr = PCSetUp(pc);CHKERRQ(ierr);
            ierr = MatGetInertia(Factor,&nneg,&nzero,&npos);CHKERRQ(ierr);CHKERRQ(ierr);
          }

          if (pdipm->deltaw >= 1.e10) {
            SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_CONV_FAILED,"Reached maximum delta w will not converge, try different inital x0");
          }
          ierr = PetscInfo1(tao,"Updated deltaw %g\n",pdipm->deltaw);CHKERRQ(ierr);
          pdipm->lastdeltaw = pdipm->deltaw;
          pdipm->deltaw     = 0.0;
        }
      }

      if (nzero) { /* Jacobian is singular */
        if (pdipm->deltac == 0.0) {
          pdipm->deltac = 1.e8*PETSC_MACHINE_EPSILON;
        } else {
          pdipm->deltac = pdipm->deltac*PetscPowReal(pdipm->mu,.25);
        }
        ierr = PetscInfo4(tao,"Updated deltac=%g, MatInertia: nneg %D, nzero %D(!=0), npos %D\n",pdipm->deltac,nneg,nzero,npos);
      }
    } else
      SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Requires an external package that supports MatGetInertia()");
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
  PetscErrorCode     ierr;
  TAO_PDIPM          *pdipm = (TAO_PDIPM*)tao->data;
  SNESLineSearch     linesearch; /* SNESLineSearch context */
  Vec                dummy;

  PetscFunctionBegin;
  if (!tao->constraints_equality && !tao->constraints_inequality) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_NULL,"Equality and inequality constraints are not set. Either set them or switch to a different algorithm");

  /* Initialize all variables */
  ierr = TaoPDIPMInitializeSolution(tao);CHKERRQ(ierr);

  /* Set linesearch */
  ierr = SNESGetLineSearch(pdipm->snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHSHELL);CHKERRQ(ierr);
  ierr = SNESLineSearchShellSetUserFunc(linesearch,PDIPMLineSearch,tao);CHKERRQ(ierr);
  ierr = SNESLineSearchSetFromOptions(linesearch);CHKERRQ(ierr);

  tao->reason = TAO_CONTINUE_ITERATING;

  /* -tao_monitor for iteration 0 and check convergence */
  ierr = VecDuplicate(pdipm->X,&dummy);CHKERRQ(ierr);
  ierr = TaoSNESFunction_PDIPM(pdipm->snes,pdipm->X,dummy,(void*)tao);CHKERRQ(ierr);

  ierr = TaoLogConvergenceHistory(tao,pdipm->obj,tao->residual,tao->cnorm,tao->niter);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,pdipm->obj,tao->residual,tao->cnorm,pdipm->mu);CHKERRQ(ierr);
  ierr = VecDestroy(&dummy);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason) {
    ierr = SNESSetConvergedReason(pdipm->snes,SNES_CONVERGED_FNORM_ABS);CHKERRQ(ierr);
  }

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    SNESConvergedReason reason;
    ierr = SNESSolve(pdipm->snes,NULL,pdipm->X);CHKERRQ(ierr);

    /* Check SNES convergence */
    ierr = SNESGetConvergedReason(pdipm->snes,&reason);CHKERRQ(ierr);
    if (reason < 0) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)pdipm->snes),"SNES solve did not converged due to reason %D\n",reason);CHKERRQ(ierr);
    }

    /* Check TAO convergence */
    if (PetscIsInfOrNanReal(pdipm->obj)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"User-provided compute function generated Inf or NaN");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->constrained = PETSC_TRUE;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Number of prime=%D, Number of dual=%D\n",pdipm->Nx+pdipm->Nci,pdipm->Nce + pdipm->Nci);CHKERRQ(ierr);
  if (pdipm->kkt_pd) {
    ierr = PetscViewerASCIIPrintf(viewer,"KKT shifts deltaw=%g, deltac=%g\n",pdipm->deltaw,pdipm->deltac);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
  PetscMPIInt       rank,size;
  PetscInt          row,col,Jcrstart,Jcrend,k,tmp,nc,proc,*nh_all,*ng_all;
  PetscInt          offset,*xa,*xb,i,j,rstart,rend;
  PetscScalar       one=1.0,neg_one=-1.0,*Xarr;
  const PetscInt    *cols,*rranges,*cranges,*aj,*ranges;
  const PetscScalar *aa;
  Mat               J,jac_equality_trans,jac_inequality_trans;
  Mat               Jce_xfixed_trans,Jci_xb_trans;
  PetscInt          *dnz,*onz,rjstart,nx_all,*nce_all,*Jranges,cols1[2];

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* (1) Setup Bounds and create Tao vectors */
  ierr = TaoPDIPMSetUpBounds(tao);CHKERRQ(ierr);

  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  }

  /* (2) Get sizes */
  /* Size of vector x - This is set by TaoSetInitialVector */
  ierr = VecGetSize(tao->solution,&pdipm->Nx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tao->solution,&pdipm->nx);CHKERRQ(ierr);

  /* Size of equality constraints and vectors */
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&pdipm->Ng);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->constraints_equality,&pdipm->ng);CHKERRQ(ierr);
  } else {
    pdipm->ng = pdipm->Ng = 0;
  }

  pdipm->nce = pdipm->ng + pdipm->nxfixed;
  pdipm->Nce = pdipm->Ng + pdipm->Nxfixed;

  /* Size of inequality constraints and vectors */
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&pdipm->Nh);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->constraints_inequality,&pdipm->nh);CHKERRQ(ierr);
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
  ierr = VecCreate(comm,&pdipm->ce);CHKERRQ(ierr);
  ierr = VecSetSizes(pdipm->ce,pdipm->nce,pdipm->Nce);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdipm->ce);CHKERRQ(ierr);

  ierr = VecCreate(comm,&pdipm->ci);CHKERRQ(ierr);
  ierr = VecSetSizes(pdipm->ci,pdipm->nci,pdipm->Nci);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdipm->ci);CHKERRQ(ierr);

  /* X=[x; lambdae; lambdai; z] for the big KKT system */
  ierr = VecCreate(comm,&pdipm->X);CHKERRQ(ierr);
  ierr = VecSetSizes(pdipm->X,pdipm->n,pdipm->N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdipm->X);CHKERRQ(ierr);

  /* Subvectors; they share local arrays with X */
  ierr = VecGetArray(pdipm->X,&Xarr);CHKERRQ(ierr);
  /* x shares local array with X.x */
  if (pdipm->Nx) {
    ierr = VecCreateMPIWithArray(comm,1,pdipm->nx,pdipm->Nx,Xarr,&pdipm->x);CHKERRQ(ierr);
  }

  /* lambdae shares local array with X.lambdae */
  if (pdipm->Nce) {
    ierr = VecCreateMPIWithArray(comm,1,pdipm->nce,pdipm->Nce,Xarr+pdipm->off_lambdae,&pdipm->lambdae);CHKERRQ(ierr);
  }

  /* tao->DE shares local array with X.lambdae_g */
  if (pdipm->Ng) {
    ierr = VecCreateMPIWithArray(comm,1,pdipm->ng,pdipm->Ng,Xarr+pdipm->off_lambdae,&tao->DE);CHKERRQ(ierr);

    ierr = VecCreate(comm,&pdipm->lambdae_xfixed);CHKERRQ(ierr);
    ierr = VecSetSizes(pdipm->lambdae_xfixed,pdipm->nxfixed,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(pdipm->lambdae_xfixed);CHKERRQ(ierr);
  }

  if (pdipm->Nci) {
    /* lambdai shares local array with X.lambdai */
    ierr = VecCreateMPIWithArray(comm,1,pdipm->nci,pdipm->Nci,Xarr+pdipm->off_lambdai,&pdipm->lambdai);CHKERRQ(ierr);

    /* z for slack variables; it shares local array with X.z */
    ierr = VecCreateMPIWithArray(comm,1,pdipm->nci,pdipm->Nci,Xarr+pdipm->off_z,&pdipm->z);CHKERRQ(ierr);
  }

  /* tao->DI which shares local array with X.lambdai_h */
  if (pdipm->Nh) {
    ierr = VecCreateMPIWithArray(comm,1,pdipm->nh,pdipm->Nh,Xarr+pdipm->off_lambdai,&tao->DI);CHKERRQ(ierr);
  }

  ierr = VecCreate(comm,&pdipm->lambdai_xb);CHKERRQ(ierr);
  ierr = VecSetSizes(pdipm->lambdai_xb,(pdipm->nci - pdipm->nh),PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdipm->lambdai_xb);CHKERRQ(ierr);

  ierr = VecRestoreArray(pdipm->X,&Xarr);CHKERRQ(ierr);

  /* (5) Create Jacobians Jce_xfixed and Jci */
  /* (5.1) PDIPM Jacobian of equality bounds cebound(x) = J_nxfixed */
  if (pdipm->Nxfixed) {
    /* Create Jce_xfixed */
    ierr = MatCreate(comm,&pdipm->Jce_xfixed);CHKERRQ(ierr);
    ierr = MatSetSizes(pdipm->Jce_xfixed,pdipm->nxfixed,pdipm->nx,PETSC_DECIDE,pdipm->Nx);CHKERRQ(ierr);
    ierr = MatSetFromOptions(pdipm->Jce_xfixed);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(pdipm->Jce_xfixed,1,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(pdipm->Jce_xfixed,1,NULL,1,NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,&Jcrend);CHKERRQ(ierr);
    ierr = ISGetIndices(pdipm->isxfixed,&cols);CHKERRQ(ierr);
    k = 0;
    for (row = Jcrstart; row < Jcrend; row++) {
      ierr = MatSetValues(pdipm->Jce_xfixed,1,&row,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdipm->isxfixed, &cols);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pdipm->Jce_xfixed,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pdipm->Jce_xfixed,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* (5.2) PDIPM inequality Jacobian Jci = [tao->jacobian_inequality; ...] */
  ierr = MatCreate(comm,&pdipm->Jci_xb);CHKERRQ(ierr);
  ierr = MatSetSizes(pdipm->Jci_xb,pdipm->nci-pdipm->nh,pdipm->nx,PETSC_DECIDE,pdipm->Nx);CHKERRQ(ierr);
  ierr = MatSetFromOptions(pdipm->Jci_xb);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(pdipm->Jci_xb,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(pdipm->Jci_xb,1,NULL,1,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,&Jcrend);CHKERRQ(ierr);
  offset = Jcrstart;
  if (pdipm->Nxub) {
    /* Add xub to Jci_xb */
    ierr = ISGetIndices(pdipm->isxub,&cols);CHKERRQ(ierr);
    k = 0;
    for (row = offset; row < offset + pdipm->nxub; row++) {
      ierr = MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdipm->isxub, &cols);CHKERRQ(ierr);
  }

  if (pdipm->Nxlb) {
    /* Add xlb to Jci_xb */
    ierr = ISGetIndices(pdipm->isxlb,&cols);CHKERRQ(ierr);
    k = 0;
    offset += pdipm->nxub;
    for (row = offset; row < offset + pdipm->nxlb; row++) {
      ierr = MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdipm->isxlb, &cols);CHKERRQ(ierr);
  }

  /* Add xbox to Jci_xb */
  if (pdipm->Nxbox) {
    ierr = ISGetIndices(pdipm->isxbox,&cols);CHKERRQ(ierr);
    k = 0;
    offset += pdipm->nxlb;
    for (row = offset; row < offset + pdipm->nxbox; row++) {
      ierr = MatSetValues(pdipm->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
      tmp = row + pdipm->nxbox;
      ierr = MatSetValues(pdipm->Jci_xb,1,&tmp,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdipm->isxbox, &cols);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(pdipm->Jci_xb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pdipm->Jci_xb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* ierr = MatView(pdipm->Jci_xb,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* (6) Set up ISs for PC Fieldsplit */
  if (pdipm->solve_reduced_kkt) {
    ierr = PetscMalloc2(pdipm->nx+pdipm->nce,&xa,2*pdipm->nci,&xb);CHKERRQ(ierr);
    for (i=0; i < pdipm->nx + pdipm->nce; i++) xa[i] = i;
    for (i=0; i < 2*pdipm->nci; i++) xb[i] = pdipm->off_lambdai + i;

    ierr = ISCreateGeneral(comm,pdipm->nx+pdipm->nce,xa,PETSC_OWN_POINTER,&pdipm->is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,2*pdipm->nci,xb,PETSC_OWN_POINTER,&pdipm->is2);CHKERRQ(ierr);
  }

  /* (7) Gather offsets from all processes */
  ierr = PetscMalloc1(size,&pdipm->nce_all);CHKERRQ(ierr);

  /* Get rstart of KKT matrix */
  ierr = MPI_Scan(&pdipm->n,&rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  rstart -= pdipm->n;

  ierr = MPI_Allgather(&pdipm->nce,1,MPIU_INT,pdipm->nce_all,1,MPIU_INT,comm);CHKERRMPI(ierr);

  ierr = PetscMalloc3(size,&ng_all,size,&nh_all,size,&Jranges);CHKERRQ(ierr);
  ierr = MPI_Allgather(&rstart,1,MPIU_INT,Jranges,1,MPIU_INT,comm);CHKERRMPI(ierr);
  ierr = MPI_Allgather(&pdipm->nh,1,MPIU_INT,nh_all,1,MPIU_INT,comm);CHKERRMPI(ierr);
  ierr = MPI_Allgather(&pdipm->ng,1,MPIU_INT,ng_all,1,MPIU_INT,comm);CHKERRMPI(ierr);

  ierr = MatGetOwnershipRanges(tao->hessian,&rranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&cranges);CHKERRQ(ierr);

  if (pdipm->Ng) {
    ierr = TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre);CHKERRQ(ierr);
    ierr = MatTranspose(tao->jacobian_equality,MAT_INITIAL_MATRIX,&pdipm->jac_equality_trans);CHKERRQ(ierr);
  }
  if (pdipm->Nh) {
    ierr = TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre);CHKERRQ(ierr);
    ierr = MatTranspose(tao->jacobian_inequality,MAT_INITIAL_MATRIX,&pdipm->jac_inequality_trans);CHKERRQ(ierr);
  }

  /* Count dnz,onz for preallocation of KKT matrix */
  jac_equality_trans   = pdipm->jac_equality_trans;
  jac_inequality_trans = pdipm->jac_inequality_trans;
  nce_all = pdipm->nce_all;

  if (pdipm->Nxfixed) {
    ierr = MatTranspose(pdipm->Jce_xfixed,MAT_INITIAL_MATRIX,&Jce_xfixed_trans);CHKERRQ(ierr);
  }
  ierr = MatTranspose(pdipm->Jci_xb,MAT_INITIAL_MATRIX,&Jci_xb_trans);CHKERRQ(ierr);

  ierr = MatPreallocateInitialize(comm,pdipm->n,pdipm->n,dnz,onz);CHKERRQ(ierr);

  /* 1st row block of KKT matrix: [Wxx; gradCe'; -gradCi'; 0] */
  ierr = TaoPDIPMEvaluateFunctionsAndJacobians(tao,pdipm->x);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);

  /* Insert tao->hessian */
  ierr = MatGetOwnershipRange(tao->hessian,&rjstart,NULL);CHKERRQ(ierr);
  for (i=0; i<pdipm->nx; i++){
    row = rstart + i;

    ierr = MatGetRow(tao->hessian,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      col = aj[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);

    if (pdipm->ng) {
      /* Insert grad g' */
      ierr = MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_equality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    /* Insert Jce_xfixed^T' */
    if (pdipm->nxfixed) {
      ierr = MatGetRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(pdipm->Jce_xfixed,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + ng_all[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    if (pdipm->nh) {
      /* Insert -grad h' */
      ierr = MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_inequality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    /* Insert Jci_xb^T' */
    ierr = MatGetRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(pdipm->Jci_xb,&ranges);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      /* find row ownership of */
      while (aj[j] >= ranges[proc+1]) proc++;
      nx_all = rranges[proc+1] - rranges[proc];
      col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc] + nh_all[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
  }

  /* 2nd Row block of KKT matrix: [grad Ce, deltac*I, 0, 0] */
  if (pdipm->Ng) {
    ierr = MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdipm->ng; i++){
      row = rstart + pdipm->off_lambdae + i;

      ierr = MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr); /* grad g */
      }
      ierr = MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }
  }
  /* Jce_xfixed */
  if (pdipm->Nxfixed) {
    ierr = MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,NULL);CHKERRQ(ierr);
    for (i=0; i < (pdipm->nce - pdipm->ng); i++){
      row = rstart + pdipm->off_lambdae + pdipm->ng + i;

      ierr = MatGetRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
      if (nc != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");

      proc = 0;
      j    = 0;
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    }
  }

  /* 3rd Row block of KKT matrix: [ gradCi, 0, deltac*I, -I] */
  if (pdipm->Nh) {
    ierr = MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdipm->nh; i++){
      row = rstart + pdipm->off_lambdai + i;

      ierr = MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr); /* grad h */
      }
      ierr = MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }
    /* I */
    for (i=0; i < pdipm->nh; i++){
      row = rstart + pdipm->off_lambdai + i;
      col = rstart + pdipm->off_z + i;
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
  }

  /* Jci_xb */
  ierr = MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,NULL);CHKERRQ(ierr);
  for (i=0; i < (pdipm->nci - pdipm->nh); i++){
    row = rstart + pdipm->off_lambdai + pdipm->nh + i;

    ierr = MatGetRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    if (nc != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    /* I */
    col = rstart + pdipm->off_z + pdipm->nh + i;
    ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
  }

  /* 4-th Row block of KKT matrix: Z and Ci */
  for (i=0; i < pdipm->nci; i++) {
    row     = rstart + pdipm->off_z + i;
    cols1[0] = rstart + pdipm->off_lambdai + i;
    cols1[1] = row;
    ierr = MatPreallocateSet(row,2,cols1,dnz,onz);CHKERRQ(ierr);
  }

  /* diagonal entry */
  for (i=0; i<pdipm->n; i++) dnz[i]++; /* diagonal entry */

  /* Create KKT matrix */
  ierr = MatCreate(comm,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,pdipm->n,pdipm->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  /* ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr); */
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  pdipm->K = J;

  /* (8) Set up nonlinear solver SNES */
  ierr = SNESSetFunction(pdipm->snes,NULL,TaoSNESFunction_PDIPM,(void*)tao);CHKERRQ(ierr);
  ierr = SNESSetJacobian(pdipm->snes,J,J,TaoSNESJacobian_PDIPM,(void*)tao);CHKERRQ(ierr);

  if (pdipm->solve_reduced_kkt) {
    PC pc;
    ierr = KSPGetPC(tao->ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"2",pdipm->is2);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"1",pdipm->is1);CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(pdipm->snes);CHKERRQ(ierr);

  /* (9) Insert constant entries to  K */
  /* Set 0.0 to diagonal of K, so that the solver does not complain *about missing diagonal value */
  ierr = MatGetOwnershipRange(J,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++){
    ierr = MatSetValue(J,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* In case Wxx has no diagonal entries preset set diagonal to deltaw given */
  if (pdipm->kkt_pd){
      for (i=0; i<pdipm->nh; i++){
        row  = rstart + i;
        ierr = MatSetValue(J,row,row,pdipm->deltaw,INSERT_VALUES);CHKERRQ(ierr);
      }
  }

  /* Row block of K: [ grad Ce, 0, 0, 0] */
  if (pdipm->Nxfixed) {
    ierr = MatGetOwnershipRange(pdipm->Jce_xfixed,&Jcrstart,NULL);CHKERRQ(ierr);
    for (i=0; i < (pdipm->nce - pdipm->ng); i++){
      row = rstart + pdipm->off_lambdae + pdipm->ng + i;

      ierr = MatGetRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (cols[j] >= cranges[proc+1]) proc++;
        col = cols[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(J,row,col,aa[j],INSERT_VALUES);CHKERRQ(ierr); /* grad Ce */
        ierr = MatSetValue(J,col,row,aa[j],INSERT_VALUES);CHKERRQ(ierr); /* grad Ce' */
      }
      ierr = MatRestoreRow(pdipm->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
    }
  }

  /* Row block of K: [ -grad Ci, 0, 0, I] */
  ierr = MatGetOwnershipRange(pdipm->Jci_xb,&Jcrstart,NULL);CHKERRQ(ierr);
  for (i=0; i < pdipm->nci - pdipm->nh; i++){
    row = rstart + pdipm->off_lambdai + pdipm->nh + i;

    ierr = MatGetRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatSetValue(J,col,row,-aa[j],INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(J,row,col,-aa[j],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(pdipm->Jci_xb,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);

    col = rstart + pdipm->off_z + pdipm->nh + i;
    ierr = MatSetValue(J,row,col,1,INSERT_VALUES);CHKERRQ(ierr);
  }

  for (i=0; i < pdipm->nh; i++){
    row = rstart + pdipm->off_lambdai + i;
    col = rstart + pdipm->off_z + i;
    ierr = MatSetValue(J,row,col,1,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Row block of K: [ 0, 0, I, ...] */
  for (i=0; i < pdipm->nci; i++){
    row = rstart + pdipm->off_z + i;
    col = rstart + pdipm->off_lambdai + i;
    ierr = MatSetValue(J,row,col,1,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (pdipm->Nxfixed) {
    ierr = MatDestroy(&Jce_xfixed_trans);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Jci_xb_trans);CHKERRQ(ierr);
  ierr = PetscFree3(ng_all,nh_all,Jranges);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Freeing Vectors assocaiated with KKT (X) */
  ierr = VecDestroy(&pdipm->x);CHKERRQ(ierr); /* Solution x */
  ierr = VecDestroy(&pdipm->lambdae);CHKERRQ(ierr); /* Equality constraints lagrangian multiplier*/
  ierr = VecDestroy(&pdipm->lambdai);CHKERRQ(ierr); /* Inequality constraints lagrangian multiplier*/
  ierr = VecDestroy(&pdipm->z);CHKERRQ(ierr);       /* Slack variables */
  ierr = VecDestroy(&pdipm->X);CHKERRQ(ierr);       /* Big KKT system vector [x; lambdae; lambdai; z] */

  /* work vectors */
  ierr = VecDestroy(&pdipm->lambdae_xfixed);CHKERRQ(ierr);
  ierr = VecDestroy(&pdipm->lambdai_xb);CHKERRQ(ierr);

  /* Legrangian equality and inequality Vec */
  ierr = VecDestroy(&pdipm->ce);CHKERRQ(ierr); /* Vec of equality constraints */
  ierr = VecDestroy(&pdipm->ci);CHKERRQ(ierr); /* Vec of inequality constraints */

  /* Matrices */
  ierr = MatDestroy(&pdipm->Jce_xfixed);CHKERRQ(ierr);
  ierr = MatDestroy(&pdipm->Jci_xb);CHKERRQ(ierr); /* Jacobian of inequality constraints Jci = [tao->jacobian_inequality ; J(nxub); J(nxlb); J(nxbx)] */
  ierr = MatDestroy(&pdipm->K);CHKERRQ(ierr);

  /* Index Sets */
  if (pdipm->Nxub) {
    ierr = ISDestroy(&pdipm->isxub);CHKERRQ(ierr);    /* Finite upper bound only -inf < x < ub */
  }

  if (pdipm->Nxlb) {
    ierr = ISDestroy(&pdipm->isxlb);CHKERRQ(ierr);    /* Finite lower bound only  lb <= x < inf */
  }

  if (pdipm->Nxfixed) {
    ierr = ISDestroy(&pdipm->isxfixed);CHKERRQ(ierr); /* Fixed variables         lb =  x = ub */
  }

  if (pdipm->Nxbox) {
    ierr = ISDestroy(&pdipm->isxbox);CHKERRQ(ierr);   /* Boxed variables         lb <= x <= ub */
  }

  if (pdipm->Nxfree) {
    ierr = ISDestroy(&pdipm->isxfree);CHKERRQ(ierr);  /* Free variables        -inf <= x <= inf */
  }

  if (pdipm->solve_reduced_kkt) {
    ierr = ISDestroy(&pdipm->is1);CHKERRQ(ierr);
    ierr = ISDestroy(&pdipm->is2);CHKERRQ(ierr);
  }

  /* SNES */
  ierr = SNESDestroy(&pdipm->snes);CHKERRQ(ierr); /* Nonlinear solver */
  ierr = PetscFree(pdipm->nce_all);CHKERRQ(ierr);
  ierr = MatDestroy(&pdipm->jac_equality_trans);CHKERRQ(ierr);
  ierr = MatDestroy(&pdipm->jac_inequality_trans);CHKERRQ(ierr);

  /* Destroy pdipm */
  ierr = PetscFree(tao->data);CHKERRQ(ierr); /* Holding locations of pdipm */

  /* Destroy Dual */
  ierr = VecDestroy(&tao->DE);CHKERRQ(ierr); /* equality dual */
  ierr = VecDestroy(&tao->DI);CHKERRQ(ierr); /* dinequality dual */
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetFromOptions_PDIPM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_PDIPM      *pdipm = (TAO_PDIPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PDIPM method for constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pdipm_push_init_slack","parameter to push initial slack variables away from bounds",NULL,pdipm->push_init_slack,&pdipm->push_init_slack,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pdipm_push_init_lambdai","parameter to push initial (inequality) dual variables away from bounds",NULL,pdipm->push_init_lambdai,&pdipm->push_init_lambdai,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_pdipm_solve_reduced_kkt","Solve reduced KKT system using Schur-complement",NULL,pdipm->solve_reduced_kkt,&pdipm->solve_reduced_kkt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pdipm_mu_update_factor","Update scalar for barrier parameter (mu) update",NULL,pdipm->mu_update_factor,&pdipm->mu_update_factor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_pdipm_symmetric_kkt","Solve non reduced symmetric KKT system",NULL,pdipm->solve_symmetric_kkt,&pdipm->solve_symmetric_kkt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_pdipm_kkt_shift_pd","Add shifts to make KKT matrix positive definite",NULL,pdipm->kkt_pd,&pdipm->kkt_pd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_PDIPM;
  tao->ops->solve          = TaoSolve_PDIPM;
  tao->ops->setfromoptions = TaoSetFromOptions_PDIPM;
  tao->ops->view           = TaoView_PDIPM;
  tao->ops->destroy        = TaoDestroy_PDIPM;

  ierr = PetscNewLog(tao,&pdipm);CHKERRQ(ierr);
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

  ierr = SNESCreate(((PetscObject)tao)->comm,&pdipm->snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(pdipm->snes,tao->hdr.prefix);CHKERRQ(ierr);
  ierr = SNESGetKSP(pdipm->snes,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
