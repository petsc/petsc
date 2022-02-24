#include "contexts.cxx"
#include "sparse.cxx"
#include "init.cxx"
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

/* --------------------------------------------------------------------------------
   Drivers for RHSJacobian and IJacobian
   ----------------------------------------------------------------------------- */

/*
  Compute Jacobian for explicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended for non-toy problems!).

  Input parameters:
  tag   - tape identifier
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeRHSJacobian(PetscInt tag,Mat A,const PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(AdolcFree2(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for explicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended for non-toy problems!).

  Input parameters:
  tag   - tape identifier
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeRHSJacobianLocal(PetscInt tag,Mat A,const PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(AdolcFree2(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended for non-toy problems!).

  Input parameters:
  tag1   - tape identifier for df/dx part
  tag2   - tape identifier for df/d(xdot) part
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeIJacobian(PetscInt tag1,PetscInt tag2,Mat A,const PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  CHKERRQ(MatZeroEntries(A));
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobian(A,ADD_VALUES,m,p,adctx->Rec,J,&a));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          J[i][j] *= a;
          CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],ADD_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(AdolcFree2(J));
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for implicit TS in the special case where it is
  known that the mass matrix is simply the identity. i.e. We have
  a problem of the form
                        du/dt = F(u).

  Input parameters:
  tag   - tape identifier for df/dx part
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeIJacobianIDMass(PetscInt tag,Mat A,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  CHKERRQ(MatZeroEntries(A));
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(AdolcFree2(J));

  /* a * dF/d(xdot) part */
  CHKERRQ(MatShift(A,a));
  PetscFunctionReturn(0);
}

/*
  Compute local portion of Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended for non-toy problems!).

  Input parameters:
  tag1   - tape identifier for df/dx part
  tag2   - tape identifier for df/d(xdot) part
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeIJacobianLocal(PetscInt tag1,PetscInt tag2,Mat A,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobianLocal(A,ADD_VALUES,m,p,adctx->Rec,J,&a));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          J[i][j] *= a;
          CHKERRQ(MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],ADD_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(AdolcFree2(J));
  PetscFunctionReturn(0);
}

/*
  Compute local portion of Jacobian for implicit TS in the special case where it is
  known that the mass matrix is simply the identity. i.e. We have
  a problem of the form
                        du/dt = F(u).

  Input parameters:
  tag   - tape identifier for df/dx part
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeIJacobianLocalIDMass(PetscInt tag,Mat A,const PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          CHKERRQ(MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(AdolcFree2(J));

  /* a * dF/d(xdot) part */
  CHKERRQ(MatShift(A,a));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------
   Drivers for Jacobian w.r.t. a parameter
   ----------------------------------------------------------------------------- */

/*
  Compute Jacobian w.r.t a parameter for explicit TS.

  Input parameters:
  tag    - tape identifier
  u_vec  - vector at which to evaluate Jacobian
  params - the parameters w.r.t. which we differentiate
  ctx    - ADOL-C context, as defined above

  Output parameter:
  A      - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeRHSJacobianP(PetscInt tag,Mat A,const PetscScalar *u_vec,PetscScalar *params,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n,p = adctx->num_params;
  PetscScalar    **J,*concat,**S;

  PetscFunctionBegin;

  /* Allocate memory and concatenate independent variable values with parameter */
  CHKERRQ(AdolcMalloc2(m,p,&J));
  CHKERRQ(PetscMalloc1(n+p,&concat));
  CHKERRQ(AdolcMalloc2(n+p,p,&S));
  CHKERRQ(Subidentity(p,n,S));
  for (i=0; i<n; i++) concat[i] = u_vec[i];
  for (i=0; i<p; i++) concat[n+i] = params[i];

  /* Propagate the appropriate seed matrix through the forward mode of AD */
  fov_forward(tag,m,n+p,p,concat,S,NULL,J);
  CHKERRQ(AdolcFree2(S));
  CHKERRQ(PetscFree(concat));

  /* Set matrix values */
  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      if (fabs(J[i][j]) > 1.e-16) {
        CHKERRQ(MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
      }
    }
  }
  CHKERRQ(AdolcFree2(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
  Compute local portion of Jacobian w.r.t a parameter for explicit TS.

  Input parameters:
  tag    - tape identifier
  u_vec  - vector at which to evaluate Jacobian
  params - the parameters w.r.t. which we differentiate
  ctx    - ADOL-C context, as defined above

  Output parameter:
  A      - Mat object corresponding to Jacobian
*/
PetscErrorCode PetscAdolcComputeRHSJacobianPLocal(PetscInt tag,Mat A,const PetscScalar *u_vec,PetscScalar *params,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n,p = adctx->num_params;
  PetscScalar    **J,*concat,**S;

  PetscFunctionBegin;

  /* Allocate memory and concatenate independent variable values with parameter */
  CHKERRQ(AdolcMalloc2(m,p,&J));
  CHKERRQ(PetscMalloc1(n+p,&concat));
  CHKERRQ(AdolcMalloc2(n+p,p,&S));
  CHKERRQ(Subidentity(p,n,S));
  for (i=0; i<n; i++) concat[i] = u_vec[i];
  for (i=0; i<p; i++) concat[n+i] = params[i];

  /* Propagate the appropriate seed matrix through the forward mode of AD */
  fov_forward(tag,m,n+p,p,concat,S,NULL,J);
  CHKERRQ(AdolcFree2(S));
  CHKERRQ(PetscFree(concat));

  /* Set matrix values */
  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      if (fabs(J[i][j]) > 1.e-16) {
        CHKERRQ(MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES));
      }
    }
  }
  CHKERRQ(AdolcFree2(J));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------
   Drivers for Jacobian diagonal
   ----------------------------------------------------------------------------- */

/*
  Compute local portion of Jacobian diagonal for implicit TS in compressed format and recover
  from this, using precomputed seed matrix and recovery vector.

  Input parameters:
  tag1  - tape identifier for df/dx part
  tag2  - tape identifier for df/d(xdot) part
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  diag  - Vec object corresponding to Jacobian diagonal
*/
PetscErrorCode PetscAdolcComputeIJacobianAndDiagonalLocal(PetscInt tag1,PetscInt tag2,Vec diag,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscInt       i,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  CHKERRQ(AdolcMalloc2(m,p,&J));

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverDiagonalLocal(diag,INSERT_VALUES,m,adctx->rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        CHKERRQ(VecSetValuesLocal(diag,1,&i,&J[i][i],INSERT_VALUES));
      }
    }
  }
  CHKERRQ(VecAssemblyBegin(diag));
  CHKERRQ(VecAssemblyEnd(diag));

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    CHKERRQ(RecoverDiagonalLocal(diag,ADD_VALUES,m,adctx->rec,J,NULL));
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        J[i][i] *= a;
        CHKERRQ(VecSetValuesLocal(diag,1,&i,&J[i][i],ADD_VALUES));
      }
    }
  }
  CHKERRQ(VecAssemblyBegin(diag));
  CHKERRQ(VecAssemblyEnd(diag));
  CHKERRQ(AdolcFree2(J));
  PetscFunctionReturn(0);
}
