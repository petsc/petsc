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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  if (adctx->sparse) {
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobian(A,ADD_VALUES,m,p,adctx->Rec,J,&a);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          J[i][j] *= a;
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  if (adctx->sparse) {
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  ierr = MatShift(A,a);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobianLocal(A,ADD_VALUES,m,p,adctx->Rec,J,&a);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          J[i][j] *= a;
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  ierr = MatShift(A,a);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n,p = adctx->num_params;
  PetscScalar    **J,*concat,**S;

  PetscFunctionBegin;

  /* Allocate memory and concatenate independent variable values with parameter */
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+p,&concat);CHKERRQ(ierr);
  ierr = AdolcMalloc2(n+p,p,&S);CHKERRQ(ierr);
  ierr = Subidentity(p,n,S);CHKERRQ(ierr);
  for (i=0; i<n; i++) concat[i] = u_vec[i];
  for (i=0; i<p; i++) concat[n+i] = params[i];

  /* Propagate the appropriate seed matrix through the forward mode of AD */
  fov_forward(tag,m,n+p,p,concat,S,NULL,J);
  ierr = AdolcFree2(S);CHKERRQ(ierr);
  ierr = PetscFree(concat);CHKERRQ(ierr);

  /* Set matrix values */
  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      if (fabs(J[i][j]) > 1.e-16) {
        ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n,p = adctx->num_params;
  PetscScalar    **J,*concat,**S;

  PetscFunctionBegin;

  /* Allocate memory and concatenate independent variable values with parameter */
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+p,&concat);CHKERRQ(ierr);
  ierr = AdolcMalloc2(n+p,p,&S);CHKERRQ(ierr);
  ierr = Subidentity(p,n,S);CHKERRQ(ierr);
  for (i=0; i<n; i++) concat[i] = u_vec[i];
  for (i=0; i<p; i++) concat[n+i] = params[i];

  /* Propagate the appropriate seed matrix through the forward mode of AD */
  fov_forward(tag,m,n+p,p,concat,S,NULL,J);
  ierr = AdolcFree2(S);CHKERRQ(ierr);
  ierr = PetscFree(concat);CHKERRQ(ierr);

  /* Set matrix values */
  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      if (fabs(J[i][j]) > 1.e-16) {
        ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(tag1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag1,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverDiagonalLocal(diag,INSERT_VALUES,m,adctx->rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        ierr = VecSetValuesLocal(diag,1,&i,&J[i][i],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(diag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diag);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(tag2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(tag2,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverDiagonalLocal(diag,ADD_VALUES,m,adctx->rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        J[i][i] *= a;
        ierr = VecSetValuesLocal(diag,1,&i,&J[i][i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(diag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diag);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

