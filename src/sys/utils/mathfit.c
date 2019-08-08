#include <petscsys.h>
#include <petscblaslapack.h>

/*@C
  PetscLinearRegression - Gives the best least-squares linear fit to some x-y data points

  Input Parameters:
+ n - The number of points
. x - The x-values
- y - The y-values

  Output Parameters:
+ slope     - The slope of the best-fit line
- intercept - The y-intercept of the best-fit line

  Level: intermediate

.seealso: PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscLinearRegression(PetscInt n, const PetscReal x[], const PetscReal y[], PetscReal *slope, PetscReal *intercept)
{
  PetscScalar    H[4];
  PetscReal     *X, *Y, beta[2];
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *slope = *intercept = 0.0;
  ierr = PetscMalloc2(n*2, &X, n*2, &Y);CHKERRQ(ierr);
  for (k = 0; k < n; ++k) {
    /* X[n,2] = [1, x] */
    X[k*2+0] = 1.0;
    X[k*2+1] = x[k];
  }
  /* H = X^T X */
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      H[i*2+j] = 0.0;
      for (k = 0; k < n; ++k) {
        H[i*2+j] += X[k*2+i] * X[k*2+j];
      }
    }
  }
  /* H = (X^T X)^{-1} */
  {
    PetscBLASInt two = 2, ipiv[2], info;
    PetscScalar  work[2];

    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&two, &two, H, &two, ipiv, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&two, H, &two, ipiv, work, &two, &info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
    /* Y = H X^T */
  for (i = 0; i < 2; ++i) {
    for (k = 0; k < n; ++k) {
      Y[i*n+k] = 0.0;
      for (j = 0; j < 2; ++j) {
        Y[i*n+k] += PetscRealPart(H[i*2+j]) * X[k*2+j];
      }
    }
  }
  /* beta = Y error = [y-intercept, slope] */
  for (i = 0; i < 2; ++i) {
    beta[i] = 0.0;
    for (k = 0; k < n; ++k) {
      beta[i] += Y[i*n+k] * y[k];
    }
  }
  ierr = PetscFree2(X, Y);CHKERRQ(ierr);
  *intercept = beta[0];
  *slope     = beta[1];
  PetscFunctionReturn(0);
}
