#include <petsc/private/petscimpl.h>
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

.seealso: `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscLinearRegression(PetscInt n, const PetscReal x[], const PetscReal y[], PetscReal *slope, PetscReal *intercept)
{
  PetscScalar  H[4];
  PetscReal   *X, *Y, beta[2];

  PetscFunctionBegin;
  if (n) {
    PetscValidRealPointer(x,2);
    PetscValidRealPointer(y,3);
  }
  PetscValidRealPointer(slope,4);
  PetscValidRealPointer(intercept,5);
  PetscCall(PetscMalloc2(n*2, &X, n*2, &Y));
  for (PetscInt k = 0; k < n; ++k) {
    /* X[n,2] = [1, x] */
    X[k*2+0] = 1.0;
    X[k*2+1] = x[k];
  }
  /* H = X^T X */
  for (PetscInt i = 0; i < 2; ++i) {
    for (PetscInt j = 0; j < 2; ++j) {
      H[i*2+j] = 0.0;
      for (PetscInt k = 0; k < n; ++k) H[i*2+j] += X[k*2+i] * X[k*2+j];
    }
  }
  /* H = (X^T X)^{-1} */
  {
    PetscBLASInt two = 2, ipiv[2], info;
    PetscScalar  work[2];

    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&two, &two, H, &two, ipiv, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&two, H, &two, ipiv, work, &two, &info));
    PetscCall(PetscFPTrapPop());
  }
    /* Y = H X^T */
  for (PetscInt i = 0; i < 2; ++i) {
    for (PetscInt k = 0; k < n; ++k) {
      Y[i*n+k] = 0.0;
      for (PetscInt j = 0; j < 2; ++j) Y[i*n+k] += PetscRealPart(H[i*2+j]) * X[k*2+j];
    }
  }
  /* beta = Y error = [y-intercept, slope] */
  for (PetscInt i = 0; i < 2; ++i) {
    beta[i] = 0.0;
    for (PetscInt k = 0; k < n; ++k) beta[i] += Y[i*n+k] * y[k];
  }
  PetscCall(PetscFree2(X, Y));
  *intercept = beta[0];
  *slope     = beta[1];
  PetscFunctionReturn(0);
}
