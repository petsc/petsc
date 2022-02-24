
static char help[] = "Solves a RBF kernel matrix with KSP and PCH2OPUS.\n\n";

#include <petscksp.h>

typedef struct {
  PetscReal sigma;
  PetscReal *l;
  PetscReal lambda;
} RBFCtx;

static PetscScalar RBF(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
  RBFCtx    *rbfctx = (RBFCtx*)ctx;
  PetscInt  d;
  PetscReal diff = 0.0;
  PetscReal s = rbfctx->sigma;
  PetscReal *l = rbfctx->l;
  PetscReal lambda = rbfctx->lambda;

  for (d = 0; d < sdim; d++) { diff += (x[d] - y[d]) * (x[d] - y[d]) / (l[d] * l[d]); }
  return s * s * PetscExpReal(-0.5 * diff) + (diff != 0.0 ? 0.0 : lambda);
}

int main(int argc,char **args)
{
  Vec            x, b, u,d;
  Mat            A,Ae = NULL, Ad = NULL;
  KSP            ksp;
  PetscRandom    r;
  PC             pc;
  PetscReal      norm,*coords,eta,scale = 0.5;
  PetscErrorCode ierr;
  PetscInt       basisord,leafsize,sdim,n,its,i;
  PetscMPIInt    size;
  RBFCtx         fctx;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  sdim = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-sdim",&sdim,NULL));
  n    = 32;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  eta  = 0.6;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL));
  leafsize = 32;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-leafsize",&leafsize,NULL));
  basisord = 8;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-basisord",&basisord,NULL));

  /* Create random points */
  CHKERRQ(PetscMalloc1(sdim*n,&coords));
  CHKERRQ(PetscRandomGetValuesReal(r,sdim*n,coords));

  fctx.lambda = 0.01;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-lambda",&fctx.lambda,NULL));
  CHKERRQ(PetscRandomGetValueReal(r,&fctx.sigma));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-sigma",&fctx.sigma,NULL));
  CHKERRQ(PetscMalloc1(sdim,&fctx.l));
  CHKERRQ(PetscRandomGetValuesReal(r,sdim,fctx.l));
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-l",fctx.l,(i=sdim,&i),NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-scale",&scale,NULL));

  /* Populate dense matrix for comparisons */
  {
    PetscInt i,j;

    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&Ad));
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        CHKERRQ(MatSetValue(Ad,i,j,RBF(sdim,coords + i*sdim,coords + j*sdim,&fctx),INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(Ad,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Ad,MAT_FINAL_ASSEMBLY));
  }

  /* Create and assemble the matrix */
  CHKERRQ(MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,sdim,coords,PETSC_FALSE,RBF,&fctx,eta,leafsize,basisord,&A));
  CHKERRQ(MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"RBF"));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(A,NULL,"-rbf_view"));

  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecDuplicate(x,&u));
  CHKERRQ(VecDuplicate(x,&d));

  {
    PetscReal norm;
    CHKERRQ(MatComputeOperator(A,MATDENSE,&Ae));
    CHKERRQ(MatAXPY(Ae,-1.0,Ad,SAME_NONZERO_PATTERN));
    CHKERRQ(MatGetDiagonal(Ae,d));
    CHKERRQ(MatViewFromOptions(Ae,NULL,"-A_view"));
    CHKERRQ(MatViewFromOptions(Ae,NULL,"-D_view"));
    CHKERRQ(MatNorm(Ae,NORM_FROBENIUS,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approx err %g\n",norm));
    CHKERRQ(VecNorm(d,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approx err (diag) %g\n",norm));
    CHKERRQ(MatDestroy(&Ae));
  }

  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(Ad,u,b));
  CHKERRQ(MatViewFromOptions(Ad,NULL,"-Ad_view"));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,Ad,A));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCH2OPUS));
  CHKERRQ(KSPSetFromOptions(ksp));
  /* we can also pass the points coordinates
     In this case it is not needed, since the preconditioning
     matrix is of type H2OPUS */
  CHKERRQ(PCSetCoordinates(pc,sdim,n,coords));

  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* change lambda and reassemble */
  CHKERRQ(VecSet(x,(scale-1.)*fctx.lambda));
  CHKERRQ(MatDiagonalSet(Ad,x,ADD_VALUES));
  fctx.lambda *= scale;
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  {
    PetscReal norm;
    CHKERRQ(MatComputeOperator(A,MATDENSE,&Ae));
    CHKERRQ(MatAXPY(Ae,-1.0,Ad,SAME_NONZERO_PATTERN));
    CHKERRQ(MatGetDiagonal(Ae,d));
    CHKERRQ(MatViewFromOptions(Ae,NULL,"-A_view"));
    CHKERRQ(MatViewFromOptions(Ae,NULL,"-D_view"));
    CHKERRQ(MatNorm(Ae,NORM_FROBENIUS,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approx err %g\n",norm));
    CHKERRQ(VecNorm(d,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Approx err (diag) %g\n",norm));
    CHKERRQ(MatDestroy(&Ae));
  }
  CHKERRQ(KSPSetOperators(ksp,Ad,A));
  CHKERRQ(MatMult(Ad,u,b));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(MatMult(Ad,x,u));
  CHKERRQ(VecAXPY(u,-1.0,b));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm error %g, Iterations %D\n",(double)norm,its));

  CHKERRQ(PetscFree(coords));
  CHKERRQ(PetscFree(fctx.l));
  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&d));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Ad));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: h2opus

  test:
    requires: h2opus !single
    suffix: 1
    args: -ksp_error_if_not_converged -pc_h2opus_monitor

  test:
    requires: h2opus !single
    suffix: 1_ns
    output_file: output/ex21_1.out
    args: -ksp_error_if_not_converged -pc_h2opus_monitor -pc_h2opus_hyperorder 2

  test:
    requires: h2opus !single
    suffix: 2
    args: -ksp_error_if_not_converged -pc_h2opus_monitor -pc_h2opus_hyperorder 4

TEST*/
