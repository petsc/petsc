
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
  PetscInt       basisord,leafsize,sdim,n,its,i;
  PetscMPIInt    size;
  RBFCtx         fctx;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetFromOptions(r));

  sdim = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-sdim",&sdim,NULL));
  n    = 32;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  eta  = 0.6;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL));
  leafsize = 32;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-leafsize",&leafsize,NULL));
  basisord = 8;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-basisord",&basisord,NULL));

  /* Create random points */
  PetscCall(PetscMalloc1(sdim*n,&coords));
  PetscCall(PetscRandomGetValuesReal(r,sdim*n,coords));

  fctx.lambda = 0.01;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-lambda",&fctx.lambda,NULL));
  PetscCall(PetscRandomGetValueReal(r,&fctx.sigma));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-sigma",&fctx.sigma,NULL));
  PetscCall(PetscMalloc1(sdim,&fctx.l));
  PetscCall(PetscRandomGetValuesReal(r,sdim,fctx.l));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-l",fctx.l,(i=sdim,&i),NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-scale",&scale,NULL));

  /* Populate dense matrix for comparisons */
  {
    PetscInt i,j;

    PetscCall(MatCreateDense(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&Ad));
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        PetscCall(MatSetValue(Ad,i,j,RBF(sdim,coords + i*sdim,coords + j*sdim,&fctx),INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Ad,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ad,MAT_FINAL_ASSEMBLY));
  }

  /* Create and assemble the matrix */
  PetscCall(MatCreateH2OpusFromKernel(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,sdim,coords,PETSC_FALSE,RBF,&fctx,eta,leafsize,basisord,&A));
  PetscCall(MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject)A,"RBF"));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A,NULL,"-rbf_view"));

  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(VecDuplicate(x,&u));
  PetscCall(VecDuplicate(x,&d));

  {
    PetscReal norm;
    PetscCall(MatComputeOperator(A,MATDENSE,&Ae));
    PetscCall(MatAXPY(Ae,-1.0,Ad,SAME_NONZERO_PATTERN));
    PetscCall(MatGetDiagonal(Ae,d));
    PetscCall(MatViewFromOptions(Ae,NULL,"-A_view"));
    PetscCall(MatViewFromOptions(Ae,NULL,"-D_view"));
    PetscCall(MatNorm(Ae,NORM_FROBENIUS,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Approx err %g\n",norm));
    PetscCall(VecNorm(d,NORM_2,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Approx err (diag) %g\n",norm));
    PetscCall(MatDestroy(&Ae));
  }

  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(Ad,u,b));
  PetscCall(MatViewFromOptions(Ad,NULL,"-Ad_view"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,Ad,A));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCH2OPUS));
  PetscCall(KSPSetFromOptions(ksp));
  /* we can also pass the points coordinates
     In this case it is not needed, since the preconditioning
     matrix is of type H2OPUS */
  PetscCall(PCSetCoordinates(pc,sdim,n,coords));

  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* change lambda and reassemble */
  PetscCall(VecSet(x,(scale-1.)*fctx.lambda));
  PetscCall(MatDiagonalSet(Ad,x,ADD_VALUES));
  fctx.lambda *= scale;
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  {
    PetscReal norm;
    PetscCall(MatComputeOperator(A,MATDENSE,&Ae));
    PetscCall(MatAXPY(Ae,-1.0,Ad,SAME_NONZERO_PATTERN));
    PetscCall(MatGetDiagonal(Ae,d));
    PetscCall(MatViewFromOptions(Ae,NULL,"-A_view"));
    PetscCall(MatViewFromOptions(Ae,NULL,"-D_view"));
    PetscCall(MatNorm(Ae,NORM_FROBENIUS,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Approx err %g\n",norm));
    PetscCall(VecNorm(d,NORM_2,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Approx err (diag) %g\n",norm));
    PetscCall(MatDestroy(&Ae));
  }
  PetscCall(KSPSetOperators(ksp,Ad,A));
  PetscCall(MatMult(Ad,u,b));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(MatMult(Ad,x,u));
  PetscCall(VecAXPY(u,-1.0,b));
  PetscCall(VecNorm(u,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm error %g, Iterations %D\n",(double)norm,its));

  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(fctx.l));
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Ad));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
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
