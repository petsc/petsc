
static char help[] = "Solves a RBF kernel matrix with KSP and PCHARA.\n\n";

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
  Vec            x, b, u;
  Mat            A,Ae = NULL, Ad = NULL;
  KSP            ksp;
  PetscRandom    r;
  PC             pc;
  PetscReal      norm,*coords,eta;
  PetscErrorCode ierr;
  PetscInt       basisord,leafsize,sdim,n,its,i;
  PetscMPIInt    size;
  RBFCtx         fctx;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  sdim = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-sdim",&sdim,NULL);CHKERRQ(ierr);
  n    = 32;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  eta  = 0.6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL);CHKERRQ(ierr);
  leafsize = 32;
  ierr = PetscOptionsGetInt(NULL,NULL,"-leafsize",&leafsize,NULL);CHKERRQ(ierr);
  basisord = 8;
  ierr = PetscOptionsGetInt(NULL,NULL,"-basisord",&basisord,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(sdim*n,&coords);CHKERRQ(ierr);
  for (i=0;i<sdim*n;i++) {
    ierr = PetscRandomGetValueReal(r,coords + i);CHKERRQ(ierr);
  }
  fctx.lambda = 0.01;
  ierr = PetscOptionsGetReal(NULL,NULL,"-lambda",&fctx.lambda,NULL);CHKERRQ(ierr);
  ierr = PetscRandomGetValueReal(r,&fctx.sigma);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-sigma",&fctx.sigma,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(sdim,&fctx.l);CHKERRQ(ierr);
  for (i=0;i<sdim;i++) {
    ierr = PetscRandomGetValueReal(r,&fctx.l[i]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-l",fctx.l,(i=sdim,&i),NULL);CHKERRQ(ierr);
  {
    PetscInt i,j;

    ierr = MatCreateDense(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&Ad);CHKERRQ(ierr);
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        ierr = MatSetValue(Ad,i,j,RBF(sdim,coords + i*sdim,coords + j*sdim,&fctx),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(Ad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatCreateHaraFromKernel(PETSC_COMM_WORLD,n,n,PETSC_DECIDE,PETSC_DECIDE,sdim,coords,RBF,&fctx,eta,leafsize,basisord,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  {
    PetscReal norm;
    ierr = MatComputeOperator(A,MATDENSE,&Ae);CHKERRQ(ierr);
    ierr = MatAXPY(Ae,-1.0,Ad,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatViewFromOptions(Ae,NULL,"-A_view");CHKERRQ(ierr);
    ierr = MatNorm(Ae,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Approx err %g\n",norm);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = MatViewFromOptions(Ad,NULL,"-Ad_view");CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,Ad ? Ad : A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCHARA);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetCoordinates(pc,sdim,n,coords);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFree(fctx.l);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&Ad);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hara

  test:
    requires: hara
    suffix: 1
    args: -pc_hara_monitor

  test:
    requires: hara
    suffix: 1_ns
    output_file: output/ex21_1.out
    args: -pc_hara_monitor -pc_hara_hyperorder 2

  test:
    requires: hara
    suffix: 2
    args: -pc_hara_monitor -pc_hara_hyperorder 4

TEST*/
