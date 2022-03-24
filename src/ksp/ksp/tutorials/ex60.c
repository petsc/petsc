static const char help[]="Example demonstrating the effect of choosing FCG over CG\n\
for a simple diagonal system with a noisy preconditioner implemented using PCShell\n\
Accepts an option -n for the problem size\n\
Accepts an option -eta for the noise amplitude (set to 0 to deactivate)\n\
Accepts an option -diagfunc [1,2,3] to select from different eigenvalue distributions\n\
\n";

/*T
   Concepts: KSP^using nested solves
   Concepts: KSP^using flexible Krylov methods
   Concepts: PC^using PCShell to define custom PCs
   Concepts: PC^using composite PCs
   Processors: n
T*/

/*
  Solve (in parallel) a diagonal linear system.

  Using PCShell, we define a preconditioner which simply adds noise to the residual.

  This example can be used to test the robustness of Krylov methods (particularly "flexible" ones for unitarily diagonalizable systems)
  to varying preconditioners. Use the command line option -diagfunc [1,2,3] to choose between some predefined eigenvalue distributions.

  The default behavior is to use a composite PC which combines (additively) an identity preconditioner with a preconditioner which
  replaces the input with scaled noise.

  To test with an inner Krylov method instead of noise, use PCKSP,  e.g.
  mpiexec -n 2 ./ex60 -eta 0 -ksp_type fcg -pc_type ksp -ksp_ksp_rtol 1e-1 -ksp_ksp_type cg -ksp_pc_type none
  (note that eta is ignored here, and we specify the analogous quantity, the tolerance of the inner KSP solve,with -ksp_ksp_rtol)

  To test by adding noise to a PC of your choosing (say ilu), run e.g.
  mpiexec -n 2 ./ex60 -eta 0.1 -ksp_type fcg -sub_0_pc_type ilu

  Contributed by Patrick Sanan
*/

#include <petscksp.h>

/* Context to use with our noise PC */
typedef struct {
  PetscReal   eta;
  PetscRandom random;
} PCNoise_Ctx;

PetscErrorCode PCApply_Noise(PC pc,Vec xin,Vec xout)
{
  PCNoise_Ctx    *ctx;
  PetscReal      nrmin, nrmnoise;

  PetscFunctionBeginUser;
  CHKERRQ(PCShellGetContext(pc,&ctx));

  /* xout is ||xin|| * ctx->eta*  f, where f is a pseudorandom unit vector
    (Note that this should always be combined additively with another PC) */
  CHKERRQ(VecSetRandom(xout,ctx->random));
  CHKERRQ(VecNorm(xin,NORM_2,&nrmin));
  CHKERRQ(VecNorm(xout,NORM_2,&nrmnoise));
  CHKERRQ(VecScale(xout,ctx->eta*(nrmin/nrmnoise)));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetup_Noise(PC pc)
{
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(PCShellGetContext(pc,&ctx));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&ctx->random));
  CHKERRQ(PetscRandomSetInterval(ctx->random,-1.0,1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_Noise(PC pc)
{
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(PCShellGetContext(pc,&ctx));
  CHKERRQ(PetscRandomDestroy(&ctx->random));
  PetscFunctionReturn(0);
}

PetscScalar diagFunc1(PetscInt i, PetscInt n)
{
  const PetscScalar kappa = 5.0;
  return 1.0 + (kappa*(PetscScalar)i)/(PetscScalar)(n-1);
}

PetscScalar diagFunc2(PetscInt i, PetscInt n)
{
  const PetscScalar kappa = 50.0;
  return 1.0 + (kappa*(PetscScalar)i)/(PetscScalar)(n-1);
}

PetscScalar diagFunc3(PetscInt i, PetscInt n)
{
  const PetscScalar kappa = 10.0;
  if (!i) {
    return 1e-2;
  }else{
    return 1. + (kappa*((PetscScalar)(i-1)))/(PetscScalar)(n-2);
  }
}

static PetscErrorCode AssembleDiagonalMatrix(Mat A, PetscScalar (*diagfunc)(PetscInt,PetscInt))
{
  PetscInt       i,rstart,rend,n;
  PetscScalar    val;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,NULL,&n));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart;i<rend;++i) {
    val = diagfunc(i,n);
    CHKERRQ(MatSetValues(A,1,&i,1,&i,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       n=10000,its,dfid=1;
  Vec            x,b,u;
  Mat            A;
  KSP            ksp;
  PC             pc,pcnoise;
  PCNoise_Ctx    ctx={0,NULL};
  PetscReal      eta=0.1,norm;
  PetscScalar(*diagfunc)(PetscInt,PetscInt);

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Process command line options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-diagfunc",&dfid,NULL));
  switch(dfid) {
    case 1:
      diagfunc = diagFunc1;
      break;
    case 2:
      diagfunc = diagFunc2;
      break;
    case 3:
      diagfunc = diagFunc3;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unrecognized diagfunc option");
  }

  /* Create a diagonal matrix with a given distribution of diagonal elements */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(AssembleDiagonalMatrix(A,diagfunc));

  /* Allocate vectors and manufacture an exact solution and rhs */
  CHKERRQ(MatCreateVecs(A,&x,NULL));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"Computed Solution"));
  CHKERRQ(MatCreateVecs(A,&b,NULL));
  CHKERRQ(PetscObjectSetName((PetscObject)b,"RHS"));
  CHKERRQ(MatCreateVecs(A,&u,NULL));
  CHKERRQ(PetscObjectSetName((PetscObject)u,"Reference Solution"));
  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(A,u,b));

  /* Create a KSP object */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /* Set up a composite preconditioner */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCOMPOSITE)); /* default composite with single Identity PC */
  CHKERRQ(PCCompositeSetType(pc,PC_COMPOSITE_ADDITIVE));
  CHKERRQ(PCCompositeAddPCType(pc,PCNONE));
  if (eta > 0) {
    CHKERRQ(PCCompositeAddPCType(pc,PCSHELL));
    CHKERRQ(PCCompositeGetPC(pc,1,&pcnoise));
    ctx.eta = eta;
    CHKERRQ(PCShellSetContext(pcnoise,&ctx));
    CHKERRQ(PCShellSetApply(pcnoise,PCApply_Noise));
    CHKERRQ(PCShellSetSetUp(pcnoise,PCSetup_Noise));
    CHKERRQ(PCShellSetDestroy(pcnoise,PCDestroy_Noise));
    CHKERRQ(PCShellSetName(pcnoise,"Noise PC"));
  }

  /* Set KSP from options (this can override the PC just defined) */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* Solve */
  CHKERRQ(KSPSolve(ksp,b,x));

  /* Compute error */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"Error"));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));

  /* Destroy objects and finalize */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type fcg -ksp_fcg_mmax 1 -eta 0.1

   test:
      suffix: 2
      nsize: 2
      args: -ksp_monitor_short -diagfunc 3 -ksp_type fcg -ksp_fcg_mmax 10000 -eta 0.3333

   test:
      suffix: 3
      nsize: 3
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 2 -ksp_type fgmres -eta 0.1

   test:
      suffix: 4
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefcg -ksp_pipefcg_mmax 1 -eta 0.1

   test:
      suffix: 5
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type pipefcg -ksp_pipefcg_mmax 10000 -eta 0.1

   test:
      suffix: 6
      nsize: 4
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type fcg -ksp_fcg_mmax 10000 -eta 0 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-1 -ksp_ksp_max_it 5 -ksp_ksp_converged_reason

   test:
      suffix: 7
      nsize: 4
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type pipefcg -ksp_pipefcg_mmax 10000 -eta 0 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-1 -ksp_ksp_max_it 5 -ksp_ksp_converged_reason

   test:
      suffix: 8
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefgmres -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-2 -ksp_ksp_converged_reason

   test:
      suffix: 9
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefgmres -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-2 -ksp_ksp_converged_reason

TEST*/
