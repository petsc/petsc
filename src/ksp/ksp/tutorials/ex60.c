static const char help[]="Example demonstrating the effect of choosing FCG over CG\n\
for a simple diagonal system with a noisy preconditioner implemented using PCShell\n\
Accepts an option -n for the problem size\n\
Accepts an option -eta for the noise amplitude (set to 0 to deactivate)\n\
Accepts an option -diagfunc [1,2,3] to select from different eigenvalue distributions\n\
\n";

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
  PetscCall(PCShellGetContext(pc,&ctx));

  /* xout is ||xin|| * ctx->eta*  f, where f is a pseudorandom unit vector
    (Note that this should always be combined additively with another PC) */
  PetscCall(VecSetRandom(xout,ctx->random));
  PetscCall(VecNorm(xin,NORM_2,&nrmin));
  PetscCall(VecNorm(xout,NORM_2,&nrmnoise));
  PetscCall(VecScale(xout,ctx->eta*(nrmin/nrmnoise)));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetup_Noise(PC pc)
{
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,&ctx));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&ctx->random));
  PetscCall(PetscRandomSetInterval(ctx->random,-1.0,1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_Noise(PC pc)
{
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,&ctx));
  PetscCall(PetscRandomDestroy(&ctx->random));
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
  PetscCall(MatGetSize(A,NULL,&n));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart;i<rend;++i) {
    val = diagfunc(i,n);
    PetscCall(MatSetValues(A,1,&i,1,&i,&val,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Process command line options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-diagfunc",&dfid,NULL));
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
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(AssembleDiagonalMatrix(A,diagfunc));

  /* Allocate vectors and manufacture an exact solution and rhs */
  PetscCall(MatCreateVecs(A,&x,NULL));
  PetscCall(PetscObjectSetName((PetscObject)x,"Computed Solution"));
  PetscCall(MatCreateVecs(A,&b,NULL));
  PetscCall(PetscObjectSetName((PetscObject)b,"RHS"));
  PetscCall(MatCreateVecs(A,&u,NULL));
  PetscCall(PetscObjectSetName((PetscObject)u,"Reference Solution"));
  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(A,u,b));

  /* Create a KSP object */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));

  /* Set up a composite preconditioner */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCCOMPOSITE)); /* default composite with single Identity PC */
  PetscCall(PCCompositeSetType(pc,PC_COMPOSITE_ADDITIVE));
  PetscCall(PCCompositeAddPCType(pc,PCNONE));
  if (eta > 0) {
    PetscCall(PCCompositeAddPCType(pc,PCSHELL));
    PetscCall(PCCompositeGetPC(pc,1,&pcnoise));
    ctx.eta = eta;
    PetscCall(PCShellSetContext(pcnoise,&ctx));
    PetscCall(PCShellSetApply(pcnoise,PCApply_Noise));
    PetscCall(PCShellSetSetUp(pcnoise,PCSetup_Noise));
    PetscCall(PCShellSetDestroy(pcnoise,PCDestroy_Noise));
    PetscCall(PCShellSetName(pcnoise,"Noise PC"));
  }

  /* Set KSP from options (this can override the PC just defined) */
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve */
  PetscCall(KSPSolve(ksp,b,x));

  /* Compute error */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(PetscObjectSetName((PetscObject)x,"Error"));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %" PetscInt_FMT "\n",(double)norm,its));

  /* Destroy objects and finalize */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
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
