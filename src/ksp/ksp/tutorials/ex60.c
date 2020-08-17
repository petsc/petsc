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
  PetscErrorCode ierr;
  PCNoise_Ctx    *ctx;
  PetscReal      nrmin, nrmnoise;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);

  /* xout is ||xin|| * ctx->eta*  f, where f is a pseudorandom unit vector
    (Note that this should always be combined additively with another PC) */
  ierr = VecSetRandom(xout,ctx->random);CHKERRQ(ierr);
  ierr = VecNorm(xin,NORM_2,&nrmin);CHKERRQ(ierr);
  ierr = VecNorm(xout,NORM_2,&nrmnoise);CHKERRQ(ierr);
  ierr = VecScale(xout,ctx->eta*(nrmin/nrmnoise));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetup_Noise(PC pc)
{
  PetscErrorCode ierr;
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&ctx->random);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(ctx->random,-1.0,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_Noise(PC pc)
{
  PetscErrorCode ierr;
  PCNoise_Ctx    *ctx;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&ctx->random);CHKERRQ(ierr);
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
  if (!i){
    return 1e-2;
  }else{
    return 1. + (kappa*((PetscScalar)(i-1)))/(PetscScalar)(n-2);
  }
}

static PetscErrorCode AssembleDiagonalMatrix(Mat A, PetscScalar (*diagfunc)(PetscInt,PetscInt))
{
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,n;
  PetscScalar    val;

  PetscFunctionBeginUser;
  ierr = MatGetSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart;i<rend;++i){
    val = diagfunc(i,n);
    ierr = MatSetValues(A,1,&i,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n=10000,its,dfid=1;
  Vec            x,b,u;
  Mat            A;
  KSP            ksp;
  PC             pc,pcnoise;
  PCNoise_Ctx    ctx={0,NULL};
  PetscReal      eta=0.1,norm;
  PetscScalar(*diagfunc)(PetscInt,PetscInt);

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Process command line options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-eta",&eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-diagfunc",&dfid,NULL);CHKERRQ(ierr);
  switch(dfid){
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
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = AssembleDiagonalMatrix(A,diagfunc);CHKERRQ(ierr);

  /* Allocate vectors and manufacture an exact solution and rhs */
  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"Computed Solution");CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&b,NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)b,"RHS");CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&u,NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Reference Solution");CHKERRQ(ierr);
  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* Create a KSP object */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /* Set up a composite preconditioner */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCOMPOSITE);CHKERRQ(ierr); /* default composite with single Identity PC */
  ierr = PCCompositeSetType(pc,PC_COMPOSITE_ADDITIVE);CHKERRQ(ierr);
  ierr = PCCompositeAddPC(pc,PCNONE);CHKERRQ(ierr);
  if (eta > 0){
    ierr = PCCompositeAddPC(pc,PCSHELL);CHKERRQ(ierr);
    ierr = PCCompositeGetPC(pc,1,&pcnoise);CHKERRQ(ierr);
    ctx.eta = eta;
    ierr = PCShellSetContext(pcnoise,&ctx);CHKERRQ(ierr);
    ierr = PCShellSetApply(pcnoise,PCApply_Noise);CHKERRQ(ierr);
    ierr = PCShellSetSetUp(pcnoise,PCSetup_Noise);CHKERRQ(ierr);
    ierr = PCShellSetDestroy(pcnoise,PCDestroy_Noise);CHKERRQ(ierr);
    ierr = PCShellSetName(pcnoise,"Noise PC");CHKERRQ(ierr);
  }

  /* Set KSP from options (this can override the PC just defined) */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Solve */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Compute error */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"Error");CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  /* Destroy objects and finalize */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
