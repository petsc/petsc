#include <petsctao.h>
/*
Description:   ADMM tomography reconstruction example .
               0.5*||Ax-b||^2 + lambda*g(x)
Reference:     BRGN Tomography Example
*/

static char help[] = "Finds the ADMM solution to the under constraint linear model Ax = b, with regularizer. \n\
                      A is a M*N real matrix (M<N), x is sparse. A good regularizer is an L1 regularizer. \n\
                      We first split the operator into 0.5*||Ax-b||^2, f(x), and lambda*||x||_1, g(z), where lambda is user specified weight. \n\
                      g(z) could be either ||z||_1, or ||z||_2^2. Default closed form solution for NORM1 would be soft-threshold, which is \n\
                      natively supported in admm.c with -tao_admm_regularizer_type soft-threshold. Or user can use regular TAO solver for  \n\
                      either NORM1 or NORM2 or TAOSHELL, with -reg {1,2,3} \n\
                      Then, we augment both f and g, and solve it via ADMM. \n\
                      D is the M*N transform matrix so that D*x is sparse. \n";

typedef struct {
  PetscInt  M,N,K,reg;
  PetscReal lambda,eps,mumin;
  Mat       A,ATA,H,Hx,D,Hz,DTD,HF;
  Vec       c,xlb,xub,x,b,workM,workN,workN2,workN3,xGT;    /* observation b, ground truth xGT, the lower bound and upper bound of x*/
} AppCtx;

/*------------------------------------------------------------*/

PetscErrorCode NullJacobian(Tao tao,Vec X,Mat J,Mat Jpre,void *ptr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoShellSolve_SoftThreshold(Tao tao)
{
  PetscReal      lambda, mu;
  AppCtx         *user;
  Vec            out,work,y,x;
  Tao            admm_tao,misfit;

  PetscFunctionBegin;
  user = NULL;
  mu   = 0;
  CHKERRQ(TaoGetADMMParentTao(tao,&admm_tao));
  CHKERRQ(TaoADMMGetMisfitSubsolver(admm_tao, &misfit));
  CHKERRQ(TaoADMMGetSpectralPenalty(admm_tao,&mu));
  CHKERRQ(TaoShellGetContext(tao,&user));

  lambda = user->lambda;
  work   = user->workN;
  CHKERRQ(TaoGetSolution(tao, &out));
  CHKERRQ(TaoGetSolution(misfit, &x));
  CHKERRQ(TaoADMMGetDualVector(admm_tao, &y));

  /* Dx + y/mu */
  CHKERRQ(MatMult(user->D,x,work));
  CHKERRQ(VecAXPY(work,1/mu,y));

  /* soft thresholding */
  CHKERRQ(TaoSoftThreshold(work, -lambda/mu, lambda/mu, out));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MisfitObjectiveAndGradient(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 */
  CHKERRQ(MatMult(user->A,X,user->workM));
  CHKERRQ(VecAXPY(user->workM,-1,user->b));
  CHKERRQ(VecDot(user->workM,user->workM,f));
  *f  *= 0.5;
  /* Gradient. ATAx-ATb */
  CHKERRQ(MatMult(user->ATA,X,user->workN));
  CHKERRQ(MatMultTranspose(user->A,user->b,user->workN2));
  CHKERRQ(VecWAXPY(g,-1.,user->workN2,user->workN));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient1(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  /* compute regularizer objective
   * f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x */
  CHKERRQ(VecCopy(X,user->workN2));
  CHKERRQ(VecPow(user->workN2,2.));
  CHKERRQ(VecShift(user->workN2,user->eps*user->eps));
  CHKERRQ(VecSqrtAbs(user->workN2));
  CHKERRQ(VecCopy(user->workN2, user->workN3));
  CHKERRQ(VecShift(user->workN2,-user->eps));
  CHKERRQ(VecSum(user->workN2,f_reg));
  *f_reg *= user->lambda;
  /* compute regularizer gradient = lambda*x */
  CHKERRQ(VecPointwiseDivide(G_reg,X,user->workN3));
  CHKERRQ(VecScale(G_reg,user->lambda));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient2(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscReal      temp;

  PetscFunctionBegin;
  /* compute regularizer objective = lambda*|z|_2^2 */
  CHKERRQ(VecDot(X,X,&temp));
  *f_reg = 0.5*user->lambda*temp;
  /* compute regularizer gradient = lambda*z */
  CHKERRQ(VecCopy(X,G_reg));
  CHKERRQ(VecScale(G_reg,user->lambda));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode HessianMisfit(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode HessianReg(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(MatMult(user->D,x,user->workN));
  CHKERRQ(VecPow(user->workN2,2.));
  CHKERRQ(VecShift(user->workN2,user->eps*user->eps));
  CHKERRQ(VecSqrtAbs(user->workN2));
  CHKERRQ(VecShift(user->workN2,-user->eps));
  CHKERRQ(VecReciprocal(user->workN2));
  CHKERRQ(VecScale(user->workN2,user->eps*user->eps));
  CHKERRQ(MatDiagonalSet(H,user->workN2,INSERT_VALUES));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode FullObjGrad(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscReal      f_reg;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 + lambda*||x||_2^2*/
  CHKERRQ(MatMult(user->A,X,user->workM));
  CHKERRQ(VecAXPY(user->workM,-1,user->b));
  CHKERRQ(VecDot(user->workM,user->workM,f));
  CHKERRQ(VecNorm(X,NORM_2,&f_reg));
  *f  *= 0.5;
  *f  += user->lambda*f_reg*f_reg;
  /* Gradient. ATAx-ATb + 2*lambda*x */
  CHKERRQ(MatMult(user->ATA,X,user->workN));
  CHKERRQ(MatMultTranspose(user->A,user->b,user->workN2));
  CHKERRQ(VecWAXPY(g,-1.,user->workN2,user->workN));
  CHKERRQ(VecAXPY(g,2*user->lambda,X));
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode HessianFull(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

PetscErrorCode InitializeUserData(AppCtx *user)
{
  char           dataFile[] = "tomographyData_A_b_xGT";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. Debug: change from "tomographyData_A_b_xGT" to "cs1Data_A_b_xGT". */
  PetscViewer    fd;   /* used to load data from file */
  PetscErrorCode ierr;
  PetscInt       k,n;
  PetscScalar    v;
  PetscFunctionBegin;

  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->A));
  CHKERRQ(MatSetType(user->A,MATAIJ));
  CHKERRQ(MatLoad(user->A,fd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->b));
  CHKERRQ(VecLoad(user->b,fd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->xGT));
  CHKERRQ(VecLoad(user->xGT,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(MatGetSize(user->A,&user->M,&user->N));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->D));
  CHKERRQ(MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N));
  CHKERRQ(MatSetFromOptions(user->D));
  CHKERRQ(MatSetUp(user->D));
  for (k=0; k<user->N; k++) {
    v = 1.0;
    n = k+1;
    if (k< user->N -1) {
      CHKERRQ(MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES));
    }
    v    = -1.0;
    CHKERRQ(MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTransposeMatMult(user->D,user->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DTD));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Hz));
  CHKERRQ(MatSetSizes(user->Hz,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N));
  CHKERRQ(MatSetFromOptions(user->Hz));
  CHKERRQ(MatSetUp(user->Hz));
  CHKERRQ(MatAssemblyBegin(user->Hz,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Hz,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&(user->x)));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&(user->workM)));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&(user->workN)));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&(user->workN2)));
  CHKERRQ(VecSetSizes(user->x,PETSC_DECIDE,user->N));
  CHKERRQ(VecSetSizes(user->workM,PETSC_DECIDE,user->M));
  CHKERRQ(VecSetSizes(user->workN,PETSC_DECIDE,user->N));
  CHKERRQ(VecSetSizes(user->workN2,PETSC_DECIDE,user->N));
  CHKERRQ(VecSetFromOptions(user->x));
  CHKERRQ(VecSetFromOptions(user->workM));
  CHKERRQ(VecSetFromOptions(user->workN));
  CHKERRQ(VecSetFromOptions(user->workN2));

  CHKERRQ(VecDuplicate(user->workN,&(user->workN3)));
  CHKERRQ(VecDuplicate(user->x,&(user->xlb)));
  CHKERRQ(VecDuplicate(user->x,&(user->xub)));
  CHKERRQ(VecDuplicate(user->x,&(user->c)));
  CHKERRQ(VecSet(user->xlb,0.0));
  CHKERRQ(VecSet(user->c,0.0));
  CHKERRQ(VecSet(user->xub,PETSC_INFINITY));

  CHKERRQ(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->ATA)));
  CHKERRQ(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->Hx)));
  CHKERRQ(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->HF)));

  CHKERRQ(MatAssemblyBegin(user->ATA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->ATA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(user->Hx,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Hx,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(user->HF,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->HF,MAT_FINAL_ASSEMBLY));

  user->lambda = 1.e-8;
  user->eps    = 1.e-3;
  user->reg    = 2;
  user->mumin  = 5.e-6;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "tomographyADMM.c");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-reg","Regularization scheme for z solver (1,2)", "tomographyADMM.c", user->reg, &(user->reg), NULL));
  CHKERRQ(PetscOptionsReal("-lambda", "The regularization multiplier. 1 default", "tomographyADMM.c", user->lambda, &(user->lambda), NULL));
  CHKERRQ(PetscOptionsReal("-eps", "L1 norm epsilon padding", "tomographyADMM.c", user->eps, &(user->eps), NULL));
  CHKERRQ(PetscOptionsReal("-mumin", "Minimum value for ADMM spectral penalty", "tomographyADMM.c", user->mumin, &(user->mumin), NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&user->A));
  CHKERRQ(MatDestroy(&user->ATA));
  CHKERRQ(MatDestroy(&user->Hx));
  CHKERRQ(MatDestroy(&user->Hz));
  CHKERRQ(MatDestroy(&user->HF));
  CHKERRQ(MatDestroy(&user->D));
  CHKERRQ(MatDestroy(&user->DTD));
  CHKERRQ(VecDestroy(&user->xGT));
  CHKERRQ(VecDestroy(&user->xlb));
  CHKERRQ(VecDestroy(&user->xub));
  CHKERRQ(VecDestroy(&user->b));
  CHKERRQ(VecDestroy(&user->x));
  CHKERRQ(VecDestroy(&user->c));
  CHKERRQ(VecDestroy(&user->workN3));
  CHKERRQ(VecDestroy(&user->workN2));
  CHKERRQ(VecDestroy(&user->workN));
  CHKERRQ(VecDestroy(&user->workM));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Tao            tao,misfit,reg;
  PetscReal      v1,v2;
  AppCtx*        user;
  PetscViewer    fd;
  char           resultFile[] = "tomographyResult_x";

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscNew(&user));
  CHKERRQ(InitializeUserData(user));

  CHKERRQ(TaoCreate(PETSC_COMM_WORLD, &tao));
  CHKERRQ(TaoSetType(tao, TAOADMM));
  CHKERRQ(TaoSetSolution(tao, user->x));
  /* f(x) + g(x) for parent tao */
  CHKERRQ(TaoADMMSetSpectralPenalty(tao,1.));
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL, FullObjGrad, (void*)user));
  CHKERRQ(MatShift(user->HF,user->lambda));
  CHKERRQ(TaoSetHessian(tao, user->HF, user->HF, HessianFull, (void*)user));

  /* f(x) for misfit tao */
  CHKERRQ(TaoADMMSetMisfitObjectiveAndGradientRoutine(tao, MisfitObjectiveAndGradient, (void*)user));
  CHKERRQ(TaoADMMSetMisfitHessianRoutine(tao, user->Hx, user->Hx, HessianMisfit, (void*)user));
  CHKERRQ(TaoADMMSetMisfitHessianChangeStatus(tao,PETSC_FALSE));
  CHKERRQ(TaoADMMSetMisfitConstraintJacobian(tao,user->D,user->D,NullJacobian,(void*)user));

  /* g(x) for regularizer tao */
  if (user->reg == 1) {
    CHKERRQ(TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient1, (void*)user));
    CHKERRQ(TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianReg, (void*)user));
    CHKERRQ(TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE));
  } else if (user->reg == 2) {
    CHKERRQ(TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient2, (void*)user));
    CHKERRQ(MatShift(user->Hz,1));
    CHKERRQ(MatScale(user->Hz,user->lambda));
    CHKERRQ(TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianMisfit, (void*)user));
    CHKERRQ(TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE));
  } else PetscCheck(user->reg == 3,PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Incorrect Reg type"); /* TaoShell case */

  /* Set type for the misfit solver */
  CHKERRQ(TaoADMMGetMisfitSubsolver(tao, &misfit));
  CHKERRQ(TaoADMMGetRegularizationSubsolver(tao, &reg));
  CHKERRQ(TaoSetType(misfit,TAONLS));
  if (user->reg == 3) {
    CHKERRQ(TaoSetType(reg,TAOSHELL));
    CHKERRQ(TaoShellSetContext(reg, (void*) user));
    CHKERRQ(TaoShellSetSolve(reg, TaoShellSolve_SoftThreshold));
  } else {
    CHKERRQ(TaoSetType(reg,TAONLS));
  }
  CHKERRQ(TaoSetVariableBounds(misfit,user->xlb,user->xub));

  /* Soft Thresholding solves the ADMM problem with the L1 regularizer lambda*||z||_1 and the x-z=0 constraint */
  CHKERRQ(TaoADMMSetRegularizerCoefficient(tao, user->lambda));
  CHKERRQ(TaoADMMSetRegularizerConstraintJacobian(tao,NULL,NULL,NullJacobian,(void*)user));
  CHKERRQ(TaoADMMSetMinimumSpectralPenalty(tao,user->mumin));

  CHKERRQ(TaoADMMSetConstraintVectorRHS(tao,user->c));
  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoSolve(tao));

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,resultFile,FILE_MODE_WRITE,&fd));
  CHKERRQ(VecView(user->x,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* compute the error */
  CHKERRQ(VecAXPY(user->x,-1,user->xGT));
  CHKERRQ(VecNorm(user->x,NORM_2,&v1));
  CHKERRQ(VecNorm(user->xGT,NORM_2,&v2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2)));

  /* Free TAO data structures */
  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(DestroyContext(user));
  CHKERRQ(PetscFree(user));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: 1
      localrunfiles: tomographyData_A_b_xGT
      args:  -lambda 1.e-8 -tao_monitor -tao_type nls -tao_nls_pc_type icc

   test:
      suffix: 2
      localrunfiles: tomographyData_A_b_xGT
      args:  -reg 2 -lambda 1.e-8 -tao_admm_dual_update update_basic -tao_admm_regularizer_type regularizer_user -tao_max_it 20 -tao_monitor -tao_admm_tolerance_update_factor 1.e-8  -misfit_tao_nls_pc_type icc -misfit_tao_monitor -reg_tao_monitor

   test:
      suffix: 3
      localrunfiles: tomographyData_A_b_xGT
      args:  -lambda 1.e-8 -tao_admm_dual_update update_basic -tao_admm_regularizer_type regularizer_soft_thresh -tao_max_it 20 -tao_monitor -tao_admm_tolerance_update_factor 1.e-8 -misfit_tao_nls_pc_type icc -misfit_tao_monitor

   test:
      suffix: 4
      localrunfiles: tomographyData_A_b_xGT
      args:  -lambda 1.e-8 -tao_admm_dual_update update_adaptive -tao_admm_regularizer_type regularizer_soft_thresh -tao_max_it 20 -tao_monitor -misfit_tao_monitor -misfit_tao_nls_pc_type icc

   test:
      suffix: 5
      localrunfiles: tomographyData_A_b_xGT
      args:  -reg 2 -lambda 1.e-8 -tao_admm_dual_update update_adaptive -tao_admm_regularizer_type regularizer_user -tao_max_it 20 -tao_monitor -tao_admm_tolerance_update_factor 1.e-8 -misfit_tao_monitor -reg_tao_monitor -misfit_tao_nls_pc_type icc

   test:
      suffix: 6
      localrunfiles: tomographyData_A_b_xGT
      args:  -reg 3 -lambda 1.e-8 -tao_admm_dual_update update_adaptive -tao_admm_regularizer_type regularizer_user -tao_max_it 20 -tao_monitor -tao_admm_tolerance_update_factor 1.e-8 -misfit_tao_monitor -reg_tao_monitor -misfit_tao_nls_pc_type icc

TEST*/
