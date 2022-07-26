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
  PetscCall(TaoGetADMMParentTao(tao,&admm_tao));
  PetscCall(TaoADMMGetMisfitSubsolver(admm_tao, &misfit));
  PetscCall(TaoADMMGetSpectralPenalty(admm_tao,&mu));
  PetscCall(TaoShellGetContext(tao,&user));

  lambda = user->lambda;
  work   = user->workN;
  PetscCall(TaoGetSolution(tao, &out));
  PetscCall(TaoGetSolution(misfit, &x));
  PetscCall(TaoADMMGetDualVector(admm_tao, &y));

  /* Dx + y/mu */
  PetscCall(MatMult(user->D,x,work));
  PetscCall(VecAXPY(work,1/mu,y));

  /* soft thresholding */
  PetscCall(TaoSoftThreshold(work, -lambda/mu, lambda/mu, out));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MisfitObjectiveAndGradient(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 */
  PetscCall(MatMult(user->A,X,user->workM));
  PetscCall(VecAXPY(user->workM,-1,user->b));
  PetscCall(VecDot(user->workM,user->workM,f));
  *f  *= 0.5;
  /* Gradient. ATAx-ATb */
  PetscCall(MatMult(user->ATA,X,user->workN));
  PetscCall(MatMultTranspose(user->A,user->b,user->workN2));
  PetscCall(VecWAXPY(g,-1.,user->workN2,user->workN));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient1(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  /* compute regularizer objective
   * f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x */
  PetscCall(VecCopy(X,user->workN2));
  PetscCall(VecPow(user->workN2,2.));
  PetscCall(VecShift(user->workN2,user->eps*user->eps));
  PetscCall(VecSqrtAbs(user->workN2));
  PetscCall(VecCopy(user->workN2, user->workN3));
  PetscCall(VecShift(user->workN2,-user->eps));
  PetscCall(VecSum(user->workN2,f_reg));
  *f_reg *= user->lambda;
  /* compute regularizer gradient = lambda*x */
  PetscCall(VecPointwiseDivide(G_reg,X,user->workN3));
  PetscCall(VecScale(G_reg,user->lambda));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient2(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscReal      temp;

  PetscFunctionBegin;
  /* compute regularizer objective = lambda*|z|_2^2 */
  PetscCall(VecDot(X,X,&temp));
  *f_reg = 0.5*user->lambda*temp;
  /* compute regularizer gradient = lambda*z */
  PetscCall(VecCopy(X,G_reg));
  PetscCall(VecScale(G_reg,user->lambda));
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
  PetscCall(MatMult(user->D,x,user->workN));
  PetscCall(VecPow(user->workN2,2.));
  PetscCall(VecShift(user->workN2,user->eps*user->eps));
  PetscCall(VecSqrtAbs(user->workN2));
  PetscCall(VecShift(user->workN2,-user->eps));
  PetscCall(VecReciprocal(user->workN2));
  PetscCall(VecScale(user->workN2,user->eps*user->eps));
  PetscCall(MatDiagonalSet(H,user->workN2,INSERT_VALUES));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode FullObjGrad(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscReal      f_reg;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 + lambda*||x||_2^2*/
  PetscCall(MatMult(user->A,X,user->workM));
  PetscCall(VecAXPY(user->workM,-1,user->b));
  PetscCall(VecDot(user->workM,user->workM,f));
  PetscCall(VecNorm(X,NORM_2,&f_reg));
  *f  *= 0.5;
  *f  += user->lambda*f_reg*f_reg;
  /* Gradient. ATAx-ATb + 2*lambda*x */
  PetscCall(MatMult(user->ATA,X,user->workN));
  PetscCall(MatMultTranspose(user->A,user->b,user->workN2));
  PetscCall(VecWAXPY(g,-1.,user->workN2,user->workN));
  PetscCall(VecAXPY(g,2*user->lambda,X));
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
  PetscInt       k,n;
  PetscScalar    v;

  PetscFunctionBegin;
  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->A));
  PetscCall(MatSetType(user->A,MATAIJ));
  PetscCall(MatLoad(user->A,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->b));
  PetscCall(VecLoad(user->b,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->xGT));
  PetscCall(VecLoad(user->xGT,fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetSize(user->A,&user->M,&user->N));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->D));
  PetscCall(MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N));
  PetscCall(MatSetFromOptions(user->D));
  PetscCall(MatSetUp(user->D));
  for (k=0; k<user->N; k++) {
    v = 1.0;
    n = k+1;
    if (k< user->N -1) {
      PetscCall(MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES));
    }
    v    = -1.0;
    PetscCall(MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY));

  PetscCall(MatTransposeMatMult(user->D,user->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DTD));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Hz));
  PetscCall(MatSetSizes(user->Hz,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N));
  PetscCall(MatSetFromOptions(user->Hz));
  PetscCall(MatSetUp(user->Hz));
  PetscCall(MatAssemblyBegin(user->Hz,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Hz,MAT_FINAL_ASSEMBLY));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&(user->x)));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&(user->workM)));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&(user->workN)));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&(user->workN2)));
  PetscCall(VecSetSizes(user->x,PETSC_DECIDE,user->N));
  PetscCall(VecSetSizes(user->workM,PETSC_DECIDE,user->M));
  PetscCall(VecSetSizes(user->workN,PETSC_DECIDE,user->N));
  PetscCall(VecSetSizes(user->workN2,PETSC_DECIDE,user->N));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecSetFromOptions(user->workM));
  PetscCall(VecSetFromOptions(user->workN));
  PetscCall(VecSetFromOptions(user->workN2));

  PetscCall(VecDuplicate(user->workN,&(user->workN3)));
  PetscCall(VecDuplicate(user->x,&(user->xlb)));
  PetscCall(VecDuplicate(user->x,&(user->xub)));
  PetscCall(VecDuplicate(user->x,&(user->c)));
  PetscCall(VecSet(user->xlb,0.0));
  PetscCall(VecSet(user->c,0.0));
  PetscCall(VecSet(user->xub,PETSC_INFINITY));

  PetscCall(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->ATA)));
  PetscCall(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->Hx)));
  PetscCall(MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->HF)));

  PetscCall(MatAssemblyBegin(user->ATA,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->ATA,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(user->Hx,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Hx,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(user->HF,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->HF,MAT_FINAL_ASSEMBLY));

  user->lambda = 1.e-8;
  user->eps    = 1.e-3;
  user->reg    = 2;
  user->mumin  = 5.e-6;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "tomographyADMM.c");
  PetscCall(PetscOptionsInt("-reg","Regularization scheme for z solver (1,2)", "tomographyADMM.c", user->reg, &(user->reg), NULL));
  PetscCall(PetscOptionsReal("-lambda", "The regularization multiplier. 1 default", "tomographyADMM.c", user->lambda, &(user->lambda), NULL));
  PetscCall(PetscOptionsReal("-eps", "L1 norm epsilon padding", "tomographyADMM.c", user->eps, &(user->eps), NULL));
  PetscCall(PetscOptionsReal("-mumin", "Minimum value for ADMM spectral penalty", "tomographyADMM.c", user->mumin, &(user->mumin), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->A));
  PetscCall(MatDestroy(&user->ATA));
  PetscCall(MatDestroy(&user->Hx));
  PetscCall(MatDestroy(&user->Hz));
  PetscCall(MatDestroy(&user->HF));
  PetscCall(MatDestroy(&user->D));
  PetscCall(MatDestroy(&user->DTD));
  PetscCall(VecDestroy(&user->xGT));
  PetscCall(VecDestroy(&user->xlb));
  PetscCall(VecDestroy(&user->xub));
  PetscCall(VecDestroy(&user->b));
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->c));
  PetscCall(VecDestroy(&user->workN3));
  PetscCall(VecDestroy(&user->workN2));
  PetscCall(VecDestroy(&user->workN));
  PetscCall(VecDestroy(&user->workM));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

int main(int argc,char **argv)
{
  Tao            tao,misfit,reg;
  PetscReal      v1,v2;
  AppCtx*        user;
  PetscViewer    fd;
  char           resultFile[] = "tomographyResult_x";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscNew(&user));
  PetscCall(InitializeUserData(user));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOADMM));
  PetscCall(TaoSetSolution(tao, user->x));
  /* f(x) + g(x) for parent tao */
  PetscCall(TaoADMMSetSpectralPenalty(tao,1.));
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL, FullObjGrad, (void*)user));
  PetscCall(MatShift(user->HF,user->lambda));
  PetscCall(TaoSetHessian(tao, user->HF, user->HF, HessianFull, (void*)user));

  /* f(x) for misfit tao */
  PetscCall(TaoADMMSetMisfitObjectiveAndGradientRoutine(tao, MisfitObjectiveAndGradient, (void*)user));
  PetscCall(TaoADMMSetMisfitHessianRoutine(tao, user->Hx, user->Hx, HessianMisfit, (void*)user));
  PetscCall(TaoADMMSetMisfitHessianChangeStatus(tao,PETSC_FALSE));
  PetscCall(TaoADMMSetMisfitConstraintJacobian(tao,user->D,user->D,NullJacobian,(void*)user));

  /* g(x) for regularizer tao */
  if (user->reg == 1) {
    PetscCall(TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient1, (void*)user));
    PetscCall(TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianReg, (void*)user));
    PetscCall(TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE));
  } else if (user->reg == 2) {
    PetscCall(TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient2, (void*)user));
    PetscCall(MatShift(user->Hz,1));
    PetscCall(MatScale(user->Hz,user->lambda));
    PetscCall(TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianMisfit, (void*)user));
    PetscCall(TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE));
  } else PetscCheck(user->reg == 3,PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Incorrect Reg type"); /* TaoShell case */

  /* Set type for the misfit solver */
  PetscCall(TaoADMMGetMisfitSubsolver(tao, &misfit));
  PetscCall(TaoADMMGetRegularizationSubsolver(tao, &reg));
  PetscCall(TaoSetType(misfit,TAONLS));
  if (user->reg == 3) {
    PetscCall(TaoSetType(reg,TAOSHELL));
    PetscCall(TaoShellSetContext(reg, (void*) user));
    PetscCall(TaoShellSetSolve(reg, TaoShellSolve_SoftThreshold));
  } else {
    PetscCall(TaoSetType(reg,TAONLS));
  }
  PetscCall(TaoSetVariableBounds(misfit,user->xlb,user->xub));

  /* Soft Thresholding solves the ADMM problem with the L1 regularizer lambda*||z||_1 and the x-z=0 constraint */
  PetscCall(TaoADMMSetRegularizerCoefficient(tao, user->lambda));
  PetscCall(TaoADMMSetRegularizerConstraintJacobian(tao,NULL,NULL,NullJacobian,(void*)user));
  PetscCall(TaoADMMSetMinimumSpectralPenalty(tao,user->mumin));

  PetscCall(TaoADMMSetConstraintVectorRHS(tao,user->c));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,resultFile,FILE_MODE_WRITE,&fd));
  PetscCall(VecView(user->x,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* compute the error */
  PetscCall(VecAXPY(user->x,-1,user->xGT));
  PetscCall(VecNorm(user->x,NORM_2,&v1));
  PetscCall(VecNorm(user->xGT,NORM_2,&v2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2)));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));
  PetscCall(DestroyContext(user));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
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
