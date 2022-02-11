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
  PetscErrorCode ierr;
  PetscReal      lambda, mu;
  AppCtx         *user;
  Vec            out,work,y,x;
  Tao            admm_tao,misfit;

  PetscFunctionBegin;
  user = NULL;
  mu   = 0;
  ierr = TaoGetADMMParentTao(tao,&admm_tao);CHKERRQ(ierr);
  ierr = TaoADMMGetMisfitSubsolver(admm_tao, &misfit);CHKERRQ(ierr);
  ierr = TaoADMMGetSpectralPenalty(admm_tao,&mu);CHKERRQ(ierr);
  ierr = TaoShellGetContext(tao,&user);CHKERRQ(ierr);

  lambda = user->lambda;
  work   = user->workN;
  ierr   = TaoGetSolutionVector(tao, &out);CHKERRQ(ierr);
  ierr   = TaoGetSolutionVector(misfit, &x);CHKERRQ(ierr);
  ierr   = TaoADMMGetDualVector(admm_tao, &y);CHKERRQ(ierr);

  /* Dx + y/mu */
  ierr = MatMult(user->D,x,work);CHKERRQ(ierr);
  ierr = VecAXPY(work,1/mu,y);CHKERRQ(ierr);

  /* soft thresholding */
  ierr = TaoSoftThreshold(work, -lambda/mu, lambda/mu, out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MisfitObjectiveAndGradient(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 */
  ierr = MatMult(user->A,X,user->workM);CHKERRQ(ierr);
  ierr = VecAXPY(user->workM,-1,user->b);CHKERRQ(ierr);
  ierr = VecDot(user->workM,user->workM,f);CHKERRQ(ierr);
  *f  *= 0.5;
  /* Gradient. ATAx-ATb */
  ierr = MatMult(user->ATA,X,user->workN);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->A,user->b,user->workN2);CHKERRQ(ierr);
  ierr = VecWAXPY(g,-1.,user->workN2,user->workN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient1(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* compute regularizer objective
   * f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x */
  ierr    = VecCopy(X,user->workN2);CHKERRQ(ierr);
  ierr    = VecPow(user->workN2,2.);CHKERRQ(ierr);
  ierr    = VecShift(user->workN2,user->eps*user->eps);CHKERRQ(ierr);
  ierr    = VecSqrtAbs(user->workN2);CHKERRQ(ierr);
  ierr    = VecCopy(user->workN2, user->workN3);CHKERRQ(ierr);
  ierr    = VecShift(user->workN2,-user->eps);CHKERRQ(ierr);
  ierr    = VecSum(user->workN2,f_reg);CHKERRQ(ierr);
  *f_reg *= user->lambda;
  /* compute regularizer gradient = lambda*x */
  ierr = VecPointwiseDivide(G_reg,X,user->workN3);CHKERRQ(ierr);
  ierr = VecScale(G_reg,user->lambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode RegularizerObjectiveAndGradient2(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscReal      temp;

  PetscFunctionBegin;
  /* compute regularizer objective = lambda*|z|_2^2 */
  ierr   = VecDot(X,X,&temp);CHKERRQ(ierr);
  *f_reg = 0.5*user->lambda*temp;
  /* compute regularizer gradient = lambda*z */
  ierr = VecCopy(X,G_reg);CHKERRQ(ierr);
  ierr = VecScale(G_reg,user->lambda);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(user->D,x,user->workN);CHKERRQ(ierr);
  ierr = VecPow(user->workN2,2.);CHKERRQ(ierr);
  ierr = VecShift(user->workN2,user->eps*user->eps);CHKERRQ(ierr);
  ierr = VecSqrtAbs(user->workN2);CHKERRQ(ierr);
  ierr = VecShift(user->workN2,-user->eps);CHKERRQ(ierr);
  ierr = VecReciprocal(user->workN2);CHKERRQ(ierr);
  ierr = VecScale(user->workN2,user->eps*user->eps);CHKERRQ(ierr);
  ierr = MatDiagonalSet(H,user->workN2,INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode FullObjGrad(Tao tao,Vec X,PetscReal *f,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscReal      f_reg;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 + lambda*||x||_2^2*/
  ierr = MatMult(user->A,X,user->workM);CHKERRQ(ierr);
  ierr = VecAXPY(user->workM,-1,user->b);CHKERRQ(ierr);
  ierr = VecDot(user->workM,user->workM,f);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_2,&f_reg);CHKERRQ(ierr);
  *f  *= 0.5;
  *f  += user->lambda*f_reg*f_reg;
  /* Gradient. ATAx-ATb + 2*lambda*x */
  ierr = MatMult(user->ATA,X,user->workN);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->A,user->b,user->workN2);CHKERRQ(ierr);
  ierr = VecWAXPY(g,-1.,user->workN2,user->workN);CHKERRQ(ierr);
  ierr = VecAXPY(g,2*user->lambda,X);CHKERRQ(ierr);
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
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->A);CHKERRQ(ierr);
  ierr = MatSetType(user->A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(user->A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->b);CHKERRQ(ierr);
  ierr = VecLoad(user->b,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->xGT);CHKERRQ(ierr);
  ierr = VecLoad(user->xGT,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatGetSize(user->A,&user->M,&user->N);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user->D);CHKERRQ(ierr);
  ierr = MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->D);CHKERRQ(ierr);
  ierr = MatSetUp(user->D);CHKERRQ(ierr);
  for (k=0; k<user->N; k++) {
    v = 1.0;
    n = k+1;
    if (k< user->N -1) {
      ierr = MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    v    = -1.0;
    ierr = MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(user->D,user->D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DTD);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user->Hz);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Hz,PETSC_DECIDE,PETSC_DECIDE,user->N,user->N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Hz);CHKERRQ(ierr);
  ierr = MatSetUp(user->Hz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->Hz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Hz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&(user->x));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workM));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workN));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workN2));CHKERRQ(ierr);
  ierr = VecSetSizes(user->x,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workM,PETSC_DECIDE,user->M);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workN,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workN2,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workM);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workN2);CHKERRQ(ierr);

  ierr = VecDuplicate(user->workN,&(user->workN3));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->xlb));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->xub));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->c));CHKERRQ(ierr);
  ierr = VecSet(user->xlb,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->c,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->xub,PETSC_INFINITY);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->ATA));CHKERRQ(ierr);
  ierr = MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->Hx));CHKERRQ(ierr);
  ierr = MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->HF));CHKERRQ(ierr);

  ierr = MatAssemblyBegin(user->ATA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->ATA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->Hx,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Hx,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->HF,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->HF,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  user->lambda = 1.e-8;
  user->eps    = 1.e-3;
  user->reg    = 2;
  user->mumin  = 5.e-6;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "tomographyADMM.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-reg","Regularization scheme for z solver (1,2)", "tomographyADMM.c", user->reg, &(user->reg), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda", "The regularization multiplier. 1 default", "tomographyADMM.c", user->lambda, &(user->lambda), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps", "L1 norm epsilon padding", "tomographyADMM.c", user->eps, &(user->eps), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mumin", "Minimum value for ADMM spectral penalty", "tomographyADMM.c", user->mumin, &(user->mumin), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->A);CHKERRQ(ierr);
  ierr = MatDestroy(&user->ATA);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Hx);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Hz);CHKERRQ(ierr);
  ierr = MatDestroy(&user->HF);CHKERRQ(ierr);
  ierr = MatDestroy(&user->D);CHKERRQ(ierr);
  ierr = MatDestroy(&user->DTD);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xGT);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xlb);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xub);CHKERRQ(ierr);
  ierr = VecDestroy(&user->b);CHKERRQ(ierr);
  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  ierr = VecDestroy(&user->c);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN3);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN2);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workM);CHKERRQ(ierr);
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
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ierr = InitializeUserData(user);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao, TAOADMM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao, user->x);CHKERRQ(ierr);
  /* f(x) + g(x) for parent tao */
  ierr = TaoADMMSetSpectralPenalty(tao,1.);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao, FullObjGrad, (void*)user);CHKERRQ(ierr);
  ierr = MatShift(user->HF,user->lambda);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao, user->HF, user->HF, HessianFull, (void*)user);CHKERRQ(ierr);

  /* f(x) for misfit tao */
  ierr = TaoADMMSetMisfitObjectiveAndGradientRoutine(tao, MisfitObjectiveAndGradient, (void*)user);CHKERRQ(ierr);
  ierr = TaoADMMSetMisfitHessianRoutine(tao, user->Hx, user->Hx, HessianMisfit, (void*)user);CHKERRQ(ierr);
  ierr = TaoADMMSetMisfitHessianChangeStatus(tao,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TaoADMMSetMisfitConstraintJacobian(tao,user->D,user->D,NullJacobian,(void*)user);CHKERRQ(ierr);

  /* g(x) for regularizer tao */
  if (user->reg == 1) {
    ierr = TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient1, (void*)user);CHKERRQ(ierr);
    ierr = TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianReg, (void*)user);CHKERRQ(ierr);
    ierr = TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE);CHKERRQ(ierr);
  } else if (user->reg == 2) {
    ierr = TaoADMMSetRegularizerObjectiveAndGradientRoutine(tao, RegularizerObjectiveAndGradient2, (void*)user);CHKERRQ(ierr);
    ierr = MatShift(user->Hz,1);CHKERRQ(ierr);
    ierr = MatScale(user->Hz,user->lambda);CHKERRQ(ierr);
    ierr = TaoADMMSetRegularizerHessianRoutine(tao, user->Hz, user->Hz, HessianMisfit, (void*)user);CHKERRQ(ierr);
    ierr = TaoADMMSetRegHessianChangeStatus(tao,PETSC_TRUE);CHKERRQ(ierr);
  } else PetscCheckFalse(user->reg != 3,PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Incorrect Reg type"); /* TaoShell case */

  /* Set type for the misfit solver */
  ierr = TaoADMMGetMisfitSubsolver(tao, &misfit);CHKERRQ(ierr);
  ierr = TaoADMMGetRegularizationSubsolver(tao, &reg);CHKERRQ(ierr);
  ierr = TaoSetType(misfit,TAONLS);CHKERRQ(ierr);
  if (user->reg == 3) {
    ierr = TaoSetType(reg,TAOSHELL);CHKERRQ(ierr);
    ierr = TaoShellSetContext(reg, (void*) user);CHKERRQ(ierr);
    ierr = TaoShellSetSolve(reg, TaoShellSolve_SoftThreshold);CHKERRQ(ierr);
  } else {
    ierr = TaoSetType(reg,TAONLS);CHKERRQ(ierr);
  }
  ierr = TaoSetVariableBounds(misfit,user->xlb,user->xub);CHKERRQ(ierr);

  /* Soft Thresholding solves the ADMM problem with the L1 regularizer lambda*||z||_1 and the x-z=0 constraint */
  ierr = TaoADMMSetRegularizerCoefficient(tao, user->lambda);CHKERRQ(ierr);
  ierr = TaoADMMSetRegularizerConstraintJacobian(tao,NULL,NULL,NullJacobian,(void*)user);CHKERRQ(ierr);
  ierr = TaoADMMSetMinimumSpectralPenalty(tao,user->mumin);CHKERRQ(ierr);

  ierr = TaoADMMSetConstraintVectorRHS(tao,user->c);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,resultFile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = VecView(user->x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* compute the error */
  ierr = VecAXPY(user->x,-1,user->xGT);CHKERRQ(ierr);
  ierr = VecNorm(user->x,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(user->xGT,NORM_2,&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2));CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = DestroyContext(user);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
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
