
static char help[] = "Tests the multigrid code.  The input parameters are:\n\
  -x N              Use a mesh in the x direction of N.  \n\
  -c N              Use N V-cycles.  \n\
  -l N              Use N Levels.  \n\
  -smooths N        Use N pre smooths and N post smooths.  \n\
  -j                Use Jacobi smoother.  \n\
  -a use additive multigrid \n\
  -f use full multigrid (preconditioner variant) \n\
This example also demonstrates matrix-free methods\n\n";

/*
  This is not a good example to understand the use of multigrid with PETSc.
*/

#include <petscksp.h>

PetscErrorCode  residual(Mat,Vec,Vec,Vec);
PetscErrorCode  gauss_seidel(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*);
PetscErrorCode  jacobi_smoother(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool,PetscInt*,PCRichardsonConvergedReason*);
PetscErrorCode  interpolate(Mat,Vec,Vec,Vec);
PetscErrorCode  restrct(Mat,Vec,Vec);
PetscErrorCode  Create1dLaplacian(PetscInt,Mat*);
PetscErrorCode  CalculateRhs(Vec);
PetscErrorCode  CalculateError(Vec,Vec,Vec,PetscReal*);
PetscErrorCode  CalculateSolution(PetscInt,Vec*);
PetscErrorCode  amult(Mat,Vec,Vec);
PetscErrorCode  apply_pc(PC,Vec,Vec);

int main(int Argc,char **Args)
{
  PetscInt       x_mesh = 15,levels = 3,cycles = 1,use_jacobi = 0;
  PetscInt       i,smooths = 1,*N,its;
  PetscErrorCode ierr;
  PCMGType       am = PC_MG_MULTIPLICATIVE;
  Mat            cmat,mat[20],fmat;
  KSP            cksp,ksp[20],kspmg;
  PetscReal      e[3];  /* l_2 error,max error, residual */
  const char     *shellname;
  Vec            x,solution,X[20],R[20],B[20];
  PC             pcmg,pc;
  PetscBool      flg;

  ierr = PetscInitialize(&Argc,&Args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-x",&x_mesh,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&levels,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-c",&cycles,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-smooths",&smooths,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-a",&flg));

  if (flg) am = PC_MG_ADDITIVE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-f",&flg));
  if (flg) am = PC_MG_FULL;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-j",&flg));
  if (flg) use_jacobi = 1;

  CHKERRQ(PetscMalloc1(levels,&N));
  N[0] = x_mesh;
  for (i=1; i<levels; i++) {
    N[i] = N[i-1]/2;
    PetscCheckFalse(N[i] < 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"Too many levels or N is not large enough");
  }

  CHKERRQ(Create1dLaplacian(N[levels-1],&cmat));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&kspmg));
  CHKERRQ(KSPGetPC(kspmg,&pcmg));
  CHKERRQ(KSPSetFromOptions(kspmg));
  CHKERRQ(PCSetType(pcmg,PCMG));
  CHKERRQ(PCMGSetLevels(pcmg,levels,NULL));
  CHKERRQ(PCMGSetType(pcmg,am));

  CHKERRQ(PCMGGetCoarseSolve(pcmg,&cksp));
  CHKERRQ(KSPSetOperators(cksp,cmat,cmat));
  CHKERRQ(KSPGetPC(cksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(KSPSetType(cksp,KSPPREONLY));

  /* zero is finest level */
  for (i=0; i<levels-1; i++) {
    Mat dummy;

    CHKERRQ(PCMGSetResidual(pcmg,levels - 1 - i,residual,NULL));
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,N[i+1],N[i],N[i+1],N[i],NULL,&mat[i]));
    CHKERRQ(MatShellSetOperation(mat[i],MATOP_MULT,(void (*)(void))restrct));
    CHKERRQ(MatShellSetOperation(mat[i],MATOP_MULT_TRANSPOSE_ADD,(void (*)(void))interpolate));
    CHKERRQ(PCMGSetInterpolation(pcmg,levels - 1 - i,mat[i]));
    CHKERRQ(PCMGSetRestriction(pcmg,levels - 1 - i,mat[i]));
    CHKERRQ(PCMGSetCycleTypeOnLevel(pcmg,levels - 1 - i,(PCMGCycleType)cycles));

    /* set smoother */
    CHKERRQ(PCMGGetSmoother(pcmg,levels - 1 - i,&ksp[i]));
    CHKERRQ(KSPGetPC(ksp[i],&pc));
    CHKERRQ(PCSetType(pc,PCSHELL));
    CHKERRQ(PCShellSetName(pc,"user_precond"));
    CHKERRQ(PCShellGetName(pc,&shellname));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"level=%D, PCShell name is %s\n",i,shellname));

    /* this is not used unless different options are passed to the solver */
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,N[i],N[i],N[i],N[i],NULL,&dummy));
    CHKERRQ(MatShellSetOperation(dummy,MATOP_MULT,(void (*)(void))amult));
    CHKERRQ(KSPSetOperators(ksp[i],dummy,dummy));
    CHKERRQ(MatDestroy(&dummy));

    /*
        We override the matrix passed in by forcing it to use Richardson with
        a user provided application. This is non-standard and this practice
        should be avoided.
    */
    CHKERRQ(PCShellSetApply(pc,apply_pc));
    CHKERRQ(PCShellSetApplyRichardson(pc,gauss_seidel));
    if (use_jacobi) {
      CHKERRQ(PCShellSetApplyRichardson(pc,jacobi_smoother));
    }
    CHKERRQ(KSPSetType(ksp[i],KSPRICHARDSON));
    CHKERRQ(KSPSetInitialGuessNonzero(ksp[i],PETSC_TRUE));
    CHKERRQ(KSPSetTolerances(ksp[i],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooths));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N[i],&x));

    X[levels - 1 - i] = x;
    if (i > 0) {
      CHKERRQ(PCMGSetX(pcmg,levels - 1 - i,x));
    }
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N[i],&x));

    B[levels -1 - i] = x;
    if (i > 0) {
      CHKERRQ(PCMGSetRhs(pcmg,levels - 1 - i,x));
    }
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N[i],&x));

    R[levels - 1 - i] = x;

    CHKERRQ(PCMGSetR(pcmg,levels - 1 - i,x));
  }
  /* create coarse level vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x));
  CHKERRQ(PCMGSetX(pcmg,0,x)); X[0] = x;
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x));
  CHKERRQ(PCMGSetRhs(pcmg,0,x)); B[0] = x;

  /* create matrix multiply for finest level */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,N[0],N[0],N[0],N[0],NULL,&fmat));
  CHKERRQ(MatShellSetOperation(fmat,MATOP_MULT,(void (*)(void))amult));
  CHKERRQ(KSPSetOperators(kspmg,fmat,fmat));

  CHKERRQ(CalculateSolution(N[0],&solution));
  CHKERRQ(CalculateRhs(B[levels-1]));
  CHKERRQ(VecSet(X[levels-1],0.0));

  CHKERRQ(residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]));
  CHKERRQ(CalculateError(solution,X[levels-1],R[levels-1],e));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"l_2 error %g max error %g resi %g\n",(double)e[0],(double)e[1],(double)e[2]));

  CHKERRQ(KSPSolve(kspmg,B[levels-1],X[levels-1]));
  CHKERRQ(KSPGetIterationNumber(kspmg,&its));
  CHKERRQ(residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]));
  CHKERRQ(CalculateError(solution,X[levels-1],R[levels-1],e));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"its %D l_2 error %g max error %g resi %g\n",its,(double)e[0],(double)e[1],(double)e[2]));

  CHKERRQ(PetscFree(N));
  CHKERRQ(VecDestroy(&solution));

  /* note we have to keep a list of all vectors allocated, this is
     not ideal, but putting it in MGDestroy is not so good either*/
  for (i=0; i<levels; i++) {
    CHKERRQ(VecDestroy(&X[i]));
    CHKERRQ(VecDestroy(&B[i]));
    if (i) CHKERRQ(VecDestroy(&R[i]));
  }
  for (i=0; i<levels-1; i++) {
    CHKERRQ(MatDestroy(&mat[i]));
  }
  CHKERRQ(MatDestroy(&cmat));
  CHKERRQ(MatDestroy(&fmat));
  CHKERRQ(KSPDestroy(&kspmg));
  ierr = PetscFinalize();
  return ierr;
}

/* --------------------------------------------------------------------- */
PetscErrorCode residual(Mat mat,Vec bb,Vec xx,Vec rr)
{
  PetscInt          i,n1;
  PetscScalar       *x,*r;
  const PetscScalar *b;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(bb,&n1));
  CHKERRQ(VecGetArrayRead(bb,&b));
  CHKERRQ(VecGetArray(xx,&x));
  CHKERRQ(VecGetArray(rr,&r));
  n1--;
  r[0]  = b[0] + x[1] - 2.0*x[0];
  r[n1] = b[n1] + x[n1-1] - 2.0*x[n1];
  for (i=1; i<n1; i++) r[i] = b[i] + x[i+1] + x[i-1] - 2.0*x[i];
  CHKERRQ(VecRestoreArrayRead(bb,&b));
  CHKERRQ(VecRestoreArray(xx,&x));
  CHKERRQ(VecRestoreArray(rr,&r));
  PetscFunctionReturn(0);
}

PetscErrorCode amult(Mat mat,Vec xx,Vec yy)
{
  PetscInt          i,n1;
  PetscScalar       *y;
  const PetscScalar *x;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(xx,&n1));
  CHKERRQ(VecGetArrayRead(xx,&x));
  CHKERRQ(VecGetArray(yy,&y));
  n1--;
  y[0] =  -x[1] + 2.0*x[0];
  y[n1] = -x[n1-1] + 2.0*x[n1];
  for (i=1; i<n1; i++) y[i] = -x[i+1] - x[i-1] + 2.0*x[i];
  CHKERRQ(VecRestoreArrayRead(xx,&x));
  CHKERRQ(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
PetscErrorCode apply_pc(PC pc,Vec bb,Vec xx)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented");
}

PetscErrorCode gauss_seidel(PC pc,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m,PetscBool guesszero,PetscInt *its,PCRichardsonConvergedReason *reason)
{
  PetscInt          i,n1;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  *its    = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  CHKERRQ(VecGetSize(bb,&n1)); n1--;
  CHKERRQ(VecGetArrayRead(bb,&b));
  CHKERRQ(VecGetArray(xx,&x));
  while (m--) {
    x[0] =  .5*(x[1] + b[0]);
    for (i=1; i<n1; i++) x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    x[n1] = .5*(x[n1-1] + b[n1]);
    for (i=n1-1; i>0; i--) x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    x[0] =  .5*(x[1] + b[0]);
  }
  CHKERRQ(VecRestoreArrayRead(bb,&b));
  CHKERRQ(VecRestoreArray(xx,&x));
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode jacobi_smoother(PC pc,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m,PetscBool guesszero,PetscInt *its,PCRichardsonConvergedReason *reason)
{
  PetscInt          i,n,n1;
  PetscScalar       *r,*x;
  const PetscScalar *b;

  PetscFunctionBegin;
  *its    = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  CHKERRQ(VecGetSize(bb,&n)); n1 = n - 1;
  CHKERRQ(VecGetArrayRead(bb,&b));
  CHKERRQ(VecGetArray(xx,&x));
  CHKERRQ(VecGetArray(w,&r));

  while (m--) {
    r[0] = .5*(x[1] + b[0]);
    for (i=1; i<n1; i++) r[i] = .5*(x[i+1] + x[i-1] + b[i]);
    r[n1] = .5*(x[n1-1] + b[n1]);
    for (i=0; i<n; i++) x[i] = (2.0*r[i] + x[i])/3.0;
  }
  CHKERRQ(VecRestoreArrayRead(bb,&b));
  CHKERRQ(VecRestoreArray(xx,&x));
  CHKERRQ(VecRestoreArray(w,&r));
  PetscFunctionReturn(0);
}
/*
   We know for this application that yy  and zz are the same
*/
/* --------------------------------------------------------------------- */
PetscErrorCode interpolate(Mat mat,Vec xx,Vec yy,Vec zz)
{
  PetscInt          i,n,N,i2;
  PetscScalar       *y;
  const PetscScalar *x;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(yy,&N));
  CHKERRQ(VecGetArrayRead(xx,&x));
  CHKERRQ(VecGetArray(yy,&y));
  n    = N/2;
  for (i=0; i<n; i++) {
    i2       = 2*i;
    y[i2]   += .5*x[i];
    y[i2+1] +=    x[i];
    y[i2+2] += .5*x[i];
  }
  CHKERRQ(VecRestoreArrayRead(xx,&x));
  CHKERRQ(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode restrct(Mat mat,Vec rr,Vec bb)
{
  PetscInt          i,n,N,i2;
  PetscScalar       *b;
  const PetscScalar *r;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(rr,&N));
  CHKERRQ(VecGetArrayRead(rr,&r));
  CHKERRQ(VecGetArray(bb,&b));
  n    = N/2;

  for (i=0; i<n; i++) {
    i2   = 2*i;
    b[i] = (r[i2] + 2.0*r[i2+1] + r[i2+2]);
  }
  CHKERRQ(VecRestoreArrayRead(rr,&r));
  CHKERRQ(VecRestoreArray(bb,&b));
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode Create1dLaplacian(PetscInt n,Mat *mat)
{
  PetscScalar    mone = -1.0,two = 2.0;
  PetscInt       i,idx;

  PetscFunctionBegin;
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,mat));

  idx  = n-1;
  CHKERRQ(MatSetValues(*mat,1,&idx,1,&idx,&two,INSERT_VALUES));
  for (i=0; i<n-1; i++) {
    CHKERRQ(MatSetValues(*mat,1,&i,1,&i,&two,INSERT_VALUES));
    idx  = i+1;
    CHKERRQ(MatSetValues(*mat,1,&idx,1,&i,&mone,INSERT_VALUES));
    CHKERRQ(MatSetValues(*mat,1,&i,1,&idx,&mone,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode CalculateRhs(Vec u)
{
  PetscInt    i,n;
  PetscReal   h;
  PetscScalar uu;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(u,&n));
  h    = 1.0/((PetscReal)(n+1));
  for (i=0; i<n; i++) {
    uu = 2.0*h*h;
    CHKERRQ(VecSetValues(u,1,&i,&uu,INSERT_VALUES));
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode CalculateSolution(PetscInt n,Vec *solution)
{
  PetscInt       i;
  PetscReal      h,x = 0.0;
  PetscScalar    uu;

  PetscFunctionBegin;
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,solution));
  h    = 1.0/((PetscReal)(n+1));
  for (i=0; i<n; i++) {
    x   += h; uu = x*(1.-x);
    CHKERRQ(VecSetValues(*solution,1,&i,&uu,INSERT_VALUES));
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
PetscErrorCode CalculateError(Vec solution,Vec u,Vec r,PetscReal *e)
{
  PetscFunctionBegin;
  CHKERRQ(VecNorm(r,NORM_2,e+2));
  CHKERRQ(VecWAXPY(r,-1.0,u,solution));
  CHKERRQ(VecNorm(r,NORM_2,e));
  CHKERRQ(VecNorm(r,NORM_1,e+1));
  PetscFunctionReturn(0);
}

/*TEST

   test:

TEST*/
