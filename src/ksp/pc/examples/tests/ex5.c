
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
#include <petscpcmg.h>

PetscErrorCode  residual(Mat,Vec,Vec,Vec);
PetscErrorCode  gauss_seidel(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*);
PetscErrorCode  jacobi(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*);
PetscErrorCode  interpolate(Mat,Vec,Vec,Vec);
PetscErrorCode  restrct(Mat,Vec,Vec);
PetscErrorCode  Create1dLaplacian(PetscInt,Mat*);
PetscErrorCode  CalculateRhs(Vec);
PetscErrorCode  CalculateError(Vec,Vec,Vec,PetscReal*);
PetscErrorCode  CalculateSolution(PetscInt,Vec*);
PetscErrorCode  amult(Mat,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int Argc,char **Args)
{
  PetscInt        x_mesh = 15,levels = 3,cycles = 1,use_jacobi = 0;
  PetscInt        i,smooths = 1,*N,its;
  PetscErrorCode  ierr;
  PCMGType        am = PC_MG_MULTIPLICATIVE;
  Mat             cmat,mat[20],fmat;
  KSP             cksp,ksp[20],kspmg;
  PetscReal       e[3]; /* l_2 error,max error, residual */
  const char      *shellname;
  Vec             x,solution,X[20],R[20],B[20];
  PC              pcmg,pc;
  PetscBool       flg;

  PetscInitialize(&Argc,&Args,(char *)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-x",&x_mesh,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-l",&levels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-c",&cycles,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-smooths",&smooths,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-a",&flg);CHKERRQ(ierr);
  if (flg) {am = PC_MG_ADDITIVE;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-f",&flg);CHKERRQ(ierr);
  if (flg) {am = PC_MG_FULL;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-j",&flg);CHKERRQ(ierr);
  if (flg) {use_jacobi = 1;}

  ierr = PetscMalloc(levels*sizeof(PetscInt),&N);CHKERRQ(ierr);
  N[0] = x_mesh;
  for (i=1; i<levels; i++) {
    N[i] = N[i-1]/2;
    if (N[i] < 1) SETERRQ(PETSC_COMM_WORLD,1,"Too many levels");
  }

  ierr = Create1dLaplacian(N[levels-1],&cmat);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspmg);CHKERRQ(ierr);
  ierr = KSPGetPC(kspmg,&pcmg);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspmg);CHKERRQ(ierr);
  ierr = PCSetType(pcmg,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pcmg,levels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pcmg,am);CHKERRQ(ierr);

  ierr = PCMGGetCoarseSolve(pcmg,&cksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(cksp,cmat,cmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(cksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetType(cksp,KSPPREONLY);CHKERRQ(ierr);

  /* zero is finest level */
  for (i=0; i<levels-1; i++) {
    ierr = PCMGSetResidual(pcmg,levels - 1 - i,residual,(Mat)0);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,N[i+1],N[i],N[i+1],N[i],(void*)0,&mat[i]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat[i],MATOP_MULT,(void(*)(void))restrct);CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat[i],MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))interpolate);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pcmg,levels - 1 - i,mat[i]);CHKERRQ(ierr);
    ierr = PCMGSetRestriction(pcmg,levels - 1 - i,mat[i]);CHKERRQ(ierr);
    ierr = PCMGSetCyclesOnLevel(pcmg,levels - 1 - i,cycles);CHKERRQ(ierr);

    /* set smoother */
    ierr = PCMGGetSmoother(pcmg,levels - 1 - i,&ksp[i]);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp[i],&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = PCShellSetName(pc,"user_precond");CHKERRQ(ierr);
    ierr = PCShellGetName(pc,&shellname);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"level=%D, PCShell name is %s\n",i,shellname);CHKERRQ(ierr);

    /* this is a dummy! since KSP requires a matrix passed in  */
    ierr = KSPSetOperators(ksp[i],mat[i],mat[i],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    /*
        We override the matrix passed in by forcing it to use Richardson with
        a user provided application. This is non-standard and this practice
        should be avoided.
    */
    ierr = PCShellSetApplyRichardson(pc,gauss_seidel);CHKERRQ(ierr);
    if (use_jacobi) {
      ierr = PCShellSetApplyRichardson(pc,jacobi);CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp[i],KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp[i],PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp[i],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooths);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    X[levels - 1 - i] = x;
    if (i > 0) {
      ierr = PCMGSetX(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
    }
    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    B[levels -1 - i] = x;
    if (i > 0) {
      ierr = PCMGSetRhs(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
    }
    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    R[levels - 1 - i] = x;
    ierr = PCMGSetR(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
  }
  /* create coarse level vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x);CHKERRQ(ierr);
  ierr = PCMGSetX(pcmg,0,x);CHKERRQ(ierr); X[0] = x;
  ierr = VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x);CHKERRQ(ierr);
  ierr = PCMGSetRhs(pcmg,0,x);CHKERRQ(ierr); B[0] = x;

  /* create matrix multiply for finest level */
  ierr = MatCreateShell(PETSC_COMM_WORLD,N[0],N[0],N[0],N[0],(void*)0,&fmat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(fmat,MATOP_MULT,(void(*)(void))amult);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspmg,fmat,fmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = CalculateSolution(N[0],&solution);CHKERRQ(ierr);
  ierr = CalculateRhs(B[levels-1]);CHKERRQ(ierr);
  ierr = VecSet(X[levels-1],0.0);CHKERRQ(ierr);

  ierr = residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]);CHKERRQ(ierr);
  ierr = CalculateError(solution,X[levels-1],R[levels-1],e);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"l_2 error %G max error %G resi %G\n",e[0],e[1],e[2]);CHKERRQ(ierr);

  ierr = KSPSolve(kspmg,B[levels-1],X[levels-1]);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspmg,&its);CHKERRQ(ierr);
  ierr = residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]);CHKERRQ(ierr);
  ierr = CalculateError(solution,X[levels-1],R[levels-1],e);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"its %D l_2 error %G max error %G resi %G\n",its,e[0],e[1],e[2]);CHKERRQ(ierr);

  ierr = PetscFree(N);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);

  /* note we have to keep a list of all vectors allocated, this is
     not ideal, but putting it in MGDestroy is not so good either*/
  for (i=0; i<levels; i++) {
    ierr = VecDestroy(&X[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&B[i]);CHKERRQ(ierr);
    if (i){ierr = VecDestroy(&R[i]);CHKERRQ(ierr);}
  }
  for (i=0; i<levels-1; i++) {
    ierr = MatDestroy(&mat[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&cmat);CHKERRQ(ierr);
  ierr = MatDestroy(&fmat);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspmg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "residual"
PetscErrorCode residual(Mat mat,Vec bb,Vec xx,Vec rr)
{
  PetscInt       i,n1;
  PetscErrorCode ierr;
  PetscScalar    *b,*x,*r;

  PetscFunctionBegin;
  ierr = VecGetSize(bb,&n1);CHKERRQ(ierr);
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
  n1--;
  r[0] = b[0] + x[1] - 2.0*x[0];
  r[n1] = b[n1] + x[n1-1] - 2.0*x[n1];
  for (i=1; i<n1; i++) {
    r[i] = b[i] + x[i+1] + x[i-1] - 2.0*x[i];
  }
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "amult"
PetscErrorCode amult(Mat mat,Vec xx,Vec yy)
{
  PetscInt       i,n1;
  PetscErrorCode ierr;
  PetscScalar    *y,*x;

  PetscFunctionBegin;
  ierr = VecGetSize(xx,&n1);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  n1--;
  y[0] =  -x[1] + 2.0*x[0];
  y[n1] = -x[n1-1] + 2.0*x[n1];
  for (i=1; i<n1; i++) {
    y[i] = -x[i+1] - x[i-1] + 2.0*x[i];
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "gauss_seidel"
PetscErrorCode gauss_seidel(PC pc,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m,PetscBool  guesszero,PetscInt *its,PCRichardsonConvergedReason *reason)
{
  PetscInt       i,n1;
  PetscErrorCode ierr;
  PetscScalar    *x,*b;

  PetscFunctionBegin;
  *its    = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  ierr    = VecGetSize(bb,&n1);CHKERRQ(ierr); n1--;
  ierr    = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr    = VecGetArray(xx,&x);CHKERRQ(ierr);
  while (m--) {
    x[0] =  .5*(x[1] + b[0]);
    for (i=1; i<n1; i++) {
      x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    x[n1] = .5*(x[n1-1] + b[n1]);
    for (i=n1-1; i>0; i--) {
      x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    x[0] =  .5*(x[1] + b[0]);
  }
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "jacobi"
PetscErrorCode jacobi(PC pc,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m,PetscBool  guesszero,PetscInt *its,PCRichardsonConvergedReason *reason)
{
  PetscInt       i,n,n1;
  PetscErrorCode ierr;
  PetscScalar    *r,*b,*x;

  PetscFunctionBegin;
  *its = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  ierr = VecGetSize(bb,&n);CHKERRQ(ierr); n1 = n - 1;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(w,&r);CHKERRQ(ierr);

  while (m--) {
    r[0] = .5*(x[1] + b[0]);
    for (i=1; i<n1; i++) {
       r[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    r[n1] = .5*(x[n1-1] + b[n1]);
    for (i=0; i<n; i++) x[i] = (2.0*r[i] + x[i])/3.0;
  }
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(w,&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
   We know for this application that yy  and zz are the same
*/
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "interpolate"
PetscErrorCode interpolate(Mat mat,Vec xx,Vec yy,Vec zz)
{
  PetscInt       i,n,N,i2;
  PetscErrorCode ierr;
  PetscScalar    *x,*y;

  PetscFunctionBegin;
  ierr = VecGetSize(yy,&N);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  n = N/2;
  for (i=0; i<n; i++) {
    i2 = 2*i;
    y[i2] +=  .5*x[i];
    y[i2+1] +=  x[i];
    y[i2+2] +=  .5*x[i];
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "restrct"
PetscErrorCode restrct(Mat mat,Vec rr,Vec bb)
{
  PetscInt       i,n,N,i2;
  PetscErrorCode ierr;
  PetscScalar    *r,*b;

  PetscFunctionBegin;
  ierr = VecGetSize(rr,&N);CHKERRQ(ierr);
  ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  n = N/2;

  for (i=0; i<n; i++) {
    i2 = 2*i;
    b[i] = (r[i2] + 2.0*r[i2+1] + r[i2+2]);
  }
  ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Create1dLaplacian"
PetscErrorCode Create1dLaplacian(PetscInt n,Mat *mat)
{
  PetscScalar    mone = -1.0,two = 2.0;
  PetscInt       i,idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,mat);CHKERRQ(ierr);

  idx= n-1;
  ierr = MatSetValues(*mat,1,&idx,1,&idx,&two,INSERT_VALUES);CHKERRQ(ierr);
  for (i=0; i<n-1; i++) {
    ierr = MatSetValues(*mat,1,&i,1,&i,&two,INSERT_VALUES);CHKERRQ(ierr);
    idx = i+1;
    ierr = MatSetValues(*mat,1,&idx,1,&i,&mone,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,1,&i,1,&idx,&mone,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CalculateRhs"
PetscErrorCode CalculateRhs(Vec u)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscReal      h,x = 0.0;
  PetscScalar    uu;

  PetscFunctionBegin;
  ierr = VecGetSize(u,&n);CHKERRQ(ierr);
  h = 1.0/((PetscReal)(n+1));
  for (i=0; i<n; i++) {
    x += h; uu = 2.0*h*h;
    ierr = VecSetValues(u,1,&i,&uu,INSERT_VALUES);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CalculateSolution"
PetscErrorCode CalculateSolution(PetscInt n,Vec *solution)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      h,x = 0.0;
  PetscScalar    uu;

  PetscFunctionBegin;
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,solution);CHKERRQ(ierr);
  h = 1.0/((PetscReal)(n+1));
  for (i=0; i<n; i++) {
    x += h; uu = x*(1.-x);
    ierr = VecSetValues(*solution,1,&i,&uu,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CalculateError"
PetscErrorCode CalculateError(Vec solution,Vec u,Vec r,PetscReal *e)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNorm(r,NORM_2,e+2);CHKERRQ(ierr);
  ierr = VecWAXPY(r,-1.0,u,solution);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,e);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_1,e+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


