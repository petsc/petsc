/*$Id: ex5.c,v 1.76 2001/08/07 21:30:37 bsmith Exp $*/

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
#include "petscmg.h"

int  residual(Mat,Vec,Vec,Vec);
int  gauss_seidel(void *,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
int  jacobi(void *,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
int  interpolate(Mat,Vec,Vec,Vec);
int  restrct(Mat,Vec,Vec);
int  Create1dLaplacian(int,Mat*);
int  CalculateRhs(Vec);
int  CalculateError(Vec,Vec,Vec,PetscReal*);
int  CalculateSolution(int,Vec*);
int  amult(Mat,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int Argc,char **Args)
{
  int         x_mesh = 15,levels = 3,cycles = 1,use_jacobi = 0;
  int         i,smooths = 1,*N;
  int         ierr,its;
  MGType      am = MGMULTIPLICATIVE;
  Mat         cmat,mat[20],fmat;
  SLES        csles,sles[20],slesmg;
  PetscReal   e[3]; /* l_2 error,max error, residual */
  char        *shellname;
  Vec         x,solution,X[20],R[20],B[20];
  PetscScalar zero = 0.0;
  KSP         cksp,ksp,kspmg,kspi;
  PC          pcmg,pc;
  PetscTruth  flg;

  PetscInitialize(&Argc,&Args,(char *)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-x",&x_mesh,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-l",&levels,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-c",&cycles,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-smooths",&smooths,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-a",&flg);CHKERRQ(ierr);
  if (flg) {am = MGADDITIVE;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-f",&flg);CHKERRQ(ierr);
  if (flg) {am = MGFULL;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-j",&flg);CHKERRQ(ierr);
  if (flg) {use_jacobi = 1;}
         
  ierr = PetscMalloc(levels*sizeof(int),&N);CHKERRQ(ierr);
  N[0] = x_mesh;
  for (i=1; i<levels; i++) {
    N[i] = N[i-1]/2;
    if (N[i] < 1) {SETERRQ(1,"Too many levels");}
  }

  ierr = Create1dLaplacian(N[levels-1],&cmat);CHKERRQ(ierr);

  ierr = SLESCreate(PETSC_COMM_WORLD,&slesmg);CHKERRQ(ierr);
  ierr = SLESGetKSP(slesmg,&kspmg);CHKERRQ(ierr);
  ierr = KSPGetPC(kspmg,&pcmg);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(slesmg);CHKERRQ(ierr);
  ierr = PCSetType(pcmg,PCMG);CHKERRQ(ierr);
  ierr = MGSetLevels(pcmg,levels,PETSC_NULL);CHKERRQ(ierr);
  ierr = MGSetType(pcmg,am);CHKERRQ(ierr);

  ierr = MGGetCoarseSolve(pcmg,&csles);CHKERRQ(ierr);
  ierr = SLESSetOperators(csles,cmat,cmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESGetKSP(csles,&cksp);CHKERRQ(ierr);
  ierr = KSPGetPC(cksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = SLESGetKSP(csles,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);

  /* zero is finest level */
  for (i=0; i<levels-1; i++) {
    ierr = MGSetResidual(pcmg,levels - 1 - i,residual,(Mat)0);CHKERRQ(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,N[i+1],N[i],N[i+1],N[i],(void *)0,&mat[i]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat[i],MATOP_MULT,(void(*)(void))restrct);CHKERRQ(ierr);
    ierr = MatShellSetOperation(mat[i],MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))interpolate);CHKERRQ(ierr);
    ierr = MGSetInterpolate(pcmg,levels - 1 - i,mat[i]);CHKERRQ(ierr);
    ierr = MGSetRestriction(pcmg,levels - 1 - i,mat[i]);CHKERRQ(ierr);
    ierr = MGSetCyclesOnLevel(pcmg,levels - 1 - i,cycles);CHKERRQ(ierr);

    /* set smoother */
    ierr = MGGetSmoother(pcmg,levels - 1 - i,&sles[i]);CHKERRQ(ierr);
    ierr = SLESGetKSP(sles[i],&kspi);CHKERRQ(ierr);
    ierr = KSPGetPC(kspi,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = PCShellSetName(pc,"user_precond");CHKERRQ(ierr);
    ierr = PCShellGetName(pc,&shellname);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"level=%d, PCShell name is %s\n",i,shellname);CHKERRQ(ierr);

    /* this is a dummy! since SLES requires a matrix passed in  */
    ierr = SLESSetOperators(sles[i],mat[i],mat[i],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    /* 
        We override the matrix passed in by forcing it to use Richardson with 
        a user provided application. This is non-standard and this practice
        should be avoided.
    */
    ierr = PCShellSetApplyRichardson(pc,gauss_seidel,(void *)0);CHKERRQ(ierr);
    if (use_jacobi) {
      ierr = PCShellSetApplyRichardson(pc,jacobi,(void *)0);CHKERRQ(ierr);
    }
    ierr = SLESGetKSP(sles[i],&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooths);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    X[levels - 1 - i] = x;
    ierr = MGSetX(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    B[levels -1 - i] = x;
    ierr = MGSetRhs(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N[i],&x);CHKERRQ(ierr);
    R[levels - 1 - i] = x;
    ierr = MGSetR(pcmg,levels - 1 - i,x);CHKERRQ(ierr);
  } 
  /* create coarse level vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x);CHKERRQ(ierr);
  ierr = MGSetX(pcmg,0,x);CHKERRQ(ierr); X[0] = x;
  ierr = VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x);CHKERRQ(ierr);
  ierr = MGSetRhs(pcmg,0,x);CHKERRQ(ierr); B[0] = x;
  ierr = VecCreateSeq(PETSC_COMM_SELF,N[levels-1],&x);CHKERRQ(ierr);
  ierr = MGSetR(pcmg,0,x);CHKERRQ(ierr); R[0] = x;

  /* create matrix multiply for finest level */
  ierr = MatCreateShell(PETSC_COMM_WORLD,N[0],N[0],N[0],N[0],(void *)0,&fmat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(fmat,MATOP_MULT,(void(*)(void))amult);CHKERRQ(ierr);
  ierr = SLESSetOperators(slesmg,fmat,fmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = CalculateSolution(N[0],&solution);CHKERRQ(ierr);
  ierr = CalculateRhs(B[levels-1]);CHKERRQ(ierr);
  ierr = VecSet(&zero,X[levels-1]);CHKERRQ(ierr);

  if (MGCheck(pcmg)) {SETERRQ(PETSC_ERR_PLIB, "MGCheck failed");}
     
  ierr = residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]);CHKERRQ(ierr);
  ierr = CalculateError(solution,X[levels-1],R[levels-1],e);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"l_2 error %g max error %g resi %g\n",e[0],e[1],e[2]);CHKERRQ(ierr);

  ierr = SLESSolve(slesmg,B[levels-1],X[levels-1]);CHKERRQ(ierr);
  ierr = SLESGetKSP(slesmg,&ksp);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = residual((Mat)0,B[levels-1],X[levels-1],R[levels-1]);CHKERRQ(ierr);
  ierr = CalculateError(solution,X[levels-1],R[levels-1],e);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"its %d l_2 error %g max error %g resi %g\n",its,e[0],e[1],e[2]);CHKERRQ(ierr);

  ierr = PetscFree(N);CHKERRQ(ierr);
  ierr = VecDestroy(solution);CHKERRQ(ierr);

  /* note we have to keep a list of all vectors allocated, this is 
     not ideal, but putting it in MGDestroy is not so good either*/
  for (i=0; i<levels; i++) {
    ierr = VecDestroy(X[i]);CHKERRQ(ierr);
    ierr = VecDestroy(B[i]);CHKERRQ(ierr);
    ierr = VecDestroy(R[i]);CHKERRQ(ierr);
  }
  for (i=0; i<levels-1; i++) {
    ierr = MatDestroy(mat[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(cmat);CHKERRQ(ierr);
  ierr = MatDestroy(fmat);CHKERRQ(ierr);
  ierr = SLESDestroy(slesmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "residual"
int residual(Mat mat,Vec bb,Vec xx,Vec rr)
{
  int         i,n1,ierr;
  PetscScalar *b,*x,*r;

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
int amult(Mat mat,Vec xx,Vec yy)
{
  int         i,n1,ierr;
  PetscScalar *y,*x;

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
int gauss_seidel(void *ptr,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal atol,PetscReal dtol,int m)
{
  int         i,n1,ierr;
  PetscScalar *x,*b;

  PetscFunctionBegin;
  ierr = VecGetSize(bb,&n1);CHKERRQ(ierr); n1--;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
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
int jacobi(void *ptr,Vec bb,Vec xx,Vec w,PetscReal rtol,PetscReal atol,PetscReal dtol,int m)
{
  int         i,n,n1,ierr;
  PetscScalar *r,*b,*x;

  PetscFunctionBegin;
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
int interpolate(Mat mat,Vec xx,Vec yy,Vec zz)
{
  int         i,n,N,i2,ierr;
  PetscScalar *x,*y;

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
int restrct(Mat mat,Vec rr,Vec bb)
{
  int         i,n,N,i2,ierr;
  PetscScalar *r,*b;

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
#define __FUNCT__ "Create2dLaplacian"
int Create1dLaplacian(int n,Mat *mat)
{
  PetscScalar mone = -1.0,two = 2.0;
  int         ierr,i,idx;

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
int CalculateRhs(Vec u)
{
  int         i,n,ierr;
  PetscReal   h,x = 0.0;
  PetscScalar uu;

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
int CalculateSolution(int n,Vec *solution)
{
  int         i,ierr;
  PetscReal   h,x = 0.0;
  PetscScalar uu;

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
int CalculateError(Vec solution,Vec u,Vec r,PetscReal *e)
{
  PetscScalar mone = -1.0;
  int    ierr;

  PetscFunctionBegin;
  ierr = VecNorm(r,NORM_2,e+2);CHKERRQ(ierr);
  ierr = VecWAXPY(&mone,u,solution,r);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,e);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_1,e+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


