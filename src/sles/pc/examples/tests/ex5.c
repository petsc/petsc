
static char help[] = "           Test of Multigrid Code\n\
    -x N              Use a mesh in the x direction of N.  \n\
    -c N              Use N V-cycles.  \n\
    -l N              Use N Levels.  \n\
    -smooths N        Use N pre smooths and N post smooths.  \n\
    -j                Use Jacobi smoother.  \n\
    -a use additive multigrid \n\
    -f use full multigrid (preconditioner variant) \n\
    This example demonstrates how matrix free methods may be written\n";

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include "draw.h"
#include "mg.h"

int  residual(Mat,Vec,Vec,Vec);
int  gauss_seidel(void *,Vec,Vec,Vec,int);
int  jacobi(void *,Vec,Vec,Vec,int);
int  interpolate(void *,Vec,Vec,Vec);
int  restrict(void *,Vec,Vec);
int  Create1dLaplacian(int,Mat*);
int  CalculateRhs(Vec);
int  CalculateError(Vec,Vec,Vec,double*);
int  CalculateSolution(int,Vec*);

int main(int Argc, char **Args)
{
  MG          *mg;
  int         x_mesh = 15,levels = 3,cycles = 1;
  int         i,smooths = 1;
  int         *N, use_jacobi = 0, am = Multiplicative;
  Mat         cmat,mat[20];
  SLES        csles,sles[20];
  double      d[3], e[3], s; /* l_2 error, max error, residual */
  DrawLGCtx   lg;
  DrawCtx     win;
  int         ierr;
  Vec         x,solution,X,R,B;
  Scalar      zero = 0.0;
  PC          pc;
  KSP         ksp;

  OptionsCreate(&Argc,&Args,0,0);

  /* open the window */
/*
  ierr = DrawOpenX(0,0,0,0,300,300,&win); CHKERR(ierr);
  ierr = DrawLGCreate(win,3,&lg); CHKERR(ierr);
*/

  OptionsGetInt(1,0,"-x",&x_mesh);  
  OptionsGetInt(1,0,"-l",&levels);  
  OptionsGetInt(1,0,"-c",&cycles);  
  OptionsGetInt(1,0,"-smooths",&smooths);  
  if (OptionsHasName(1,0,"-help")) {fprintf(stderr,"%s",help); exit(0);}
  if (OptionsHasName(1,0,"-a")) {am = Additive;}
  if (OptionsHasName(1,0,"-f")) {am = FullMultigrid;}
  if (OptionsHasName(1,0,"-j")) {use_jacobi = 1;}
         
  N = (int *) MALLOC(levels*sizeof(int)); CHKPTR(N);
  N[0] = x_mesh;
  for ( i=1; i<levels; i++ ) {
    N[i] = N[i-1]/2;
    if (N[i] < 1) {SETERR(1,"Too many levels");}
  }

  Create1dLaplacian(N[levels-1],&cmat);
  SLESCreate(&csles);
  SLESSetMat(csles,cmat);

  SLESGetPC(csles,&pc); PCSetMethod(pc,PCDIRECT);
  SLESGetKSP(csles,&ksp); KSPSetMethod(ksp,KSPPREONLY);

  ierr = MGCreate(levels,&mg);

  /* zero is finest level */
  MGSetCoarseSolve(mg,csles);
  for ( i=0; i<levels-1; i++ ) {
      MGSetResidual(mg,levels - 1 - i,residual,(Mat)0);
      MatShellCreate(N[i],N[i+1],(void *)0,&mat[i]);
      MatShellSetMult(mat[i],restrict);
      MatShellSetMultTransAdd(mat[i],interpolate);
      MGSetInterpolate(mg,levels - 1 - i,mat[i]);
      MGSetRestriction(mg,levels - 1 - i,mat[i]);
      MGSetCyclesOnLevel(mg,levels - 1 - i,cycles);
      SLESCreate(&sles[i]);
      SLESGetPC(sles[i],&pc);
      PCSetMethod(pc,PCSHELL);
      SLESSetMat(sles[i],mat[i]); /* this is a dummy! */
      PCShellSetApplyRichardson(pc,gauss_seidel,(void *)0);
      if (use_jacobi) { PCShellSetApplyRichardson(pc,jacobi,(void *)0); }
      SLESGetKSP(sles[i],&ksp);
      KSPSetMethod(ksp,KSPRICHARDSON);
      KSPSetInitialGuessNonZero(ksp);
      KSPSetIterations(ksp,smooths);
      MGSetSmootherDown(mg,levels - 1 - i,sles[i]);
      MGSetSmootherUp(mg,levels - 1 - i,sles[i]);
      VecCreateSequential(N[i],&x); if (!i) X = x;
      MGSetX(mg,levels - 1 - i,x);
      VecCreateSequential(N[i],&x); if (!i) B = x;
      MGSetRhs(mg,levels - 1 - i,x);
      VecCreateSequential(N[i],&x); if (!i) R = x;
      MGSetR(mg,levels - 1 - i,x);
  } 
  /* create coarse level vectors */
  VecCreateSequential(N[levels-1],&x); MGSetX(mg,0,x); if (levels==1) X = x;
  VecCreateSequential(N[levels-1],&x); MGSetRhs(mg,0,x);if (levels==1) B = x;
  VecCreateSequential(N[levels-1],&x); MGSetR(mg,0,x);if (levels==1) R = x;

  CalculateSolution(N[0],&solution);
  CalculateRhs(B);
  VecSet(&zero,X);

  if (MGCheck(mg)) {SETERR(1,0);}
     
  residual((void*)0,B,X,R);
  CalculateError(solution,X,R,e); i = 0; s = e[0];
  d[0] = d[1] = d[2] = (double) i;
  printf("l_2 error %g max error %g resi %g\n",e[0],e[1],e[2]);
  e[0] = log10(e[0]);e[1] = log10(e[1]);e[2] = log10(e[2]);
/*
  awLGAddPoint(lg,d,e); 
*/

  while (s > 1.e-13) {
    ierr = MGCycle(mg,am); CHKERR(ierr);
    i++;
    residual((void*)0,B,X,R);
    CalculateError(solution,X,R,e); 
    d[0] = d[1] = d[2] = (double) i;
    printf("l_2 error %g max error %g resi %g\n",e[0],e[1],e[2]);
    s = e[0];
    e[0] = log10(e[0]);e[1] = log10(e[1]);e[2] = log10(e[2]);

/*
    DrawLGAddPoint(lg,d,e);
    DrawLG(lg); 
*/
  }
/*
  DrawLGDestroy(lg);
  DrawDestroy(win);
*/
  MGDestroy(mg);
  return 0;
}

/* --------------------------------------------------------------------- */
int residual(Mat mat,Vec bb,Vec xx,Vec rr)
{
  int    i, n1;
  Scalar *b,*x,*r;

  VecGetSize(bb,&n1);
  VecGetArray(bb,&b); VecGetArray(xx,&x); VecGetArray(rr,&r);
  n1--;
  r[0] = b[0] + x[1] - 2.0*x[0];
  r[n1] = b[n1] + x[n1-1] - 2.0*x[n1];
  for ( i=1; i<n1; i++ ) {
    r[i] = b[i] + x[i+1] + x[i-1] - 2.0*x[i];
  }
  return 0;
}
/* --------------------------------------------------------------------- */
int gauss_seidel(void *ptr,Vec bb,Vec xx,Vec w,int m)
{
  int      i, n1;
  double *x, *b;
  VecGetSize(bb,&n1);n1--;
  VecGetArray(bb,&b); VecGetArray(xx,&x);
  while (m--) {
    x[0] =  .5*(x[1] + b[0]);
    for ( i=1; i<n1; i++ ) {
      x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    x[n1] = .5*(x[n1-1] + b[n1]);
    for ( i=n1-1; i>0; i-- ) {
      x[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    x[0] =  .5*(x[1] + b[0]);
  }
  return 0;
}
/* --------------------------------------------------------------------- */
int jacobi(void *ptr,Vec bb,Vec xx,Vec w,int m)
{
  int      i, n, n1;
  double   *r,*b,*x;

  VecGetSize(bb,&n); n1 = n - 1;
  VecGetArray(bb,&b); VecGetArray(xx,&x);
  VecGetArray(w,&r);

  while (m--) {
    r[0] = .5*(x[1] + b[0]);
    for ( i=1; i<n1; i++ ) {
       r[i] = .5*(x[i+1] + x[i-1] + b[i]);
    }
    r[n1] = .5*(x[n1-1] + b[n1]);
    for ( i=0; i<n; i++ ) x[i] = (2.0*r[i] + x[i])/3.0;
  }
  return 0;
}
/*
   We know for this application that yy  and zz are the same
*/
/* --------------------------------------------------------------------- */
int interpolate(void *ptr,Vec xx,Vec yy,Vec zz)
{
  int    i, n, N, i2;
  double *x,*y;

  VecGetSize(yy,&N);
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  n = N/2;
  for ( i=0; i<n; i++ ) {
    i2 = 2*i;
    y[i2] +=  .5*x[i];
    y[i2+1] +=  x[i];
    y[i2+2] +=  .5*x[i];
  }
  return 0;
}
/* --------------------------------------------------------------------- */
int restrict(void *ptr,Vec rr,Vec bb)
{
  int    i, n, N, i2;
  double *r,*b;

  VecGetSize(rr,&N);
  VecGetArray(rr,&r); VecGetArray(bb,&b);
  n = N/2;

  for ( i=0; i<n; i++ ) {
    i2 = 2*i;
    b[i] = ( r[i2] + 2.0*r[i2+1] + r[i2+2] );
  }
  return 0;
}

int Create1dLaplacian(int n,Mat *mat)
{
  Scalar mone = -1.0, two = 2.0;
  int    ierr,i,idx;
  ierr = MatCreateSequentialAIJ(n,n,3,0,mat); CHKERR(ierr);
  
  idx= n-1;
  MatSetValues(*mat,1,&idx,1,&idx,&two,InsertValues);
  for ( i=0; i<n-1; i++ ) {
    MatSetValues(*mat,1,&i,1,&i,&two,InsertValues);
    idx = i+1;
    MatSetValues(*mat,1,&idx,1,&i,&mone,InsertValues);
    MatSetValues(*mat,1,&i,1,&idx,&mone,InsertValues);
  }
  ierr = MatBeginAssembly(*mat); CHKERR(ierr);
  ierr = MatEndAssembly(*mat); CHKERR(ierr);
  return 0;
}

int CalculateRhs(Vec u)
{
  int    i,n, ierr;
  double h,x = 0.0,uu;
  VecGetSize(u,&n);
  h = 1.0/((double) (n+1));
  for ( i=0; i<n; i++ ) {
    x += h; uu = 2.0*h*h; 
    ierr = VecSetValues(u,1,&i,&uu,InsertValues); CHKERR(ierr);
  }

  return 0;
}

int CalculateSolution(int n,Vec *solution)
{
  int    i,ierr;
  double h,x = 0.0,uu;
  VecCreateSequential(n,solution);
  h = 1.0/((double) (n+1));
  for ( i=0; i<n; i++ ) {
    x += h; uu = x*(1.-x); 
    ierr = VecSetValues(*solution,1,&i,&uu,InsertValues); CHKERR(ierr);
  }

  return 0;
}

int CalculateError(Vec solution,Vec u,Vec r,double *e)
{
  int    i;
  Scalar mone = -1.0;

  VecNorm(r,e+2);
  VecWAXPY(&mone,u,solution,r);
  VecNorm(r,e);
  VecASum(r,e+1);
  return 0;
}


