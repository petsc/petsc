#include <petsc.h>

EXTERN_C_BEGIN
extern void formInitial(int*,int*,int*,double*,
                        double*,double*);
extern void formFunction(const int*,const int*,const int*,const double*,
                         const double*,const double[],const double[],double[]);
EXTERN_C_END

typedef struct AppCtx {
  PetscInt    nx,ny,nz;
  PetscScalar h[3];
} AppCtx;

PetscErrorCode FormInitial(PetscReal t, Vec X, void *ctx)
{
  PetscScalar    *x;
  AppCtx         *app = (AppCtx*) ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  /**/
  formInitial(&app->nx,&app->ny,&app->nz,app->h,&t,x);
  /**/
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec Xdot,Vec F, void *ctx)
{
  const PetscScalar *x;
  const PetscScalar    *xdot;
  PetscScalar    *f;
  AppCtx         *app = (AppCtx*) ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  /**/
  formFunction(&app->nx,&app->ny,&app->nz,app->h,&t,x,xdot,f);
  /**/
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RunTest(int nx, int ny, int nz, int loops, double *wt)
{
  Vec            x,f;
  TS             ts;
  AppCtx         _app,*app=&_app;
  double         t1,t2;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  app->nx = nx; app->h[0] = 1./(nx-1);
  app->ny = ny; app->h[1] = 1./(ny-1);
  app->nz = nz; app->h[2] = 1./(nz-1);

  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,nx*ny*nz,nx*ny*nz);CHKERRQ(ierr);
  ierr = VecSetUp(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,f,FormFunction,app);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-snes_mf","1");CHKERRQ(ierr);
  {
    SNES snes;
    KSP  ksp;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  *wt = 1e300;
  while (loops-- > 0) {
    ierr = FormInitial(0.0,x,app);CHKERRQ(ierr);
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    ierr = TSSolve(ts,x);CHKERRQ(ierr);
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    *wt = PetscMin(*wt,t2-t1);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode GetInt(const char* name, PetscInt *v, PetscInt defv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *v = defv;
  ierr = PetscOptionsGetInt(NULL,NULL,name,v,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  double         wt;
  PetscInt       n,start,step,stop,samples;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  ierr = GetInt("-start",   &start,   12);CHKERRQ(ierr);
  ierr = GetInt("-step",    &step,    4);CHKERRQ(ierr);
  ierr = GetInt("-stop",    &stop,    start);CHKERRQ(ierr);
  ierr = GetInt("-samples", &samples, 1);CHKERRQ(ierr);

  for (n=start; n<=stop; n+=step){
    int nx=n+1, ny=n+1, nz=n+1;
    ierr = RunTest(nx,ny,nz,samples,&wt);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,
                       "Grid  %3d x %3d x %3d -> %f seconds (%2d samples)\n",
                       nx,ny,nz,wt,samples);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
