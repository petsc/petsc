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
  CHKERRQ(VecGetArray(X,&x));
  /**/
  formInitial(&app->nx,&app->ny,&app->nz,app->h,&t,x);
  /**/
  CHKERRQ(VecRestoreArray(X,&x));
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
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  /**/
  formFunction(&app->nx,&app->ny,&app->nz,app->h,&t,x,xdot,f);
  /**/
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
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

  CHKERRQ(VecCreate(PETSC_COMM_SELF,&x));
  CHKERRQ(VecSetSizes(x,nx*ny*nz,nx*ny*nz));
  CHKERRQ(VecSetUp(x));
  CHKERRQ(VecDuplicate(x,&f));

  CHKERRQ(TSCreate(PETSC_COMM_SELF,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSTHETA));
  CHKERRQ(TSThetaSetTheta(ts,1.0));
  CHKERRQ(TSSetTimeStep(ts,0.01));
  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetMaxTime(ts,1.0));
  CHKERRQ(TSSetMaxSteps(ts,10));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  CHKERRQ(TSSetSolution(ts,x));
  CHKERRQ(TSSetIFunction(ts,f,FormFunction,app));
  CHKERRQ(PetscOptionsSetValue(NULL,"-snes_mf","1"));
  {
    SNES snes;
    KSP  ksp;
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESGetKSP(snes,&ksp));
    CHKERRQ(KSPSetType(ksp,KSPCG));
  }
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetUp(ts));

  *wt = 1e300;
  while (loops-- > 0) {
    CHKERRQ(FormInitial(0.0,x,app));
    CHKERRQ(PetscTime(&t1));
    CHKERRQ(TSSolve(ts,x));
    CHKERRQ(PetscTime(&t2));
    *wt = PetscMin(*wt,t2-t1);
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(TSDestroy(&ts));

  PetscFunctionReturn(0);
}

PetscErrorCode GetInt(const char* name, PetscInt *v, PetscInt defv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *v = defv;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,name,v,NULL));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  double         wt;
  PetscInt       n,start,step,stop,samples;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,NULL));

  CHKERRQ(GetInt("-start",   &start,   12));
  CHKERRQ(GetInt("-step",    &step,    4));
  CHKERRQ(GetInt("-stop",    &stop,    start));
  CHKERRQ(GetInt("-samples", &samples, 1));

  for (n=start; n<=stop; n+=step) {
    int nx=n+1, ny=n+1, nz=n+1;
    CHKERRQ(RunTest(nx,ny,nz,samples,&wt));
    ierr = PetscPrintf(PETSC_COMM_SELF,
                       "Grid  %3d x %3d x %3d -> %f seconds (%2d samples)\n",
                       nx,ny,nz,wt,samples);CHKERRQ(ierr);
  }

  CHKERRQ(PetscFinalize());
  return 0;
}
