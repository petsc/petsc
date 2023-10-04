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
  PetscFunctionBegin;
  PetscCall(VecGetArray(X,&x));
  /**/
  formInitial(&app->nx,&app->ny,&app->nz,app->h,&t,x);
  /**/
  PetscCall(VecRestoreArray(X,&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec Xdot,Vec F, void *ctx)
{
  const PetscScalar *x;
  const PetscScalar    *xdot;
  PetscScalar    *f;
  AppCtx         *app = (AppCtx*) ctx;
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(Xdot,&xdot));
  PetscCall(VecGetArray(F,&f));
  /**/
  formFunction(&app->nx,&app->ny,&app->nz,app->h,&t,x,xdot,f);
  /**/
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArrayRead(Xdot,&xdot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RunTest(int nx, int ny, int nz, int loops, double *wt)
{
  Vec            x,f;
  TS             ts;
  AppCtx         _app,*app=&_app;
  double         t1,t2;
  PetscFunctionBegin;

  app->nx = nx; app->h[0] = 1./(nx-1);
  app->ny = ny; app->h[1] = 1./(ny-1);
  app->nz = nz; app->h[2] = 1./(nz-1);

  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(VecSetSizes(x,nx*ny*nz,nx*ny*nz));
  PetscCall(VecSetUp(x));
  PetscCall(VecDuplicate(x,&f));

  PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSTHETA));
  PetscCall(TSThetaSetTheta(ts,1.0));
  PetscCall(TSSetTimeStep(ts,0.01));
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetMaxTime(ts,1.0));
  PetscCall(TSSetMaxSteps(ts,10));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  PetscCall(TSSetSolution(ts,x));
  PetscCall(TSSetIFunction(ts,f,FormFunction,app));
  PetscCall(PetscOptionsSetValue(NULL,"-snes_mf","1"));
  {
    SNES snes;
    KSP  ksp;
    PetscCall(TSGetSNES(ts,&snes));
    PetscCall(SNESGetKSP(snes,&ksp));
    PetscCall(KSPSetType(ksp,KSPCG));
  }
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetUp(ts));

  *wt = 1e300;
  while (loops-- > 0) {
    PetscCall(FormInitial(0.0,x,app));
    PetscCall(PetscTime(&t1));
    PetscCall(TSSolve(ts,x));
    PetscCall(PetscTime(&t2));
    *wt = PetscMin(*wt,t2-t1);
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(TSDestroy(&ts));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetInt(const char* name, PetscInt *v, PetscInt defv)
{
  PetscFunctionBegin;
  *v = defv;
  PetscCall(PetscOptionsGetInt(NULL,NULL,name,v,NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  double         wt;
  PetscInt       n,start,step,stop,samples;

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  PetscCall(GetInt("-start",   &start,   12));
  PetscCall(GetInt("-step",    &step,    4));
  PetscCall(GetInt("-stop",    &stop,    start));
  PetscCall(GetInt("-samples", &samples, 1));

  for (n=start; n<=stop; n+=step) {
    int nx=n+1, ny=n+1, nz=n+1;
    PetscCall(RunTest(nx,ny,nz,samples,&wt));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Grid  %3d x %3d x %3d -> %f seconds (%2d samples)\n",nx,ny,nz,wt,samples));
  }
  PetscCall(PetscFinalize());
  return 0;
}
