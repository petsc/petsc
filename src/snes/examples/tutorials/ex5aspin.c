
static char help[] = "Bratu nonlinear PDE in 2d.\n\
This particular implementation of the problem has an ASPIN residual for the outer solver\n";

/*T
   Concepts: SNES^parallel Bratu example
   Concepts: DMDA^using distributed arrays;
   Concepts: IS coloirng types;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  

 This particular modification of the example has an ASPIN nonlinear function with dirichlet 
 boundary conditions on the subdomain boundaries.  use with -snes_mf.

The subdomain DA has one row of ghost points; the global patch has overlap + 1 ghost points
to allow for communication.

The amount of overlap may be controlled with -overlap

  ------------------------------------------------------------------------- */

/* 
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/
#include <petscdmda.h>
#include <petscsnes.h>

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PassiveReal param;          /* test problem parameter */
  SNES        sneslocal;      /* the local SNES */
  PetscInt    overlap;        /* the amount of subdomain overlap */
} AppCtx;

/* 
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(DM,AppCtx*,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
extern PetscErrorCode FormFunctionASPIN(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                   snes,sneslocal;
  Vec                    x;
  AppCtx                 user;
  PetscInt               its;
  PetscErrorCode         ierr;
  PetscReal              bratu_lambda_max = 6.81;
  PetscReal              bratu_lambda_min = 0.;
  DM                     da,dalocal;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.param = 6.0;
  user.overlap = 0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-overlap",&user.overlap,PETSC_NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) SETERRQ3(PETSC_COMM_SELF,1,"Lambda, %g, is out of range, [%g, %g]", user.param, bratu_lambda_min, bratu_lambda_max);

  /* Create global discretization and solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_GHOSTED,
                      DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1+user.overlap,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,PETSC_NULL,FormFunctionASPIN,(void*)&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Create local discretization and solver by getting the info of the global discretization and extrapolating */
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_SELF, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX,info.xm+2*(user.overlap+1),info.ym+2*(user.overlap+1),
                      PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&dalocal);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_SELF,&sneslocal);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dalocal,
                                   (PetscReal)(info.xs-user.overlap-1)/(info.mx-1), (PetscReal)(info.xs+info.xm+user.overlap)/(info.mx-1),
                                   (PetscReal)(info.ys-user.overlap-1)/(info.my-1), (PetscReal)(info.ys+info.ym+user.overlap)/(info.my-1), 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dalocal,&user);CHKERRQ(ierr);
  ierr = SNESSetDM(sneslocal,dalocal);CHKERRQ(ierr);
  ierr = SNESAppendOptionsPrefix(sneslocal,"sub_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)sneslocal,(PetscObject)snes,2);CHKERRQ(ierr);
  user.sneslocal = sneslocal;

  ierr = DMDASetLocalFunction(dalocal,(DMDALocalFunction1)FormFunctionLocal);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(sneslocal);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecSet(x,0.);CHKERRQ(ierr);
  ierr = FormInitialGuess(da,&user,x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

 ierr = SNESDestroy(&sneslocal);CHKERRQ(ierr);
 ierr = DMDestroy(&dalocal);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(DM da,AppCtx *user,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,temp,hx,hy;
  PetscScalar    **x;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  lambda = user->param;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  temp1  = lambda/(lambda + 1.0);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0; 
      } else {
        x[j][i] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp)); 
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch


 */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      lambda,hx,hy,hxdhy,hydhx,sc;
  PetscScalar    u,uxx,uyy;
  PetscReal      xl[3],xh[3],xp,yp;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetBoundingBox(info->da,xl,xh);CHKERRQ(ierr);
  lambda = user->param;
  hx     = (xh[0] - xl[0]) / (info->mx-1);
  hy     = (xh[1] - xl[1]) / (info->my-1);
  sc     = hx*hy*lambda;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      xp = xl[0] + hx*i;
      yp = xl[1] + hy*j;
      if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
        f[j][i] = 0.;
        /* PetscPrintf(PETSC_COMM_WORLD, "%f %f: %f\n", xp,yp,x[j][i]); */
      } else if (xp <= 0. || yp <= 0. || xp >= 1.0 || yp >= 1.0) {
        f[j][i] = 2.0*(hydhx+hxdhy)*x[j][i];
      } else {
        u       = x[j][i];
        uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
        uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
        f[j][i] = uxx + uyy - sc*PetscExpScalar(u);
      }
    }
  }
  ierr = PetscLogFlops(11.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionASPIN"
/*
 This forms the function:

 G(X) = X - R_l^T M_l(R_l X_l);

 And proceeds in three stages:

 1. Compute the whole function
 2. Solve the local equation
 3. Inject it back into the global solution

 */
PetscErrorCode FormFunctionASPIN(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm,dmlocal;
  Vec            Xlocal,Xlocalloc,Xgloballoc;
  AppCtx         *user = (AppCtx*)ctx;
  SNES           sneslocal=user->sneslocal;
  PetscInt       i,j;
  DMDALocalInfo  info,ginfo;
  PetscScalar    **xloc,**xglob;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = SNESGetDM(sneslocal,&dmlocal);CHKERRQ(ierr);
  DMGetGlobalVector(dmlocal,&Xlocal);CHKERRQ(ierr);

  /* get work vectors */
  ierr = DMDAGetLocalInfo(dm,&ginfo);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dmlocal,&info);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmlocal,&Xlocalloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xgloballoc);CHKERRQ(ierr);
  ierr = VecSet(Xgloballoc,0.);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xgloballoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xgloballoc);CHKERRQ(ierr);

  /* local for both DMs should be the same size without overlap */
  ierr = VecCopy(Xgloballoc,Xlocalloc);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmlocal,Xlocalloc,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmlocal,Xlocalloc,INSERT_VALUES,Xlocal);CHKERRQ(ierr);

  /* local solve */
  ierr = SNESSolve(sneslocal,PETSC_NULL,Xlocal);CHKERRQ(ierr);

  /* copy the local solution back over and redistribute */
  ierr = DMGlobalToLocalBegin(dmlocal,Xlocal,INSERT_VALUES,Xlocalloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmlocal,Xlocal,INSERT_VALUES,Xlocalloc);CHKERRQ(ierr);
  ierr = VecAYPX(Xgloballoc,-1.0,Xlocalloc);CHKERRQ(ierr);

  /* restrict and subtract */
  ierr = DMLocalToGlobalBegin(dm,Xgloballoc,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,Xgloballoc,INSERT_VALUES,F);CHKERRQ(ierr);


  /* restore work vectors */
  ierr = DMRestoreLocalVector(dmlocal,&Xlocalloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xgloballoc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmlocal,&Xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
