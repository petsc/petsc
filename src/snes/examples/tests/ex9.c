
static char help[] = "This program demonstrates use of the SNES package. Solve systems of\n\
nonlinear equations in parallel.  This example uses matrix-free Krylov\n\
Newton methods with no preconditioner to solve the Bratu (SFI - solid fuel\n\
ignition) test problem. The command line options are:\n\
   -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
      problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
   -mx <xg>, where <xg> = number of grid points in the x-direction\n\
   -my <yg>, where <yg> = number of grid points in the y-direction\n\
   -mz <zg>, where <zg> = number of grid points in the z-direction\n\n";

/*  
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y,z < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
   
    A finite difference approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
*/

#include "petscsnes.h"
#include "petscda.h"

typedef struct {
    PetscReal param;           /* test problem nonlinearity parameter */
    PetscInt  mx,my,mz;      /* discretization in x,y,z-directions */
    Vec       localX,localF;  /* ghosted local vectors */
    DA        da;              /* distributed array datastructure */
} AppCtx;

extern PetscErrorCode  FormFunction1(SNES,Vec,Vec,void*),FormInitialGuess1(AppCtx*,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  KSP            ksp;                 /* linear solver */
  PC             pc;                   /* preconditioner */
  Mat            J;                    /* Jacobian matrix */
  AppCtx         user;                 /* user-defined application context */
  Vec            x,r;                  /* vectors */
  DAStencilType  stencil = DA_STENCIL_BOX;
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscInt       Nx = PETSC_DECIDE,Ny = PETSC_DECIDE,Nz = PETSC_DECIDE,its;
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsHasName(PETSC_NULL,"-star",&flg);CHKERRQ(ierr);
  if (flg) stencil = DA_STENCIL_STAR;

  user.mx    = 4; 
  user.my    = 4; 
  user.mz    = 4; 
  user.param = 6.0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mz",&user.mz,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRQ(1,"Lambda is out of range");
  }
  
  /* Set up distributed array */
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,stencil,user.mx,user.my,user.mz,
                    Nx,Ny,Nz,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localF);CHKERRQ(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  /* Set various routines and options */
  ierr = SNESSetFunction(snes,r,FormFunction1,(void*)&user);CHKERRQ(ierr);
  ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Force no preconditioning to be used.  Note that this overrides whatever
     choices may have been specified in the options database. */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);

  /* Solve nonlinear system */
  ierr = FormInitialGuess1(&user,x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D\n",its);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(user.localX);CHKERRQ(ierr);
  ierr = VecDestroy(user.localF);CHKERRQ(ierr);
  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}/* --------------------  Form initial approximation ----------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode  FormInitialGuess1(AppCtx *user,Vec X)
{
  PetscInt       i,j,k,loc,mx,my,mz,xs,ys,zs,xm,ym,zm,Xm,Ym,Zm,Xs,Ys,Zs,base1;
  PetscErrorCode ierr;
  PetscReal      one = 1.0,lambda,temp1,temp,Hx,Hy;
  PetscScalar    *x;
  Vec            localX = user->localX;

  mx	 = user->mx; my	 = user->my; mz = user->mz; lambda = user->param;
  Hx     = one / (PetscReal)(mx-1);     Hy     = one / (PetscReal)(my-1);

  ierr  = VecGetArray(localX,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  ierr  = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr  = DAGetGhostCorners(user->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
 
  for (k=zs; k<zs+zm; k++) {
    base1 = (Xm*Ym)*(k-Zs);
    for (j=ys; j<ys+ym; j++) {
      temp = (PetscReal)(PetscMin(j,my-j-1))*Hy;
      for (i=xs; i<xs+xm; i++) {
        loc = base1 + i-Xs + (j-Ys)*Xm; 
        if (i == 0 || j == 0 || k == 0 || i==mx-1 || j==my-1 || k==mz-1) {
          x[loc] = 0.0; 
          continue;
        }
        x[loc] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*Hx,temp)); 
      }
    }
  }

  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  /* stick values into global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
}/* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
PetscErrorCode  FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k,loc,mx,my,mz,xs,ys,zs,xm,ym,zm,Xs,Ys,Zs,Xm,Ym,Zm,base1,base2;
  PetscReal      two = 2.0,one = 1.0,lambda,Hx,Hy,Hz,HxHzdHy,HyHzdHx,HxHydHz;
  PetscScalar    u,uxx,uyy,sc,*x,*f,uzz;
  Vec            localX = user->localX,localF = user->localF; 

  mx      = user->mx; my = user->my; mz = user->mz; lambda = user->param;
  Hx      = one / (PetscReal)(mx-1);
  Hy      = one / (PetscReal)(my-1);
  Hz      = one / (PetscReal)(mz-1);
  sc      = Hx*Hy*Hz*lambda; HxHzdHy  = Hx*Hz/Hy; HyHzdHx  = Hy*Hz/Hx;
  HxHydHz = Hx*Hy/Hz;

  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  ierr = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    base1 = (Xm*Ym)*(k-Zs); 
    for (j=ys; j<ys+ym; j++) {
      base2 = base1 + (j-Ys)*Xm; 
      for (i=xs; i<xs+xm; i++) {
        loc = base2 + (i-Xs);
        if (i == 0 || j == 0 || k== 0 || i == mx-1 || j == my-1 || k == mz-1) {
          f[loc] = x[loc]; 
        }
        else {
          u = x[loc];
          uxx = (two*u - x[loc-1] - x[loc+1])*HyHzdHx;
          uyy = (two*u - x[loc-Xm] - x[loc+Xm])*HxHzdHy;
          uzz = (two*u - x[loc-Xm*Ym] - x[loc+Xm*Ym])*HxHydHz;
          f[loc] = uxx + uyy + uzz - sc*PetscExpScalar(u);
        }
      }  
    }
  }  
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  /* stick values into global vector */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm*zm);CHKERRQ(ierr);
  return 0; 
}
 




 





















