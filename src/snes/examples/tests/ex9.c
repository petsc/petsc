/* Peter Mell Modified this file   8/95 */

#ifndef lint
static char vcid[] = "$Id: ex9.c,v 1.12 1996/01/01 01:05:24 bsmith Exp bsmith $";
#endif

static char help[] =
"This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel.  This example uses matrix free Krylov\n\
Newton methods with no preconditioner.\n\
The Bratu (SFI - solid fuel ignition) test problem\n\
is solved.  The command line options are:\n\
   -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
      problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
   -mx <xg>, where <xg> = number of grid points in the x-direction\n\
   -my <yg>, where <yg> = number of grid points in the y-direction\n\
   -mz <zg>, where <zg> = number of grid points in the y-direction\n\n";

/*  
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y,z < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
   
    A finite difference approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
*/

#include "snes.h"
#include "da.h"
#include <math.h>
#include <stdio.h>

typedef struct {
    double    param;           /* test problem nonlinearity parameter */
    int       mx, my, mz;      /* discretization in x,y,z-directions */
    Vec       localX, localF;  /* ghosted local vectors */
    DA        da;              /* distributed array datastructure */
} AppCtx;

int FormFunction1(SNES,Vec,Vec,void*),FormInitialGuess1(SNES,Vec,void*);

int main( int argc, char **argv )
{
  SNES          snes;
  SLES          sles;
  PC            pc;
  Mat           J;
  SNESType      method = SNES_EQ_NLS;  /* nonlinear solution method */
  Vec           x,r;
  int           ierr, its, N, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE, Nz = PETSC_DECIDE; 
  AppCtx        user;
  double        bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  DAStencilType stencil = DA_STENCIL_BOX;

  PetscInitialize( &argc, &argv, 0,0,help );
  if (OptionsHasName(PETSC_NULL,"-star")) stencil = DA_STENCIL_STAR;

  user.mx    = 4; 
  user.my    = 4; 
  user.mz    = 4; 
  user.param = 6.0;
  OptionsGetInt(PETSC_NULL,"-mx",&user.mx); 
  OptionsGetInt(PETSC_NULL,"-my",&user.my);
  OptionsGetInt(PETSC_NULL,"-mz",&user.mz);
  OptionsGetInt(PETSC_NULL,"-Nx",&Nx); 
  OptionsGetInt(PETSC_NULL,"-Ny",&Ny);
  OptionsGetInt(PETSC_NULL,"-Nz",&Nz);
  OptionsGetDouble(PETSC_NULL,"-par",&user.param);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRA(1,"Lambda is out of range");
  }
  N = user.mx*user.my*user.mz;
  
  /* Set up distributed array */
  ierr = DACreate3d(MPI_COMM_WORLD,DA_NONPERIODIC,stencil,user.mx,
                    user.my,user.mz,Nx,Ny,Nz,1,1,&user.da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(user.da,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = DAGetLocalVector(user.da,&user.localX); CHKERRA(ierr);
  ierr = VecDuplicate(user.localX,&user.localF); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess1,(void *)&user); CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction1,(void *)&user); CHKERRA(ierr);
  ierr = SNESDefaultMatrixFreeMatCreate(snes,x,&J); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,0,(void *)&user); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Force no preconditioning to be used. */
  ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);

  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its);  CHKERRA(ierr);

  MPIU_printf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n", its );

  /* Free data structures */
  ierr = VecDestroy(user.localX); CHKERRA(ierr);
  ierr = VecDestroy(user.localF); CHKERRA(ierr);
  ierr = DADestroy(user.da); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr); ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr); ierr = SNESDestroy(snes); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}/* --------------------  Form initial approximation ----------------- */
int FormInitialGuess1(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i,j,k, loc, mx, my, mz, ierr,xs,ys,zs,xm,ym,zm,Xm,Ym,Zm,Xs,Ys,Zs,base1;
  double  one = 1.0, lambda, temp1, temp, hx, hy, hxdhy, hydhx,sc;
  Scalar  *x;
  Vec     localX = user->localX;

  mx	 = user->mx; my	 = user->my; mz = user->mz; lambda = user->param;
  hx     = one / (double)(mx-1);     hy     = one / (double)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy; hydhx  = hy/hx;

  ierr  = VecGetArray(localX,&x); CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  ierr  = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr  = DAGetGhostCorners(user->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
 
  for (k=zs; k<zs+zm; k++) {
    base1 = (Xm*Ym)*(k-Zs);
    for (j=ys; j<ys+ym; j++) {
      temp = (double)(PetscMin(j,my-j-1))*hy;
      for (i=xs; i<xs+xm; i++) {
        loc = base1 + i-Xs + (j-Ys)*Xm; 
        if (i == 0 || j == 0 || k == 0 || i==mx-1 || j==my-1 || k==mz-1) {
          x[loc] = 0.0; 
          continue;
        }
        x[loc] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
      }
    }
  }

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  /* stick values into global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
}/* --------------------  Evaluate Function F(x) --------------------- */
int FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, k,loc, mx,my,mz,xs,ys,zs,xm,ym,zm,Xs,Ys,Zs,Xm,Ym,Zm;
  int     base1, base2;
  double  two = 2.0, one = 1.0, lambda,hx, hy, hz, hxhzdhy, hyhzdhx,hxhydhz;
  Scalar  u, uxx, uyy, sc,*x,*f,uzz;
  Vec     localX = user->localX, localF = user->localF; 

  mx      = user->mx; my = user->my; mz = user->mz; lambda = user->param;
  hx      = one / (double)(mx-1);
  hy      = one / (double)(my-1);
  hz      = one / (double)(mz-1);
  sc      = hx*hy*hz*lambda; hxhzdhy  = hx*hz/hy; hyhzdhx  = hy*hz/hx;
  hxhydhz = hx*hy/hz;

  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);

  ierr = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm); CHKERRQ(ierr);

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
          uxx = (two*u - x[loc-1] - x[loc+1])*hyhzdhx;
          uyy = (two*u - x[loc-Xm] - x[loc+Xm])*hxhzdhy;
          uzz = (two*u - x[loc-Xm*Ym] - x[loc+Xm*Ym])*hxhydhz;
          f[loc] = uxx + uyy + uzz - sc*exp(u);
        }
      }  
    }
  }  
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);
  /* stick values into global vector */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F);
  PLogFlops(11*ym*xm*zm);
  return 0; 
}
 




 





















