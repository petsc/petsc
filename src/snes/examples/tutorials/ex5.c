#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.17 1995/08/01 19:10:03 bsmith Exp bsmith $";
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
   -my <yg>, where <yg> = number of grid points in the y-direction\n\n";

/*  
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
*/

#include "draw.h"
#include "snes.h"
#include "da.h"
#include <math.h>
#include <stdio.h>

typedef struct {
      double      param;         /* test problem parameter */
      int         mx,my;         /* Discretization in x,y-direction */
      Vec         localX,localF; /* ghosted local vector */
      DA          da;            /* regular array datastructure */
} AppCtx;

int  FormFunction1(SNES,Vec,Vec,void*),FormInitialGuess1(SNES,Vec,void*);

int main( int argc, char **argv )
{
  SLES         sles;
  PC           pc;
  SNES         snes;
  SNESMethod   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r;
  int          ierr, its, N; 
  AppCtx       user;
  double       bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  Mat          J;

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);

  user.mx    = 4; user.my    = 4; user.param = 6.0;
  OptionsGetInt(0,"-mx",&user.mx); OptionsGetInt(0,"-my",&user.my);
  OptionsGetDouble(0,"-par",&user.param);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRA(1,"Lambda is out of range");
  }
  N          = user.mx*user.my;
  
  /* Set up distributed array */
  ierr = DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
               user.mx,user.my,PETSC_DECIDE,PETSC_DECIDE,1,1,&user.da); 
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(user.da,&x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = DAGetLocalVector(user.da,&user.localX); CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localF); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);
  CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess1,(void *)&user); 
           CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction1,(void *)&user,0); 
           CHKERRA(ierr);
  ierr =  SNESDefaultMatrixFreeMatCreate(snes,x,&J);CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,0,(void *)&user); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Force no preconditioning to be used. */
  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRQ(ierr);
  ierr = PCSetMethod(pc,PCNONE); CHKERRQ(ierr);

  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its);  CHKERRA(ierr);

  MPIU_printf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n", its );

  /* Free data structures */
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(user.localX); CHKERRA(ierr);
  ierr = VecDestroy(user.localF); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = DADestroy(user.da); CHKERRQ(ierr);
  PetscFinalize();

  return 0;
}/* --------------------  Form initial approximation ----------------- */
int FormInitialGuess1(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i, j, row, mx, my, ierr,xs,ys,xm,ym,Xm,Ym,Xs,Ys;
  double  one = 1.0, lambda, temp1, temp, hx, hy, hxdhy, hydhx,sc;
  Scalar  *x;
  Vec     localX = user->localX;

  mx	 = user->mx; my	 = user->my; lambda = user->param;
  hx     = one / (double)(mx-1);     hy     = one / (double)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy; hydhx  = hy/hx;

  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);
  DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);
  for (j=ys; j<ys+ym; j++) {
    temp = (double)(PETSCMIN(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Xm; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PETSCMIN( (double)(PETSCMIN(i,mx-i-1))*hx,temp) ); 
    }
  }
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  /* stick values into global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERTVALUES,X);
  return 0;
}/* --------------------  Evaluate Function F(x) --------------------- */
int FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my,xs,ys,xm,ym,Xs,Ys,Xm,Ym;
  double  two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx,sc;
  Scalar  u, uxx, uyy, *x,*f;
  Vec     localX = user->localX, localF = user->localF; 

  mx	 = user->mx; my	 = user->my;lambda = user->param;
  hx     = one / (double)(mx-1);
  hy     = one / (double)(my-1);
  sc     = hx*hy*lambda; hxdhy  = hx/hy; hydhx  = hy/hx;

  ierr = DAGlobalToLocalBegin(user->da,X,INSERTVALUES,localX);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERTVALUES,localX);
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);
  DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);
  DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);

  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-Xm] - x[row+Xm])*hxdhy;
      f[row] = uxx + uyy - sc*exp(u);
    }
  }
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);
  /* stick values into global vector */







  ierr = DALocalToGlobal(user->da,localF,INSERTVALUES,F);
  PLogFlops(11*ym*xm);
  return 0; 
}

