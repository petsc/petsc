
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

#include "vec.h"
#include "draw.h"
#include "snes.h"
#include "options.h"
#include "ra.h"
#include <math.h>
#include <stdio.h>
#define MIN(a,b) ( ((a)<(b)) ? a : b )

typedef struct {
      double      param;         /* test problem parameter */
      int         mx;            /* Discretization in x-direction */
      int         my;            /* Discretization in y-direction */
      Vec         localX,localF; /* ghosted local vector */
      RA          ra;            /* regular array datastructure */
} AppCtx;

int  FormFunction1(SNES,Vec,Vec,void*),
     FormInitialGuess1(SNES,Vec,void*);

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

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  user.mx    = 4;
  user.my    = 4;
  user.param = 6.0;
  OptionsGetInt(0,0,"-mx",&user.mx);
  OptionsGetInt(0,0,"-my",&user.my);
  OptionsGetDouble(0,0,"-param",&user.param);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERR(1,"Lambda is out of range");
  }
  N          = user.mx*user.my;
  
  /* Set up distributed array */
  ierr = RACreate2d(MPI_COMM_WORLD,user.mx,user.my,PETSC_DECIDE,PETSC_DECIDE,
                    1,1,&user.ra); CHKERRA(ierr);
  ierr = RAGetDistributedVector(user.ra,&x); CHKERR(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = RAGetLocalVector(user.ra,&user.localX); CHKERR(ierr);
  ierr = VecDuplicate(user.localX,&user.localF); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess1,(void *)&user); 
           CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction1,(void *)&user,0); 
           CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,0,0,SNESDefaultMatrixFreeComputeJacobian,
                         (void *)&user); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Force no preconditioning to be used. */
  ierr = SNESGetSLES(snes,&sles); CHKERR(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERR(ierr);
  ierr = PCSetMethod(pc,PCNONE); CHKERR(ierr);

  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its);  CHKERRA(ierr);

  MPE_printf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = RADestroy(user.ra); CHKERR(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess1(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i, j, row, mx, my, ierr,xs,ys,xm,ym,Xm,Ym,Xs,Ys;
  double  one = 1.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  sc;
  double  *x;
  Vec     localX = user->localX;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(localX,&x); CHKERR(ierr);
  temp1 = lambda/(lambda + one);
  RAGetCorners(user->ra,&xs,&ys,0,&xm,&ym,0);
  RAGetGhostCorners(user->ra,&Xs,&Ys,0,&Xm,&Ym,0);

  for (j=ys; j<ys+ym; j++) {
    temp = (double)(MIN(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Ym; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( MIN( (double)(MIN(i,mx-i-1))*hx,temp) ); 
    }
  }
  ierr = VecRestoreArray(localX,&x); CHKERR(ierr);
  /* stick values into global vector */
  ierr = RALocalToGlobal(user->ra,localX,INSERTVALUES,X);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */
 
int FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my,xs,ys,xm,ym,Xs,Ys,Xm,Ym;
  double  two = 2.0, one = 1.0, lambda;
  double  hx, hy, hxdhy, hydhx;
  double  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;
  Vec     localX = user->localX, localF = user->localF; 

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = RAGlobalToLocal(user->ra,X,INSERTVALUES,localX);
  ierr = VecGetArray(localX,&x); CHKERR(ierr);
  ierr = VecGetArray(localF,&f); CHKERR(ierr);
  RAGetCorners(user->ra,&xs,&ys,0,&xm,&ym,0);
  RAGetGhostCorners(user->ra,&Xs,&Ys,0,&Xm,&Ym,0);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Ym; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      ub = x[row - Xm];
      ul = x[row - 1];
      ut = x[row + Xm];
      ur = x[row + 1];
      uxx = (-ur + two*u - ul)*hydhx;
      uyy = (-ut + two*u - ub)*hxdhy;
      f[row] = uxx + uyy - sc*lambda*exp(u);
    }
  }
  ierr = VecRestoreArray(localX,&x); CHKERR(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERR(ierr);
  /* stick values into global vector */
  ierr = RALocalToGlobal(user->ra,localF,INSERTVALUES,F);
  return 0; 
}

