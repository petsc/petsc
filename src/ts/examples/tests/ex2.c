#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.4 1996/04/20 04:21:08 bsmith Exp bsmith $";
#endif

static char help[] ="Solves the time dependent Bratu problem";

/*
     This code demonstrates how one may solve a nonlinear problem 
   with pseudo time-stepping. In this simple example, the pseudo-time
   step is the same for all grid points, i.e. this is equivalent to 
   a backward Euler with variable time-step.
*/

#include "draw.h"
#include "ts.h"
#include "options.h"
#include <math.h>

typedef struct {
      double      param;        /* test problem parameter */
      int         mx;           /* Discretization in x-direction */
      int         my;           /* Discretization in y-direction */
} AppCtx;

typedef struct {
  SNES snes;
  Vec  w;
} MFCtx_Private;


int  FormJacobian(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(TS,double,Vec,Vec,void*),
     FormInitialGuess(Vec,AppCtx*);

int main( int argc, char **argv )
{
  TS           ts;
  Vec          x,r;
  Mat          J;
  int          ierr, N, its,flg; 
  AppCtx       user;
  double       bratu_lambda_max = 6.81, bratu_lambda_min = 0.,dt = 1.e-6;
  double       ftime;

  PetscInitialize( &argc, &argv, (char *)0,help );
  user.mx        = 4;
  user.my        = 4;
  user.param     = 6.0;
  
  OptionsGetInt(0,"-mx",&user.mx,&flg);
  OptionsGetInt(0,"-my",&user.my,&flg);
  OptionsGetDouble(0,"-param",&user.param,&flg);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRQ(1,"Lambda is out of range");
  }
  OptionsGetDouble(0,"-dt",&dt,&flg);
  N          = user.mx*user.my;
  
  /* Set up data structures */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,N,N,0,0,&J); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = TSCreate(MPI_COMM_WORLD,TS_NONLINEAR,&ts); CHKERRA(ierr);

  /* Set various routines */
  ierr = TSSetSolution(ts,x); CHKERRA(ierr);
  ierr = TSSetRHSFunction(ts,FormFunction,(void *)&user); CHKERRA(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormJacobian,(void *)&user);CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = FormInitialGuess(x,&user);
  ierr = TSSetType(ts,TS_PSEUDO_POSITION_INDEPENDENT_TIMESTEP); CHKERRA(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt); CHKERRA(ierr);
  ierr = TSSetDuration(ts,1000,1.e12);
  ierr = TSPseudoSetPositionIndependentTimeStep(ts,
                    TSPseudoDefaultPositionIndependentTimeStep,0); CHKERRA(ierr);

  ierr = TSSetFromOptions(ts); CHKERRA(ierr);
  ierr = TSSetUp(ts); CHKERRA(ierr);
  ierr = TSStep(ts,&its,&ftime); CHKERRA(ierr);
  printf( "number of pseudo time-steps = %d\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = TSDestroy(ts); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(Vec X,AppCtx *user)
{
  int     i, j, row, mx, my, ierr;
  double  one = 1.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  sc;
  Scalar  *x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  for (j=0; j<my; j++) {
    temp = (double)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;  
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction(TS ts,double t,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my;
  double  two = 2.0, one = 1.0, lambda;
  double  hx, hy, hxdhy, hydhx;
  Scalar  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  ierr = VecGetArray(F,&f); CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      ub = x[row - mx];
      ul = x[row - 1];
      ut = x[row + mx];
      ur = x[row + 1];
      uxx = (-ur + two*u - ul)*hydhx;
      uyy = (-ut + two*u - ub)*hxdhy;
      f[row] = -uxx + -uyy + sc*lambda*exp(u);
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f); CHKERRQ(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian(TS ts,double t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  Mat     jac = *J;
  int     i, j, row, mx, my, col[5], ierr;
  Scalar  two = 2.0, one = 1.0, lambda, v[5],sc, *x;
  double  hx, hy, hxdhy, hydhx;


  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = 1.0 / (double)(mx-1);
  hy    = 1.0 / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&row,1,&row,&one,INSERT_VALUES); CHKERRQ(ierr);
        continue;
      }
      v[0] = hxdhy; col[0] = row - mx;
      v[1] = hydhx; col[1] = row - 1;
      v[2] = -two*(hydhx + hxdhy) + sc*lambda*exp(x[row]); col[2] = row;
      v[3] = hydhx; col[3] = row + 1;
      v[4] = hxdhy; col[4] = row + mx;
      ierr = MatSetValues(jac,1,&row,5,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}




