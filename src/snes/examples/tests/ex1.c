#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.1 1995/04/17 20:11:39 bsmith Exp bsmith $";
#endif

/*  
    This program computes on a single processor the solution for the 
    Solid Fuel Ignition (SFI) problem from the MINPACK-2 suite.  This 
    problem is modeled by the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
  
    The nonzero terms of the Jacobian are stored in a sparse format. 
 */
#include "vec.h"
#include "draw.h"
#include "snes.h"
#include "options.h"
#include <math.h>
#define MIN(a,b) ( ((a)<(b)) ? a : b )

typedef struct {
      double      param;        /* test problem parameter */
      int         mx;           /* Discretization in x-direction */
      int         my;           /* Discretization in y-direction */
} AppCtx;
int  FormJacobian(SNES,Vec,Mat*,Mat*,int*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec,void*);

int main( int argc, char **argv )
{
  SNES         snes;
  SNESMethod   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r;
  Mat          J;
  int          ierr, its, N; 
  AppCtx       user;
  DrawCtx      win;

  PetscInitialize( &argc, &argv, 0,0 );
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"Crab",300,0,300,300,&win);CHKERRA(ierr);

  user.mx    = 4;
  user.my    = 4;
  user.param = 6.0;
  OptionsGetInt(0,0,"-mx",&user.mx);
  OptionsGetInt(0,0,"-my",&user.my);
  OptionsGetDouble(0,0,"-param",&user.param);
  N          = user.mx*user.my;
  

  ierr = VecCreateSequential(MPI_COMM_SELF,N,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,N,N,0,0,&J); CHKERRA(ierr);

  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  SNESSetSolution( snes, x,FormInitialGuess,(void *)&user );
  SNESSetFunction( snes, r,FormFunction,  (void *)&user,0);
  SNESSetJacobian( snes, J, J, FormJacobian, (void *)&user);

  ierr = SNESSetFromOptions(snes); CHKERR(ierr);
  SNESSetUp( snes );                                   

  /* Execute solution method */
  ierr = SNESSolve( snes,&its );  CHKERR(ierr);
                                     
  printf( "number of Newton iterations = %d\n\n", its );

  DrawTensorContour(win,user.mx,user.my,0,0,x);
  DrawSyncFlush(win);

  VecDestroy(x);
  VecDestroy(r);
  MatDestroy(J);
  SNESDestroy( snes );                                 
  PetscFinalize();

  return 0;
}

int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i, j, row, mx, my;
  double  one = 1.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  sc;
  double  bratu_lambda_max = 6.81, bratu_lambda_min = 0.,*x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  /* Test for invalid input parameters */
  if ((mx <= 0) || (my <= 0)) SETERR(1,0);
  if ((lambda > bratu_lambda_max)||(lambda < bratu_lambda_min)) SETERR(2,0); 

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  VecGetArray(X,&x);
  temp1 = lambda/(lambda + one);
  for (j=0; j<my; j++) {
    temp = (double)(MIN(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;  
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( MIN( (double)(MIN(i,mx-i-1))*hx,temp) ); 
    }
  }
  VecRestoreArray(X,&x);
  return 0;
}
 
     /* Evaluate Function */
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i, j, row, mx, my;
  double  two = 2.0, one = 1.0, lambda;
  double  hx, hy, hxdhy, hydhx;
  double  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;
  double  bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  /* Test for invalid input parameters */
  if ((mx <= 0) || (my <= 0)) SETERR(1,0);
  if ((lambda > bratu_lambda_max)||(lambda < bratu_lambda_min)) SETERR(2,0); 

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;


  VecGetArray(X,&x);  VecGetArray(F,&f);
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
      f[row] = uxx + uyy - sc*lambda*exp(x[row]);
    }
  }
  VecRestoreArray(X,&x);  VecRestoreArray(F,&f);
  return 0; 
}

     /* Evaluate Jacobian matrix */
int FormJacobian(SNES snes,Vec X,Mat *J, Mat *B, int *flag, void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  Mat     jac = *J;
  int     i, j, row, mx, my,col;
  double  two = 2.0, one = 1.0, lambda, v;
  double  hx, hy, hxdhy, hydhx;
  double  sc, *x;
  double  bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  /* Test for invalid input parameters */
  if ((mx <= 0) || (my <= 0)) SETERR(1,0);
  if ((lambda > bratu_lambda_max)||(lambda < bratu_lambda_min)) SETERR(2,0); 

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  VecGetArray(X,&x);  
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        MatSetValues( jac, 1, &row,1,&row, &one, INSERTVALUES);
        continue;
      }
      v = -hxdhy; col = row - mx;
      MatSetValues( jac, 1, &row, 1, &col,&v,INSERTVALUES);
      v = -hydhx; col = row - 1;
      MatSetValues( jac, 1, &row, 1, &col,&v,INSERTVALUES);
      v = two*(hydhx + hxdhy) - sc*lambda*exp(x[row]);
      MatSetValues( jac, 1, &row, 1, &row,&v,INSERTVALUES);
      v = -hydhx; col = row + 1;
      MatSetValues( jac, 1, &row, 1, &col,&v,INSERTVALUES);
      v = -hxdhy; col = row + mx;
      MatSetValues( jac, 1, &row, 1, &col,&v,INSERTVALUES);
    }
  }
  MatAssemblyBegin(jac,FINAL_ASSEMBLY);
  VecRestoreArray(X,&x);  
  MatAssemblyEnd(jac,FINAL_ASSEMBLY);
  return 0;
}
