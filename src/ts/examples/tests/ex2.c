#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.2 1996/01/31 03:49:36 bsmith Exp bsmith $";
#endif

static char help[] ="Solves the time dependent Bratu problem";

#include "vec.h"
#include "draw.h"
#include "snes.h"
#include "options.h"
#include <math.h>

typedef struct {
      double      param;        /* test problem parameter */
      int         mx;           /* Discretization in x-direction */
      int         my;           /* Discretization in y-direction */
      int         max_steps;    /* Number of time-steps */
      double	  dt;		/* Size of time step */
      double      dt_init;	/* Size of first time step */
      double      ssnorm;	/* Steady-state nonlinear norm (for SER) */
      double      ssnorm_init;  /* Initial steady-state nonlinear norm */
      Vec	  Xold;		/* needed for timestepping*/
} AppCtx;

typedef struct {
  SNES snes;
  Vec  w;
} MFCtx_Private;


int  Monitor(SNES,int,double,void *);

int  FormJacobianBE(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormFunctionBE(SNES ,Vec ,Vec ,void *),
     FormInitialGuessBE(SNES,Vec,void*),
     FormInitialGuess(SNES,Vec,void*);

int main( int argc, char **argv )
{
  SNES         snes; 
  SLES         sles;
  KSP	       itP;
  SNESType     type = SNES_EQ_NLS;  /* nonlinear solution type */
  Vec          x,r;
  Mat          MFJ,J;
  int          ierr, N, its,flg; 
  int	       itime, i; 	 /* counter for timesteps */
  AppCtx       user;
  double       bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  FILE	       *sol_file;
  double	*xx;
  long int	start, secs;

  PetscInitialize( &argc, &argv, 0,0,help );
  if ((OptionsHasName(0,"-help", &flg), flg)) fprintf(stderr,"%s",help);

  start = PetscGetTime();

  user.mx    = 4;
  user.my    = 4;
  user.param = 6.0;
  user.dt_init = 1.0e-6;
  user.max_steps = 10;
  
  OptionsGetInt(0,"-mx",&user.mx,&flg);
  OptionsGetInt(0,"-my",&user.my,&flg);
  OptionsGetDouble(0,"-param",&user.param,&flg);
  OptionsGetDouble(0,"-dt",&user.dt_init,&flg);
  OptionsGetInt(0,"-max_steps",&user.max_steps,&flg);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRQ(1,"Lambda is out of range");
  }
  N          = user.mx*user.my;
  
  /* Set up data structures */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&user.Xold); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,N,N,0,0,&J); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,type); CHKERRA(ierr);
  /*ierr = SNESSetMonitor(snes,Monitor,0); CHKERRA(ierr);*/
  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&itP); CHKERRQ(ierr);
  ierr = KSPSetPreconditionerSide(itP, PC_RIGHT);  CHKERRQ(ierr);

  /*ierr = KSPSetMonitor(itP,KSPDefaultMonitor ,0);*/

  /* Set various routines */
  ierr = SNESSetFunction(snes,r,FormFunctionBE,(void *)&user); CHKERRA(ierr);
  ierr = SNESDefaultMatrixFreeMatCreate(snes,x,&MFJ); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,MFJ,J,FormJacobianBE,(void *)&user);CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = FormInitialGuess(snes,x,&user);
  ierr = VecCopy( x, user.Xold);  CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes, x); CHKERRA(ierr);
  ierr = FormFunction(snes,x,r,&user);
  ierr = VecNorm(r,NORM_2,&user.ssnorm_init);
  user.ssnorm = user.ssnorm_init;
  user.dt = user.dt_init;
  /*for(itime=0; itime<user.max_steps; ++itime)*/
  for (itime=0; user.ssnorm>= 1.0e-10 ; ++itime) {
    user.dt = 1.1*user.dt*user.ssnorm_init/user.ssnorm;
    printf("ssnorm_init = %e , ssnorm = %e  dt = %e \n",user.ssnorm_init,user.ssnorm, user.dt);
    ierr = SNESSolve(snes, x, &its);  CHKERRA(ierr);
 printf( "number of Newton iterations = %d\n", its );
   ierr = FormFunction(snes,x,r,&user); CHKERRA(ierr);
    ierr = VecNorm(r,NORM_2,&user.ssnorm); CHKERRA(ierr);
  }
    printf("ssnorm_init = %e , ssnorm = %e  dt = %e \n",user.ssnorm_init,user.ssnorm, user.dt);

  /*printf( "number of Newton iterations = %d\n\n", its );*/

  sol_file = fopen("data","w");
  if (sol_file != 0) {
        ierr = VecGetArray(x,&xx); CHKERRA(ierr);
        for(i = 0; i < user.mx*user.my ; ++i)
           fprintf(sol_file,"%G \n",xx[i]);
  }
  fclose(sol_file);
        
  /*DrawTensorContour(win,user.mx,user.my,0,0,x);
  DrawSyncFlush(win);
  scanf("%c", &c);*/

  secs = PetscGetTime() - start;
  printf("That took %6.2f sec \n",(double) secs*1.0e-6);
  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(MFJ); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     i, j, row, mx, my, ierr;
  double  one = 1.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  sc;
  double  *x;

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
 
int FormFunctionBE(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i;
  double  *x,*f, *xold;
  ierr = FormFunction(snes,X,F,ptr);  CHKERRQ(ierr);
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  ierr = VecGetArray(F,&f); CHKERRQ(ierr);
  ierr = VecGetArray(user->Xold,&xold); CHKERRQ(ierr);
  for (i=0; i<user->mx*user->my; ++i)
  {
    f[i]+=(x[i]-xold[i])/user->dt;
  }

  ierr = VecRestoreArray(F,&f); CHKERRQ(ierr);
  return 0;
}


int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my;
  double  two = 2.0, one = 1.0, lambda;
  double  hx, hy, hxdhy, hydhx;
  double  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;

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
      f[row] = uxx + uyy - sc*lambda*exp(u);
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f); CHKERRQ(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobianBE(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  Mat     jac = *B;
  int     i, j, row, mx, my, col, ierr;
  double  two = 2.0, one = 1.0, lambda, v;
  double  hx, hy, hxdhy, hydhx;
  double  sc, *x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
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
      v = -hxdhy; col = row - mx;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
      v = -hydhx; col = row - 1;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
      v = one/user->dt + two*(hydhx + hxdhy) - sc*lambda*exp(x[row]);
      ierr = MatSetValues(jac,1,&row,1,&row,&v,INSERT_VALUES); CHKERRQ(ierr);
      v = -hydhx; col = row + 1;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
      v = -hxdhy; col = row + mx;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  /*ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);*/
  ierr = MatAssemblyEnd(jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  fprintf( stdout, "iter = %d, Function norm %g \n",its,fnorm);
  /*ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,STDOUT_VIEWER); CHKERRQ(ierr);*/
  return 0;
}

/*
int MFMultBE(void *ptr,Vec dx,Vec y)
{
  int ierr;
  AppCtx *user = (AppCtx *) ptr;
  double factor = -1.0/user->dt;

  ierr = MFMult(ptr,dx,y); CHKERRQ(ierr);
  ierr = VecAXPY(&factor, dx, y); CHKERRQ(ierr);
  return 0;
}
*/

