#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.17 1999/10/01 21:22:34 bsmith Exp bsmith $";
#endif

static char help[] =
"This program is a replica of ex6.c except that it does 2 solves to avoid paging\n\
This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel, using 2-dimensional distributed arrays.\n\
The 2-dim Bratu (SFI - solid fuel ignition) test problem is used, where\n\
analytic formation of the Jacobian is the default.  The command line\n\
options are:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

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

#include "snes.h"
#include "da.h"

/* User-defined application context */
typedef struct {
   double      param;         /* test problem parameter */
   int         mx,my;         /* discretization in x, y directions */
   Vec         localX,localF; /* ghosted local vector */
   DA          da;            /* distributed array data structure */
} AppCtx;

extern int FormFunction1(SNES,Vec,Vec,void*), FormInitialGuess1(AppCtx*,Vec);
extern int FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES          snes;                      /* nonlinear solver */
  SNESType      type = SNESEQLS;           /* nonlinear solution method */
  Vec           x, r;                      /* solution, residual vectors */
  Mat           J;                         /* Jacobian matrix */
  AppCtx        user;                      /* user-defined work context */
  int           i,ierr, its, N, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE;
  int           matrix_free, size, flg; 
  double        bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  PetscInitialize( &argc, &argv,(char *)0,help );

  for ( i=0; i<2; i++ ) {
    PLogStagePush(i);
    user.mx = 4; user.my = 4; user.param = 6.0;
    
    if (i!=0) {
      ierr = OptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg);CHKERRA(ierr);
      ierr = OptionsGetInt(PETSC_NULL,"-my",&user.my,&flg);CHKERRA(ierr);
      ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg);CHKERRA(ierr);
      if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
        SETERRA(1,0,"Lambda is out of range");
      }
    }
    N = user.mx*user.my;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
    ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);CHKERRA(ierr);
    ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);CHKERRA(ierr);
    if (Nx*Ny != size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
      SETERRQ(1,0,"Incompatible number of processors:  Nx * Ny != size");
    
    /* Set up distributed array */
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,user.my,Nx,Ny,1,1,
                      PETSC_NULL,PETSC_NULL,&user.da);CHKERRA(ierr);
    ierr = DACreateGlobalVector(user.da,&x);CHKERRA(ierr);
    ierr = VecDuplicate(x,&r);CHKERRA(ierr);
    ierr = DACreateLocalVector(user.da,&user.localX);CHKERRA(ierr);
    ierr = VecDuplicate(user.localX,&user.localF);CHKERRA(ierr);

    /* Create nonlinear solver and set function evaluation routine */
    ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
    ierr = SNESSetType(snes,type);CHKERRA(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction1,&user);CHKERRA(ierr);

    /* Set default Jacobian evaluation routine.  User can override with:
       -snes_mf : matrix-free Newton-Krylov method with no preconditioning
       (unless user explicitly sets preconditioner) 
       -snes_fd : default finite differencing approximation of Jacobian
       */
    ierr = OptionsHasName(PETSC_NULL,"-snes_mf",&matrix_free);CHKERRA(ierr);
    if (!matrix_free) {
      ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&J);CHKERRA(ierr);
      ierr = SNESSetJacobian(snes,J,J,FormJacobian1,&user);CHKERRA(ierr);
    }

    /* Set options, then solve nonlinear system */
    ierr = SNESSetFromOptions(snes);CHKERRA(ierr);
    ierr = FormInitialGuess1(&user,x);CHKERRA(ierr);
    ierr = SNESSolve(snes,x,&its);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n",its);CHKERRA(ierr);

  /* Free data structures */
    if (!matrix_free) {
      ierr = MatDestroy(J);CHKERRA(ierr);
    }
    ierr = VecDestroy(x);CHKERRA(ierr);
    ierr = VecDestroy(r);CHKERRA(ierr);
    ierr = VecDestroy(user.localX);CHKERRA(ierr);
    ierr = VecDestroy(user.localF);CHKERRA(ierr);
    ierr = SNESDestroy(snes);CHKERRA(ierr);
    ierr = DADestroy(user.da);CHKERRA(ierr);
  }
  PetscFinalize();

  return 0;
}/* --------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess1(AppCtx *user,Vec X)
{
  int     i, j, row, mx, my, ierr, xs, ys, xm, ym, Xm, Ym, Xs, Ys;
  double  one = 1.0, lambda, temp1, temp, hx, hy;
  Scalar  *x;
  Vec     localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);

  /* Get ghost points */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + one);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    temp = (double)(PetscMin(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Xm; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
} /* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction1"
int FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym;
  double  two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx,sc;
  Scalar  u, uxx, uyy, *x,*f;
  Vec     localX = user->localX, localF = user->localF; 

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);

  /* Evaluate function */
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
      f[row] = uxx + uyy - sc*PetscExpScalar(u);
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PLogFlops(11*ym*xm);
  return 0; 
} /* --------------------  Evaluate Jacobian F'(x) --------------------- */
#undef __FUNC__
#define __FUNC__ "FormJacobian1"
int FormJacobian1(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;
  Mat     jac = *J;
  int     ierr, i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym, col[5];
  int     nloc, *ltog, grow;
  Scalar  two = 2.0, one = 1.0, lambda, v[5], hx, hy, hxdhy, hydhx, sc, *x;
  Vec     localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&grow,1,&grow,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      v[0] = -hxdhy; col[0] = ltog[row - Xm];
      v[1] = -hydhx; col[1] = ltog[row - 1];
      v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = grow;
      v[3] = -hydhx; col[3] = ltog[row + 1];
      v[4] = -hxdhy; col[4] = ltog[row + Xm];
      ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
