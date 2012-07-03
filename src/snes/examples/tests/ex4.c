
/* NOTE:  THIS PROGRAM HAS NOT YET BEEN SET UP IN TUTORIAL STYLE. */

static char help[] ="This program demonstrates use of the SNES package to solve systems of nonlinear equations on a single processor.\n\
  Both of these examples employ\n\
sparse storage of the Jacobian matrices and are taken from the MINPACK-2\n\
test suite.  By default the Bratu (SFI - solid fuel ignition) test problem\n\
is solved.  The command line options are:\n\
   -cavity : Solve FDC (flow in a driven cavity) problem\n\
   -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
      problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
      problem FDC:  <parameter> = Reynolds number (par > 0)\n\
   -mx <xg>, where <xg> = number of grid points in the x-direction\n\
   -my <yg>, where <yg> = number of grid points in the y-direction\n\n";

/*  
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
  
    2) Flow in a Driven Cavity (FDC) problem. The problem is
    formulated as a boundary value problem, which is discretized by
    standard finite difference approximations to obtain a system of
    nonlinear equations. 
*/

#include <petscsnes.h>

typedef struct {
      PetscReal   param;        /* test problem parameter */
      PetscInt    mx;           /* Discretization in x-direction */
      PetscInt    my;           /* Discretization in y-direction */
} AppCtx;

extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
                      FormFunction1(SNES,Vec,Vec,void*),
                      FormInitialGuess1(AppCtx*,Vec);
extern PetscErrorCode FormJacobian2(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
                      FormFunction2(SNES,Vec,Vec,void*),
                      FormInitialGuess2(AppCtx*,Vec);
 
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* SNES context */
  const SNESType method = SNESLS;      /* default nonlinear solution method */
  Vec            x, r;                 /* solution, residual vectors */
  Mat            J;                    /* Jacobian matrix */
  AppCtx         user;                 /* user-defined application context */
  PetscDraw      draw;                 /* drawing context */
  PetscInt       its, N, nfails;
  PetscErrorCode ierr;
  PetscBool      cavity; 
  PetscReal      bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  PetscScalar    *xvalues;

  PetscInitialize(&argc, &argv,(char *)0,help);
  /* ierr = PetscDrawOpenX(PETSC_COMM_WORLD,0,"Solution",300,0,300,300,&draw);CHKERRQ(ierr); */
  ierr = PetscDrawCreate(PETSC_COMM_WORLD,0,"Solution",300,0,300,300,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetType(draw,PETSC_DRAW_X);CHKERRQ(ierr);

  user.mx    = 4;
  user.my    = 4;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-cavity",&cavity);CHKERRQ(ierr);
  if (cavity) user.param = 100.0;
  else        user.param = 6.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
  if (!cavity && (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min)) SETERRQ(PETSC_COMM_SELF,1,"Lambda is out of range");
  N = user.mx*user.my;
  
  /* Set up data structures */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,PETSC_NULL,&J);CHKERRQ(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetType(snes,method);CHKERRQ(ierr);

  /* Set various routines and compute initial guess for nonlinear solver */
  if (cavity){
    ierr = FormInitialGuess2(&user,x);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction2,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian2,(void*)&user);CHKERRQ(ierr);
  } else {
    ierr = FormInitialGuess1(&user,x);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction1,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian1,(void*)&user);CHKERRQ(ierr);
  }

  /* Set options and solve nonlinear system */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetNonlinearStepFailures(snes,&nfails);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D, ",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"number of unsuccessful steps = %D\n\n",nfails);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xvalues);CHKERRQ(ierr);
  ierr = PetscDrawTensorContour(draw,user.mx,user.my,0,0,xvalues);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xvalues);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------ */
/*           Bratu (Solid Fuel Ignition) Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode  FormInitialGuess1(AppCtx *user,Vec X)
{
  PetscInt       i, j, row, mx, my;
  PetscErrorCode ierr;
  PetscReal      lambda, temp1, temp, hx, hy, hxdhy, hydhx,sc;
  PetscScalar    *x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = 1.0 / (PetscReal)(mx-1);
  hy    = 1.0 / (PetscReal)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + 1.0);
  for (j=0; j<my; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;  
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp)); 
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */
 
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
PetscErrorCode  FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i, j, row, mx, my;
  PetscErrorCode ierr;
  PetscReal      two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx;
  PetscScalar    ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = one / (PetscReal)(mx-1);
  hy    = one / (PetscReal)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
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
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

#undef __FUNCT__
#define __FUNCT__ "FormJacobian1"
PetscErrorCode  FormJacobian1(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  Mat            jac = *J;
  PetscInt       i, j, row, mx, my, col[5];
  PetscErrorCode ierr;
  PetscScalar    two = 2.0, one = 1.0, lambda, v[5],sc, *x;
  PetscReal      hx, hy, hxdhy, hydhx;


  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = 1.0 / (PetscReal)(mx-1);
  hy    = 1.0 / (PetscReal)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(jac,1,&row,1,&row,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      v[0] = -hxdhy; col[0] = row - mx;
      v[1] = -hydhx; col[1] = row - 1;
      v[2] = two*(hydhx + hxdhy) - sc*lambda*exp(x[row]); col[2] = row;
      v[3] = -hydhx; col[3] = row + 1;
      v[4] = -hxdhy; col[4] = row + mx;
      ierr = MatSetValues(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
/* ------------------------------------------------------------------ */
/*                       Driven Cavity Test Problem                   */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode  FormInitialGuess2(AppCtx *user,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i, j, row, mx, my;
  PetscScalar    xx,yy,*x;
  PetscReal      hx, hy;

  mx	 = user->mx; 
  my	 = user->my;

  /* Test for invalid input parameters */
  if ((mx <= 0) || (my <= 0)) SETERRQ(PETSC_COMM_SELF,1,0);

  hx    = 1.0 / (PetscReal)(mx-1);
  hy    = 1.0 / (PetscReal)(my-1);

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  yy = 0.0;
  for (j=0; j<my; j++) {
    xx = 0.0;
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0;
      } 
      else {
        x[row] = - xx*(1.0 - xx)*yy*(1.0 - yy);
      }
      xx = xx + hx;
    }
    yy = yy + hy;
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

#undef __FUNCT__
#define __FUNCT__ "FormFunction2"
PetscErrorCode  FormFunction2(SNES snes,Vec X,Vec F,void *pptr)
{
  AppCtx         *user = (AppCtx*)pptr;
  PetscInt       i, j, row, mx, my;
  PetscErrorCode ierr;
  PetscScalar    two = 2.0, zero = 0.0, pb, pbb,pbr, pl,pll,p,pr,prr;
  PetscScalar    ptl,pt,ptt,dpdy,dpdx,pblap,ptlap,rey,pbl,ptr,pllap,plap,prlap;
  PetscScalar    *x,*f, hx2, hy2, hxhy2;
  PetscReal      hx, hy;

  mx	 = user->mx; 
  my	 = user->my;
  hx     = 1.0 / (PetscReal)(mx-1);
  hy     = 1.0 / (PetscReal)(my-1);
  hx2    = hx*hx;
  hy2    = hy*hy;
  hxhy2  = hx2*hy2;
  rey    = user->param;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      }
      if (i == 1 || j == 1) {
           pbl = zero;
      } 
      else {
           pbl = x[row-mx-1];
      }
      if (j == 1) {
           pb = zero;
           pbb = x[row];
      } 
      else if (j == 2) {
           pb = x[row-mx];
           pbb = zero;
      } 
      else {
           pb = x[row-mx];
           pbb = x[row-2*mx];
      }
      if (j == 1 || i == mx-2) {
           pbr = zero;
      }
      else {
           pbr = x[row-mx+1];
      }
      if (i == 1) {
           pl = zero;
           pll = x[row];
      } 
      else if (i == 2) {
           pl = x[row-1];
           pll = zero;
      } 
      else {
           pl = x[row-1];
           pll = x[row-2];
      }
      p = x[row];
      if (i == mx-3) {
           pr = x[row+1];
           prr = zero;
      } 
      else if (i == mx-2) {
           pr = zero;
           prr = x[row];
      } 
      else {
           pr = x[row+1];
           prr = x[row+2];
      }
      if (j == my-2 || i == 1) {
           ptl = zero;
      } 
      else {
           ptl = x[row+mx-1];
      }
      if (j == my-3) {
           pt = x[row+mx];
           ptt = zero;
      } 
      else if (j == my-2) {
           pt = zero;
           ptt = x[row] + two*hy;
      } 
      else {
           pt = x[row+mx];
           ptt = x[row+2*mx];
      }
      if (j == my-2 || i == mx-2) {
           ptr = zero;
      } 
      else {
           ptr = x[row+mx+1];
      }

      dpdy = (pt - pb)/(two*hy);
      dpdx = (pr - pl)/(two*hx);

      /*  Laplacians at each point in the 5 point stencil */
      pblap = (pbr - two*pb + pbl)/hx2 + (p   - two*pb + pbb)/hy2;
      pllap = (p   - two*pl + pll)/hx2 + (ptl - two*pl + pbl)/hy2;
      plap =  (pr  - two*p  + pl)/hx2 + (pt  - two*p  + pb)/hy2;
      prlap = (prr - two*pr + p)/hx2 + (ptr - two*pr + pbr)/hy2;
      ptlap = (ptr - two*pt + ptl)/hx2 + (ptt - two*pt + p)/hy2;

      f[row] = hxhy2*((prlap - two*plap + pllap)/hx2
                        + (ptlap - two*plap + pblap)/hy2
                        - rey*(dpdy*(prlap - pllap)/(two*hx)
                        - dpdx*(ptlap - pblap)/(two*hy)));
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

#undef __FUNCT__
#define __FUNCT__ "FormJacobian2"
PetscErrorCode  FormJacobian2(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *pptr)
{
  AppCtx         *user = (AppCtx*)pptr;
  PetscInt       i, j, row, mx, my, col;
  PetscErrorCode ierr;
  PetscScalar    two = 2.0, one = 1.0, zero = 0.0, pb, pbb,pbr, pl,pll,p,pr,prr;
  PetscScalar    ptl,pt,ptt,dpdy,dpdx,pblap,ptlap,rey,pbl,ptr,pllap,plap,prlap;
  PetscScalar    val,four = 4.0, three = 3.0,*x;
  PetscReal      hx, hy,hx2, hy2, hxhy2;
  PetscBool      assembled;

  mx	 = user->mx; 
  my	 = user->my;
  hx     = 1.0 / (PetscReal)(mx-1);
  hy     = 1.0 / (PetscReal)(my-1);
  hx2    = hx*hx;
  hy2    = hy*hy;
  hxhy2  = hx2*hy2;
  rey    = user->param;

  ierr = MatAssembled(*J,&assembled);CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(*J);CHKERRQ(ierr);
  }
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(*J,1,&row,1,&row,&one,ADD_VALUES);CHKERRQ(ierr);
        continue;
      }
      if (i == 1 || j == 1) {
           pbl = zero;
      } 
      else {
           pbl = x[row-mx-1];
      }
      if (j == 1) {
           pb = zero;
           pbb = x[row];
      } 
      else if (j == 2) {
           pb = x[row-mx];
           pbb = zero;
      } 
      else {
           pb = x[row-mx];
           pbb = x[row-2*mx];
      }
      if (j == 1 || i == mx-2) {
           pbr = zero;
      }
      else {
           pbr = x[row-mx+1];
      }
      if (i == 1) {
           pl = zero;
           pll = x[row];
      } 
      else if (i == 2) {
           pl = x[row-1];
           pll = zero;
      } 
      else {
           pl = x[row-1];
           pll = x[row-2];
      }
      p = x[row];
      if (i == mx-3) {
           pr = x[row+1];
           prr = zero;
      } 
      else if (i == mx-2) {
           pr = zero;
           prr = x[row];
      } 
      else {
           pr = x[row+1];
           prr = x[row+2];
      }
      if (j == my-2 || i == 1) {
           ptl = zero;
      } 
      else {
           ptl = x[row+mx-1];
      }
      if (j == my-3) {
           pt = x[row+mx];
           ptt = zero;
      } 
      else if (j == my-2) {
           pt = zero;
           ptt = x[row] + two*hy;
      } 
      else {
           pt = x[row+mx];
           ptt = x[row+2*mx];
      }
      if (j == my-2 || i == mx-2) {
           ptr = zero;
      } 
      else {
           ptr = x[row+mx+1];
      }

      dpdy = (pt - pb)/(two*hy);
      dpdx = (pr - pl)/(two*hx);

      /*  Laplacians at each point in the 5 point stencil */
      pblap = (pbr - two*pb + pbl)/hx2 + (p   - two*pb + pbb)/hy2;
      pllap = (p   - two*pl + pll)/hx2 + (ptl - two*pl + pbl)/hy2;
      plap =  (pr  - two*p  + pl)/hx2 + (pt  - two*p  + pb)/hy2;
      prlap = (prr - two*pr + p)/hx2 + (ptr - two*pr + pbr)/hy2;
      ptlap = (ptr - two*pt + ptl)/hx2 + (ptt - two*pt + p)/hy2;

      if (j > 2) {
        val = hxhy2*(one/hy2/hy2 - rey*dpdx/hy2/(two*hy));
        col = row - 2*mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i > 2) {
        val = hxhy2*(one/hx2/hx2 + rey*dpdy/hx2/(two*hx));
        col = row - 2;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i < mx-3) {
        val = hxhy2*(one/hx2/hx2 - rey*dpdy/hx2/(two*hx));
        col = row + 2;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j < my-3) {
        val = hxhy2*(one/hy2/hy2 + rey*dpdx/hy2/(two*hy));
        col = row + 2*mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != 1 && j != 1) {
        val = hxhy2*(two/hy2/hx2 + rey*(dpdy/hy2/(two*hx) - dpdx/hx2/(two*hy)));
        col = row - mx - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j != 1 && i != mx-2) {
        val = hxhy2*(two/hy2/hx2 - rey*(dpdy/hy2/(two*hx) + dpdx/hx2/(two*hy)));
        col = row - mx + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j != my-2 && i != 1) {
        val = hxhy2*(two/hy2/hx2 + rey*(dpdy/hy2/(two*hx) + dpdx/hx2/(two*hy)));
        col = row + mx - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j != my-2 && i != mx-2) {
        val = hxhy2*(two/hy2/hx2 - rey*(dpdy/hy2/(two*hx) - dpdx/hx2/(two*hy)));
        col = row + mx + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j != 1) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hy2/hy2) 
                     + rey*((prlap - pllap)/(two*hx)/(two*hy) 
                     + dpdx*(one/hx2 + one/hy2)/hy));
        col = row - mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != 1) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hx2/hx2) 
                     - rey*((ptlap - pblap)/(two*hx)/(two*hy) 
                     + dpdy*(one/hx2 + one/hy2)/hx));
        col = row - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != mx-2) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hx2/hx2) 
                     + rey*((ptlap - pblap)/(two*hx)/(two*hy) 
                     + dpdy*(one/hx2 + one/hy2)/hx));
        col = row + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j != my-2) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hy2/hy2) 
                     - rey*((prlap - pllap)/(two*hx)/(two*hy) 
                     + dpdx*(one/hx2 + one/hy2)/hy));
        col = row + mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      val = hxhy2*(two*(four/hx2/hy2 + three/hx2/hx2 + three/hy2/hy2));
      ierr = MatSetValues(*J,1,&row,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
      if (j == 1) {
        val = hxhy2*(one/hy2/hy2 - rey*(dpdx/hy2/(two*hy)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i == 1) {
        val = hxhy2*(one/hx2/hx2 + rey*(dpdy/hx2/(two*hx)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i == mx-2) {
        val = hxhy2*(one/hx2/hx2 - rey*(dpdy/hx2/(two*hx)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (j == my-2) {
        val = hxhy2*(one/hy2/hy2 + rey*(dpdx/hy2/(two*hy)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}

