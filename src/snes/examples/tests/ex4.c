
static char help[] =
"This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations on a single processor.  Both of these examples employ\n\
sparse storage of the Jacobian matrices and are taken from the MINPACK-2\n\
test suite.  By default the Bratu (SFI - solid fuel ignition) test problem\n\
is solved.  The command line options are:\n\
   -cavity : Solve FDC (flow in a driven cavity) problem\n\
   -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
      problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
      problem FDC:  <parameter> = Reynolds number ( par > 0 )\n\
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
  
    2) Flow in a Driven Cavity (FDC) problem. The problem is
    formulated as a boundary value problem, which is discretized by
    standard finite difference approximations to obtain a system of
    nonlinear equations. 
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

int  FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction1(SNES,Vec,Vec,void*),
     FormInitialGuess1(SNES,Vec,void*);
int  FormJacobian2(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction2(SNES,Vec,Vec,void*),
     FormInitialGuess2(SNES,Vec,void*);

int main( int argc, char **argv )
{
  SNES         snes;
  SNESMethod   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r;
  Mat          J;
  int          ierr, its, N; 
  AppCtx       user;
  DrawCtx      win;
  double       bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"Solution",300,0,300,300,&win);
  CHKERRA(ierr);

  user.mx    = 4;
  user.my    = 4;
  user.param = 6.0;
  OptionsGetInt(0,0,"-mx",&user.mx);
  OptionsGetInt(0,0,"-my",&user.my);
  OptionsGetDouble(0,0,"-param",&user.param);
  if (!OptionsHasName(0,0,"-cavity") && 
      (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min)) {
    SETERR(1,"Lambda is out of range");
  }
  N          = user.mx*user.my;
  
  /* Set up data structures */
  ierr = VecCreateSequential(MPI_COMM_SELF,N,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,N,N,0,0,&J); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  if (OptionsHasName(0,0,"-cavity")){
    ierr = SNESSetSolution(snes,x,FormInitialGuess2,(void *)&user); 
           CHKERRA(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction2,(void *)&user,0); 
           CHKERRA(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian2,(void *)&user); 
           CHKERRA(ierr);
  }
  else {
    ierr = SNESSetSolution(snes,x,FormInitialGuess1,(void *)&user); 
           CHKERRA(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction1,(void *)&user,0); 
           CHKERRA(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian1,(void *)&user);
           CHKERRA(ierr);
  }

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its);  CHKERRA(ierr);

  printf( "number of Newton iterations = %d\n\n", its );
  DrawTensorContour(win,user.mx,user.my,0,0,x);
  DrawSyncFlush(win);

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
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

  ierr = VecGetArray(X,&x); CHKERR(ierr);
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
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */
 
int FormFunction1(SNES snes,Vec X,Vec F,void *ptr)
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

  ierr = VecGetArray(X,&x); CHKERR(ierr);
  ierr = VecGetArray(F,&f); CHKERR(ierr);
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
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  ierr = VecRestoreArray(F,&f); CHKERR(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian1(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  Mat     jac = *J;
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

  ierr = VecGetArray(X,&x); CHKERR(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&row,1,&row,&one,INSERTVALUES); CHKERR(ierr);
        continue;
      }
      v = -hxdhy; col = row - mx;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERTVALUES); CHKERR(ierr);
      v = -hydhx; col = row - 1;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERTVALUES); CHKERR(ierr);
      v = two*(hydhx + hxdhy) - sc*lambda*exp(x[row]);
      ierr = MatSetValues(jac,1,&row,1,&row,&v,INSERTVALUES); CHKERR(ierr);
      v = -hydhx; col = row + 1;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERTVALUES); CHKERR(ierr);
      v = -hxdhy; col = row + mx;
      ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERTVALUES); CHKERR(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  ierr = MatAssemblyEnd(jac,FINAL_ASSEMBLY); CHKERR(ierr);
  return 0;
}
/* ------------------------------------------------------------------ */
/*                       Driven Cavity Test Problem                   */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess2(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my;
  double  one = 1.0, xx,yy,*x;
  double  hx, hy;

  mx	 = user->mx; 
  my	 = user->my;

  /* Test for invalid input parameters */
  if ((mx <= 0) || (my <= 0)) SETERR(1,0);

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);

  ierr = VecGetArray(X,&x); CHKERR(ierr);
  yy = 0.0;
  for (j=0; j<my; j++) {
    xx = 0.0;
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0;
      } 
      else {
        x[row] = - xx*(1.0 - xx)*yy*(1.0 - yy);
      }
      xx = xx + hx;
    }
    yy = yy + hy;
  }
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction2(SNES snes,Vec X,Vec F,void *pptr)
{
  AppCtx *user = (AppCtx *) pptr;
  int     i, j, row, mx, my, ierr;
  double  two = 2.0, one = 1.0, zero = 0.0, pb, pbb,pbr, pl,pll,p,pr,prr;
  double  ptl,pt,ptt,dpdy,dpdx,pblap,ptlap,rey,pbl,ptr,pllap,plap,prlap;
  double  hx, hy;
  double  *x,*f, hx2, hy2, hxhy2;

  mx	 = user->mx; 
  my	 = user->my;
  hx     = one / (double)(mx-1);
  hy     = one / (double)(my-1);
  hx2    = hx*hx;
  hy2    = hy*hy;
  hxhy2  = hx2*hy2;
  rey    = user->param;

  ierr = VecGetArray(X,&x); CHKERR(ierr);
  ierr = VecGetArray(F,&f); CHKERR(ierr);
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
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
      plap =  (pr  - two*p  + pl )/hx2 + (pt  - two*p  + pb )/hy2;
      prlap = (prr - two*pr + p  )/hx2 + (ptr - two*pr + pbr)/hy2;
      ptlap = (ptr - two*pt + ptl)/hx2 + (ptt - two*pt + p  )/hy2;

      f[row] = hxhy2*( (prlap - two*plap + pllap)/hx2
                        + (ptlap - two*plap + pblap)/hy2
                        - rey*(dpdy*(prlap - pllap)/(two*hx)
                        - dpdx*(ptlap - pblap)/(two*hy)));
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  ierr = VecRestoreArray(F,&f); CHKERR(ierr);
  return 0; 
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian2(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *pptr)
{
  AppCtx *user = (AppCtx *) pptr;
  int     i, j, row, mx, my, col, ierr;
  double  two = 2.0, one = 1.0, zero = 0.0, pb, pbb,pbr, pl,pll,p,pr,prr;
  double  ptl,pt,ptt,dpdy,dpdx,pblap,ptlap,rey,pbl,ptr,pllap,plap,prlap;
  double  hx, hy,val,four = 4.0, three = 3.0;
  double  *x, hx2, hy2, hxhy2;

  mx	 = user->mx; 
  my	 = user->my;
  hx     = one / (double)(mx-1);
  hy     = one / (double)(my-1);
  hx2    = hx*hx;
  hy2    = hy*hy;
  hxhy2  = hx2*hy2;
  rey    = user->param;

  MatZeroEntries(*J);
  VecGetArray(X,&x); 
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(*J,1,&row,1,&row,&one,ADDVALUES); CHKERR(ierr);
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
      plap =  (pr  - two*p  + pl )/hx2 + (pt  - two*p  + pb )/hy2;
      prlap = (prr - two*pr + p  )/hx2 + (ptr - two*pr + pbr)/hy2;
      ptlap = (ptr - two*pt + ptl)/hx2 + (ptt - two*pt + p  )/hy2;

      if (j > 2) {
        val = hxhy2*(one/hy2/hy2 - rey*dpdx/hy2/(two*hy));
        col = row - 2*mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i > 2) {
        val = hxhy2*(one/hx2/hx2 + rey*dpdy/hx2/(two*hx));
        col = row - 2;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i < mx-3) {
        val = hxhy2*(one/hx2/hx2 - rey*dpdy/hx2/(two*hx));
        col = row + 2;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j < my-3) {
        val = hxhy2*(one/hy2/hy2 + rey*dpdx/hy2/(two*hy));
        col = row + 2*mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i != 1 && j != 1) {
        val = hxhy2*(two/hy2/hx2 + rey*(dpdy/hy2/(two*hx) - dpdx/hx2/(two*hy)));
        col = row - mx - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j != 1 && i != mx-2) {
        val = hxhy2*(two/hy2/hx2 - rey*(dpdy/hy2/(two*hx) + dpdx/hx2/(two*hy)));
        col = row - mx + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j != my-2 && i != 1) {
        val = hxhy2*(two/hy2/hx2 + rey*(dpdy/hy2/(two*hx) + dpdx/hx2/(two*hy)));
        col = row + mx - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j != my-2 && i != mx-2) {
        val = hxhy2*(two/hy2/hx2 - rey*(dpdy/hy2/(two*hx) - dpdx/hx2/(two*hy)));
        col = row + mx + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j != 1) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hy2/hy2) 
                     + rey*((prlap - pllap)/(two*hx)/(two*hy) 
                     + dpdx*(one/hx2 + one/hy2)/hy));
        col = row - mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i != 1) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hx2/hx2) 
                     - rey*((ptlap - pblap)/(two*hx)/(two*hy) 
                     + dpdy*(one/hx2 + one/hy2)/hx));
        col = row - 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i != mx-2) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hx2/hx2) 
                     + rey*((ptlap - pblap)/(two*hx)/(two*hy) 
                     + dpdy*(one/hx2 + one/hy2)/hx));
        col = row + 1;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j != my-2) {
        val = hxhy2*(-four*(one/hy2/hx2 + one/hy2/hy2) 
                     - rey*((prlap - pllap)/(two*hx)/(two*hy) 
                     + dpdx*(one/hx2 + one/hy2)/hy));
        col = row + mx;
        ierr = MatSetValues(*J,1,&row,1,&col,&val,ADDVALUES); CHKERR(ierr);
      }
      val = hxhy2*(two*(four/hx2/hy2 + three/hx2/hx2 + three/hy2/hy2));
      ierr = MatSetValues(*J,1,&row,1,&row,&val,ADDVALUES); CHKERR(ierr);
      if (j == 1) {
        val = hxhy2*(one/hy2/hy2 - rey*(dpdx/hy2/(two*hy)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i == 1) {
        val = hxhy2*(one/hx2/hx2 + rey*(dpdy/hx2/(two*hx)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADDVALUES); CHKERR(ierr);
      }
      if (i == mx-2) {
        val = hxhy2*(one/hx2/hx2 - rey*(dpdy/hx2/(two*hx)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADDVALUES); CHKERR(ierr);
      }
      if (j == my-2) {
        val = hxhy2*(one/hy2/hy2 + rey*(dpdx/hy2/(two*hy)));
        ierr = MatSetValues(*J,1,&row,1,&row,&val,ADDVALUES); CHKERR(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(*J,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = VecRestoreArray(X,&x); CHKERR(ierr);
  ierr = MatAssemblyEnd(*J,FINAL_ASSEMBLY); CHKERR(ierr);
  return 0;
}

