#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.1 1995/08/29 19:53:36 curfman Exp curfman $";
#endif

static char help[] = "\n\
ex2:\n\
This program demonstrates use of the SNES package to solve several\n\
unconstrained minimization problems on a single processor.  All of these\n\
examples are taken from the MINPACK-2 test suite, and employ sparse MATAIJ\n\
storage of the Hessian matrices by default.  The command line options are:\n\
  -mx xg, where xg = number of grid points in the 1st coordinate direction\n\
  -my yg, where yg = number of grid points in the 2nd coordinate direction\n\
     3: Minimal Surface Area (dmsa)\n\
        Note:  param is not an option for this problem.\n";

#include "petsc.h"
#include "snes.h"
#include <string.h>
#include "viewer.h"
#include <math.h>

/* User-defined application context */
   typedef struct {
      int     mx;       /* discretization in x-direction */
      int     my;       /* discretization in y-direction */
      int     ndim;     /* problem dimension */
      int     number;   /* test problem number */
      Scalar  *work;    /* work space */
      Vec     s,y;      /* work space for computing Hessian */
      double  hx, hy;
   } AppCtx;

typedef enum {FunctionEval=1, GradientEval=2} FctGradFlag;

int FormHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormMinimizationFunction(SNES,Vec,Scalar*,void*);
int FormGradient(SNES,Vec,Vec,void*);
int FormInitialGuess(SNES,Vec,void*);
int FunctionGradient(SNES,Vec,Scalar*,Vec,FctGradFlag,AppCtx*);
int BoundaryValues(AppCtx*);
int HessProduct(AppCtx*,Scalar*,Scalar*,Vec);

int main(int argc,char **argv)
{
  SNES       snes;                  /* SNES context */
  SNESMethod method = SNES_UM_NTR;  /* nonlinear solution method */
  Vec        x, g;                  /* solution, gradient vectors */
  Mat        H;                     /* Hessian matrix */
  AppCtx     user;                  /* application context */
  int        ierr, its, nfails;
  int        mx=10;   /* discretization of problem in x-direction */
  int        my=10;   /* discretization of problem in y-direction */
  double     one = 1.0;

  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);

  OptionsGetInt(0,"-my",&my);
  OptionsGetInt(0,"-mx",&mx);
  user.ndim = mx * my;
  user.mx = mx;
  user.my = my;
  user.hx = one/(double)(mx+1);
  user.hy = one/(double)(my+1);

  /* Set up data structures and allocate work space */
  ierr = MatCreate(MPI_COMM_SELF,user.ndim,user.ndim,&H); CHKERRA(ierr);
  ierr = VecCreate(MPI_COMM_SELF,user.ndim,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&g); CHKERRA(ierr);
  ierr = VecDuplicate(x,&user.y); CHKERRA(ierr);
  ierr = VecDuplicate(x,&user.s); CHKERRA(ierr);
  user.work = (Scalar*)PETSCMALLOC(2*(mx+my+4)*sizeof(Scalar)); CHKPTRQ(user.work);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_SELF,SNES_UNCONSTRAINED_MINIMIZATION,&snes);
         CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess,(void *)&user); CHKERRA(ierr);
  ierr = SNESSetMinimizationFunction(snes,FormMinimizationFunction,
         (void *)&user); CHKERRA(ierr);
  ierr = SNESSetGradient(snes,g,FormGradient,(void *)&user); CHKERRA(ierr);
  ierr = SNESSetHessian(snes,H,H,FormHessian,(void *)&user); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its);  CHKERRA(ierr);
  ierr = SNESGetNumberUnsuccessfulSteps(snes,&nfails); CHKERRA(ierr);
  ierr = SNESView(snes,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
  printf("number of Newton iterations = %d, ",its);
  printf("number of unsuccessful steps = %d\n\n",nfails);

  /* Free data structures */
  PETSCFREE(user.work); 
  ierr = VecDestroy(user.s); CHKERRA(ierr);
  ierr = VecDestroy(user.y); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(g); CHKERRA(ierr);
  ierr = MatDestroy(H); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* -------------------------------------------------------------------- */
/*
    Evaluate function f(x) on a single processor
 */

int FormMinimizationFunction(SNES snes,Vec xvec,Scalar *f,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  return FunctionGradient(snes,xvec,f,NULL,FunctionEval,user); 
}
/* -------------------------------------------------------------------- */
/*
    Evaluate gradient g(x) on a single processor
 */

int FormGradient(SNES snes,Vec xvec,Vec gvec,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  return FunctionGradient(snes,xvec,NULL,gvec,GradientEval,user); 
}
/* -------------------------------------------------------------------- */
/*
    Evaluate function f(x) on a single processor
 */

int FunctionGradient(SNES snes,Vec xvec,Scalar *f,Vec gvec,FctGradFlag fg,
                     AppCtx *user)
{
  int    ierr, nx = user->mx, ny = user->my, nx1 = nx+1, ny1 = ny+1;
  int    ind, i, j, k;
  double zero = 0.0, one = 1.0, p5 = 0.5, area, dvdx, dvdy, fl, fu;
  double v, vb, vl, vr, vt, hx = user->hx, hy = user->hy;
  Scalar *bottom, *top, *left, *right, *x, val;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = VecGetArray(xvec,&x); CHKERRQ(ierr);
  if (fg & FunctionEval) {
    *f = zero;
  }
  if (fg & GradientEval) {
    ierr = VecSet(&zero,gvec); CHKERRQ(ierr);
  }
  ierr =  BoundaryValues(user); CHKERRQ(ierr);

  /* Compute function and gradient over the lower triangular elements */
  for (j=0; j<ny1; j++) {
    for (i=0; i<nx1; i++) {
      k = nx*(j-1) + i-1;
      if (i >= 1 && j >= 1) {
        v = x[k];
      } else {
        if (j == 0) v = bottom[i];
        if (i == 0) v = left[j];
      }
      if (i<nx && j>0) {
        vr = x[k+1];
      } else {
        if (i == nx) vr = right[j];
        if (j == 0)  vr = bottom[i+1];
      }
      if (i>0 && j<ny) {
         vt = x[k+nx];
      } else {
         if (i == 0)  vt = left[j+1];
         if (j == ny) vt = top[i];
      }
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      fl = sqrt(one + dvdx*dvdx + dvdy*dvdy);
      if (fg & FunctionEval) {
        *f += fl;
      }
      if (fg & GradientEval) {
        if (i>=1 && j>=1) {
          ind = k; val = -(dvdx/hx+dvdy/hy)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i<nx && j>0) {
          ind = k+1; val = (dvdx/hx)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i>0 && j<ny) {
          ind = k+nx; val = (dvdy/hy)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
      }
    }
  }

  /* Compute function and gradient over the upper triangular elements */
  for (j=1; j<=ny1; j++) {
    for (i=1; i<=nx1; i++) {
      k = nx*(j-1) + i-1;
      if (i<=nx && j>1) {
        vb = x[k-nx];
      } else {
        if (j == 1)    vb = bottom[i];
        if (i == nx+1) vb = right[j-1];
      }
      if (i>1 && j<=ny) {
         vl = x[k-1];
      } else {
         if (j == ny+1) vl = top[i-1];
         if (i == 1)    vl = left[j];
      }
      if (i<=nx && j<=ny) {
         v = x[k];
      } else {
         if (i == nx+1) v = right[j];
         if (j == ny+1) v = top[i];
      }
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      fu = sqrt(one + dvdx*dvdx + dvdy*dvdy);
      if (fg & FunctionEval) {
        *f += fu;
      } if (fg & GradientEval) {
        if (i<= nx && j>1) {
          ind = k-nx; val = -(dvdy/hy)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i>1 && j<=ny) {
          ind = k-1; val = -(dvdx/hx)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i<=nx && j<=ny) {
          ind = k; val = (dvdx/hx+dvdy/hy)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(xvec,&x); CHKERRQ(ierr);
  area = p5*hx*hy;
  if (fg & FunctionEval) {
    *f *= area;                                  /* Scale the function */
  } if (fg & GradientEval) {
    ierr = VecAssemblyBegin(gvec); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gvec); CHKERRQ(ierr);
    ierr = VecScale(&area,gvec); CHKERRQ(ierr);     /* Scale gradient */
    printf("gradient");
    ierr = VecView(gvec,STDOUT_VIEWER_WORLD); CHKERRQ(ierr);
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/*
    Form initial guess for nonlinear solver
 */
int FormInitialGuess(SNES snes,Vec x,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int    ierr, i, j, *k, nx = user->mx, ny = user->my;
  double one = 1.0, p5 = 0.5, alphaj, betai, xline, yline;
  double hx = user->hx, hy = user->hy;
  Scalar *bottom, *top, *left, *right, *val;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];
  k      = (int*)PETSCMALLOC(nx*sizeof(int)); CHKPTRQ(k);
  val    = (Scalar*)PETSCMALLOC(nx*sizeof(Scalar)); CHKPTRQ(val);

  ierr = BoundaryValues(user); CHKERRQ(ierr);
  for (j=1; j<=ny; j++) {
    alphaj = j*hy;
    for (i=1; i<=nx; i++) {
      betai = i*hx;
      yline = alphaj*top[i] + (one-alphaj)*bottom[i];
      xline = betai*right[j] + (one-betai)*left[j];
      k[i-1] = nx*(j-1) + i-1;
      val[i-1] = (yline+xline)*p5;
    }
    ierr = VecSetValues(x,nx,k,val,INSERTVALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
  PETSCFREE(k); PETSCFREE(val);
  ierr = VecAssemblyEnd(x); CHKERRQ(ierr);
  ierr = VecView(x,STDOUT_VIEWER_WORLD); CHKERRQ(ierr);
  return 0;
}
/* -------------------------------------------------------------------- */
/*
   Form Hessian matrix
 */
int FormHessian(SNES snes,Vec xvec,Mat *H,Mat *PrecH,MatStructure *flag,
                void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;
  double     gamma1, zero = 0.0, one = 1.0;
  int        i, j, ierr, ndim;
  Scalar     *s, *x, *y;
  SNESMethod method;

  ndim = user->ndim;
  ierr = VecSet(&zero,user->s); CHKERRQ(ierr);
  ierr = VecGetArray(user->s,&s); CHKERRQ(ierr);
  ierr = VecGetArray(xvec,&x); CHKERRQ(ierr);
  for (j=0; j<ndim; j++) {   /* loop over columns */
    s[j] = one;
    ierr = HessProduct(user,x,s,user->y); CHKERRQ(ierr);
    ierr = VecGetArray(user->y,&y); CHKERRQ(ierr);
    s[j] = zero;
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        ierr = MatSetValues(*H,1,&i,1,&j,&y[i],INSERTVALUES); CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(user->y,&y); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(user->s,&s); CHKERRQ(ierr);
  ierr = VecRestoreArray(xvec,&x); CHKERRQ(ierr);

  /* Modify diagonal if necessary */
  ierr = SNESGetMethodFromContext(snes,&method); CHKERRQ(ierr);
  if (method == SNES_UM_NLS) {
    SNESGetLineSearchDampingParameter(snes,&gamma1);
    printf("  gamma1 = %g\n",gamma1);
    for (i=0; i<ndim; i++) {
      ierr = MatSetValues(*H,1,&i,1,&i,&gamma1,ADDVALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*H,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*H,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatView(*H,STDOUT_VIEWER_WORLD); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   HessProduct - Computes y = f''(x)*s
 */
int HessProduct(AppCtx *user,Scalar *x,Scalar *s,Vec y)
{
  int nx = user->mx, ny = user->my, nx1 = nx+1, ny1 = ny+1, i, j, k, ierr, ind;
  double area, dvdx, dvdxhx, dvdy, dvdyhy, dzdx, dzdxhx;
  double one = 1.0, p5 = 0.5, zero = 0.0;
  double dzdy, dzdyhy, fl, fl3, fu, fu3, tl, tu;
  double v, vb, vl, vr, vt, z, zb, zl, zr, zt, hx = user->hx, hy = user->hy;
  Scalar *bottom, *top, *left, *right, val;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = BoundaryValues(user); CHKERRQ(ierr);
  ierr = VecSet(&zero,y); CHKERRQ(ierr);

  /* Compute f''(x)*s over the lower triangular elements */
  for (j=0; j<ny1; j++) {
    for (i=0; i<nx1; i++) {
       k = nx*(j-1) + i-1;
       if (i != 0 && j != 0) {
         v = x[k];
         z = s[k];
       } else {
         if (j == 0) v = bottom[i];
         if (i == 0) v = left[j];
         z = zero;
       }
       if (i != nx && j != 0) {
         vr = x[k+1];
         zr = s[k+1];
       } else {
         if (i == nx) vr = right[j];
         if (j == 0)  vr = bottom[i+1];
         zr = zero;
       }
       if (i != 0 && j != ny) {
          vt = x[k+nx];
          zt = s[k+nx];
       } else {
         if (i == 0)  vt = left[j+1];
         if (j == ny) vt = top[i];
         zt = zero;
       }
       dvdx = (vr-v)/hx;
       dvdy = (vt-v)/hy;
       dzdx = (zr-z)/hx;
       dzdy = (zt-z)/hy;
       dvdxhx = dvdx/hx;
       dvdyhy = dvdy/hy;
       dzdxhx = dzdx/hx;
       dzdyhy = dzdy/hy;
       tl = one + dvdx*dvdx + dvdy*dvdy;
       fl = sqrt(tl);
       fl3 = fl*tl;
       if (i != 0 && j != 0) {
         ind = k;
         val = (dvdx*dzdx+dvdy*dzdy)*(dvdxhx+dvdyhy)/fl3 - (dzdxhx+dzdyhy)/fl;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
       if (i != nx && j != 0) {
         ind = k+1;
         val = dzdxhx/fl - (dvdx*dzdx+dvdy*dzdy)*dvdxhx/fl3;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
       if (i != 0 && j != ny) {
         ind = k+nx;
         val = dzdyhy/fl - (dvdx*dzdx+dvdy*dzdy)*dvdyhy/fl3;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
     }
   }

  /* Compute f''(x)*s over the upper triangular elements */
  for (j=1; j<=ny1; j++) {
    for (i=1; i<=nx1; i++) {
       k = nx*(j-1) + i-1;
       if (i != nx+1 && j != 1) {
         vb = x[k-nx];
         zb = s[k-nx];
       } else {
         if (j == 1) vb = bottom[i];
         if (i == nx+1) vb = right[j-1];
         zb = zero;
       }
       if (i != 1 && j != ny+1) {
         vl = x[k-1];
         zl = s[k-1];
       } else {
         if (j == ny+1) vl = top[i-1];
         if (i == 1)    vl = left[j];
         zl = zero;
       }
       if (i != nx+1 && j != ny+1) {
         v = x[k];
         z = s[k];
       } else {
         if (i == nx+1) v = right[j];
         if (j == ny+1) v = top[i];
         z = zero;
       }
       dvdx = (v-vl)/hx;
       dvdy = (v-vb)/hy;
       dzdx = (z-zl)/hx;
       dzdy = (z-zb)/hy;
       dvdxhx = dvdx/hx;
       dvdyhy = dvdy/hy;
       dzdxhx = dzdx/hx;
       dzdyhy = dzdy/hy;
       tu = one + dvdx*dvdx + dvdy*dvdy;
       fu = sqrt(tu);
       fu3 = fu*tu;
       if (i != nx+1 && j != ny+1) {
         ind = k;
         val = (dzdxhx+dzdyhy)/fu - (dvdx*dzdx+dvdy*dzdy)*(dvdxhx+dvdyhy)/fu3;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
       if (i != 1 && j != ny+1) {
         ind = k-1;
         val = (dvdx*dzdx+dvdy*dzdy)*dvdxhx/fu3 - dzdxhx/fu;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
       if (i != nx+1 && j != 1) {
         ind = k-nx;
         val = (dvdx*dzdx+dvdy*dzdy)*dvdyhy/fu3 - dzdyhy/fu;
         ierr = VecSetValues(y,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
       }
    }
  }

  /* Scale result by area */
  ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
  area = p5*hx*hy;
  ierr = VecAssemblyEnd(y); CHKERRQ(ierr);
  ierr = VecScale(&area,y); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   BoundaryValues - Computes Enneper's boundary conditions: bottom, 
   top, left, right.  Enneper's boundary values are obtained by defining
     bv(x,y) = u**2 - v**2, where u and v are the unique solutions of
     x = u + u*(v**2) - (u**3)/3, y = -v - (u**2)*v + (v**3)/3. 
 */
int BoundaryValues(AppCtx *user)
{
  int    maxit=5, i, j, k, limit, nx = user->mx, ny = user->my;
  double one=1.0, two=2.0, three=3.0, tol=1.0e-10;
  double b=-.50, t=.50, l=-.50, r=.50, det, fnorm, xt, yt;
  double nf[2], njac[2][2], u[2], hx = user->hx, hy = user->hy;
  Scalar *bottom, *top, *left, *right;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  for (j=0; j<4; j++) {
    switch (j) {
      case 0:
        yt = b; xt = l; limit = nx + 2; break;
      case 1:
        yt = t; xt = l; limit = nx + 2; break;
      case 2:
        yt = b; xt = l; limit = ny + 2; break;
      case 3:
        yt = b; xt = r; limit = ny + 2; break;
      default:
        SETERRQ(1,"Only cases 0,1,2,3 are valid");
    }
    /* Use Newton's method to solve xt = u + u*(v**2) - (u**3)/3,
       yt = -v - (u**2)*v + (v**3)/3. */

    for (i=0; i<limit; i++) {
      u[0] = xt;
      u[1] = -yt;
      for (k=0; k<maxit; k++) {
        nf[0] = u[0] + u[0]*u[1]*u[1] - pow(u[0],three)/three - xt;
        nf[1] = -u[1] - u[0]*u[0]*u[1] + pow(u[1],three)/three - yt;
        fnorm = sqrt(nf[0]*nf[0]+nf[1]*nf[1]);
        if (fnorm <= tol) break;
        njac[0][0] = one + u[1]*u[1] - u[0]*u[0];
        njac[0][1] = two*u[0]*u[1];
        njac[1][0] = -two*u[0]*u[1];
        njac[1][1] = -one - u[0]*u[0] + u[1]*u[1];
        det = njac[0][0]*njac[1][1] - njac[0][1]*njac[1][0];
        u[0] -= (njac[1][1]*nf[0]-njac[0][1]*nf[1])/det;
        u[1] -= (njac[0][0]*nf[1]-njac[1][0]*nf[0])/det;
      }
      switch (j) {
        case 0:
          bottom[i] = u[0]*u[0] - u[1]*u[1]; xt += hx; break;
        case 1:
          top[i] = u[0]*u[0] - u[1]*u[1];    xt += hx; break;
        case 2:
          left[i] = u[0]*u[0] - u[1]*u[1];   yt += hy; break;
        case 3:
          right[i] = u[0]*u[0] - u[1]*u[1];  yt += hy; break;
        default:
          SETERRQ(1,"Only cases 0,1,2,3 are valid");
      }
    }
  }
  return 0;
}


