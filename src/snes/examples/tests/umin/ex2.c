#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.5 1995/08/31 00:22:09 curfman Exp curfman $";
#endif

static char help[] = "\n\
ex2:\n\
This program demonstrates use of the SNES package to solve an unconstrained\n\
minimization problem on a single processor.  This example is based on the\n\
Minimal Surface Area problem (dmsa) from the MINPACK-2 test suite.  The\n\
command line options are:\n\
  -mx xg, where xg = number of grid points in the 1st coordinate direction\n\
  -my yg, where yg = number of grid points in the 2nd coordinate direction\n";

#include "petsc.h"
#include "snes.h"
#include <string.h>
#include <math.h>

/* User-defined application context */
   typedef struct {
      int     mx;        /* discretization in x-direction */
      int     my;        /* discretization in y-direction */
      int     ndim;      /* problem dimension */
      int     number;    /* test problem number */
      double  *work;     /* work space */
      Vec     s,y,xvec;  /* work space for computing Hessian */
      double  hx, hy;
   } AppCtx;

typedef enum {FunctionEval=1, GradientEval=2} FctGradFlag;

int FormHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int MatrixFreeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormMinimizationFunction(SNES,Vec,double*,void*);
int FormGradient(SNES,Vec,Vec,void*);
int FormInitialGuess(SNES,Vec,void*);
int EvalFunctionGradient(SNES,Vec,double*,Vec,FctGradFlag,AppCtx*);
int BoundaryValues(AppCtx*);
int HessianProduct(void*,Vec,Vec);

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
  SLES       sles;
  PC         pc;

  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);

  /* Set up user-defined work space */
  OptionsGetInt(0,"-my",&my);
  OptionsGetInt(0,"-mx",&mx);
  user.ndim = mx * my;
  user.mx = mx;
  user.my = my;
  user.hx = one/(double)(mx+1);
  user.hy = one/(double)(my+1);
  user.work = (double*)PETSCMALLOC(2*(mx+my+4)*sizeof(double)); CHKPTRQ(user.work);

  /* Allocate vectors */
  ierr = VecCreate(MPI_COMM_SELF,user.ndim,&user.y); CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&user.s); CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&g); CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&x); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_SELF,SNES_UNCONSTRAINED_MINIMIZATION,&snes);
         CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess,(void *)&user); CHKERRA(ierr);
  ierr = SNESSetMinimizationFunction(snes,FormMinimizationFunction,
         (void *)&user); CHKERRA(ierr);
  ierr = SNESSetGradient(snes,g,FormGradient,(void *)&user); CHKERRA(ierr);

  /* Either explicitly form Hessian matrix approx or use matrix-free version */
  if (OptionsHasName(0,"-snes_mf")) {
    ierr = MatShellCreate(MPI_COMM_SELF,user.ndim,user.ndim,(void*)&user,&H);
           CHKERRA(ierr);
    ierr = MatShellSetMult(H,HessianProduct); CHKERRA(ierr);
    ierr = SNESSetHessian(snes,H,H,MatrixFreeHessian,(void *)&user); 
           CHKERRA(ierr);

    /* Set null preconditioner.  Alternatively, set user-provided 
       preconditioner or explicitly form preconditioning matrix */
    ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
    ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
    ierr = PCSetMethod(pc,PCNONE); CHKERRA(ierr);
  } else {
    ierr = MatCreate(MPI_COMM_SELF,user.ndim,user.ndim,&H); CHKERRA(ierr);
    ierr = SNESSetHessian(snes,H,H,FormHessian,(void *)&user); CHKERRA(ierr);
  }

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
    Form initial guess for nonlinear solver
 */
int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int    ierr, i, j, k, nx = user->mx, ny = user->my;
  double one = 1.0, p5 = 0.5, alphaj, betai;
  double hx = user->hx, hy = user->hy;
  double *bottom, *top, *left, *right, xline, yline;
  Scalar *x;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  /* Compute the boundary values once only */
  ierr = BoundaryValues(user); CHKERRQ(ierr);
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  for (j=1; j<=ny; j++) {
    alphaj = j*hy;
    for (i=1; i<=nx; i++) {
      betai = i*hx;
      yline = alphaj*top[i] + (one-alphaj)*bottom[i];
      xline = betai*right[j] + (one-betai)*left[j];
      k = nx*(j-1) + i-1;
      x[k] = (yline+xline)*p5;
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  return 0;
}
/* -------------------------------------------------------------------- */
/*
    Evaluate function f(x)
 */

int FormMinimizationFunction(SNES snes,Vec x,double *f,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  return EvalFunctionGradient(snes,x,f,NULL,FunctionEval,user); 
}
/* -------------------------------------------------------------------- */
/*
    Evaluate gradient g(x)
 */

int FormGradient(SNES snes,Vec x,Vec g,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  return EvalFunctionGradient(snes,x,NULL,g,GradientEval,user); 
}
/* -------------------------------------------------------------------- */
/*
    Evaluate function f(x) and/or gradient g(x)
 */

int EvalFunctionGradient(SNES snes,Vec X,double *f,Vec gvec,FctGradFlag fg,
                         AppCtx *user)
{
  int    ierr, nx = user->mx, ny = user->my, nx1 = nx+1, ny1 = ny+1;
  int    ind, i, j, k;
  double one = 1.0, p5 = 0.5, hx = user->hx, hy = user->hy, fl, fu, area;
  double *bottom, *top, *left, *right;
  double v=0.0, vb=0.0, vl=0.0, vr=0.0, vt=0.0, dvdx, dvdy;
  Scalar zero = 0.0, val, *x;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  if (fg & FunctionEval) {
    *f = 0.0;
  }
  if (fg & GradientEval) {
    ierr = VecSet(&zero,gvec); CHKERRQ(ierr);
    ierr = VecView(gvec,STDOUT_VIEWER_SELF); CHKERRQ(ierr);
  }

  /* Compute function and gradient over the lower triangular elements */
  for (j=0; j<ny1; j++) {
    for (i=0; i<nx1; i++) {
      k = nx*(j-1) + i-1;
      if (i >= 1 && j >= 1) {
#if defined(PETSC_COMPLEX)
        v = real(x[k]);
#else
        v = x[k];
#endif
      } else {
        if (j == 0) v = bottom[i];
        if (i == 0) v = left[j];
      }
      if (i<nx && j>0) {
#if defined(PETSC_COMPLEX)
        vr = real(x[k+1]);
#else
        vr = x[k+1];
#endif
      } else {
        if (i == nx) vr = right[j];
        if (j == 0)  vr = bottom[i+1];
      }
      if (i>0 && j<ny) {
#if defined(PETSC_COMPLEX)
         vt =real( x[k+nx]);
#else
         vt = x[k+nx];
#endif
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
 printf("real=%g, imag=%g\n",real(val),imag(val));
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i<nx && j>0) {
          ind = k+1; val = (dvdx/hx)/fl;
 printf("real=%g, imag=%g\n",real(val),imag(val));
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i>0 && j<ny) {
          ind = k+nx; val = (dvdy/hy)/fl;
 printf("real=%g, imag=%g\n",real(val),imag(val));
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
#if defined(PETSC_COMPLEX)
        vb = real(x[k-nx]);
#else
        vb = x[k-nx];
#endif
      } else {
        if (j == 1)    vb = bottom[i];
        if (i == nx+1) vb = right[j-1];
      }
      if (i>1 && j<=ny) {
#if defined(PETSC_COMPLEX)
         vl = real(x[k-1]);
#else
         vl = x[k-1];
#endif
      } else {
         if (j == ny+1) vl = top[i-1];
         if (i == 1)    vl = left[j];
      }
      if (i<=nx && j<=ny) {
#if defined(PETSC_COMPLEX)
         v = real(x[k]);
#else
         v = x[k];
#endif
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
 printf("real=%g, imag=%g\n",real(val),imag(val));
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i>1 && j<=ny) {
          ind = k-1; val = -(dvdx/hx)/fu;
 printf("real=%g, imag=%g\n",real(val),imag(val));
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
        if (i<=nx && j<=ny) {
          ind = k; val = (dvdx/hx+dvdy/hy)/fu;
 printf("real=%g, imag=%g\n",real(val),imag(val));
          ierr = VecSetValues(gvec,1,&ind,&val,ADDVALUES); CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  area = p5*hx*hy;
  if (fg & FunctionEval) {   /* Scale the function */
    *f *= area;
  } if (fg & GradientEval) { /* Scale the gradient */
    ierr = VecAssemblyBegin(gvec); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gvec); CHKERRQ(ierr);
    ierr = VecScale((Scalar*)&area,gvec); CHKERRQ(ierr);
    ierr = VecView(gvec,STDOUT_VIEWER_SELF); CHKERRQ(ierr);
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/*
   FormHessian - Forms Hessian matrix by computing a column at a time.
 */
int FormHessian(SNES snes,Vec X,Mat *H,Mat *PrecH,MatStructure *flag,
                void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;
  int        i, j, ierr, ndim;
  Scalar     *y, zero = 0.0, one = 1.0;
  double     gamma1;
  SNESMethod method;

  ndim = user->ndim;
  ierr = VecSet(&zero,user->s); CHKERRQ(ierr);
  user->xvec = X; /* Set location of vector */

  for (j=0; j<ndim; j++) {   /* loop over columns */

    ierr = VecSetValues(user->s,1,&j,&one,INSERTVALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s); CHKERRQ(ierr);

    ierr = HessianProduct(ptr,user->s,user->y); CHKERRQ(ierr);

    ierr = VecSetValues(user->s,1,&j,&zero,INSERTVALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s); CHKERRQ(ierr);

    ierr = VecGetArray(user->y,&y); CHKERRQ(ierr);
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        ierr = MatSetValues(*H,1,&i,1,&j,&y[i],INSERTVALUES); CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(user->y,&y); CHKERRQ(ierr);
  }

  /* Modify diagonal if necessary */
  ierr = SNESGetMethodFromContext(snes,&method); CHKERRQ(ierr);
  if (method == SNES_UM_NLS) {
    SNESGetLineSearchDampingParameter(snes,&gamma1);
    printf("  gamma1 = %g\n",gamma1);
    for (i=0; i<ndim; i++) {
      ierr = MatSetValues(*H,1,&i,1,&i,(Scalar*)&gamma1,ADDVALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*H,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*H,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
/* -------------------------------------------------------------------- */
/*
   FormHessian - Forms Hessian matrix by computing a column at a time.
 */
int MatrixFreeHessian(SNES snes,Vec X,Mat *H,Mat *PrecH,MatStructure *flag,
                      void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;

  /* Sets location of vector for use in computing matrix-vector products
     of the form H(X)*y  */
  user->xvec = X;   
  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   HessianProduct - Computes the matrix-vector product: y = f''(x)*s
 */
int HessianProduct(void *ptr,Vec svec,Vec y)
{
  AppCtx *user = (AppCtx *) ptr;
  int    nx = user->mx, ny = user->my, nx1 = nx+1, ny1 = ny+1;
  int    i, j, k, ierr, ind;
  double one = 1.0, p5 = 0.5, hx = user->hx, hy = user->hy;
  double dzdy, dzdyhy, fl, fl3, fu, fu3, tl, tu, z, zb, zl, zr, zt;
  double *bottom, *top, *left, *right, *s, *x;
  double dvdx, dvdxhx, dvdy, dvdyhy, dzdx, dzdxhx;
  double v=0.0, vb=0.0, vl=0.0, vr=0.0, vt=0.0, zerod = 0.0;
  Scalar val, area, zero = 0.0;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = VecGetArray(user->xvec,(Scalar**)&x); CHKERRQ(ierr);
  ierr = VecGetArray(svec,(Scalar**)&s); CHKERRQ(ierr);
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
         z = zerod;
       }
       if (i != nx && j != 0) {
         vr = x[k+1];
         zr = s[k+1];
       } else {
         if (i == nx) vr = right[j];
         if (j == 0)  vr = bottom[i+1];
         zr = zerod;
       }
       if (i != 0 && j != ny) {
          vt = x[k+nx];
          zt = s[k+nx];
       } else {
         if (i == 0)  vt = left[j+1];
         if (j == ny) vt = top[i];
         zt = zerod;
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
         zb = zerod;
       }
       if (i != 1 && j != ny+1) {
         vl = x[k-1];
         zl = s[k-1];
       } else {
         if (j == ny+1) vl = top[i-1];
         if (i == 1)    vl = left[j];
         zl = zerod;
       }
       if (i != nx+1 && j != ny+1) {
         v = x[k];
         z = s[k];
       } else {
         if (i == nx+1) v = right[j];
         if (j == ny+1) v = top[i];
         z = zerod;
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
  ierr = VecRestoreArray(svec,(Scalar**)&s); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->xvec,(Scalar**)&x); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y); CHKERRQ(ierr);

  /* Scale result by area */
  area = p5*hx*hy;
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
  double *bottom, *top, *left, *right;

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


