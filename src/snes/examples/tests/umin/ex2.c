#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.48 1999/05/04 20:36:14 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates use of the SNES package to solve unconstrained\n\
minimization problems on a single processor.  These examples are based on\n\
problems from the MINPACK-2 test suite.  The command line options are:\n\
  -mx xg, where xg = number of grid points in the 1st coordinate direction\n\
  -my yg, where yg = number of grid points in the 2nd coordinate direction\n\
  -p problem_number, where the possible problem numbers are:\n\
     1: Elastic-Plastic Torsion (dept)\n\
     2: Minimal Surface Area (dmsa)\n\
  -snes_mf: use matrix-free methods\n\
  -par param, where param = angle of twist per unit length (problem 1 only)\n\n";

#include "snes.h"

/* User-defined application context */
   typedef struct {
      int     problem;    /* test problem number */
      double  param;      /* nonlinearity parameter */
      int     mx;         /* discretization in x-direction */
      int     my;         /* discretization in y-direction */
      int     ndim;       /* problem dimension */
      int     number;     /* test problem number */
      Scalar  *work;      /* work space */
      Vec     s, y, xvec; /* work space for computing Hessian */
      Scalar  hx, hy;
   } AppCtx;

/* Flag to indicate evaluation of function and/or gradient */
typedef enum {FunctionEval=1, GradientEval=2} FctGradFlag;

/* General routines */
extern int FormHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int MatrixFreeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int FormMinimizationFunction(SNES,Vec,double*,void*);
extern int FormGradient(SNES,Vec,Vec,void*);

/* For Elastic-Plastic Torsion test problem */
extern int HessianProduct1(void *,Vec,Vec);
extern int HessianProductMat1(Mat,Vec,Vec);
extern int FormInitialGuess1(AppCtx*,Vec);
extern int EvalFunctionGradient1(SNES,Vec,double*,Vec,FctGradFlag,AppCtx*);

/* For Minimal Surface Area test problem */
extern int HessianProduct2(void *,Vec,Vec);
extern int HessianProductMat2(Mat,Vec,Vec);
extern int FormInitialGuess2(AppCtx*,Vec);
extern int EvalFunctionGradient2(SNES,Vec,double*,Vec,FctGradFlag,AppCtx*);
extern int BoundaryValues(AppCtx*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  SNES       snes;                 /* SNES context */
  SNESType   method = SNES_UM_TR;  /* nonlinear solution method */
  Vec        x, g;                 /* solution, gradient vectors */
  Mat        H;                    /* Hessian matrix */
  SLES       sles;                 /* linear solver */
  PC         pc;                   /* preconditioner */
  AppCtx     user;                 /* application context */
  int        mx=10;   /* discretization of problem in x-direction */
  int        my=10;   /* discretization of problem in y-direction */
  double     one = 1.0;
  int        ierr, its, nfails, flg, ldim;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* Set up user-defined work space */
  user.problem = 1;
  ierr = OptionsGetInt(PETSC_NULL,"-p",&user.problem,&flg);CHKERRA(ierr);
  user.param = 5.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg);CHKERRA(ierr);
  if (user.problem != 1 && user.problem != 2) SETERRA(1,0,"Invalid problem number");
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,&flg);CHKERRA(ierr);
  user.ndim = mx * my; user.mx = mx; user.my = my;
  user.hx = one/(mx+1); user.hy = one/(my+1);
  if (user.problem == 2) {
    user.work = (Scalar*)PetscMalloc(2*(mx+my+4)*sizeof(Scalar));CHKPTRQ(user.work);
  } else {
    user.work = 0;
  }

  /* Allocate vectors */
  ierr = VecCreate(PETSC_COMM_SELF,PETSC_DECIDE,user.ndim,&user.y);CHKERRA(ierr);
  ierr = VecSetFromOptions(user.y);CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&user.s);CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&g);CHKERRA(ierr);
  ierr = VecDuplicate(user.y,&x);CHKERRA(ierr);
  ierr = VecGetLocalSize(x,&ldim);CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_SELF,SNES_UNCONSTRAINED_MINIMIZATION,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,method);CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetMinimizationFunction(snes,FormMinimizationFunction,
         (void *)&user);CHKERRA(ierr);
  ierr = SNESSetGradient(snes,g,FormGradient,(void *)&user);CHKERRA(ierr);

  /* Form Hessian matrix approx, using one of three methods:
      (default)   : explicitly form Hessian approximation
      -snes_mf    : employ default PETSc matrix-free code
      -my_snes_mf : employ user-defined matrix-free code (since we just happen to
                    have a routine for matrix-vector products in this example) 
   */
  ierr = OptionsHasName(PETSC_NULL,"-my_snes_mf",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MatCreateShell(PETSC_COMM_SELF,ldim,user.ndim,user.ndim,user.ndim,
           (void*)&user,&H);CHKERRA(ierr);
    if (user.problem == 1) {
      ierr = MatShellSetOperation(H,MATOP_MULT,(void *)HessianProductMat1);
     CHKERRA(ierr);
    } else if (user.problem == 2) {
      ierr = MatShellSetOperation(H,MATOP_MULT,(void*)HessianProductMat2);
     CHKERRA(ierr);
    }
    ierr = SNESSetHessian(snes,H,H,MatrixFreeHessian,(void *)&user);CHKERRA(ierr);

    /* Set null preconditioner.  Alternatively, set user-provided 
       preconditioner or explicitly form preconditioning matrix */
    ierr = SNESGetSLES(snes,&sles);CHKERRA(ierr);
    ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRA(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,user.ndim,user.ndim,&H);CHKERRA(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC);CHKERRA(ierr);
    ierr = SNESSetHessian(snes,H,H,FormHessian,(void *)&user);CHKERRA(ierr);
  }

  /* Set options; then solve minimization problem */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);
  if (user.problem == 1) {
    ierr = FormInitialGuess1(&user,x);CHKERRA(ierr);
  } else if (user.problem == 2) {
    ierr = FormInitialGuess2(&user,x);CHKERRA(ierr);
  }
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  ierr = SNESGetNumberUnsuccessfulSteps(snes,&nfails);CHKERRA(ierr);
  ierr = SNESView(snes,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %d, ",its);
  PetscPrintf(PETSC_COMM_SELF,"number of unsuccessful steps = %d\n\n",nfails);

  /* Free data structures */
  if (user.work) PetscFree(user.work); 
  ierr = VecDestroy(user.s);CHKERRA(ierr);
  ierr = VecDestroy(user.y);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(g);CHKERRA(ierr);
  ierr = MatDestroy(H);CHKERRA(ierr);
  ierr = SNESDestroy(snes);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormMinimzationFunction"
/*
    FormMinimizationFunction - Evaluates function f(x).
*/
int FormMinimizationFunction(SNES snes,Vec x,double *f,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int ierr;

  if (user->problem == 1) {
    ierr = EvalFunctionGradient1(snes,x,f,NULL,FunctionEval,user);CHKERRQ(ierr);
  } else if (user->problem == 2) {
    ierr = EvalFunctionGradient2(snes,x,f,NULL,FunctionEval,user);CHKERRQ(ierr);
  } else SETERRQ(1,0,"FormMinimizationFunction: Invalid problem number.");
  return 0;
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormGradient"
/*
    FormGradient - Evaluates gradient g(x).
*/
int FormGradient(SNES snes,Vec x,Vec g,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int ierr;

  if (user->problem == 1) {
    ierr = EvalFunctionGradient1(snes,x,NULL,g,GradientEval,user);CHKERRQ(ierr);
  } else if (user->problem == 2) {
    ierr = EvalFunctionGradient2(snes,x,NULL,g,GradientEval,user);CHKERRQ(ierr);
  } else SETERRQ(1,0,"FormGradient: Invalid problem number.");
  return 0;
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormHessian"
/*
   FormHessian - Forms Hessian matrix by computing a column at a time.
*/
int FormHessian(SNES snes,Vec X,Mat *H,Mat *PrecH,MatStructure *flag,void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;
  int        i, j, ierr, ndim;
  Scalar     *y, zero = 0.0, one = 1.0;

  ndim = user->ndim;
  ierr = VecSet(&zero,user->s);CHKERRQ(ierr);
  user->xvec = X; /* Set location of vector */

  ierr = MatZeroEntries(*H);CHKERRQ(ierr);
  for (j=0; j<ndim; j++) {   /* loop over columns */

    ierr = VecSetValues(user->s,1,&j,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    if (user->problem == 1) {
      ierr = HessianProduct1(ptr,user->s,user->y);CHKERRQ(ierr);
    } else if (user->problem == 2) {
      ierr = HessianProduct2(ptr,user->s,user->y);CHKERRQ(ierr);
    }

    ierr = VecSetValues(user->s,1,&j,&zero,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    ierr = VecGetArray(user->y,&y);CHKERRQ(ierr);
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        ierr = MatSetValues(*H,1,&i,1,&j,&y[i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(user->y,&y);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "MatrxFreeHessian"
/*
  MatrixFreeHessian
 */
int MatrixFreeHessian(SNES snes,Vec X,Mat *H,Mat *PrecH,MatStructure *flag,void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;

  /* Sets location of vector for use in computing matrix-vector products
     of the form H(X)*y  */
  user->xvec = X;   
  return 0;
}

/* ------------------------------------------------------------------ */
/*               Elastic-Plastic Torsion Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess1(AppCtx *user,Vec X)
{
  int    ierr, i, j, k, nx = user->mx, ny = user->my;
  Scalar hx = user->hx, hy = user->hy, temp;
  Scalar *x;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (j=0; j<ny; j++) {
    temp = PetscMin(j+1,ny-j)*hy;
    for (i=0; i<nx; i++) {
      k = nx*j + i;
#if !defined(PETSC_USE_COMPLEX)
      x[k] = PetscMin((PetscMin(i+1,nx-i))*hx,temp);
#else
      x[k] = PetscMin(PetscReal(PetscMin(i+1,nx-i)*hx),PetscReal(temp));
#endif
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* ---------- Evaluate function f(x) and/or gradient g(x) ----------- */

#undef __FUNC__
#define __FUNC__ "EvalFunctionGradient1"
int EvalFunctionGradient1(SNES snes,Vec X,double *f,Vec gvec,FctGradFlag fg,
                         AppCtx *user)
{
  int    ierr, nx = user->mx, ny = user->my, ind, i, j, k;
  Scalar hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5, cdiv3;
  Scalar zero = 0.0, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  Scalar val, v, *x;

  cdiv3 = user->param/three;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  if (fg & GradientEval) {
    ierr = VecSet(&zero,gvec);CHKERRQ(ierr);
  }

  /* Compute function and gradient over the lower triangular elements */
  for (j=-1; j<ny; j++) {
    for (i=-1; i<nx; i++) {
      k = nx*j + i;
      v = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k];
      if (i < nx-1 && j > -1) vr = x[k+1];
      if (i > -1 && j < ny-1) vt = x[k+nx];
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      if (fg & FunctionEval) {
        fquad += dvdx*dvdx + dvdy*dvdy;
        flin -= cdiv3*(v+vr+vt);
      }
      if (fg & GradientEval) {
        if (i != -1 && j != -1) {
          ind = k; val = - dvdx/hx - dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != nx-1 && j != -1) {
          ind = k+1; val =  dvdx/hx - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != -1 && j != ny-1) {
          ind = k+nx; val = dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Compute function and gradient over the upper triangular elements */
  for (j=0; j<=ny; j++) {
    for (i=0; i<=nx; i++) {
      k = nx*j + i;
      vb = zero;
      vl = zero;
      v = zero;
      if (i < nx && j > 0) vb = x[k-nx];
      if (i > 0 && j < ny) vl = x[k-1];
      if (i < nx && j < ny) v = x[k];
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      if (fg & FunctionEval) {
        fquad = fquad + dvdx*dvdx + dvdy*dvdy;
        flin = flin - cdiv3*(vb+vl+v);
      } if (fg & GradientEval) {
        if (i != nx && j != 0) {
          ind = k-nx; val = - dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != 0 && j != ny) {
          ind = k-1; val =  - dvdx/hx - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != nx && j != ny) {
          ind = k; val =  dvdx/hx + dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  area = p5*hx*hy;
  if (fg & FunctionEval) {   /* Scale the function */
#if !defined(PETSC_USE_COMPLEX)
    *f = area*(p5*fquad+flin);
#else
    *f = PetscReal(area*(p5*fquad+flin));
#endif
  } if (fg & GradientEval) { /* Scale the gradient */
    ierr = VecAssemblyBegin(gvec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gvec);CHKERRQ(ierr);
    ierr = VecScale((Scalar*)&area,gvec);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "HessianProductMat1"
int HessianProductMat1(Mat mat,Vec svec,Vec y)
{
  void *ptr;
  MatShellGetContext(mat,&ptr);
  HessianProduct1(ptr,svec,y);
  return 0;
}
  
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "HessianProduct1"
/* 
   HessianProduct - Computes the matrix-vector product: y = f''(x)*s
 */
int HessianProduct1(void *ptr,Vec svec,Vec y)
{
  AppCtx *user = (AppCtx *)ptr;
  int    nx, ny, i, j, k, ierr, ind;
  Scalar p5 = 0.5, one = 1.0, hx, hy;
  Scalar v, vb, vl, vr, vt, hxhx, hyhy, zero = 0.0;
  Scalar val, area, *x, *s, szero = 0.0;

  nx = user->mx;
  ny = user->my;
  hx = user->hx;
  hy = user->hy;

  hxhx = one/(hx*hx);
  hyhy = one/(hy*hy);

  ierr = VecGetArray(user->xvec,&x);CHKERRQ(ierr);
  ierr = VecGetArray(svec,&s);CHKERRQ(ierr);
  ierr = VecSet(&szero,y);CHKERRQ(ierr);

  /* Compute f''(x)*s over the lower triangular elements */
  for (j=-1; j<ny; j++) {
    for (i=-1; i<nx; i++) {
       k = nx*j + i;
       v = zero;
       vr = zero;
       vt = zero;
       if (i != -1 && j != -1) v = s[k];
       if (i != nx-1 && j != -1) {
         vr = s[k+1];
         ind = k+1; val = hxhx*(vr-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != -1 && j != ny-1) {
         vt = s[k+nx];
         ind = k+nx; val = hyhy*(vt-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != -1 && j != -1) {
         ind = k; val = hxhx*(v-vr) + hyhy*(v-vt);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
     }
   }

  /* Compute f''(x)*s over the upper triangular elements */
  for (j=0; j<=ny; j++) {
    for (i=0; i<=nx; i++) {
       k = nx*j + i;
       v = zero;
       vl = zero;
       vb = zero;
       if (i != nx && j != ny) v = s[k];
       if (i != nx && j != 0) {
         vb = s[k-nx];
         ind = k-nx; val = hyhy*(vb-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != 0 && j != ny) {
         vl = s[k-1];
         ind = k-1; val = hxhx*(vl-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != nx && j != ny) {
         ind = k; val = hxhx*(v-vl) + hyhy*(v-vb);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
    }
  }
  ierr = VecRestoreArray(svec,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->xvec,&x);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  /* Scale result by area */
  area = p5*hx*hy;
  ierr = VecScale(&area,y);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------ */
/*                 Minimal Surface Area Test Problem                  */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess2(AppCtx *user,Vec X)
{
  int    ierr, i, j, k, nx = user->mx, ny = user->my;
  Scalar one = 1.0, p5 = 0.5, alphaj, betai;
  Scalar hx = user->hx, hy = user->hy, *x;
  Scalar *bottom, *top, *left, *right, xline, yline;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  /* Compute the boundary values once only */
  ierr = BoundaryValues(user);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (j=0; j<ny; j++) {
    alphaj = (j+1)*hy;
    for (i=0; i<nx; i++) {
      betai = (i+1)*hx;
      yline = alphaj*top[i+1] + (one-alphaj)*bottom[i+1];
      xline = betai*right[j+1] + (one-betai)*left[j+1];
      k = nx*j + i;
      x[k] = (yline+xline)*p5;
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}

/* ---------- Evaluate function f(x) and/or gradient g(x) ----------- */

#undef __FUNC__
#define __FUNC__ "EvalFunctionGradient2"
int EvalFunctionGradient2(SNES snes,Vec X,double *f,Vec gvec,FctGradFlag fg,
                         AppCtx *user)
{
  int    ierr, nx = user->mx, ny = user->my, ind, i, j, k;
  Scalar one = 1.0, p5 = 0.5, hx = user->hx, hy = user->hy, fl, fu, area;
  Scalar *bottom, *top, *left, *right;
  Scalar v=0.0, vb=0.0, vl=0.0, vr=0.0, vt=0.0, dvdx, dvdy;
  Scalar zero = 0.0, val, *x;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  if (fg & FunctionEval) {
    *f = 0.0;
  }
  if (fg & GradientEval) {
    ierr = VecSet(&zero,gvec);CHKERRQ(ierr);
  }

  /* Compute function and gradient over the lower triangular elements */
  for (j=-1; j<ny; j++) {
    for (i=-1; i<nx; i++) {
      k = nx*j + i;
      if (i >= 0 && j >= 0) {
        v = x[k];
      } else {
        if (j == -1) v = bottom[i+1];
        if (i == -1) v = left[j+1];
      }
      if (i<nx-1 && j>-1) {
        vr = x[k+1];
      } else {
        if (i == nx-1) vr = right[j+1];
        if (j == -1)  vr = bottom[i+2];
      }
      if (i>-1 && j<ny-1) {
         vt = x[k+nx];
      } else {
         if (i == -1)  vt = left[j+2];
         if (j == ny-1) vt = top[i+1];
      }
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      fl = sqrt(one + dvdx*dvdx + dvdy*dvdy);
      if (fg & FunctionEval) {
#if !defined(PETSC_USE_COMPLEX)
        *f += fl;
#else
        *f += PetscReal(fl);
#endif
      }
      if (fg & GradientEval) {
        if (i>-1 && j>-1) {
          ind = k; val = -(dvdx/hx+dvdy/hy)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i<nx-1 && j>-1) {
          ind = k+1; val = (dvdx/hx)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i>-1 && j<ny-1) {
          ind = k+nx; val = (dvdy/hy)/fl;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Compute function and gradient over the upper triangular elements */
  for (j=0; j<=ny; j++) {
    for (i=0; i<=nx; i++) {
      k = nx*j + i;
      if (i<nx && j>0) {
        vb = x[k-nx];
      } else {
        if (j == 0)    vb = bottom[i+1];
        if (i == nx) vb = right[j];
      }
      if (i>0 && j<ny) {
         vl = x[k-1];
      } else {
         if (j == ny) vl = top[i];
         if (i == 0)    vl = left[j+1];
      }
      if (i<nx && j<ny) {
         v = x[k];
      } else {
         if (i == nx) v = right[j+1];
         if (j == ny) v = top[i+1];
      }
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      fu = sqrt(one + dvdx*dvdx + dvdy*dvdy);
      if (fg & FunctionEval) {
#if !defined(PETSC_USE_COMPLEX)
        *f += fu;
#else
        *f += PetscReal(fu);
#endif
      } if (fg & GradientEval) {
        if (i<nx && j>0) {
          ind = k-nx; val = -(dvdy/hy)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i>0 && j<ny) {
          ind = k-1; val = -(dvdx/hx)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i<nx && j<ny) {
          ind = k; val = (dvdx/hx+dvdy/hy)/fu;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  area = p5*hx*hy;
  if (fg & FunctionEval) {   /* Scale the function */
#if !defined(PETSC_USE_COMPLEX)
    *f *= area;
#else
    *f *= PetscReal(area);
#endif
  } if (fg & GradientEval) { /* Scale the gradient */
    ierr = VecAssemblyBegin(gvec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gvec);CHKERRQ(ierr);
    ierr = VecScale(&area,gvec);CHKERRQ(ierr);
  }
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "HessianProductMat2"
int HessianProductMat2(Mat mat,Vec svec,Vec y)
{
  void *ptr;
  MatShellGetContext(mat,&ptr);
  HessianProduct2(ptr,svec,y);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "HessianProduct2"
/* 
   HessianProduct2 - Computes the matrix-vector product: y = f''(x)*s
 */
int HessianProduct2(void *ptr,Vec svec,Vec y)
{
  AppCtx *user = (AppCtx *) ptr;
  int    nx, ny, i, j, k, ierr, ind;
  Scalar one = 1.0, p5 = 0.5, hx, hy;
  Scalar dzdy, dzdyhy, fl, fl3, fu, fu3, tl, tu, z, zb, zl, zr, zt;
  Scalar *bottom, *top, *left, *right;
  Scalar dvdx, dvdxhx, dvdy, dvdyhy, dzdx, dzdxhx;
  Scalar v=0.0, vb=0.0, vl=0.0, vr=0.0, vt=0.0, zerod = 0.0;
  Scalar val, area, zero = 0.0, *s, *x;

  nx = user->mx;
  ny = user->my;
  hx = user->hx;
  hy = user->hy;

  bottom = user->work;
  top    = &user->work[nx+2];
  left   = &user->work[2*nx+4];
  right  = &user->work[2*nx+ny+6];

  ierr = VecGetArray(user->xvec,&x);CHKERRQ(ierr);
  ierr = VecGetArray(svec,&s);CHKERRQ(ierr);
  ierr = VecSet(&zero,y);CHKERRQ(ierr);

  /* Compute f''(x)*s over the lower triangular elements */
  for (j=-1; j<ny; j++) {
    for (i=-1; i<nx; i++) {
       k = nx*j + i;
       if (i != -1 && j != -1) {
         v = x[k];
         z = s[k];
       } else {
         if (j == -1) v = bottom[i+1];
         if (i == -1) v = left[j+1];
         z = zerod;
       }
       if (i != nx-1 && j != -1) {
         vr = x[k+1];
         zr = s[k+1];
       } else {
         if (i == nx-1) vr = right[j+1];
         if (j == -1)  vr = bottom[i+2];
         zr = zerod;
       }
       if (i != -1 && j != ny-1) {
          vt = x[k+nx];
          zt = s[k+nx];
       } else {
         if (i == -1)  vt = left[j+2];
         if (j == ny-1) vt = top[i+1];
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
       if (i != -1 && j != -1) {
         ind = k;
         val = (dvdx*dzdx+dvdy*dzdy)*(dvdxhx+dvdyhy)/fl3 - (dzdxhx+dzdyhy)/fl;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != nx-1 && j != -1) {
         ind = k+1;
         val = dzdxhx/fl - (dvdx*dzdx+dvdy*dzdy)*dvdxhx/fl3;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != -1 && j != ny-1) {
         ind = k+nx;
         val = dzdyhy/fl - (dvdx*dzdx+dvdy*dzdy)*dvdyhy/fl3;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
     }
   }

  /* Compute f''(x)*s over the upper triangular elements */
  for (j=0; j<=ny; j++) {
    for (i=0; i<=nx; i++) {
       k = nx*j + i;
       if (i != nx && j != 0) {
         vb = x[k-nx];
         zb = s[k-nx];
       } else {
         if (j == 0) vb = bottom[i+1];
         if (i == nx) vb = right[j];
         zb = zerod;
       }
       if (i != 0 && j != ny) {
         vl = x[k-1];
         zl = s[k-1];
       } else {
         if (j == ny) vl = top[i];
         if (i == 0)    vl = left[j+1];
         zl = zerod;
       }
       if (i != nx && j != ny) {
         v = x[k];
         z = s[k];
       } else {
         if (i == nx) v = right[j+1];
         if (j == ny) v = top[i+1];
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
       if (i != nx && j != ny) {
         ind = k;
         val = (dzdxhx+dzdyhy)/fu - (dvdx*dzdx+dvdy*dzdy)*(dvdxhx+dvdyhy)/fu3;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != 0 && j != ny) {
         ind = k-1;
         val = (dvdx*dzdx+dvdy*dzdy)*dvdxhx/fu3 - dzdxhx/fu;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
       if (i != nx && j != 0) {
         ind = k-nx;
         val = (dvdx*dzdx+dvdy*dzdy)*dvdyhy/fu3 - dzdyhy/fu;
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
       }
    }
  }
  ierr = VecRestoreArray(svec,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->xvec,&x);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  /* Scale result by area */
  area = p5*hx*hy;
  ierr = VecScale(&area,y);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "BoundaryValues"
/* 
   BoundaryValues - For Minimal Surface Area problem.  Computes Enneper's 
   boundary conditions (bottom, top, left, right) which are obtained by 
   defining:
     bv(x,y) = u**2 - v**2, where u and v are the unique solutions of
     x = u + u*(v**2) - (u**3)/3, y = -v - (u**2)*v + (v**3)/3. 
 */
int BoundaryValues(AppCtx *user)
{
  int    maxit=5, i, j, k, limit=0, nx = user->mx, ny = user->my;
  double three=3.0, tol=1.0e-10;
  Scalar one=1.0, two=2.0;
  Scalar b=-.50, t=.50, l=-.50, r=.50, det, fnorm, xt=0.0, yt=0.0;
  Scalar nf[2], njac[2][2], u[2], hx = user->hx, hy = user->hy;
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
        SETERRQ(1,0,"BoundaryValues: Only cases 0,1,2,3 are valid");
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
#if !defined(PETSC_USE_COMPLEX)
        if (fnorm <= tol) break;
#else
        if (PetscReal(fnorm) <= tol) break;
#endif
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
          SETERRQ(1,0,"Only cases 0,1,2,3 are valid");
      }
    }
  }
  return 0;
}


