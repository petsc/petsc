#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.54 1999/07/08 14:44:30 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates use of the SNES package to solve unconstrained\n\
minimization problems in parallel.  This example is based on the\n\
Elastic-Plastic Torsion (dept) problem from the MINPACK-2 test suite.\n\
The command line options are:\n\
  -mx xg, where xg = number of grid points in the 1st coordinate direction\n\
  -my yg, where yg = number of grid points in the 2nd coordinate direction\n\
  -par param, where param = angle of twist per unit length\n\
  -snes_mf: use matrix-free methods\n\
  -defaultH: use default finite difference approximation of Hessian\n\n";

#include "snes.h"
#include "da.h"

/* User-defined application context */
   typedef struct {
      double  param;          /* nonlinearity parameter */
      int     mx;             /* discretization in x-direction */
      int     my;             /* discretization in y-direction */
      int     ndim;           /* problem dimension */
      int     number;         /* test problem number */
      Vec     s,y,xvec;       /* work space for computing Hessian */
      Scalar  hx, hy;    
      Vec     localX, localS; /* ghosted local vector */
      DA      da;             /* distributed array data structure */
   } AppCtx;

/* Flag to indicate evaluation of function and/or gradient */
typedef enum {FunctionEval=1, GradientEval=2} FctGradFlag;

extern int FormHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int MatrixFreeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int FormMinimizationFunction(SNES,Vec,double*,void*);
extern int FormGradient(SNES,Vec,Vec,void*);
extern int HessianProduct(void *,Vec,Vec);
extern int HessianProductMat(Mat,Vec,Vec);
extern int FormInitialGuess(AppCtx*,Vec);
extern int EvalFunctionGradient(SNES,Vec,double*,Vec,FctGradFlag,AppCtx*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  SNES       snes;                 /* SNES context */
  SNESType   type = SNES_UM_TR;  /* nonlinear solution method */
  Vec        x, g;                 /* solution, gradient vectors */
  Mat        H;                    /* Hessian matrix */
  AppCtx     user;                 /* application context */
  int        mx=10;                /* discretization in x-direction */
  int        my=10;                /* discretization in y-direction */
  int        Nx=PETSC_DECIDE;      /* processors in x-direction */
  int        Ny=PETSC_DECIDE;      /* processors in y-direction */
  int        ierr, its, ldim, nfails, size,flg;
  double     one = 1.0;
  SLES       sles;
  PC         pc;

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);CHKERRA(ierr);
  if (Nx*Ny != size && (Nx != PETSC_DECIDE && Ny != PETSC_DECIDE))
    SETERRQ(1,0,"Incompatible number of processors:  Nx * Ny != size");

  /* Set up user-defined work space */
  user.param = 5.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,&flg);CHKERRA(ierr);
  user.ndim = mx * my; user.mx = mx; user.my = my;
  user.hx = one/(mx+1); user.hy = one/(my+1);

  /* Set up distributed array and vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,user.mx,
         user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(user.da,&x);CHKERRA(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX);CHKERRA(ierr);
  ierr = VecDuplicate(x,&user.s);CHKERRA(ierr);
  ierr = VecDuplicate(user.localX,&user.localS);CHKERRA(ierr);
  ierr = VecDuplicate(x,&g);CHKERRA(ierr);
  ierr = VecDuplicate(g,&user.y);CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_UNCONSTRAINED_MINIMIZATION,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,type);CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetMinimizationFunction(snes,FormMinimizationFunction,
         (void *)&user);CHKERRA(ierr);
  ierr = SNESSetGradient(snes,g,FormGradient,(void *)&user);CHKERRA(ierr);

  /* Either explicitly form Hessian matrix approx or use matrix-free version */
  ierr = OptionsHasName(PETSC_NULL,"-snes_mf",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = VecGetLocalSize(x,&ldim);CHKERRA(ierr);
    ierr = MatCreateShell(PETSC_COMM_WORLD,ldim,user.ndim,user.ndim,user.ndim,
           (void*)&user,&H);CHKERRA(ierr);
    ierr = MatShellSetOperation(H,MATOP_MULT,(void*)HessianProductMat);CHKERRA(ierr);
    ierr = SNESSetHessian(snes,H,H,MatrixFreeHessian,(void *)&user);CHKERRA(ierr);

    /* Set null preconditioner.  Alternatively, set user-provided 
       preconditioner or explicitly form preconditioning matrix */
    ierr = SNESGetSLES(snes,&sles);CHKERRA(ierr);
    ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRA(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,user.ndim,user.ndim,&H);CHKERRA(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC);CHKERRA(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-defaultH",&flg);CHKERRA(ierr);
    if (flg) ierr = SNESSetHessian(snes,H,H,SNESDefaultComputeHessian,(void *)&user);
    else     ierr = SNESSetHessian(snes,H,H,FormHessian,(void *)&user);CHKERRA(ierr);
  }

  /* Set options; then solve minimization problem */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);
  ierr = FormInitialGuess(&user,x);CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  ierr = SNESGetNumberUnsuccessfulSteps(snes,&nfails);CHKERRA(ierr);
  ierr = SNESView(snes,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"number of Newton iterations = %d, ",its);
  PetscPrintf(PETSC_COMM_WORLD,"number of unsuccessful steps = %d\n\n",nfails);

  /* Free data structures */
  ierr = VecDestroy(user.s);CHKERRA(ierr);
  ierr = VecDestroy(user.localX);CHKERRA(ierr);
  ierr = VecDestroy(user.localS);CHKERRA(ierr);
  ierr = VecDestroy(user.y);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(g);CHKERRA(ierr);
  ierr = MatDestroy(H);CHKERRA(ierr); 
  ierr = SNESDestroy(snes);CHKERRA(ierr); 
  ierr = DADestroy(user.da);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormMinimizationFunction"
/*
    FormMinimizationFunction - Evaluates function f(x).
*/
int FormMinimizationFunction(SNES snes,Vec x,double *f,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  return EvalFunctionGradient(snes,x,f,NULL,FunctionEval,user); 
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
  return EvalFunctionGradient(snes,x,NULL,g,GradientEval,user); 
}
/* -------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormHessian"
/*
   FormHessian - Forms Hessian matrix by computing a column at a time.
 */
int FormHessian(SNES snes,Vec X,Mat *H,Mat *PrecH,MatStructure *flag,
                void *ptr)
{
  AppCtx   *user = (AppCtx *) ptr;
  int      i, j, ierr, ndim, xs, ys,  xm, ym, rstart, rend, ldim, iglob;
  Scalar   *y, zero = 0.0, one = 1.0;

  ierr = MatZeroEntries(*H);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ndim = user->ndim;
  ierr = VecSet(&zero,user->s);CHKERRQ(ierr);
  user->xvec = X; /* Set location of vector */
  ierr = VecGetOwnershipRange(user->y,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->y,&ldim);CHKERRQ(ierr);

  for (j=0; j<ndim; j++) {   /* loop over columns */

    ierr = VecSetValues(user->s,1,&j,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    ierr = HessianProduct(ptr,user->s,user->y);CHKERRQ(ierr);

    ierr = VecSetValues(user->s,1,&j,&zero,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    ierr = VecGetArray(user->y,&y);CHKERRQ(ierr);
    for (i=0; i<ldim; i++) {
      if (y[i] != zero) {
        iglob = i+rstart;
        ierr = MatSetValues(*H,1,&iglob,1,&j,&y[i],ADD_VALUES);CHKERRQ(ierr);
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
#define __FUNC__ "MatrixFreeHessian"
/*
  MatrixFreeHessian
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

/* ------------------------------------------------------------------ */
/*               Elastic-Plastic Torsion Test Problem                 */
/* ------------------------------------------------------------------ */

/* --------------------  Form initial approximation ----------------- */

#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(AppCtx *user,Vec X)
{
  int    ierr, i, j, k, nx = user->mx, ny = user->my;
  Scalar hx = user->hx, hy = user->hy, temp, *x;
  int    xs, ys, xm, ym, Xm, Ym, Xs, Ys, xe, ye;

  /* Get local vector (including ghost points) */
  ierr = VecGetArray(user->localX,&x);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);

  xe = xs+xm;
  ye = ys+ym;
  for (j=ys; j<ye; j++) {  /*  for (j=0; j<ny; j++) */
    temp = (double)PetscMin(j+1,ny-j)*hy;
    for (i=xs; i<xe; i++) {  /*  for (i=0; i<nx; i++) */
      k = (j-Ys)*Xm + i-Xs;
#if !defined(PETSC_USE_COMPLEX)
      x[k] = PetscMin((PetscMin(i+1,nx-i))*hx,temp);
#else
      x[k] = PetscMin(PetscReal((double)(PetscMin(i+1,nx-i))*hx),PetscReal(temp));
#endif
    }
  }
  ierr = VecRestoreArray(user->localX,&x);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal(user->da,user->localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
}
/* ---------- Evaluate function f(x) and/or gradient g(x) ----------- */

#undef __FUNC__
#define __FUNC__ "EvalFunctionGradient"
int EvalFunctionGradient(SNES snes,Vec X,double *f,Vec gvec,FctGradFlag fg,
                         AppCtx *user)
{
  Scalar hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5, cdiv3;
  Scalar zero = 0.0, v, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  Scalar val, *x, szero = 0.0, floc;
  Vec    localX = user->localX;
  int    xs, ys, xm, ym, Xm, Ym, Xs, Ys, xe, ye, xsm, ysm, xep, yep;
  int    ierr, nx = user->mx, ny = user->my, ind, i, j, k, *ltog, nloc; 

  cdiv3 = user->param/three;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);
  xe = xs+xm;
  ye = ys+ym;
  if (xs == 0)  xsm = xs-1;
  else          xsm = xs;
  if (ys == 0)  ysm = ys-1;
  else          ysm = ys;
  if (xe == nx) xep = xe+1;
  else          xep = xe;
  if (ye == ny) yep = ye+1;
  else          yep = ye;

  if (fg & GradientEval) {
    ierr = VecSet(&szero,gvec);CHKERRQ(ierr);
  }

  /* Compute function and gradient over the lower triangular elements */
  for (j=ysm; j<ye; j++) {  /*  for (j=-1; j<ny; j++) */
    for (i=xsm; i<xe; i++) {  /*   for (i=-1; i<nx; i++) */
      k = (j-Ys)*Xm + i-Xs;
      v = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k];
      if (i < nx-1 && j > -1) vr = x[k+1];
      if (i > -1 && j < ny-1) vt = x[k+Xm];
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      if (fg & FunctionEval) {
        fquad += dvdx*dvdx + dvdy*dvdy;
        flin -= cdiv3*(v+vr+vt);
      }
      if (fg & GradientEval) {
        if (i != -1 && j != -1) {
          ind = ltog[k]; val = - dvdx/hx - dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != nx-1 && j != -1) {
          ind = ltog[k+1]; val =  dvdx/hx - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != -1 && j != ny-1) {
          ind = ltog[k+Xm]; val = dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Compute function and gradient over the upper triangular elements */
  for (j=ys; j<yep; j++) { /*  for (j=0; j<=ny; j++) */
    for (i=xs; i<xep; i++) {  /*   for (i=0; i<=nx; i++) */
      k = (j-Ys)*Xm + i-Xs;
      vb = zero;
      vl = zero;
      v = zero;
      if (i < nx && j > 0) vb = x[k-Xm];
      if (i > 0 && j < ny) vl = x[k-1];
      if (i < nx && j < ny) v = x[k];
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      if (fg & FunctionEval) {
        fquad = fquad + dvdx*dvdx + dvdy*dvdy;
        flin = flin - cdiv3*(vb+vl+v);
      } if (fg & GradientEval) {
        if (i != nx && j != 0) {
          ind = ltog[k-Xm]; val = - dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != 0 && j != ny) {
          ind = ltog[k-1]; val =  - dvdx/hx - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
        if (i != nx && j != ny) {
          ind = ltog[k]; val =  dvdx/hx + dvdy/hy - cdiv3;
          ierr = VecSetValues(gvec,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  area = p5*hx*hy;
  if (fg & FunctionEval) {   /* Scale the function */
#if !defined(PETSC_USE_COMPLEX)
    floc = area*(p5*fquad+flin);
#else
    floc = PetscReal(area*(p5*fquad+flin));
#endif
    MPI_Allreduce((void*)&floc,(void*)f,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
  } if (fg & GradientEval) { /* Scale the gradient */
    ierr = VecAssemblyBegin(gvec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gvec);CHKERRQ(ierr);
    ierr = VecScale((Scalar*)&area,gvec);CHKERRQ(ierr);
  }
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "HessianProductMat"
int HessianProductMat(Mat mat,Vec svec,Vec y)
{
  void *ptr;
  MatShellGetContext(mat,&ptr);
  HessianProduct(ptr,svec,y);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "HessianProduct"
/* 
   HessianProduct - Computes the matrix-vector product: y = f''(x)*s
 */
int HessianProduct(void *ptr,Vec svec,Vec y)
{
  AppCtx *user = (AppCtx *) ptr;
  Scalar p5 = 0.5, one = 1.0, zero = 0.0, hx, hy;
  Scalar val, area, *x, *s, szero = 0.0, v, vb, vl, vr, vt, hxhx, hyhy;
  Vec    localX, localS;
  int    xs, ys, xm, ym, Xm, Ym, Xs, Ys, xe, ye, xsm, ysm, xep, yep;
  int    nx, ny, i, j, k, ierr, ind, nloc, *ltog;

  hx = user->hx;
  hy = user->hy;
  localX = user->localX;
  localS = user->localS;
  nx = user->mx;
  ny = user->my;

  hxhx = one/(hx*hx);
  hyhy = one/(hy*hy);

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(user->da,user->xvec,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,user->xvec,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(user->da,svec,INSERT_VALUES,localS);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,svec,INSERT_VALUES,localS);CHKERRQ(ierr);
  ierr = VecGetArray(localS,&s);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);
  xe = xs+xm;
  ye = ys+ym;
  if (xs == 0)  xsm = xs-1;
  else          xsm = xs;
  if (ys == 0)  ysm = ys-1;
  else          ysm = ys;
  if (xe == nx) xep = xe+1;
  else          xep = xe;
  if (ye == ny) yep = ye+1;
  else          yep = ye;

  ierr = VecSet(&szero,y);CHKERRQ(ierr);

  /* Compute f''(x)*s over the lower triangular elements */
  for (j=ysm; j<ye; j++) {  /*  for (j=-1; j<ny; j++) */
    for (i=xsm; i<xe; i++) {  /*   for (i=-1; i<nx; i++) */
      k = (j-Ys)*Xm + i-Xs;
      v = zero;
      vr = zero;
      vt = zero;
      if (i != -1 && j != -1) v = s[k];
      if (i != nx-1 && j != -1) {
        vr = s[k+1];
        ind = ltog[k+1]; val = hxhx*(vr-v);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != -1 && j != ny-1) {
        vt = s[k+Xm];
        ind = ltog[k+Xm]; val = hyhy*(vt-v);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != -1 && j != -1) {
        ind = ltog[k]; val = hxhx*(v-vr) + hyhy*(v-vt);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* Compute f''(x)*s over the upper triangular elements */
  for (j=ys; j<yep; j++) { /*  for (j=0; j<=ny; j++) */
    for (i=xs; i<xep; i++) {  /*   for (i=0; i<=nx; i++) */
      k = (j-Ys)*Xm + i-Xs;
      v = zero;
      vl = zero;
      vb = zero;
      if (i != nx && j != ny) v = s[k];
      if (i != nx && j != 0) {
        vb = s[k-Xm];
        ind = ltog[k-Xm]; val = hyhy*(vb-v);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != 0 && j != ny) {
        vl = s[k-1];
        ind = ltog[k-1]; val = hxhx*(vl-v);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != nx && j != ny) {
        ind = ltog[k]; val = hxhx*(v-vl) + hyhy*(v-vb);
        ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localS,&x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  /* Scale result by area */
  area = p5*hx*hy;
  ierr = VecScale(&area,y);CHKERRQ(ierr);
  return 0;
}



