/* Program usage: mpiexec -n 1 eptorsion1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------

  Elastic-plastic torsion problem.

  The elastic plastic torsion problem arises from the determination
  of the stress field on an infinitely long cylindrical bar, which is
  equivalent to the solution of the following problem:

  min{ .5 * integral(||gradient(v(x))||^2 dx) - C * integral(v(x) dx)}

  where C is the torsion angle per unit length.

  The multiprocessor version of this code is eptorsion2.c.

---------------------------------------------------------------------- */

/*
  Include "petsctao.h" so that we can use TAO solvers.  Note that this
  file automatically includes files for lower-level support, such as those
  provided by the PETSc library:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/

#include <petsctao.h>


static  char help[]=
"Demonstrates use of the TAO package to solve \n\
unconstrained minimization problems on a single processor.  This example \n\
is based on the Elastic-Plastic Torsion (dept) problem from the MINPACK-2 \n\
test suite.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -par <param>, where <param> = angle of twist per unit length\n\n";

/*T
   Concepts: TAO^Solving an unconstrained minimization problem
   Routines: TaoCreate(); TaoSetType();
   Routines: TaoSetInitialVector();
   Routines: TaoSetObjectiveAndGradientRoutine();
   Routines: TaoSetHessianRoutine(); TaoSetFromOptions();
   Routines: TaoGetKSP(); TaoSolve();
   Routines: TaoDestroy();
   Processors: 1
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunction(),
   FormGradient(), and FormHessian().
*/

typedef struct {
   PetscReal  param;      /* nonlinearity parameter */
   PetscInt   mx, my;     /* discretization in x- and y-directions */
   PetscInt   ndim;       /* problem dimension */
   Vec        s, y, xvec; /* work space for computing Hessian */
   PetscReal  hx, hy;     /* mesh spacing in x- and y-directions */
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode FormInitialGuess(AppCtx*,Vec);
PetscErrorCode FormFunction(Tao,Vec,PetscReal*,void*);
PetscErrorCode FormGradient(Tao,Vec,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode HessianProductMat(Mat,Vec,Vec);
PetscErrorCode HessianProduct(void*,Vec,Vec);
PetscErrorCode MatrixFreeHessian(Tao,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal *,Vec,void *);

PetscErrorCode main(int argc,char **argv)
{
  PetscErrorCode     ierr;                /* used to check for functions returning nonzeros */
  PetscInt           mx=10;               /* discretization in x-direction */
  PetscInt           my=10;               /* discretization in y-direction */
  Vec                x;                   /* solution, gradient vectors */
  PetscBool          flg;                 /* A return value when checking for use options */
  Tao                tao;                 /* Tao solver context */
  Mat                H;                   /* Hessian matrix */
  AppCtx             user;                /* application context */
  PetscMPIInt        size;                /* number of processes */
  PetscReal          one=1.0;

  /* Initialize TAO,PETSc */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size >1) SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");

  /* Specify default parameters for the problem, check for command-line overrides */
  user.param = 5.0;
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-par",&user.param,&flg);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n---- Elastic-Plastic Torsion Problem -----\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"mx: %D     my: %D   \n\n",mx,my); CHKERRQ(ierr);
  user.ndim = mx * my; user.mx = mx; user.my = my;
  user.hx = one/(mx+1); user.hy = one/(my+1);

  /* Allocate vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.ndim,&user.y);CHKERRQ(ierr);
  ierr = VecDuplicate(user.y,&user.s);CHKERRQ(ierr);
  ierr = VecDuplicate(user.y,&x);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);

  /* From command line options, determine if using matrix-free hessian */
  ierr = PetscOptionsHasName(NULL,NULL,"-my_tao_mf",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCreateShell(PETSC_COMM_SELF,user.ndim,user.ndim,user.ndim,user.ndim,(void*)&user,&H);CHKERRQ(ierr);
    ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)(void))HessianProductMat);CHKERRQ(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

    ierr = TaoSetHessianRoutine(tao,H,H,MatrixFreeHessian,(void *)&user);CHKERRQ(ierr);

    /* Set null preconditioner.  Alternatively, set user-provided
       preconditioner or explicitly form preconditioning matrix */
    ierr = PetscOptionsSetValue(NULL,"-pc_type","none");CHKERRQ(ierr);
  } else {
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user.ndim,user.ndim,5,NULL,&H);CHKERRQ(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,(void *)&user);CHKERRQ(ierr);
  }


  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&user.s);CHKERRQ(ierr);
  ierr = VecDestroy(&user.y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */
/*
    FormInitialGuess - Computes an initial approximation to the solution.

    Input Parameters:
.   user - user-defined application context
.   X    - vector

    Output Parameters:
.   X    - vector
*/
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscReal      hx = user->hx, hy = user->hy, temp;
  PetscReal      val;
  PetscErrorCode ierr;
  PetscInt       i, j, k, nx = user->mx, ny = user->my;

  /* Compute initial guess */
  for (j=0; j<ny; j++) {
    temp = PetscMin(j+1,ny-j)*hy;
    for (i=0; i<nx; i++) {
      k   = nx*j + i;
      val = PetscMin((PetscMin(i+1,nx-i))*hx,temp);
      ierr = VecSetValues(X,1,&k,&val,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f,Vec G,void *ptr)
{
  PetscErrorCode ierr;

  ierr = FormFunction(tao,X,f,ptr);CHKERRQ(ierr);
  ierr = FormGradient(tao,X,G,ptr);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates the function, f(X).

   Input Parameters:
.  tao - the Tao context
.  X   - the input vector
.  ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
.  f    - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;
  PetscReal      hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5;
  PetscReal      zero = 0.0, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  PetscReal      v, cdiv3 = user->param/three;
  PetscReal      *x;
  PetscErrorCode ierr;
  PetscInt       nx = user->mx, ny = user->my, i, j, k;

  /* Get pointer to vector data */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /* Compute function contributions over the lower triangular elements */
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
      fquad += dvdx*dvdx + dvdy*dvdy;
      flin -= cdiv3*(v+vr+vt);
    }
  }

  /* Compute function contributions over the upper triangular elements */
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
      fquad = fquad + dvdx*dvdx + dvdy*dvdy;
      flin = flin - cdiv3*(vb+vl+v);
    }
  }

  /* Restore vector */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  /* Scale the function */
  area = p5*hx*hy;
  *f = area*(p5*fquad+flin);

  ierr = PetscLogFlops(nx*ny*24);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
    FormGradient - Evaluates the gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context

    Output Parameters:
.   G - vector containing the newly evaluated gradient
*/
PetscErrorCode FormGradient(Tao tao,Vec X,Vec G,void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;
  PetscReal      zero=0.0, p5=0.5, three = 3.0, area, val, *x;
  PetscErrorCode ierr;
  PetscInt       nx = user->mx, ny = user->my, ind, i, j, k;
  PetscReal      hx = user->hx, hy = user->hy;
  PetscReal      vb, vl, vr, vt, dvdx, dvdy;
  PetscReal      v, cdiv3 = user->param/three;

  /* Initialize gradient to zero */
  ierr = VecSet(G, zero);CHKERRQ(ierr);

  /* Get pointer to vector data */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /* Compute gradient contributions over the lower triangular elements */
  for (j=-1; j<ny; j++) {
    for (i=-1; i<nx; i++) {
      k  = nx*j + i;
      v  = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0)    v = x[k];
      if (i < nx-1 && j > -1) vr = x[k+1];
      if (i > -1 && j < ny-1) vt = x[k+nx];
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      if (i != -1 && j != -1) {
        ind = k; val = - dvdx/hx - dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != nx-1 && j != -1) {
        ind = k+1; val =  dvdx/hx - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != -1 && j != ny-1) {
        ind = k+nx; val = dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* Compute gradient contributions over the upper triangular elements */
  for (j=0; j<=ny; j++) {
    for (i=0; i<=nx; i++) {
      k = nx*j + i;
      vb = zero;
      vl = zero;
      v  = zero;
      if (i < nx && j > 0) vb = x[k-nx];
      if (i > 0 && j < ny) vl = x[k-1];
      if (i < nx && j < ny) v = x[k];
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      if (i != nx && j != 0) {
        ind = k-nx; val = - dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != 0 && j != ny) {
        ind = k-1; val =  - dvdx/hx - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != nx && j != ny) {
        ind = k; val =  dvdx/hx + dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  /* Assemble gradient vector */
  ierr = VecAssemblyBegin(G);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(G);CHKERRQ(ierr);

  /* Scale the gradient */
  area = p5*hx*hy;
  ierr = VecScale(G, area);CHKERRQ(ierr);
  ierr = PetscLogFlops(nx*ny*24);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Forms the Hessian matrix.

   Input Parameters:
.  tao - the Tao context
.  X    - the input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix
.  PrecH - optionally different preconditioning Hessian
.  flag  - flag indicating matrix structure

   Notes:
   This routine is intended simply as an example of the interface
   to a Hessian evaluation routine.  Since this example compute the
   Hessian a column at a time, it is not particularly efficient and
   is not recommended.
*/
PetscErrorCode FormHessian(Tao tao,Vec X,Mat H,Mat Hpre, void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;
  PetscErrorCode ierr;
  PetscInt       i,j, ndim = user->ndim;
  PetscReal      *y, zero = 0.0, one = 1.0;
  PetscBool      assembled;

  user->xvec = X;

  /* Initialize Hessian entries and work vector to zero */
  ierr = MatAssembled(H,&assembled);CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(H); CHKERRQ(ierr);}

  ierr = VecSet(user->s, zero);CHKERRQ(ierr);

  /* Loop over matrix columns to compute entries of the Hessian */
  for (j=0; j<ndim; j++) {
    ierr = VecSetValues(user->s,1,&j,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    ierr = HessianProduct(ptr,user->s,user->y);CHKERRQ(ierr);

    ierr = VecSetValues(user->s,1,&j,&zero,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s);CHKERRQ(ierr);

    ierr = VecGetArray(user->y,&y);CHKERRQ(ierr);
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        ierr = MatSetValues(H,1,&i,1,&j,&y[i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(user->y,&y);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   MatrixFreeHessian - Sets a pointer for use in computing Hessian-vector
   products.

   Input Parameters:
.  tao - the Tao context
.  X    - the input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix
.  PrecH - optionally different preconditioning Hessian
.  flag  - flag indicating matrix structure
*/
PetscErrorCode MatrixFreeHessian(Tao tao,Vec X,Mat H,Mat PrecH, void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;

  /* Sets location of vector for use in computing matrix-vector products  of the form H(X)*y  */
  user->xvec = X;
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   HessianProductMat - Computes the matrix-vector product
   y = mat*svec.

   Input Parameters:
.  mat  - input matrix
.  svec - input vector

   Output Parameters:
.  y    - solution vector
*/
PetscErrorCode HessianProductMat(Mat mat,Vec svec,Vec y)
{
  void           *ptr;
  PetscErrorCode ierr;

  ierr = MatShellGetContext(mat,&ptr);CHKERRQ(ierr);
  ierr = HessianProduct(ptr,svec,y);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   Hessian Product - Computes the matrix-vector product:
   y = f''(x)*svec.

   Input Parameters
.  ptr  - pointer to the user-defined context
.  svec - input vector

   Output Parameters:
.  y    - product vector
*/
PetscErrorCode HessianProduct(void *ptr,Vec svec,Vec y)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscReal      p5 = 0.5, zero = 0.0, one = 1.0, hx, hy, val, area, *x, *s;
  PetscReal      v, vb, vl, vr, vt, hxhx, hyhy;
  PetscErrorCode ierr;
  PetscInt       nx, ny, i, j, k, ind;

  nx   = user->mx;
  ny   = user->my;
  hx   = user->hx;
  hy   = user->hy;
  hxhx = one/(hx*hx);
  hyhy = one/(hy*hy);

  /* Get pointers to vector data */
  ierr = VecGetArray(user->xvec,&x);CHKERRQ(ierr);
  ierr = VecGetArray(svec,&s);CHKERRQ(ierr);

  /* Initialize product vector to zero */
  ierr = VecSet(y, zero);CHKERRQ(ierr);

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
  /* Restore vector data */
  ierr = VecRestoreArray(svec,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->xvec,&x);CHKERRQ(ierr);

  /* Assemble vector */
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  /* Scale resulting vector by area */
  area = p5*hx*hy;
  ierr = VecScale(y, area);CHKERRQ(ierr);
  ierr = PetscLogFlops(nx*ny*18);CHKERRQ(ierr);
  return 0;
}


