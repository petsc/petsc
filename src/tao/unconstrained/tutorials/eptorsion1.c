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
     petscsys.h    - system routines        petscmat.h - matrices
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
   Routines: TaoSetSolution();
   Routines: TaoSetObjectiveAndGradient();
   Routines: TaoSetHessian(); TaoSetFromOptions();
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
  PetscInt           mx=10;               /* discretization in x-direction */
  PetscInt           my=10;               /* discretization in y-direction */
  Vec                x;                   /* solution, gradient vectors */
  PetscBool          flg;                 /* A return value when checking for use options */
  Tao                tao;                 /* Tao solver context */
  Mat                H;                   /* Hessian matrix */
  AppCtx             user;                /* application context */
  PetscMPIInt        size;                /* number of processes */
  PetscReal          one=1.0;

  PetscBool          test_lmvm = PETSC_FALSE;
  KSP                ksp;
  PC                 pc;
  Mat                M;
  Vec                in, out, out2;
  PetscReal          mult_solve_dist;

  /* Initialize TAO,PETSc */
  PetscCall(PetscInitialize(&argc,&argv,(char *)0,help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Incorrect number of processors");

  /* Specify default parameters for the problem, check for command-line overrides */
  user.param = 5.0;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-my",&my,&flg));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&flg));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-par",&user.param,&flg));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_lmvm",&test_lmvm,&flg));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n---- Elastic-Plastic Torsion Problem -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"mx: %D     my: %D   \n\n",mx,my));
  user.ndim = mx * my; user.mx = mx; user.my = my;
  user.hx = one/(mx+1); user.hy = one/(my+1);

  /* Allocate vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,user.ndim,&user.y));
  PetscCall(VecDuplicate(user.y,&user.s));
  PetscCall(VecDuplicate(user.y,&x));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF,&tao));
  PetscCall(TaoSetType(tao,TAOLMVM));

  /* Set solution vector with an initial guess */
  PetscCall(FormInitialGuess(&user,x));
  PetscCall(TaoSetSolution(tao,x));

  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

  /* From command line options, determine if using matrix-free hessian */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-my_tao_mf",&flg));
  if (flg) {
    PetscCall(MatCreateShell(PETSC_COMM_SELF,user.ndim,user.ndim,user.ndim,user.ndim,(void*)&user,&H));
    PetscCall(MatShellSetOperation(H,MATOP_MULT,(void(*)(void))HessianProductMat));
    PetscCall(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE));

    PetscCall(TaoSetHessian(tao,H,H,MatrixFreeHessian,(void *)&user));
  } else {
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,user.ndim,user.ndim,5,NULL,&H));
    PetscCall(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(TaoSetHessian(tao,H,H,FormHessian,(void *)&user));
  }

  /* Test the LMVM matrix */
  if (test_lmvm) {
    PetscCall(PetscOptionsSetValue(NULL, "-tao_type", "bntr"));
    PetscCall(PetscOptionsSetValue(NULL, "-tao_bnk_pc_type", "lmvm"));
  }

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Test the LMVM matrix */
  if (test_lmvm) {
    PetscCall(TaoGetKSP(tao, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCLMVMGetMatLMVM(pc, &M));
    PetscCall(VecDuplicate(x, &in));
    PetscCall(VecDuplicate(x, &out));
    PetscCall(VecDuplicate(x, &out2));
    PetscCall(VecSet(in, 5.0));
    PetscCall(MatMult(M, in, out));
    PetscCall(MatSolve(M, out, out2));
    PetscCall(VecAXPY(out2, -1.0, in));
    PetscCall(VecNorm(out2, NORM_2, &mult_solve_dist));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between MatMult and MatSolve: %e\n", mult_solve_dist));
    PetscCall(VecDestroy(&in));
    PetscCall(VecDestroy(&out));
    PetscCall(VecDestroy(&out2));
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&user.s));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));

  PetscCall(PetscFinalize());
  return 0;
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
  PetscInt       i, j, k, nx = user->mx, ny = user->my;

  /* Compute initial guess */
  PetscFunctionBeginUser;
  for (j=0; j<ny; j++) {
    temp = PetscMin(j+1,ny-j)*hy;
    for (i=0; i<nx; i++) {
      k   = nx*j + i;
      val = PetscMin((PetscMin(i+1,nx-i))*hx,temp);
      PetscCall(VecSetValues(X,1,&k,&val,ADD_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscFunctionReturn(0);
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
  PetscFunctionBeginUser;
  PetscCall(FormFunction(tao,X,f,ptr));
  PetscCall(FormGradient(tao,X,G,ptr));
  PetscFunctionReturn(0);
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
  AppCtx            *user = (AppCtx *) ptr;
  PetscReal         hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5;
  PetscReal         zero = 0.0, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  PetscReal         v, cdiv3 = user->param/three;
  const PetscScalar *x;
  PetscInt          nx = user->mx, ny = user->my, i, j, k;

  PetscFunctionBeginUser;
  /* Get pointer to vector data */
  PetscCall(VecGetArrayRead(X,&x));

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
  PetscCall(VecRestoreArrayRead(X,&x));

  /* Scale the function */
  area = p5*hx*hy;
  *f = area*(p5*fquad+flin);

  PetscCall(PetscLogFlops(24.0*nx*ny));
  PetscFunctionReturn(0);
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
  AppCtx            *user = (AppCtx *) ptr;
  PetscReal         zero=0.0, p5=0.5, three = 3.0, area, val;
  PetscInt          nx = user->mx, ny = user->my, ind, i, j, k;
  PetscReal         hx = user->hx, hy = user->hy;
  PetscReal         vb, vl, vr, vt, dvdx, dvdy;
  PetscReal         v, cdiv3 = user->param/three;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Initialize gradient to zero */
  PetscCall(VecSet(G, zero));

  /* Get pointer to vector data */
  PetscCall(VecGetArrayRead(X,&x));

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
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
      }
      if (i != nx-1 && j != -1) {
        ind = k+1; val =  dvdx/hx - cdiv3;
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
      }
      if (i != -1 && j != ny-1) {
        ind = k+nx; val = dvdy/hy - cdiv3;
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
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
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
      }
      if (i != 0 && j != ny) {
        ind = k-1; val =  - dvdx/hx - cdiv3;
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
      }
      if (i != nx && j != ny) {
        ind = k; val =  dvdx/hx + dvdy/hy - cdiv3;
        PetscCall(VecSetValues(G,1,&ind,&val,ADD_VALUES));
      }
    }
  }
  PetscCall(VecRestoreArrayRead(X,&x));

  /* Assemble gradient vector */
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));

  /* Scale the gradient */
  area = p5*hx*hy;
  PetscCall(VecScale(G, area));
  PetscCall(PetscLogFlops(24.0*nx*ny));
  PetscFunctionReturn(0);
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
  PetscInt       i,j, ndim = user->ndim;
  PetscReal      *y, zero = 0.0, one = 1.0;
  PetscBool      assembled;

  PetscFunctionBeginUser;
  user->xvec = X;

  /* Initialize Hessian entries and work vector to zero */
  PetscCall(MatAssembled(H,&assembled));
  if (assembled)PetscCall(MatZeroEntries(H));

  PetscCall(VecSet(user->s, zero));

  /* Loop over matrix columns to compute entries of the Hessian */
  for (j=0; j<ndim; j++) {
    PetscCall(VecSetValues(user->s,1,&j,&one,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(user->s));
    PetscCall(VecAssemblyEnd(user->s));

    PetscCall(HessianProduct(ptr,user->s,user->y));

    PetscCall(VecSetValues(user->s,1,&j,&zero,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(user->s));
    PetscCall(VecAssemblyEnd(user->s));

    PetscCall(VecGetArray(user->y,&y));
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        PetscCall(MatSetValues(H,1,&i,1,&j,&y[i],ADD_VALUES));
      }
    }
    PetscCall(VecRestoreArray(user->y,&y));
  }
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
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
  AppCtx *user = (AppCtx *) ptr;

  /* Sets location of vector for use in computing matrix-vector products  of the form H(X)*y  */
  PetscFunctionBeginUser;
  user->xvec = X;
  PetscFunctionReturn(0);
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

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat,&ptr));
  PetscCall(HessianProduct(ptr,svec,y));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   Hessian Product - Computes the matrix-vector product:
   y = f''(x)*svec.

   Input Parameters:
.  ptr  - pointer to the user-defined context
.  svec - input vector

   Output Parameters:
.  y    - product vector
*/
PetscErrorCode HessianProduct(void *ptr,Vec svec,Vec y)
{
  AppCtx            *user = (AppCtx *)ptr;
  PetscReal         p5 = 0.5, zero = 0.0, one = 1.0, hx, hy, val, area;
  const PetscScalar *x, *s;
  PetscReal         v, vb, vl, vr, vt, hxhx, hyhy;
  PetscInt          nx, ny, i, j, k, ind;

  PetscFunctionBeginUser;
  nx   = user->mx;
  ny   = user->my;
  hx   = user->hx;
  hy   = user->hy;
  hxhx = one/(hx*hx);
  hyhy = one/(hy*hy);

  /* Get pointers to vector data */
  PetscCall(VecGetArrayRead(user->xvec,&x));
  PetscCall(VecGetArrayRead(svec,&s));

  /* Initialize product vector to zero */
  PetscCall(VecSet(y, zero));

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
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
       }
       if (i != -1 && j != ny-1) {
         vt = s[k+nx];
         ind = k+nx; val = hyhy*(vt-v);
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
       }
       if (i != -1 && j != -1) {
         ind = k; val = hxhx*(v-vr) + hyhy*(v-vt);
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
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
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
       }
       if (i != 0 && j != ny) {
         vl = s[k-1];
         ind = k-1; val = hxhx*(vl-v);
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
       }
       if (i != nx && j != ny) {
         ind = k; val = hxhx*(v-vl) + hyhy*(v-vb);
         PetscCall(VecSetValues(y,1,&ind,&val,ADD_VALUES));
       }
    }
  }
  /* Restore vector data */
  PetscCall(VecRestoreArrayRead(svec,&s));
  PetscCall(VecRestoreArrayRead(user->xvec,&x));

  /* Assemble vector */
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  /* Scale resulting vector by area */
  area = p5*hx*hy;
  PetscCall(VecScale(y, area));
  PetscCall(PetscLogFlops(18.0*nx*ny));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      args: -tao_smonitor -tao_type ntl -tao_gatol 1.e-4

   test:
      suffix: 2
      args: -tao_smonitor -tao_type ntr -tao_gatol 1.e-4

   test:
      suffix: 3
      args: -tao_smonitor -tao_type bntr -tao_gatol 1.e-4 -my_tao_mf -tao_test_hessian

   test:
     suffix: 4
     args: -tao_smonitor -tao_gatol 1e-3 -tao_type bqnls

   test:
     suffix: 5
     args: -tao_smonitor -tao_gatol 1e-3 -tao_type blmvm

   test:
     suffix: 6
     args: -tao_smonitor -tao_gatol 1e-3 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1

TEST*/
