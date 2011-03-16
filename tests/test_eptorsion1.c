/* Program usage: mpirun -np 1 eptorsion1 [-help] [all TAO options] */

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
  Include "taosolver.h" so that we can use TAO solvers.  Note that this 
  file automatically includes files for lower-level support, such as those
  provided by the PETSc library:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/

#include "taosolver.h"


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
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoInitialize(); TaoFinalize(); 
   Routines: TaoApplicationCreate(); TaoAppDestroy();
   Routines: TaoCreate(); TaoDestroy(); 
   Routines: TaoAppSetObjectiveAndGradientRoutine();
   Routines: TaoAppSetHessianMat(); TaoAppSetHessianRoutine();
   Routines: TaoSetOptions();
   Routines: TaoAppSetInitialSolutionVec();
   Routines: TaoSolveApplication();
   Routines: TaoGetSolutionStatus(); TaoAppGetKSP();
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
   Vec     s, y, xvec; /* work space for computing Hessian */
   PetscReal  hx, hy;     /* mesh spacing in x- and y-directions */
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode FormInitialGuess(AppCtx*,Vec);
PetscErrorCode FormFunction(TaoSolver,Vec,double*,void*);
PetscErrorCode FormGradient(TaoSolver,Vec,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*, MatStructure *,void*);
PetscErrorCode HessianProductMat(Mat,Vec,Vec);
PetscErrorCode HessianProduct(void*,Vec,Vec);
PetscErrorCode MatrixFreeHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,double *,Vec,void *);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc,char **argv)
{
  PetscErrorCode ierr;                /* used to check for functions returning nonzeros */
  PetscInt      mx=10;               /* discretization in x-direction */
  PetscInt      my=10;               /* discretization in y-direction */
  Vec         x;                   /* solution, gradient vectors */
  PetscBool   flg;                 /* A return value when checking for use options */
  TaoSolver   tao;                 /* TAO_SOLVER solver context */
  Mat         H;                   /* Hessian matrix */
  TaoSolverTerminationReason reason;        
  KSP         ksp;                 /* PETSc Krylov subspace solver */
  AppCtx      user;                /* application context */
  PetscMPIInt size;                /* number of processes */
  PetscScalar one=1.0;


  /* Initialize TAO,PETSc */
  PetscInitialize(&argc,&argv,(char *)0,help);
  TaoInitialize(&argc,&argv,(char *)0,help);

  /* Optional:  Check  that only one processor is being used. */
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size); CHKERRQ(ierr);
  if (size >1) {
    PetscPrintf(PETSC_COMM_SELF,"This example is intended for single processor use!\n");
    PetscPrintf(PETSC_COMM_SELF,"Try the example eptorsion2!\n");
    SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");
  }

  /* Specify default parameters for the problem, check for command-line overrides */
  user.param = 5.0;
  ierr = PetscOptionsGetInt(TAO_NULL,"-my",&my,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(TAO_NULL,"-mx",&mx,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(TAO_NULL,"-par",&user.param,&flg); CHKERRQ(ierr);


  PetscPrintf(PETSC_COMM_SELF,"\n---- Elastic-Plastic Torsion Problem -----\n");
  PetscPrintf(PETSC_COMM_SELF,"mx: %d     my: %d   \n\n",mx,my);  
  user.ndim = mx * my; user.mx = mx; user.my = my;

  user.hx = one/(mx+1); user.hy = one/(my+1);


  /* Allocate vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.ndim,&user.y); CHKERRQ(ierr);
  ierr = VecDuplicate(user.y,&user.s); CHKERRQ(ierr);
  ierr = VecDuplicate(user.y,&x); CHKERRQ(ierr);

  /* The TAO code begins here */

  /* Create TAO solver and set desired solution method */
  ierr = TaoSolverCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSolverSetType(tao,"tao_lmvm"); CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = FormInitialGuess(&user,x); CHKERRQ(ierr);
  ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSolverSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user); CHKERRQ(ierr);

  /* From command line options, determine if using matrix-free hessian */
  ierr = PetscOptionsHasName(TAO_NULL,"-my_tao_mf",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatCreateShell(PETSC_COMM_SELF,user.ndim,user.ndim,user.ndim,
                          user.ndim,(void*)&user,&H); CHKERRQ(ierr);
    ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)())HessianProductMat); CHKERRQ
(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

    ierr = TaoSolverSetHessianRoutine(tao,H,H,MatrixFreeHessian,(void *)&user); CHKERRQ(ierr);


    /* Set null preconditioner.  Alternatively, set user-provided 
       preconditioner or explicitly form preconditioning matrix */
    ierr = PetscOptionsSetValue("-tao_pc_type","none"); CHKERRQ(ierr);

  } else {

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user.ndim,user.ndim,5,TAO_NULL,&H); CHKERRQ(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

    ierr = TaoSolverSetHessianRoutine(tao,H,H,FormHessian,(void *)&user); CHKERRQ(ierr);

  }



  /* Modify the PETSc KSP structure */
  ierr = PetscOptionsSetValue("-tao_ksp_type","cg"); CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);


  /* SOLVE THE APPLICATION */
  ierr = TaoSolverSolve(tao);  CHKERRQ(ierr);
  ierr = TaoSolverGetKSP(tao,&ksp); CHKERRQ(ierr);
  if (ksp) {
    KSPView(ksp,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }

  /* 
     To View TAO solver information use
      ierr = TaoView(tao); CHKERRQ(ierr);
  */

  /* Get information on termination */
  ierr = TaoSolverGetConvergedReason(tao,&reason); CHKERRQ(ierr);
  if (reason <= 0){
    PetscPrintf(PETSC_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");
  }

  /* Free TAO data structures */
  ierr = TaoSolverDestroy(tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(user.s); CHKERRQ(ierr);
  ierr = VecDestroy(user.y); CHKERRQ(ierr);
  ierr = VecDestroy(x); CHKERRQ(ierr);
  ierr = MatDestroy(H); CHKERRQ(ierr);

  /* Finalize TAO, PETSc */
  TaoFinalize();
  PetscFinalize();

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
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
  PetscScalar hx = user->hx, hy = user->hy, temp;
  PetscScalar val;
  PetscErrorCode ierr;
  PetscInt i, j, k, nx = user->mx, ny = user->my;

  /* Compute initial guess */
  for (j=0; j<ny; j++) {
    temp = PetscMin(j+1,ny-j)*hy;
    for (i=0; i<nx; i++) {
      k   = nx*j + i;
      val = PetscMin((PetscMin(i+1,nx-i))*hx,temp);
      ierr = VecSetValues(X,1,&k,&val,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/* 
   FormFunctionGradient - Evaluates the function and corresponding gradient.
    
   Input Parameters:
   tao - the TaoSolver context
   X   - the input vector 
   ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(TaoSolver tao,Vec X,double *f,Vec G,void *ptr)
{
  PetscErrorCode ierr;
  ierr = FormFunction(tao,X,f,ptr);CHKERRQ(ierr);
  ierr = FormGradient(tao,X,G,ptr);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   FormFunction - Evaluates the function, f(X).

   Input Parameters:
.  tao - the TaoSolver context
.  X   - the input vector 
.  ptr - optional user-defined context, as set by TaoSetFunction()

   Output Parameters:
.  f    - the newly evaluated function
*/
PetscErrorCode FormFunction(TaoSolver tao,Vec X,PetscScalar *f,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  PetscScalar hx = user->hx, hy = user->hy, area, three = 3.0, p5 = 0.5;
  PetscScalar zero = 0.0, vb, vl, vr, vt, dvdx, dvdy, flin = 0.0, fquad = 0.0;
  PetscScalar v, cdiv3 = user->param/three;
  PetscScalar *x;
  PetscErrorCode ierr;
  PetscInt  nx = user->mx, ny = user->my, i, j, k;

  /* Get pointer to vector data */
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);

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
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);

  /* Scale the function */
  area = p5*hx*hy;
  *f = area*(p5*fquad+flin);

  ierr = PetscLogFlops(nx*ny*24); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormGradient"
/*  
    FormGradient - Evaluates the gradient, G(X).              

    Input Parameters:
.   tao  - the TaoSolver context
.   X    - input vector
.   ptr  - optional user-defined context
    
    Output Parameters:
.   G - vector containing the newly evaluated gradient
*/
PetscErrorCode FormGradient(TaoSolver tao,Vec X,Vec G,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  PetscScalar zero=0.0, p5=0.5, three = 3.0, area, val, *x;
  PetscErrorCode ierr;
  PetscInt nx = user->mx, ny = user->my, ind, i, j, k;
  PetscScalar hx = user->hx, hy = user->hy;
  PetscScalar vb, vl, vr, vt, dvdx, dvdy;
  PetscScalar v, cdiv3 = user->param/three;

  /* Initialize gradient to zero */
  ierr = VecSet(G, zero); CHKERRQ(ierr);

  /* Get pointer to vector data */
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);

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
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
      }
      if (i != nx-1 && j != -1) {
        ind = k+1; val =  dvdx/hx - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
      }
      if (i != -1 && j != ny-1) {
        ind = k+nx; val = dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
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
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
      }
      if (i != 0 && j != ny) {
        ind = k-1; val =  - dvdx/hx - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
      }
      if (i != nx && j != ny) {
        ind = k; val =  dvdx/hx + dvdy/hy - cdiv3;
        ierr = VecSetValues(G,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
      }
    }
  }

  /* Restore vector */
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);

  /* Assemble gradient vector */
  ierr = VecAssemblyBegin(G); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(G); CHKERRQ(ierr);

  /* Scale the gradient */
  area = p5*hx*hy;
  ierr = VecScale(G, area); CHKERRQ(ierr);
  
  ierr = PetscLogFlops(nx*ny*24); CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/* 
   FormHessian - Forms the Hessian matrix.

   Input Parameters:
.  tao - the TaoSolver context
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
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *HH,Mat *Hpre, MatStructure *flg, void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;
  PetscErrorCode ierr;
  PetscInt   i,j, ndim = user->ndim;
  PetscScalar  *y, zero = 0.0, one = 1.0;
  Mat H=*HH;
  *Hpre = H;
  PetscBool assembled;

  /* Set location of vector */
  user->xvec = X;

  /* Initialize Hessian entries and work vector to zero */
  ierr = MatAssembled(H,&assembled); CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(H);  CHKERRQ(ierr);}

  ierr = VecSet(user->s, zero); CHKERRQ(ierr);

  /* Loop over matrix columns to compute entries of the Hessian */
  for (j=0; j<ndim; j++) {

    ierr = VecSetValues(user->s,1,&j,&one,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s); CHKERRQ(ierr);

    ierr = HessianProduct(ptr,user->s,user->y); CHKERRQ(ierr);

    ierr = VecSetValues(user->s,1,&j,&zero,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(user->s); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->s); CHKERRQ(ierr);

    ierr = VecGetArray(user->y,&y); CHKERRQ(ierr);
    for (i=0; i<ndim; i++) {
      if (y[i] != zero) {
        ierr = MatSetValues(H,1,&i,1,&j,&y[i],ADD_VALUES); CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(user->y,&y); CHKERRQ(ierr);

  }

  *flg=SAME_NONZERO_PATTERN;

  /* Assemble matrix  */
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatrixFreeHessian"
/* 
   MatrixFreeHessian - Sets a pointer for use in computing Hessian-vector
   products.
    
   Input Parameters:
.  tao - the TaoSolver context
.  X    - the input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()
   
   Output Parameters:
.  H     - Hessian matrix
.  PrecH - optionally different preconditioning Hessian
.  flag  - flag indicating matrix structure
*/
PetscErrorCode MatrixFreeHessian(TaoSolver tao,Vec X,Mat *H,Mat *PrecH,
                      MatStructure *flag,void *ptr)
{
  AppCtx     *user = (AppCtx *) ptr;

  /* Sets location of vector for use in computing matrix-vector products
     of the form H(X)*y  */

  user->xvec = X;   
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "HessianProductMat"
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
  void *ptr;
  PetscErrorCode ierr;
  ierr = MatShellGetContext(mat,&ptr); CHKERRQ(ierr);
  ierr = HessianProduct(ptr,svec,y); CHKERRQ(ierr);
  

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "HessianProduct"
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
  AppCtx *user = (AppCtx *)ptr;
  PetscScalar p5 = 0.5, zero = 0.0, one = 1.0, hx, hy, val, area, *x, *s;
  PetscScalar v, vb, vl, vr, vt, hxhx, hyhy;
  PetscErrorCode ierr;
  PetscInt nx, ny, i, j, k, ind;

  nx   = user->mx;
  ny   = user->my;
  hx   = user->hx;
  hy   = user->hy;
  hxhx = one/(hx*hx);
  hyhy = one/(hy*hy);

  /* Get pointers to vector data */
  ierr = VecGetArray(user->xvec,&x); CHKERRQ(ierr);
  ierr = VecGetArray(svec,&s); CHKERRQ(ierr);

  /* Initialize product vector to zero */
  ierr = VecSet(y, zero); CHKERRQ(ierr);

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
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
       }
       if (i != -1 && j != ny-1) {
         vt = s[k+nx];
         ind = k+nx; val = hyhy*(vt-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
       }
       if (i != -1 && j != -1) {
         ind = k; val = hxhx*(v-vr) + hyhy*(v-vt);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
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
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
       }
       if (i != 0 && j != ny) {
         vl = s[k-1];
         ind = k-1; val = hxhx*(vl-v);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
       }
       if (i != nx && j != ny) {
         ind = k; val = hxhx*(v-vl) + hyhy*(v-vb);
         ierr = VecSetValues(y,1,&ind,&val,ADD_VALUES); CHKERRQ(ierr);
       }
    }
  }
  /* Restore vector data */
  ierr = VecRestoreArray(svec,&s); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->xvec,&x); CHKERRQ(ierr);

  /* Assemble vector */
  ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y); CHKERRQ(ierr);
 
  /* Scale resulting vector by area */
  area = p5*hx*hy;
  ierr = VecScale(y, area); CHKERRQ(ierr);

  ierr = PetscLogFlops(nx*ny*18); CHKERRQ(ierr);
  
  return 0;
}


