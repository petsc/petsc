/* Program usage: mpiexec -n <proc> eptorsion2 [-help] [all TAO options] */

/* ----------------------------------------------------------------------

  Elastic-plastic torsion problem.

  The elastic plastic torsion problem arises from the determination
  of the stress field on an infinitely long cylindrical bar, which is
  equivalent to the solution of the following problem:

  min{ .5 * integral(||gradient(v(x))||^2 dx) - C * integral(v(x) dx)}

  where C is the torsion angle per unit length.

  The uniprocessor version of this code is eptorsion1.c; the Fortran
  version of this code is eptorsion2f.F.

  This application solves the problem without calculating hessians
---------------------------------------------------------------------- */

/*
  Include "petsctao.h" so that we can use TAO solvers.  Note that this
  file automatically includes files for lower-level support, such as those
  provided by the PETSc library:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
  Include "petscdmda.h" so that we can use distributed arrays (DMs) for managing
  the parallel mesh.
*/

#include <petsctao.h>
#include <petscdmda.h>

static  char help[] =
"Demonstrates use of the TAO package to solve \n\
unconstrained minimization problems in parallel.  This example is based on \n\
the Elastic-Plastic Torsion (dept) problem from the MINPACK-2 test suite.\n\
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
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: n
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunction() and
   FormGradient().
*/
typedef struct {
  /* parameters */
   PetscInt      mx, my;         /* global discretization in x- and y-directions */
   PetscReal     param;          /* nonlinearity parameter */

  /* work space */
   Vec           localX;         /* local vectors */
   DM            dm;             /* distributed array data structure */
} AppCtx;

PetscErrorCode FormInitialGuess(AppCtx*, Vec);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

int main(int argc, char **argv)
{
    PetscErrorCode     ierr;
    Vec                x;
    Mat                H;
    PetscInt           Nx, Ny;
    Tao                tao;
    PetscBool          flg;
    KSP                ksp;
    PC                 pc;
    AppCtx             user;

    PetscInitialize(&argc, &argv, (char *)0, help);

    /* Specify default dimension of the problem */
    user.param = 5.0; user.mx = 10; user.my = 10;
    Nx = Ny = PETSC_DECIDE;

    /* Check for any command line arguments that override defaults */
    ierr = PetscOptionsGetReal(NULL,NULL,"-par",&user.param,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-my",&user.my,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,&flg);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n---- Elastic-Plastic Torsion Problem -----\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"mx: %D     my: %D   \n\n",user.mx,user.my);CHKERRQ(ierr);

    /* Set up distributed array */
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.mx,user.my,Nx,Ny,1,1,NULL,NULL,&user.dm);CHKERRQ(ierr);
    ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);
    ierr = DMSetUp(user.dm);CHKERRQ(ierr);

    /* Create vectors */
    ierr = DMCreateGlobalVector(user.dm,&x);CHKERRQ(ierr);

    ierr = DMCreateLocalVector(user.dm,&user.localX);CHKERRQ(ierr);

    /* Create Hessian */
    ierr = DMCreateMatrix(user.dm,&H);CHKERRQ(ierr);
    ierr = MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

    /* The TAO code begins here */

    /* Create TAO solver and set desired solution method */
    ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

    /* Set initial solution guess */
    ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

    /* Set routine for function and gradient evaluation */
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);

    ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,(void*)&user);CHKERRQ(ierr);

    /* Check for any TAO command line options */
    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

    ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
    if (ksp) {
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    }

    /* SOLVE THE APPLICATION */
    ierr = TaoSolve(tao);CHKERRQ(ierr);

    /* Free TAO data structures */
    ierr = TaoDestroy(&tao);CHKERRQ(ierr);

    /* Free PETSc data structures */
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);

    ierr = VecDestroy(&user.localX);CHKERRQ(ierr);
    ierr = DMDestroy(&user.dm);CHKERRQ(ierr);

    PetscFinalize();
    return 0;
}

/* ------------------------------------------------------------------- */
/*
    FormInitialGuess - Computes an initial approximation to the solution.

    Input Parameters:
.   user - user-defined application context
.   X    - vector

    Output Parameters:
    X    - vector
*/
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i, j, k, mx = user->mx, my = user->my;
  PetscInt       xs, ys, xm, ym, gxm, gym, gxs, gys, xe, ye;
  PetscReal      hx = 1.0/(mx+1), hy = 1.0/(my+1), temp, val;

  PetscFunctionBegin;
  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Compute initial guess over locally owned part of mesh */
  xe = xs+xm;
  ye = ys+ym;
  for (j=ys; j<ye; j++) {  /*  for (j=0; j<my; j++) */
    temp = PetscMin(j+1,my-j)*hy;
    for (i=xs; i<xe; i++) {  /*  for (i=0; i<mx; i++) */
      k   = (j-gys)*gxm + i-gxs;
      val = PetscMin((PetscMin(i+1,mx-i))*hx,temp);
      ierr = VecSetValuesLocal(X,1,&k,&val,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f,Vec G,void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k,ind;
  PetscInt       xe,ye,xsm,ysm,xep,yep;
  PetscInt       xs, ys, xm, ym, gxm, gym, gxs, gys;
  PetscInt       mx = user->mx, my = user->my;
  PetscReal      three = 3.0, zero = 0.0, *x, floc, cdiv3 = user->param/three;
  PetscReal      p5 = 0.5, area, val, flin, fquad;
  PetscReal      v,vb,vl,vr,vt,dvdx,dvdy;
  PetscReal      hx = 1.0/(user->mx + 1);
  PetscReal      hy = 1.0/(user->my + 1);
  Vec            localX = user->localX;

  PetscFunctionBegin;
  /* Initialize */
  flin = fquad = zero;

  ierr = VecSet(G, zero);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointer to vector data */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Get local mesh boundaries */
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Set local loop dimensions */
  xe = xs+xm;
  ye = ys+ym;
  if (xs == 0)  xsm = xs-1;
  else          xsm = xs;
  if (ys == 0)  ysm = ys-1;
  else          ysm = ys;
  if (xe == mx) xep = xe+1;
  else          xep = xe;
  if (ye == my) yep = ye+1;
  else          yep = ye;

  /* Compute local gradient contributions over the lower triangular elements */
  for (j=ysm; j<ye; j++) {  /*  for (j=-1; j<my; j++) */
    for (i=xsm; i<xe; i++) {  /*   for (i=-1; i<mx; i++) */
      k = (j-gys)*gxm + i-gxs;
      v = zero;
      vr = zero;
      vt = zero;
      if (i >= 0 && j >= 0) v = x[k];
      if (i < mx-1 && j > -1) vr = x[k+1];
      if (i > -1 && j < my-1) vt = x[k+gxm];
      dvdx = (vr-v)/hx;
      dvdy = (vt-v)/hy;
      if (i != -1 && j != -1) {
        ind = k; val = - dvdx/hx - dvdy/hy - cdiv3;
        ierr = VecSetValuesLocal(G,1,&k,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != mx-1 && j != -1) {
        ind = k+1; val =  dvdx/hx - cdiv3;
        ierr = VecSetValuesLocal(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != -1 && j != my-1) {
        ind = k+gxm; val = dvdy/hy - cdiv3;
        ierr = VecSetValuesLocal(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      fquad += dvdx*dvdx + dvdy*dvdy;
      flin -= cdiv3 * (v + vr + vt);
    }
  }

  /* Compute local gradient contributions over the upper triangular elements */
  for (j=ys; j<yep; j++) { /*  for (j=0; j<=my; j++) */
    for (i=xs; i<xep; i++) {  /*   for (i=0; i<=mx; i++) */
      k = (j-gys)*gxm + i-gxs;
      vb = zero;
      vl = zero;
      v  = zero;
      if (i < mx && j > 0) vb = x[k-gxm];
      if (i > 0 && j < my) vl = x[k-1];
      if (i < mx && j < my) v = x[k];
      dvdx = (v-vl)/hx;
      dvdy = (v-vb)/hy;
      if (i != mx && j != 0) {
        ind = k-gxm; val = - dvdy/hy - cdiv3;
        ierr = VecSetValuesLocal(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != 0 && j != my) {
        ind = k-1; val =  - dvdx/hx - cdiv3;
        ierr = VecSetValuesLocal(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      if (i != mx && j != my) {
        ind = k; val =  dvdx/hx + dvdy/hy - cdiv3;
        ierr = VecSetValuesLocal(G,1,&ind,&val,ADD_VALUES);CHKERRQ(ierr);
      }
      fquad += dvdx*dvdx + dvdy*dvdy;
      flin -= cdiv3 * (vb + vl + v);
    }
  }

  /* Restore vector */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /* Assemble gradient vector */
  ierr = VecAssemblyBegin(G);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(G);CHKERRQ(ierr);

  /* Scale the gradient */
  area = p5*hx*hy;
  floc = area * (p5 * fquad + flin);
  ierr = VecScale(G, area);CHKERRQ(ierr);

  /* Sum function contributions from all processes */
  ierr = (PetscErrorCode)MPI_Allreduce((void*)&floc,(void*)f,1,MPIU_REAL,MPIU_SUM,MPI_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscLogFlops((ye-ysm)*(xe-xsm)*20+(xep-xs)*(yep-ys)*16);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec X, Mat A, Mat Hpre, void*ctx)
{
  AppCtx         *user= (AppCtx*) ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       col[5],row;
  PetscInt       xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      v[5];
  PetscReal      hx=1.0/(user->mx+1), hy=1.0/(user->my+1), hxhx=1.0/(hx*hx), hyhy=1.0/(hy*hy), area=0.5*hx*hy;

  /* Compute the quadratic term in the objective function */

  /*
     Get local grid boundaries
  */

  PetscFunctionBegin;
  ierr = DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {

    for (i=xs; i< xs+xm; i++) {

      row=(j-gys)*gxm + (i-gxs);

      k=0;
      if (j>gys) {
        v[k]=-2*hyhy; col[k]=row - gxm; k++;
      }

      if (i>gxs) {
        v[k]= -2*hxhx; col[k]=row - 1; k++;
      }

      v[k]= 4.0*(hxhx+hyhy); col[k]=row; k++;

      if (i+1 < gxs+gxm) {
        v[k]= -2.0*hxhx; col[k]=row+1; k++;
      }

      if (j+1 <gys+gym) {
        v[k]= -2*hyhy; col[k] = row+gxm; k++;
      }

      ierr = MatSetValuesLocal(A,1,&row,k,col,v,INSERT_VALUES);CHKERRQ(ierr);

    }
  }
  /*
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  ierr = MatScale(A,area);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscLogFlops(9*xm*ym+49*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      nsize: 2
      args: -tao_smonitor -tao_type nls -mx 16 -my 16 -tao_gatol 1.e-4

TEST*/
