
/*T
   Concepts: SNES^parallel Surface process example
   Concepts: DA^using distributed arrays;
   Concepts: IS coloirng types;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,

static char help[] = "u`` + u^{2} = f. \n\
Demonstrates the use of Newton-Krylov methods \n\
with different jacobian evaluation routines and matrix colorings. \n\
Modified from ex6.c \n\
Input arguments are:\n\
   -snes_jacobian_default : Jacobian using finite differences. Slow and expensive, not take advantage of sparsity \n\
   -fd_jacobian_coloring:   Jacobian using finite differences with matrix coloring\n\
   -my_jacobian_struct:     use user-provided Jacobian data structure to create matcoloring context \n\n";
    Program usage:  mpiexec -n <procs> ex5 [-help] [all PETSc options] 
     e.g.,
      ./ex5 -fd_jacobian -mat_fd_coloring_view_draw -draw_pause -1
      mpiexec -n 2 ./ex5 -fd_jacobian_ghosted -log_summary

  ------------------------------------------------------------------------- */

/*
  Example: 
  ./ex45 -n 10 -snes_monitor -ksp_monitor
  ./ex45 -n 10 -snes_monitor -ksp_monitor -snes_jacobian_default -pc_type jacobi
  ./ex45 -n 10 -snes_monitor -ksp_monitor -snes_jacobian_default -pc_type ilu
  ./ex45 -n 10 -snes_jacobian_default -log_summary |grep SNESFunctionEval 
  ./ex45 -n 10 -snes_jacobian_default -fd_jacobian_coloring -my_jacobian_struct -log_summary |grep SNESFunctionEval 
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
 */
#include "petscdmmg.h"
#include "petscsnes.h"

#include "petscsnes.h"
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  DA          da; /* distributed array data structure */
  PassiveReal D;  /* The diffusion coefficient */
  PassiveReal K;  /* The advection coefficient */
  PetscInt    m;  /* Exponent for A */
} AppCtx;

/* 
   User-defined routines
*/
PetscErrorCode MyJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode MyApproxJacobianStructure(Mat *,void *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                /* SNES context */
  Vec            x,r,F;               /* vectors */
  Mat            J,JPrec;             /* Jacobian,preconditioner matrices */
  PetscInt               its;                  /* iterations for convergence */
  PetscErrorCode ierr;
  PetscInt       it,n = 5,i;
  PetscMPIInt    size;
  PetscReal      h,xp = 0.0; 
  PetscScalar    v,pfive = .5;
  PetscTruth     flg;
  MatFDColoring  matfdcoloring = 0;
  PetscTruth     fd_jacobian_coloring;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    user.D = 1.0;
    ierr = PetscOptionsReal("-D", "The diffusion coefficient D", __FILE__, user.D, &user.D, PETSC_NULL);CHKERRQ(ierr);
    user.K = 1.0;
    ierr = PetscOptionsReal("-K", "The advection coefficient K", __FILE__, user.K, &user.K, PETSC_NULL);CHKERRQ(ierr);
    user.m = 1;
    ierr = PetscOptionsInt("-m", "The exponent for A", __FILE__, user.m, &user.m, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  xp = 0.0;
  for (i=0; i<n; i++) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    xp += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(x,pfive);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Function evaluation routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&J);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,1,PETSC_NULL,&JPrec);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_jacobian_default",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg){ 
    /* Jacobian using finite differences. Slow and expensive, not take advantage of sparsity */ 
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobian,PETSC_NULL);CHKERRQ(ierr);
  } else {
    /* User provided Jacobian and preconditioner(diagonal part of Jacobian) */
    ierr = SNESSetJacobian(snes,J,JPrec,MyJacobian,0);CHKERRQ(ierr);
  } 

  fd_jacobian_coloring = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-fd_jacobian_coloring",&fd_jacobian_coloring,PETSC_NULL);CHKERRQ(ierr);  
  if (fd_jacobian_coloring){ 
    /* Jacobian using finite differences with matfdcoloring based on the sparse structure.
     In this case, only three calls to FormFunction() for each Jacobian evaluation - very fast! */
    ISColoring    iscoloring;
    
    /* Get the data structure of J */
    flg = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-my_jacobian_struct",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg){ 
      /* use user-provided jacobian data structure */
      ierr = MyApproxJacobianStructure(&J,PETSC_NULL);CHKERRQ(ierr);
    } else {
      /* use previously defined jacobian: SNESDefaultComputeJacobian() or MyJacobian()  */
      MatStructure  flag;
      ierr = SNESComputeJacobian(snes,x,&J,&J,&flag);CHKERRQ(ierr);
    }

    /* Create coloring context */
    ierr = MatGetColoring(J,MATCOLORING_SL,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,(void*)F);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    /* ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    
    /* Use SNESDefaultComputeJacobianColor() for Jacobian evaluation */
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr); 
    ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Newton iterations = %D\n\n",it);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(DMMGGetx(dmmg));CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(x);CHKERRQ(ierr);     ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(F);CHKERRQ(ierr);     ierr = MatDestroy(J);CHKERRQ(ierr);
  ierr = MatDestroy(JPrec);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);
  if (fd_jacobian_coloring){
    ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar    *xx,*ff,*FF,d;
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    **x;

  PetscFunctionBegin;
  ierr = DAGetInfo(user->da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  D      = user->D;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  temp1  = D/(D + 1.0);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(user->da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArray((Vec)dummy,&FF);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;
  ff[0]   = xx[0];
  for (i=1; i<n-1; i++) {
    ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
        x[j][i] = 0.0; 
      } else {
        x[j][i] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp)); 
      }
  }
  ff[n-1] = xx[n-1] - 1.0;
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArray((Vec)dummy,&FF);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   MyJacobian - This routine demonstrates the use of different
   matrices for the Jacobian and preconditioner
  ierr = DAVecRestoreArray(user->da,X,&x);CHKERRQ(ierr);

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode MyJacobian(SNES snes,Vec x,Mat *jac,Mat *prejac,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[3],d;
  PetscInt       i,n,j[3];
  PetscErrorCode ierr;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;
{
  PetscScalar v = 1.0;

  /* Form Jacobian.  Also form a different preconditioning matrix that 
     has only the diagonal elements. */
  i = 0; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
    A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(*prejac,1,&i,1,&i,&A[1],INSERT_VALUES);CHKERRQ(ierr);
  }
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscScalar v = 1.0;

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
  return user->m*v;
}

/* ------------------------------------------------------------------- */
#define __FUNCT__ "FormFunctionLocal"
/*
  Create an approximate data structure for Jacobian matrix to be used with matcoloring
*/
PetscErrorCode FormFunctionLocal(DALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  DA             coordDA;
  Vec            coordinates;
  DACoor2d     **coords;
  PetscScalar    u, ux, uy, uxx, uyy;
  PetscReal      D, K, hx, hy, hxdhy, hydhx;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

   Input Parameters:
.    A - dummy jacobian matrix 
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;
  /*
     Compute function over the locally owned part of the grid
  */
  ierr = DAGetCoordinateDA(user->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetCoordinates(user->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i];
      } else {
        u       = x[j][i];
        ux      = (x[j][i+1] - x[j][i])/hx;
        uy      = (x[j+1][i] - x[j][i])/hy;
        uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
        uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
        f[j][i] = D*(uxx + uyy) - (K*funcA(x[j][i], user)*sqrt(ux*ux + uy*uy) + funcU(&coords[j][i]))*hx*hy;
        if (PetscIsInfOrNanScalar(f[j][i])) {SETERRQ1(PETSC_ERR_FP, "Invalid residual: %g", PetscRealPart(f[j][i]));}
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  ierr = VecDestroy(coordinates);CHKERRQ(ierr);
  ierr = DADestroy(coordDA);CHKERRQ(ierr);
  ierr = PetscLogFlops(11*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

   Output Parameters:
     A -  jacobian matrix with assigned non-zero structure
/*
   FormJacobianLocal - Evaluates Jacobian matrix.
 */
PetscErrorCode MyApproxJacobianStructure(Mat *jac,void *dummy)
{
  PetscScalar    zeros[3];
  PetscInt       i,n,j[3];
  PetscInt       i, j;
  PetscErrorCode ierr;

  ierr = MatGetSize(*jac,&n,&n);CHKERRQ(ierr);
  D      = user->D;
  K      = user->K;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;

  zeros[0] = zeros[1] = zeros[2] = 0.0;
  i = 0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,zeros,INSERT_VALUES);CHKERRQ(ierr);
        contiguous chunks of rows across the processors. 
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      row.j = j; row.i = i;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        /* boundary points */
        v[0] = 1.0;
        ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* interior grid points */
        ux        = (x[j][i+1] - x[j][i])/hx;
        uy        = (x[j+1][i] - x[j][i])/hy;
        normGradZ = sqrt(ux*ux + uy*uy);
        PetscPrintf(PETSC_COMM_SELF, "i: %d j: %d normGradZ: %g\n", i, j, normGradZ);
        if (normGradZ < 1.0e-8) {
          normGradZ = 1.0e-8;
        }
        A         = funcA(x[j][i], user);
  
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1; 
    ierr = MatSetValues(*jac,1,&i,3,j,zeros,INSERT_VALUES);CHKERRQ(ierr);
        v[3] = -D*hydhx + K*A*hx*hy/(2.0*normGradZ);                                              col[3].j = j;     col[3].i = i+1;
        v[4] = -D*hxdhy + K*A*hx*hy/(2.0*normGradZ);                                              col[4].j = j + 1; col[4].i = i;
        for(int k = 0; k < 5; ++k) {
          if (PetscIsInfOrNanScalar(v[k])) {SETERRQ1(PETSC_ERR_FP, "Invalid residual: %g", PetscRealPart(v[k]));}
        }
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  i = n-1; 
  ierr = MatSetValues(*jac,1,&i,1,&i,zeros,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
