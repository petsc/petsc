
static char help[] ="Tests ML interface. Modified from ~src/ksp/ksp/tests/ex19.c \n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*
    This problem is modeled by
    the partial differential equation

            -Laplacian u  = g,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

    Usage: ./ex29 -ksp_type gmres -ksp_monitor
           -pc_mg_type <multiplicative> (one of) additive multiplicative full kascade
           -mg_coarse_ksp_max_it 10 -mg_levels_3_ksp_max_it 10 -mg_levels_2_ksp_max_it 10
           -mg_levels_1_ksp_max_it 10 -mg_fine_ksp_max_it 10
*/

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */
typedef struct {
  PetscInt mx,my;              /* number grid points in x and y direction */
  Vec      localX,localF;      /* local vectors with ghost region */
  DM       da;
  Vec      x,b,r;              /* global vectors */
  Mat      J;                  /* Jacobian on grid */
  Mat      A,P,R;
  KSP      ksp;
} GridCtx;
extern int FormJacobian_Grid(GridCtx*,Mat*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       its,n,Nx=PETSC_DECIDE,Ny=PETSC_DECIDE,nlocal,i;
  PetscMPIInt    size;
  PC             pc;
  PetscInt       mx,my;
  Mat            A;
  GridCtx        fine_ctx;
  KSP            ksp;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  /* set up discretization matrix for fine grid */
  /* ML requires input of fine-grid matrix. It determines nlevels. */
  fine_ctx.mx = 9; fine_ctx.my = 9;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&flg));
  if (flg) fine_ctx.mx = mx;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-my",&my,&flg));
  if (flg) fine_ctx.my = my;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",fine_ctx.mx,fine_ctx.my));
  n    = fine_ctx.mx*fine_ctx.my;

  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL));

  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,fine_ctx.mx,fine_ctx.my,Nx,Ny,1,1,NULL,NULL,&fine_ctx.da));
  CHKERRQ(DMSetFromOptions(fine_ctx.da));
  CHKERRQ(DMSetUp(fine_ctx.da));
  CHKERRQ(DMCreateGlobalVector(fine_ctx.da,&fine_ctx.x));
  CHKERRQ(VecDuplicate(fine_ctx.x,&fine_ctx.b));
  CHKERRQ(VecGetLocalSize(fine_ctx.x,&nlocal));
  CHKERRQ(DMCreateLocalVector(fine_ctx.da,&fine_ctx.localX));
  CHKERRQ(VecDuplicate(fine_ctx.localX,&fine_ctx.localF));
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,NULL,3,NULL,&A));
  CHKERRQ(FormJacobian_Grid(&fine_ctx,&A));

  /* create linear solver */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCML));

  /* set options, then solve system */
  CHKERRQ(KSPSetFromOptions(ksp)); /* calls PCSetFromOptions_MG/ML */

  for (i=0; i<3; i++) {
    if (i<2) {
      /* set values for rhs vector */
      CHKERRQ(VecSet(fine_ctx.b,i+1.0));
      /* modify A */
      CHKERRQ(MatShift(A,1.0));
      CHKERRQ(MatScale(A,2.0));
      CHKERRQ(KSPSetOperators(ksp,A,A));
    } else {  /* test SAME_NONZERO_PATTERN */
      CHKERRQ(KSPSetOperators(ksp,A,A));
    }
    CHKERRQ(KSPSolve(ksp,fine_ctx.b,fine_ctx.x));
    CHKERRQ(KSPGetIterationNumber(ksp,&its));
    if (its > 6) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Number of iterations = %D greater than expected\n",its));
    }
  }

  /* free data structures */
  CHKERRQ(VecDestroy(&fine_ctx.x));
  CHKERRQ(VecDestroy(&fine_ctx.b));
  CHKERRQ(DMDestroy(&fine_ctx.da));
  CHKERRQ(VecDestroy(&fine_ctx.localX));
  CHKERRQ(VecDestroy(&fine_ctx.localF));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}

int FormJacobian_Grid(GridCtx *grid,Mat *J)
{
  Mat                    jac = *J;
  PetscInt               i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  PetscInt               grow;
  const PetscInt         *ltog;
  PetscScalar            two = 2.0,one = 1.0,v[5],hx,hy,hxdhy,hydhx,value;
  ISLocalToGlobalMapping ltogm;

  mx    = grid->mx;               my = grid->my;
  hx    = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx = hy/hx;

  /* Get ghost points */
  CHKERRQ(DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0));
  CHKERRQ(DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0));
  CHKERRQ(DMGetLocalToGlobalMapping(grid->da,&ltogm));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(ltogm,&ltog));

  /* Evaluate Jacobian of function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1;
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {
        v[0] = -hxdhy; col[0] = ltog[row - Xm];
        v[1] = -hydhx; col[1] = ltog[row - 1];
        v[2] = two*(hydhx + hxdhy); col[2] = grow;
        v[3] = -hydhx; col[3] = ltog[row + 1];
        v[4] = -hxdhy; col[4] = ltog[row + Xm];
        CHKERRQ(MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES));
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        CHKERRQ(MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES));
      } else {
        value = .25*two*(hydhx + hxdhy);
        CHKERRQ(MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog));
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  return 0;
}

/*TEST

    test:
      output_file: output/ex29.out
      args: -mat_no_inode
      requires: ml

    test:
      suffix: 2
      nsize: 3
      requires: ml

TEST*/
