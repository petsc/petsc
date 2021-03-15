static char help[] ="Solves Laplacian with multigrid. Tests block API for PCMG\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*  Modified from ~src/ksp/tests/ex19.c. Used for testing ML 6.2 interface.

    This problem is modeled by
    the partial differential equation

            -Laplacian u  = g,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a linear
    system of equations.

    Usage: ./ex26 -ksp_monitor_short -pc_type ml
           -mg_coarse_ksp_max_it 10
           -mg_levels_1_ksp_max_it 10 -mg_levels_2_ksp_max_it 10
           -mg_fine_ksp_max_it 10
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

static PetscErrorCode FormJacobian_Grid(GridCtx*,Mat);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,its,Nx=PETSC_DECIDE,Ny=PETSC_DECIDE,nlocal,nrhs = 1;
  PetscScalar    one = 1.0;
  Mat            A,B,X;
  GridCtx        fine_ctx;
  KSP            ksp;
  PetscBool      Brand = PETSC_FALSE,flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* set up discretization matrix for fine grid */
  fine_ctx.mx = 9;
  fine_ctx.my = 9;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&fine_ctx.mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&fine_ctx.my,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-rand",&Brand,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",fine_ctx.mx,fine_ctx.my);CHKERRQ(ierr);

  /* Set up distributed array for fine grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,fine_ctx.mx,fine_ctx.my,Nx,Ny,1,1,NULL,NULL,&fine_ctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(fine_ctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(fine_ctx.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fine_ctx.da,&fine_ctx.x);CHKERRQ(ierr);
  ierr = VecDuplicate(fine_ctx.x,&fine_ctx.b);CHKERRQ(ierr);
  ierr = VecGetLocalSize(fine_ctx.x,&nlocal);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fine_ctx.da,&fine_ctx.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(fine_ctx.localX,&fine_ctx.localF);CHKERRQ(ierr);
  ierr = DMCreateMatrix(fine_ctx.da,&A);CHKERRQ(ierr);
  ierr = FormJacobian_Grid(&fine_ctx,A);CHKERRQ(ierr);

  /* create linear solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,fine_ctx.da);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_FALSE);CHKERRQ(ierr);

  /* set values for rhs vector */
  ierr = VecSet(fine_ctx.b,one);CHKERRQ(ierr);

  /* set options, then solve system */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr); /* calls PCSetFromOptions_ML if 'pc_type=ml' */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,fine_ctx.b,fine_ctx.x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(fine_ctx.x,NULL,"-debug");CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %D\n",its);CHKERRQ(ierr);

  /* test multiple right-hand side */
  ierr = MatCreateDense(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,fine_ctx.mx*fine_ctx.my,nrhs,NULL,&B);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"rhs_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);
  if (Brand) {
    ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  } else {
    PetscScalar *b;

    ierr = MatDenseGetArrayWrite(B,&b);CHKERRQ(ierr);
    for (i=0;i<nlocal*nrhs;i++) b[i] = 1.0;
    ierr = MatDenseRestoreArrayWrite(B,&b);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
  ierr = MatViewFromOptions(X,NULL,"-debug");CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&flg);CHKERRQ(ierr);
  if ((flg || nrhs == 1) && !Brand) {
    PetscInt          n;
    const PetscScalar *xx,*XX;

    ierr = VecGetArrayRead(fine_ctx.x,&xx);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(X,&XX);CHKERRQ(ierr);
    for (n=0;n<nrhs;n++) {
      for (i=0;i<nlocal;i++) {
        if (PetscAbsScalar(xx[i] - XX[nlocal*n + i]) > PETSC_SMALL) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error local solve %D, entry %D -> %g + i %g != %g + i %g\n",PetscGlobalRank,n,i,(double)PetscRealPart(xx[i]),(double)PetscImaginaryPart(xx[i]),(double)PetscRealPart(XX[i]),(double)PetscImaginaryPart(XX[i]));CHKERRQ(ierr);
        }
      }
    }
    ierr = MatDenseRestoreArrayRead(X,&XX);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(fine_ctx.x,&xx);CHKERRQ(ierr);
  }

  /* free data structures */
  ierr = VecDestroy(&fine_ctx.x);CHKERRQ(ierr);
  ierr = VecDestroy(&fine_ctx.b);CHKERRQ(ierr);
  ierr = DMDestroy(&fine_ctx.da);CHKERRQ(ierr);
  ierr = VecDestroy(&fine_ctx.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&fine_ctx.localF);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FormJacobian_Grid(GridCtx *grid,Mat jac)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5];
  PetscInt               grow;
  const PetscInt         *ltog;
  PetscScalar            two = 2.0,one = 1.0,v[5],hx,hy,hxdhy,hydhx,value;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  mx    = grid->mx;            my = grid->my;
  hx    = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(grid->da,&ltogm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogm,&ltog);CHKERRQ(ierr);

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
        ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        value = .25*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&ltog);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ksp_monitor_short

    test:
      suffix: 2
      args:  -ksp_monitor_short
      nsize: 3

    test:
      suffix: ml_1
      args:  -ksp_monitor_short -pc_type ml -mat_no_inode
      nsize: 3
      requires: ml

    test:
      suffix: ml_2
      args:  -ksp_monitor_short -pc_type ml -mat_no_inode -ksp_max_it 3
      nsize: 3
      requires: ml

    test:
      suffix: ml_3
      args:  -ksp_monitor_short -pc_type ml -mat_no_inode -pc_mg_type ADDITIVE -ksp_max_it 7
      nsize: 1
      requires: ml

    test:
      suffix: cycles
      nsize: {{1 2}}
      args: -ksp_view_final_residual -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -ksp_monitor -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}separate output} -nrhs 1

    test:
      suffix: matcycles
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type preonly -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -ksp_monitor -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}separate output} -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

    test:
      requires: ml
      suffix: matcycles_ml
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type preonly -pc_type ml -mx 5 -my 5 -ksp_monitor -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}separate output} -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

    test:
      requires: hpddm
      suffix: matcycles_hpddm_mg
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type hpddm -pc_type mg -pc_mg_levels 3 -pc_mg_galerkin -mx 5 -my 5 -ksp_monitor -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}separate output} -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

    test:
      requires: hpddm
      nsize: {{1 2}}
      suffix: matcycles_hpddm_ilu
      args: -ksp_view_final_residual -ksp_type hpddm -pc_type redundant -redundant_pc_type ilu -mx 5 -my 5 -ksp_monitor -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

TEST*/
