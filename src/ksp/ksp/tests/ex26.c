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
  PetscInt       i,its,Nx=PETSC_DECIDE,Ny=PETSC_DECIDE,nlocal,nrhs = 1;
  PetscScalar    one = 1.0;
  Mat            A,B,X;
  GridCtx        fine_ctx;
  KSP            ksp;
  PetscBool      Brand = PETSC_FALSE,flg;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* set up discretization matrix for fine grid */
  fine_ctx.mx = 9;
  fine_ctx.my = 9;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mx",&fine_ctx.mx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-my",&fine_ctx.my,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-rand",&Brand,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",fine_ctx.mx,fine_ctx.my));

  /* Set up distributed array for fine grid */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,fine_ctx.mx,fine_ctx.my,Nx,Ny,1,1,NULL,NULL,&fine_ctx.da));
  CHKERRQ(DMSetFromOptions(fine_ctx.da));
  CHKERRQ(DMSetUp(fine_ctx.da));
  CHKERRQ(DMCreateGlobalVector(fine_ctx.da,&fine_ctx.x));
  CHKERRQ(VecDuplicate(fine_ctx.x,&fine_ctx.b));
  CHKERRQ(VecGetLocalSize(fine_ctx.x,&nlocal));
  CHKERRQ(DMCreateLocalVector(fine_ctx.da,&fine_ctx.localX));
  CHKERRQ(VecDuplicate(fine_ctx.localX,&fine_ctx.localF));
  CHKERRQ(DMCreateMatrix(fine_ctx.da,&A));
  CHKERRQ(FormJacobian_Grid(&fine_ctx,A));

  /* create linear solver */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetDM(ksp,fine_ctx.da));
  CHKERRQ(KSPSetDMActive(ksp,PETSC_FALSE));

  /* set values for rhs vector */
  CHKERRQ(VecSet(fine_ctx.b,one));

  /* set options, then solve system */
  CHKERRQ(KSPSetFromOptions(ksp)); /* calls PCSetFromOptions_ML if 'pc_type=ml' */
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSolve(ksp,fine_ctx.b,fine_ctx.x));
  CHKERRQ(VecViewFromOptions(fine_ctx.x,NULL,"-debug"));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %D\n",its));

  /* test multiple right-hand side */
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,fine_ctx.mx*fine_ctx.my,nrhs,NULL,&B));
  CHKERRQ(MatSetOptionsPrefix(B,"rhs_"));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X));
  if (Brand) {
    CHKERRQ(MatSetRandom(B,NULL));
  } else {
    PetscScalar *b;

    CHKERRQ(MatDenseGetArrayWrite(B,&b));
    for (i=0;i<nlocal*nrhs;i++) b[i] = 1.0;
    CHKERRQ(MatDenseRestoreArrayWrite(B,&b));
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(KSPMatSolve(ksp,B,X));
  CHKERRQ(MatViewFromOptions(X,NULL,"-debug"));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&flg));
  if ((flg || nrhs == 1) && !Brand) {
    PetscInt          n;
    const PetscScalar *xx,*XX;

    CHKERRQ(VecGetArrayRead(fine_ctx.x,&xx));
    CHKERRQ(MatDenseGetArrayRead(X,&XX));
    for (n=0;n<nrhs;n++) {
      for (i=0;i<nlocal;i++) {
        if (PetscAbsScalar(xx[i] - XX[nlocal*n + i]) > PETSC_SMALL) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error local solve %D, entry %D -> %g + i %g != %g + i %g\n",PetscGlobalRank,n,i,(double)PetscRealPart(xx[i]),(double)PetscImaginaryPart(xx[i]),(double)PetscRealPart(XX[i]),(double)PetscImaginaryPart(XX[i])));
        }
      }
    }
    CHKERRQ(MatDenseRestoreArrayRead(X,&XX));
    CHKERRQ(VecRestoreArrayRead(fine_ctx.x,&xx));
  }

  /* free data structures */
  CHKERRQ(VecDestroy(&fine_ctx.x));
  CHKERRQ(VecDestroy(&fine_ctx.b));
  CHKERRQ(DMDestroy(&fine_ctx.da));
  CHKERRQ(VecDestroy(&fine_ctx.localX));
  CHKERRQ(VecDestroy(&fine_ctx.localF));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(KSPDestroy(&ksp));

  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode FormJacobian_Grid(GridCtx *grid,Mat jac)
{
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

    testset:
      requires: hpddm
      args: -ksp_view_final_residual -ksp_type hpddm -pc_type mg -pc_mg_levels 3 -pc_mg_galerkin -mx 5 -my 5 -ksp_monitor -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -nrhs 7
      test:
        suffix: matcycles_hpddm_mg
        nsize: {{1 2}}
        args: -pc_mg_type {{additive multiplicative full kaskade}separate output} -ksp_matsolve_batch_size {{4 7}separate output}
      test:
        suffix: hpddm_mg_mixed_precision
        nsize: 2
        output_file: output/ex26_matcycles_hpddm_mg_pc_mg_type-multiplicative_ksp_matsolve_batch_size-4.out
        args: -ksp_matsolve_batch_size 4 -ksp_hpddm_precision {{single double}shared output}

    test:
      requires: hpddm
      nsize: {{1 2}}
      suffix: matcycles_hpddm_ilu
      args: -ksp_view_final_residual -ksp_type hpddm -pc_type redundant -redundant_pc_type ilu -mx 5 -my 5 -ksp_monitor -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

TEST*/
