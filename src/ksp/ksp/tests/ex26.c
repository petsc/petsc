static char help[] = "Solves Laplacian with multigrid. Tests block API for PCMG\n\
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

    Usage: ./ex26 -ksp_monitor -pc_type ml
           -mg_coarse_ksp_max_it 10
           -mg_levels_1_ksp_max_it 10 -mg_levels_2_ksp_max_it 10
           -mg_fine_ksp_max_it 10
*/

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */
typedef struct {
  PetscInt mx, my;         /* number grid points in x and y direction */
  Vec      localX, localF; /* local vectors with ghost region */
  DM       da;
  Vec      x, b, r; /* global vectors */
  Mat      J;       /* Jacobian on grid */
  Mat      A, P, R;
  KSP      ksp;
} GridCtx;

static PetscErrorCode FormJacobian_Grid(GridCtx *, Mat);

/* Copies A into a new dense matrix of the same type, VecType, and layout, stored with a leading dimension larger by shift */
static PetscErrorCode DuplicateWithLDA(Mat A, PetscInt shift, Mat *B)
{
  MatType  type;
  VecType  vtype;
  PetscInt m, n, M, N;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &type));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, m, n, M, N));
  PetscCall(MatSetType(*B, type));
  PetscCall(MatSetVecType(*B, vtype));
  PetscCall(MatDenseSetLDA(*B, m + shift));
  PetscCall(MatSetUp(*B));
  PetscCall(MatCopy(A, *B, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    i, its, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE, nlocal, nrhs = 1, lda_shift = 0;
  PetscScalar one = 1.0;
  Mat         A, B, X;
  GridCtx     fine_ctx;
  KSP         ksp;
  PetscBool   Brand = PETSC_FALSE, transpose = PETSC_FALSE, new_interpolation = PETSC_FALSE, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* set up discretization matrix for fine grid */
  fine_ctx.mx = 9;
  fine_ctx.my = 9;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &fine_ctx.mx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &fine_ctx.my, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nx", &Nx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ny", &Ny, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rand", &Brand, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", &transpose, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lda_shift", &lda_shift, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-new_interpolation", &new_interpolation, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Fine grid size %" PetscInt_FMT " by %" PetscInt_FMT "\n", fine_ctx.mx, fine_ctx.my));

  /* Set up distributed array for fine grid */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, fine_ctx.mx, fine_ctx.my, Nx, Ny, 1, 1, NULL, NULL, &fine_ctx.da));
  PetscCall(DMSetFromOptions(fine_ctx.da));
  PetscCall(DMSetUp(fine_ctx.da));
  PetscCall(DMCreateGlobalVector(fine_ctx.da, &fine_ctx.x));
  PetscCall(VecDuplicate(fine_ctx.x, &fine_ctx.b));
  PetscCall(VecGetLocalSize(fine_ctx.x, &nlocal));
  PetscCall(DMCreateLocalVector(fine_ctx.da, &fine_ctx.localX));
  PetscCall(VecDuplicate(fine_ctx.localX, &fine_ctx.localF));
  PetscCall(DMCreateMatrix(fine_ctx.da, &A));
  PetscCall(FormJacobian_Grid(&fine_ctx, A));

  /* create linear solver */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, fine_ctx.da));
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_ALL, PETSC_FALSE));

  /* set values for rhs vector */
  PetscCall(VecSet(fine_ctx.b, one));

  /* set options, then solve system */
  PetscCall(KSPSetFromOptions(ksp)); /* calls PCSetFromOptions_ML if 'pc_type=ml' */
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSolve(ksp, fine_ctx.b, fine_ctx.x));
  PetscCall(VecViewFromOptions(fine_ctx.x, NULL, "-debug"));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %" PetscInt_FMT "\n", its));

  /* test multiple right-hand side */
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, nlocal, PETSC_DECIDE, fine_ctx.mx * fine_ctx.my, nrhs, NULL, &B));
  PetscCall(MatSetOptionsPrefix(B, "rhs_"));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
  /* start the -ksp_initial_guess_nonzero runs from a known state, MatDuplicate() zeroing is not guaranteed for all -rhs_mat_type values */
  PetscCall(MatZeroEntries(X));
  if (Brand) {
    PetscCall(MatSetRandom(B, NULL));
  } else {
    PetscScalar *b;

    PetscCall(MatDenseGetArrayWrite(B, &b));
    for (i = 0; i < nlocal * nrhs; i++) b[i] = 1.0;
    PetscCall(MatDenseRestoreArrayWrite(B, &b));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  if (transpose) PetscCall(KSPMatSolveTranspose(ksp, B, X));
  else PetscCall(KSPMatSolve(ksp, B, X));
  PetscCall(MatViewFromOptions(X, NULL, "-debug"));

  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPPREONLY, &flg));
  if ((flg || nrhs == 1) && !Brand && !transpose) { /* the operator is not symmetric, so KSPMatSolveTranspose() does not return the solution of KSPSolve() */
    const PetscScalar *xx, *XX;

    PetscCall(VecGetArrayRead(fine_ctx.x, &xx));
    PetscCall(MatDenseGetArrayRead(X, &XX));
    for (PetscInt n = 0; n < nrhs; n++) {
      for (i = 0; i < nlocal; i++) {
        if (PetscAbsScalar(xx[i] - XX[nlocal * n + i]) > PETSC_SMALL) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error local solve %" PetscInt_FMT ", entry %" PetscInt_FMT " -> %g + i %g != %g + i %g\n", PetscGlobalRank, n, i, (double)PetscRealPart(xx[i]), (double)PetscImaginaryPart(xx[i]), (double)PetscRealPart(XX[i]), (double)PetscImaginaryPart(XX[i])));
        }
      }
    }
    PetscCall(MatDenseRestoreArrayRead(X, &XX));
    PetscCall(VecRestoreArrayRead(fine_ctx.x, &xx));
  }

  if (lda_shift > 0) { /* solve again with the same blocks stored with a larger leading dimension, which the solver must handle with what it kept from the first solve */
    Mat       B2, X2;
    PetscReal norm, err;

    PetscCall(DuplicateWithLDA(B, lda_shift, &B2));
    PetscCall(DuplicateWithLDA(X, lda_shift, &X2));
    PetscCall(MatZeroEntries(X2));
    if (transpose) PetscCall(KSPMatSolveTranspose(ksp, B2, X2));
    else PetscCall(KSPMatSolve(ksp, B2, X2));
    PetscCall(MatNorm(X, NORM_FROBENIUS, &norm));
    PetscCall(MatAXPY(X2, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(X2, NORM_FROBENIUS, &err));
    PetscCheck(err <= PETSC_SMALL * norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution with a larger leading dimension has error %g relative to %g", (double)err, (double)norm);
    PetscCall(MatDestroy(&B2));
    PetscCall(MatDestroy(&X2));
  }

  if (new_interpolation) { /* solve again after replacing the finest interpolation and restriction by a copy with an explicit zero in a new off-process column */
    PC        pc;
    Mat       Pf, P2, X2;
    PetscInt  nlevels, rstart, cend, N;
    PetscReal norm, err;

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCMGGetLevels(pc, &nlevels));
    PetscCall(PCMGGetInterpolation(pc, nlevels - 1, &Pf));
    PetscCall(MatDuplicate(Pf, MAT_COPY_VALUES, &P2));
    PetscCall(MatGetOwnershipRange(P2, &rstart, NULL));
    PetscCall(MatGetOwnershipRangeColumn(P2, NULL, &cend));
    PetscCall(MatGetSize(P2, NULL, &N));
    PetscCall(MatSetOption(P2, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetValue(P2, rstart, cend < N ? N - 1 : 0, 0.0, ADD_VALUES));
    PetscCall(MatAssemblyBegin(P2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P2, MAT_FINAL_ASSEMBLY));
    PetscCall(PCMGSetInterpolation(pc, nlevels - 1, P2));
    PetscCall(PCMGSetRestriction(pc, nlevels - 1, P2));
    PetscCall(MatDestroy(&P2));
    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &X2));
    PetscCall(MatZeroEntries(X2));
    if (transpose) PetscCall(KSPMatSolveTranspose(ksp, B, X2));
    else PetscCall(KSPMatSolve(ksp, B, X2));
    PetscCall(MatNorm(X, NORM_FROBENIUS, &norm));
    PetscCall(MatAXPY(X2, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(X2, NORM_FROBENIUS, &err));
    PetscCheck(err <= PETSC_SMALL * norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution after replacing the interpolation has error %g relative to %g", (double)err, (double)norm);
    PetscCall(MatDestroy(&X2));
  }

  /* free data structures */
  PetscCall(VecDestroy(&fine_ctx.x));
  PetscCall(VecDestroy(&fine_ctx.b));
  PetscCall(DMDestroy(&fine_ctx.da));
  PetscCall(VecDestroy(&fine_ctx.localX));
  PetscCall(VecDestroy(&fine_ctx.localF));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormJacobian_Grid(GridCtx *grid, Mat jac)
{
  PetscInt               i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym, col[5];
  PetscInt               grow;
  const PetscInt        *ltog;
  PetscScalar            two = 2.0, one = 1.0, v[5], hx, hy, hxdhy, hydhx, value;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  mx    = grid->mx;
  my    = grid->my;
  hx    = one / (PetscReal)(mx - 1);
  hy    = one / (PetscReal)(my - 1);
  hxdhy = hx / hy;
  hydhx = hy / hx;

  /* Get ghost points */
  PetscCall(DMDAGetCorners(grid->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(grid->da, &Xs, &Ys, NULL, &Xm, &Ym, NULL));
  PetscCall(DMGetLocalToGlobalMapping(grid->da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &ltog));

  /* Evaluate Jacobian of function */
  for (j = ys; j < ys + ym; j++) {
    row = (j - Ys) * Xm + xs - Xs - 1;
    for (i = xs; i < xs + xm; i++) {
      row++;
      grow = ltog[row];
      if (i > 0 && i < mx - 1 && j > 0 && j < my - 1) {
        v[0]   = -hxdhy;
        col[0] = ltog[row - Xm];
        v[1]   = -hydhx;
        col[1] = ltog[row - 1];
        v[2]   = two * (hydhx + hxdhy);
        col[2] = grow;
        v[3]   = -hydhx;
        col[3] = ltog[row + 1];
        v[4]   = -hxdhy;
        col[4] = ltog[row + Xm];
        PetscCall(MatSetValues(jac, 1, &grow, 5, col, v, INSERT_VALUES));
      } else if ((i > 0 && i < mx - 1) || (j > 0 && j < my - 1)) {
        value = .5 * two * (hydhx + hxdhy);
        PetscCall(MatSetValues(jac, 1, &grow, 1, &grow, &value, INSERT_VALUES));
      } else {
        value = .25 * two * (hydhx + hxdhy);
        PetscCall(MatSetValues(jac, 1, &grow, 1, &grow, &value, INSERT_VALUES));
      }
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &ltog));
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -ksp_monitor

    test:
      suffix: 2
      args: -ksp_monitor
      nsize: 3

    test:
      suffix: ml_1
      args: -ksp_monitor -pc_type ml -mat_no_inode
      nsize: 3
      requires: ml

    test:
      suffix: ml_2
      args: -ksp_monitor -pc_type ml -mat_no_inode -ksp_max_it 3
      nsize: 3
      requires: ml

    test:
      suffix: ml_3
      args: -ksp_monitor -pc_type ml -mat_no_inode -pc_mg_type ADDITIVE -ksp_max_it 7
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
      suffix: matcycles_richardson
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type richardson -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

    test:
      suffix: matcycles_richardson_guess
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type richardson -ksp_initial_guess_nonzero -ksp_norm_type unpreconditioned -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

    test:
      # a second solve after the finest interpolation and restriction are replaced by a matrix with an extra off-process column
      suffix: matcycles_new_interpolation
      nsize: 2
      args: -ksp_type preonly -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}shared output} -nrhs 7 -new_interpolation

    test:
      # a second solve with blocks of a larger leading dimension
      suffix: matcycles_lda
      nsize: 2
      args: -ksp_type preonly -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}shared output} -nrhs 7 -ksp_matsolve_batch_size {{4 7}shared output} -lda_shift 3

    # the same on the device, solving the whole block at once so that the kept products get the new leading dimension
    testset:
      nsize: 2
      args: -ksp_type preonly -pc_type mg -mx 5 -my 5 -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_coarse_pc_type jacobi -pc_mg_type {{additive multiplicative full kaskade}shared output} -nrhs 7 -lda_shift 3
      output_file: output/ex26_matcycles_lda.out

      test:
        suffix: matcycles_lda_kokkos_cuda
        requires: cuda kokkos_kernels
        args: -dm_mat_type aijkokkos -dm_vec_type kokkos -rhs_mat_type densecuda -rhs_mat_vec_type kokkos

      test:
        suffix: matcycles_lda_cuda
        requires: cuda
        args: -dm_mat_type aijcusparse -dm_vec_type cuda -rhs_mat_type densecuda

      test:
        suffix: matcycles_lda_kokkos_hip
        requires: hip kokkos_kernels
        args: -dm_mat_type aijkokkos -dm_vec_type kokkos -rhs_mat_type densehip -rhs_mat_vec_type kokkos

      test:
        suffix: matcycles_lda_hip
        requires: hip
        args: -dm_mat_type aijhipsparse -dm_vec_type hip -rhs_mat_type densehip

    test:
      suffix: matcycles_richardson_transpose
      nsize: {{1 2}}
      args: -ksp_view_final_residual -ksp_type richardson -transpose -pc_type jacobi -mx 5 -my 5 -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

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
        requires: !__float128 !__fp16
        suffix: hpddm_mg_mixed_precision
        nsize: 2
        output_file: output/ex26_matcycles_hpddm_mg_pc_mg_type-multiplicative_ksp_matsolve_batch_size-4.out
        args: -ksp_matsolve_batch_size 4 -ksp_hpddm_precision {{single double}shared output}
      test:
        requires: __float128
        suffix: hpddm_mg_mixed_precision___float128
        nsize: 2
        output_file: output/ex26_matcycles_hpddm_mg_pc_mg_type-multiplicative_ksp_matsolve_batch_size-4.out
        args: -ksp_matsolve_batch_size 4 -ksp_hpddm_precision {{double __float128}shared output}
      test:
        requires: double defined(PETSC_HAVE_F2CBLASLAPACK___FLOAT128_BINDINGS)
        suffix: hpddm_mg_mixed_precision_double
        nsize: 2
        output_file: output/ex26_matcycles_hpddm_mg_pc_mg_type-multiplicative_ksp_matsolve_batch_size-4.out
        args: -ksp_matsolve_batch_size 4 -ksp_hpddm_precision __float128
      test:
        requires: single defined(PETSC_HAVE_F2CBLASLAPACK___FP16_BINDINGS)
        suffix: hpddm_mg_mixed_precision_single
        nsize: 2
        output_file: output/ex26_matcycles_hpddm_mg_pc_mg_type-multiplicative_ksp_matsolve_batch_size-4.out
        args: -ksp_matsolve_batch_size 4 -ksp_hpddm_precision __fp16 -ksp_rtol 1e-3

    test:
      requires: hpddm
      nsize: {{1 2}}
      suffix: matcycles_hpddm_ilu
      args: -ksp_view_final_residual -ksp_type hpddm -pc_type redundant -redundant_pc_type ilu -mx 5 -my 5 -ksp_monitor -nrhs 7 -ksp_matsolve_batch_size {{4 7}separate output}

TEST*/
