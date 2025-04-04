const char help[] = "CUDA backend of rosenbrock4cu.cu\n";

/* ------------------------------------------------------------------------

  Copy of rosenbrock1.c.
  Once PETSc test harness supports conditional linking, we can remove this duplicate.
  See https://gitlab.com/petsc/petsc/-/issues/1173
  ------------------------------------------------------------------------- */

#include "rosenbrock4.h"

int main(int argc, char **argv)
{
  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(RosenbrockMain());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex cuda !single !__float128 !defined(PETSC_USE_65BIT_INDICES)

  test:
    suffix: 1
    nsize: {{1 2 3}}
    args: -mat_type aijcusparse -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_nls_pc_type pbjacobi
    requires: !single
    output_file: output/rosenbrock1_1.out

  test:
    suffix: 2
    args: -mat_type aijcusparse -tao_monitor_short -tao_type lmvm -tao_gatol 1.e-3
    output_file: output/rosenbrock1_2.out

  test:
    suffix: 3
    args: -mat_type aijcusparse -tao_monitor_short -tao_type ntr -tao_gatol 1.e-4
    requires: !single
    output_file: output/rosenbrock1_3.out

  test:
    suffix: 5
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bntr -tao_gatol 1.e-4
    output_file: output/rosenbrock1_5.out

  test:
    suffix: 6
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bntl -tao_gatol 1.e-4
    output_file: output/rosenbrock1_6.out

  test:
    suffix: 7
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bnls -tao_gatol 1.e-4
    output_file: output/rosenbrock1_7.out

  test:
    suffix: 8
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bntr -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_8.out

  test:
    suffix: 9
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bntl -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_9.out

  test:
    suffix: 10
    args: -mat_type aijcusparse -tao_monitor_short -tao_type bnls -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_10.out

  test:
    suffix: 11
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbroyden
    output_file: output/rosenbrock1_11.out

  test:
    suffix: 12
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbadbroyden
    output_file: output/rosenbrock1_12.out

  test:
    suffix: 13
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbroyden
    output_file: output/rosenbrock1_13.out

  test:
    suffix: 14
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbfgs
    output_file: output/rosenbrock1_14.out

  test:
    suffix: 15
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmdfp
    output_file: output/rosenbrock1_15.out

  test:
    suffix: 16
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsr1
    output_file: output/rosenbrock1_16.out

  test:
    suffix: 17
    args: -mat_type aijcusparse -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnls
    output_file: output/rosenbrock1_17.out

  test:
    suffix: 18
    args: -mat_type aijcusparse -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm
    output_file: output/rosenbrock1_18.out

  test:
    suffix: 19
    args: -mat_type aijcusparse -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1
    output_file: output/rosenbrock1_19.out

  test:
    suffix: 20
    args: -mat_type aijcusparse -tao_monitor -tao_gatol 1e-4 -tao_type blmvm -tao_ls_monitor
    output_file: output/rosenbrock1_20.out

  test:
    suffix: 21
    args: -mat_type aijcusparse -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbadbroyden
    output_file: output/rosenbrock1_21.out

  test:
    suffix: 22
    args: -mat_type aijcusparse -tao_max_it 1 -tao_converged_reason
    output_file: output/rosenbrock1_22.out

  test:
    suffix: 23
    args: -mat_type aijcusparse -tao_max_funcs 0 -tao_converged_reason
    output_file: output/rosenbrock1_23.out

  test:
    suffix: 24
    args: -mat_type aijcusparse -tao_gatol 10 -tao_converged_reason
    output_file: output/rosenbrock1_24.out

  test:
    suffix: 25
    args: -mat_type aijcusparse -tao_grtol 10 -tao_converged_reason
    output_file: output/rosenbrock1_25.out

  test:
    suffix: 26
    args: -mat_type aijcusparse -tao_gttol 10 -tao_converged_reason
    output_file: output/rosenbrock1_26.out

  test:
    suffix: 27
    args: -mat_type aijcusparse -tao_steptol 10 -tao_converged_reason
    output_file: output/rosenbrock1_27.out

  test:
    suffix: 28
    args: -mat_type aijcusparse -tao_fmin 10 -tao_converged_reason
    output_file: output/rosenbrock1_28.out

  test:
    suffix: test_dbfgs
    nsize: {{1 2 3}}
    output_file: output/rosenbrock1_14.out
    args: -mat_type aijcusparse -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdbfgs -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lbfgs_type {{inplace reorder}} -tao_bqnk_mat_lbfgs_recursive {{0 1}}

  test:
    suffix: test_ddfp
    nsize: {{1 2 3}}
    output_file: output/rosenbrock1_14.out
    args: -mat_type aijcusparse -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmddfp -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_ldfp_type {{inplace reorder}} -tao_bqnk_mat_ldfp_recursive {{0 1}}

  test:
    suffix: test_dqn_1
    nsize: 1
    output_file: output/rosenbrock1_29.out
    args: -mat_type aijcusparse -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

  test:
    suffix: test_dqn_2
    nsize: 2
    output_file: output/rosenbrock1_30.out
    args: -mat_type aijcusparse -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

  test:
    suffix: test_dqn_3
    nsize: 3
    output_file: output/rosenbrock1_31.out
    args: -mat_type aijcusparse -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

TEST*/
