static char help[] = "This example demonstrates the use of different performance portable backends in user-defined callbacks in Tao.\n";

#include "rosenbrock4.h"

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(RosenbrockMain());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !__float128 !single !defined(PETSC_USE_64BIT_INDICES)

  test:
    suffix: 1
    nsize: {{1 2 3}}
    args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4
    output_file: output/rosenbrock1_1.out

  test:
    suffix: 2
    args: -tao_monitor_short -tao_type lmvm -tao_gatol 1.e-3
    output_file: output/rosenbrock1_2.out

  test:
    suffix: 3
    args: -tao_monitor_short -tao_type ntr -tao_gatol 1.e-4
    output_file: output/rosenbrock1_3.out

  test:
    suffix: 4
    args: -tao_monitor_short -tao_type ntr -tao_mf_hessian -tao_ntr_pc_type none -tao_gatol 1.e-4
    output_file: output/rosenbrock1_4.out

  test:
    suffix: 5
    args: -tao_monitor_short -tao_type bntr -tao_gatol 1.e-4
    output_file: output/rosenbrock1_5.out

  test:
    suffix: 6
    args: -tao_monitor_short -tao_type bntl -tao_gatol 1.e-4
    output_file: output/rosenbrock1_6.out

  test:
    suffix: 7
    args: -tao_monitor_short -tao_type bnls -tao_gatol 1.e-4
    output_file: output/rosenbrock1_7.out

  test:
    suffix: 8
    args: -tao_monitor_short -tao_type bntr -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_8.out

  test:
    suffix: 9
    args: -tao_monitor_short -tao_type bntl -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_9.out

  test:
    suffix: 10
    args: -tao_monitor_short -tao_type bnls -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4
    output_file: output/rosenbrock1_10.out

  test:
    suffix: 11
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbroyden
    output_file: output/rosenbrock1_11.out

  test:
    suffix: 12
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbadbroyden
    output_file: output/rosenbrock1_12.out

  test:
    suffix: 13
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbroyden
    output_file: output/rosenbrock1_13.out

  test:
    suffix: 14
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbfgs
    output_file: output/rosenbrock1_14.out

  test:
    suffix: 15
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmdfp
    output_file: output/rosenbrock1_15.out

  test:
    suffix: 16
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsr1
    output_file: output/rosenbrock1_16.out

  test:
    suffix: 17
    args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnls
    output_file: output/rosenbrock1_17.out

  test:
    suffix: 18
    args: -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm
    output_file: output/rosenbrock1_18.out

  test:
    suffix: 19
    args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1
    output_file: output/rosenbrock1_19.out

  test:
    suffix: 20
    args: -tao_monitor -tao_gatol 1e-4 -tao_type blmvm -tao_ls_monitor
    output_file: output/rosenbrock1_20.out

  test:
    suffix: 21
    args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbadbroyden
    output_file: output/rosenbrock1_21.out

  test:
    suffix: 22
    args: -tao_max_it 1 -tao_converged_reason
    output_file: output/rosenbrock1_22.out

  test:
    suffix: 23
    args: -tao_max_funcs 0 -tao_converged_reason
    output_file: output/rosenbrock1_23.out

  test:
    suffix: 24
    args: -tao_gatol 10 -tao_converged_reason
    output_file: output/rosenbrock1_24.out

  test:
    suffix: 25
    args: -tao_grtol 10 -tao_converged_reason
    output_file: output/rosenbrock1_25.out

  test:
    suffix: 26
    args: -tao_gttol 10 -tao_converged_reason
    output_file: output/rosenbrock1_26.out

  test:
    suffix: 27
    args: -tao_steptol 10 -tao_converged_reason
    output_file: output/rosenbrock1_27.out

  test:
    suffix: 28
    args: -tao_fmin 10 -tao_converged_reason
    output_file: output/rosenbrock1_28.out

  test:
    suffix: test_dbfgs
    nsize: {{1 2 3}}
    output_file: output/rosenbrock1_14.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdbfgs -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lbfgs_type {{inplace reorder}} -tao_bqnk_mat_lbfgs_recursive {{0 1}}

  test:
    suffix: test_ddfp
    nsize: {{1 2 3}}
    output_file: output/rosenbrock1_14.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmddfp -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_ldfp_type {{inplace reorder}} -tao_bqnk_mat_ldfp_recursive {{0 1}}

  test:
    suffix: test_dqn_1
    nsize: 1
    output_file: output/rosenbrock1_29.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

  test:
    suffix: test_dqn_2
    nsize: 2
    output_file: output/rosenbrock1_30.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

  test:
    suffix: test_dqn_3
    nsize: 3
    output_file: output/rosenbrock1_31.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type none -tao_bqnk_mat_lqn_type {{inplace reorder}}

  test:
    suffix: test_dqn_diag
    output_file: output/rosenbrock1_32.out
    args: -n 10 -tao_type bqnktr -test_lmvm -tao_max_it 10 -tao_bqnk_mat_type lmvmdqn -tao_bqnk_mat_lmvm_scale_type diagonal -tao_bqnk_mat_lqn_type {{inplace reorder}}

  # test with a large update rejection tolerance to test that portion of code
  test:
    suffix: bfgs_rejects
    output_file: output/rosenbrock4_bfgs_rejects.out
    args: -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm -tao_view -n 10 -bs 10 -alpha 8.0 -tao_blmvm_mat_lmvm_eps 1.2 -tao_blmvm_mat_type lmvmbfgs -tao_blmvm_mat_lmvm_scale_type scalar -tao_blmvm_mat_lmvm_mult_algorithm {{recursive dense}}

  test:
    suffix: dbfgs_rejects
    output_file: output/rosenbrock4_dbfgs_rejects.out
    args: -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm -tao_view -n 10 -bs 10 -alpha 8.0 -tao_blmvm_mat_lmvm_eps 0.01 -tao_blmvm_mat_type lmvmdbfgs -tao_blmvm_mat_lmvm_scale_type scalar

TEST*/
