static const char help[] = "Solves a Q2-Q1 Navier-Stokes problem.\n\n";

#include <petscksp.h>
#include <petscdmda.h>

PetscErrorCode LSCLoadOperators(Mat *A, Mat *Q, Mat *L, Vec *rhs, IS *velocity, IS *pressure)
{
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN];
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatCreate(PETSC_COMM_WORLD, Q));
  if (L) PetscCall(MatCreate(PETSC_COMM_WORLD, L));
  PetscCall(ISCreate(PETSC_COMM_WORLD, velocity));
  PetscCall(ISCreate(PETSC_COMM_WORLD, pressure));
  PetscCall(VecCreate(PETSC_COMM_WORLD, rhs));
  /* Load matrices from a Q2-Q1 discretisation of Navier-Stokes. The data is packed into one file. */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must provide a data file with -f");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(*A, viewer));
  PetscCall(VecLoad(*rhs, viewer));
  PetscCall(ISLoad(*velocity, viewer));
  PetscCall(ISLoad(*pressure, viewer));
  PetscCall(MatLoad(*Q, viewer));
  if (L) PetscCall(MatLoad(*L, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode port_lsd_bfbt(void)
{
  Mat       A, Q, L = NULL;
  Vec       x, b;
  KSP       ksp_A;
  PC        pc_A;
  IS        isu, isp;
  PetscBool commute_lsc = PETSC_FALSE;
  KSP      *subksp; // This will be length two, with the former being the A KSP and the latter being the
                    // Schur complement KSP
  KSP      schur_complement_ksp;
  PC       lsc_pc;
  PetscInt num_splits;
  Mat      lsc_pc_pmat;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-commute_lsc", &commute_lsc, NULL));
  if (commute_lsc) PetscCall(LSCLoadOperators(&A, &Q, &L, &b, &isu, &isp));
  else PetscCall(LSCLoadOperators(&A, &Q, NULL, &b, &isu, &isp));
  PetscCall(VecDuplicate(b, &x));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_A));
  PetscCall(KSPSetOptionsPrefix(ksp_A, "fc_"));
  PetscCall(KSPSetOperators(ksp_A, A, A));

  PetscCall(KSPSetFromOptions(ksp_A));
  PetscCall(KSPGetPC(ksp_A, &pc_A));
  PetscCall(PCFieldSplitSetBlockSize(pc_A, 2));
  PetscCall(PCFieldSplitSetIS(pc_A, "velocity", isu));
  PetscCall(PCFieldSplitSetIS(pc_A, "pressure", isp));

  // Need to call this before getting sub ksps, etc.
  PetscCall(PCSetUp(pc_A));
  PetscCall(PCFieldSplitGetSubKSP(pc_A, &num_splits, &subksp));
  schur_complement_ksp = subksp[1];

  PetscCall(KSPGetPC(schur_complement_ksp, &lsc_pc));
  PetscCall(PCGetOperators(lsc_pc, NULL, &lsc_pc_pmat));
  PetscCall(PetscObjectCompose((PetscObject)lsc_pc_pmat, "LSC_Qscale", (PetscObject)Q));
  if (commute_lsc) PetscCall(PetscObjectCompose((PetscObject)lsc_pc_pmat, "LSC_L", (PetscObject)L));

  PetscCall(KSPSolve(ksp_A, b, x));

  PetscCall(KSPDestroy(&ksp_A));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Q));
  if (L) PetscCall(MatDestroy(&L));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&isu));
  PetscCall(ISDestroy(&isp));
  PetscCall(PetscFree(subksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(port_lsd_bfbt());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: elman
      args: -f ${DATAFILESPATH}/matrices/elman -fc_ksp_monitor -fc_ksp_type fgmres -fc_ksp_max_it 100 -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type full -fc_fieldsplit_velocity_ksp_type gmres -fc_fieldsplit_velocity_pc_type lu -fc_fieldsplit_pressure_ksp_max_it 100 -fc_fieldsplit_pressure_ksp_monitor -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type gmres -fc_fieldsplit_pressure_lsc_ksp_max_it 100 -fc_fieldsplit_pressure_lsc_pc_type hypre -fc_fieldsplit_pressure_lsc_pc_hypre_type boomeramg -commute_lsc 0 -fc_ksp_converged_reason -fc_fieldsplit_pressure_ksp_converged_reason -fc_fieldsplit_pressure_ksp_type fgmres -fc_fieldsplit_pressure_pc_lsc_scale_diag
      requires: datafilespath double hypre !complex defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: olshanskii
      args: -f ${DATAFILESPATH}/matrices/olshanskii -fc_ksp_monitor -fc_ksp_type fgmres -fc_ksp_max_it 100 -fc_pc_type fieldsplit -fc_pc_fieldsplit_type SCHUR -fc_pc_fieldsplit_schur_fact_type full -fc_fieldsplit_velocity_ksp_type gmres -fc_fieldsplit_velocity_pc_type lu -fc_fieldsplit_pressure_ksp_max_it 100 -fc_fieldsplit_pressure_ksp_monitor -fc_fieldsplit_pressure_pc_type lsc -fc_fieldsplit_pressure_lsc_ksp_type gmres -fc_fieldsplit_pressure_lsc_ksp_max_it 100 -fc_fieldsplit_pressure_lsc_pc_type hypre -fc_fieldsplit_pressure_lsc_pc_hypre_type boomeramg -commute_lsc 1 -fc_ksp_converged_reason -fc_fieldsplit_pressure_ksp_converged_reason -fc_fieldsplit_pressure_ksp_type fgmres -fc_fieldsplit_pressure_lsc_mass_pc_type lu -fc_fieldsplit_pressure_lsc_mass_ksp_type gmres -fc_fieldsplit_pressure_lsc_mass_ksp_pc_side right  -fc_fieldsplit_pressure_pc_lsc_commute 1
      requires: datafilespath double hypre !complex defined(PETSC_USE_64BIT_INDICES)

TEST*/
