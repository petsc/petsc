#include <petscksp.h>
#include <petscpc.h>
#include <petscviewer.h>

typedef struct {
  PetscInt  num_levels;
  PetscInt *n_per_level;
  Mat       stiff;
  Mat      *ProlongationOps;
  PetscBT  *CFMarkers;
  KSP       kspHypre;
} *DataCompression;

PetscErrorCode Create1dLaplacian(PetscInt, Mat *);
PetscErrorCode DataCompExportMats(DataCompression);
PetscErrorCode DataCompDestroy(DataCompression);

int main(int Argc, char **Args)
{
  PetscInt        n_nodes = 33;
  Vec             x, b;
  PC              pcHypre;
  DataCompression data_comp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&Argc, &Args, NULL, NULL));

  PetscCall(PetscNew(&data_comp));

  // Creating stiffness matrix
  PetscCall(Create1dLaplacian(n_nodes, &data_comp->stiff));
  PetscCall(PetscObjectSetName((PetscObject)data_comp->stiff, "Stiffness"));

  // Set-up BoomerAMG PC to get Prolongation Operators and Coarse/Fine splittings
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &data_comp->kspHypre));
  PetscCall(KSPSetType(data_comp->kspHypre, KSPRICHARDSON));
  PetscCall(KSPGetPC(data_comp->kspHypre, &pcHypre));
  PetscCall(PCSetType(pcHypre, PCHYPRE));
  PetscCall(PCHYPRESetType(pcHypre, "boomeramg"));
  PetscCall(PCSetFromOptions(pcHypre));
  PetscCall(PCSetOperators(pcHypre, data_comp->stiff, data_comp->stiff));
  PetscCall(PCSetUp(pcHypre));

  PetscCall(MatCreateVecs(data_comp->stiff, &x, &b));
  PetscCall(PCApply(pcHypre, x, b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));

  //Viewing the PC and Extracting the Prolongation Operators and CFMarkers
  PetscCall(PCView(pcHypre, NULL));
  PetscCall(PCGetInterpolations(pcHypre, &data_comp->num_levels, &data_comp->ProlongationOps));
  PetscCall(PCHYPREGetCFMarkers(pcHypre, &data_comp->n_per_level, &data_comp->CFMarkers));

  PetscCall(DataCompExportMats(data_comp));
  PetscCall(DataCompDestroy(data_comp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode Create1dLaplacian(PetscInt n, Mat *mat)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 3, NULL, mat));
  PetscCall(MatSetValue(*mat, n - 1, n - 1, 2.0, INSERT_VALUES));
  for (PetscInt i = 0; i < n - 1; i++) {
    PetscCall(MatSetValue(*mat, i, i, 2.0, INSERT_VALUES));
    PetscCall(MatSetValue(*mat, i + 1, i, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(*mat, i, i + 1, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCompExportMats(DataCompression data_comp)
{
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Num levels: %" PetscInt_FMT "\n", data_comp->num_levels));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " -- Nodes per level --\n"));
  for (PetscInt i = 0; i < data_comp->num_levels; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Level %" PetscInt_FMT ": %" PetscInt_FMT "\n", i, data_comp->n_per_level[i]));

  for (PetscInt i = 0; i < data_comp->num_levels - 1; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Prolongation Operator - Level %" PetscInt_FMT "\n", i));
    PetscCall(PetscObjectSetName((PetscObject)data_comp->ProlongationOps[i], "P"));
    PetscCall(MatView(data_comp->ProlongationOps[i], PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }

  for (PetscInt i = 0; i < data_comp->num_levels - 1; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coarse/Fine splitting - Level %" PetscInt_FMT "\n", i + 1));
    PetscCall(PetscBTView(data_comp->n_per_level[i + 1], data_comp->CFMarkers[i], PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stiffness matrix, sparse format:\n"));
  PetscCall(MatViewFromOptions(data_comp->stiff, NULL, "-mat_view_stiff"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Finished calling the Viewer functions\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCompDestroy(DataCompression data_comp)
{
  PetscFunctionBeginUser;
  if (data_comp == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&data_comp->stiff));
  PetscCall(KSPDestroy(&data_comp->kspHypre));
  for (PetscInt i = 0; i < data_comp->num_levels - 1; i++) {
    PetscCall(MatDestroy(&data_comp->ProlongationOps[i]));
    PetscCall(PetscBTDestroy(&data_comp->CFMarkers[i]));
  }
  PetscCall(PetscFree(data_comp->ProlongationOps));
  PetscCall(PetscFree(data_comp->n_per_level));
  PetscCall(PetscFree(data_comp->CFMarkers));
  PetscCall(PetscFree(data_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      requires: hypre !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -pc_hypre_boomeramg_coarsen_type modifiedRuge-Stueben -pc_hypre_boomeramg_interp_type classical -pc_hypre_boomeramg_strong_threshold 0.25 pc_hypre_boomeramg_numfunctions 1 -pc_hypre_boomeramg_max_row_sum 1.0 -mat_view_stiff

TEST*/
