static char help[] = "Tests DMPlexTransformCreateSplitCellLabel().\n\n";

#include <petscdmplex.h>
#include <petscdmplextransform.h>

// Flags a single cell for refinement, which is what an error estimator would do
static PetscErrorCode CreateAdaptLabel(DM dm, DMLabel *adaptLabel)
{
  PetscInt cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", adaptLabel));
  PetscCall(DMLabelSetDefaultValue(*adaptLabel, DM_ADAPT_KEEP));
  if (cEnd > cStart) PetscCall(DMLabelSetValue(*adaptLabel, cStart, DM_ADAPT_REFINE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM              dm, rdm;
  DMPlexTransform tr;
  DMLabel         adaptLabel, splitLabel;
  PetscInt        cStart, cEnd, rcStart, rcEnd, numSplit;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  // Without this the refined mesh does not keep the transformation that produced it
  PetscCall(DMPlexSetSaveTransform(dm, PETSC_TRUE));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  PetscCall(CreateAdaptLabel(dm, &adaptLabel));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &rdm));
  PetscCheck(rdm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Adaptation did not produce a mesh");
  PetscCall(DMPlexGetHeightStratum(rdm, 0, &rcStart, &rcEnd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coarse cells: %" PetscInt_FMT " Refined cells: %" PetscInt_FMT "\n", cEnd - cStart, rcEnd - rcStart));

  PetscCall(DMPlexGetTransform(rdm, &tr));
  PetscCheck(tr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Refined mesh did not keep its transform");
  PetscCall(DMPlexTransformCreateSplitCellLabel(tr, rdm, &splitLabel));
  PetscCall(DMLabelGetStratumSize(splitLabel, 1, &numSplit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Cells from a split parent: %" PetscInt_FMT "\n", numSplit));
  // Every cell of a genuinely refined parent is marked, and cells passed through unchanged are not
  PetscCheck(numSplit > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "No cell came from a split parent, but the mesh grew");
  PetscCheck(numSplit <= rcEnd - rcStart, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Marked %" PetscInt_FMT " cells, but the mesh only has %" PetscInt_FMT, numSplit, rcEnd - rcStart);

  // The closure of the split cells is the set of entities whose star contains one, which is what PCPATCH wants
  PetscCall(DMPlexLabelComplete(rdm, splitLabel));
  PetscCall(DMLabelView(splitLabel, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMLabelDestroy(&splitLabel));
  PetscCall(DMLabelDestroy(&adaptLabel));
  PetscCall(DMDestroy(&rdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # A doublet is two tetrahedra, so SBR splits the flagged cell and its neighbor to stay conforming
  test:
    suffix: sbr
    args: -dm_adaptor cellrefiner -dm_plex_shape doublet -dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_transform_type refine_sbr

  test:
    suffix: regular
    args: -dm_adaptor cellrefiner -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_transform_type refine_regular

TEST*/
