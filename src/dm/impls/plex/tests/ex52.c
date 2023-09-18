const char help[] = "Test clearing stale AMR data (example contributed by Berend van Wachem)";

#include <petscdmplex.h>
#include <petscdmforest.h>

PetscErrorCode CloneDMWithNewSection(DM OriginalDM, DM *NewDM, PetscInt NFields)
{
  PetscFunctionBegin;

  PetscSection section;
  PetscInt    *NumComp, *NumDof;
  PetscCall(DMClone(OriginalDM, NewDM));
  PetscCall(DMPlexDistributeSetDefault(*NewDM, PETSC_FALSE));
  PetscCall(DMClearDS(*NewDM));
  PetscCall(PetscCalloc2(1, &NumComp, 3, &NumDof));
  NumComp[0] = 1;
  NumDof[2]  = NFields;
  PetscCall(DMSetNumFields(*NewDM, 1));
  PetscCall(DMSetFromOptions(*NewDM));
  PetscCall(DMPlexCreateSection(*NewDM, NULL, NumComp, NumDof, 0, NULL, NULL, NULL, NULL, &section));
  PetscCall(DMSetLocalSection(*NewDM, section));
  PetscCall(PetscFree2(NumComp, NumDof));
  PetscCall(PetscSectionDestroy(&section));
  PetscFE fe;
  PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, 2, 1, PETSC_FALSE, NULL, PETSC_DEFAULT, &fe));
  PetscCall(DMSetField(*NewDM, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(*NewDM));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm       comm            = PETSC_COMM_WORLD;
  PetscInt       dim             = 2;
  PetscInt       cells_per_dir[] = {1, 1};
  PetscReal      dir_min[]       = {0.0, 0.0};
  PetscReal      dir_max[]       = {1.0, 1.0};
  DMBoundaryType bcs[]           = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  DM             forest;
  DM             NewDM;
  Vec            NewDMVecGlobal, NewDMVecLocal;

  PetscCall(DMCreate(comm, &forest));
  PetscCall(DMSetType(forest, DMP4EST));
  {
    DM dm_base;
    PetscCall(DMPlexCreateBoxMesh(comm, dim, /* simplex */ PETSC_FALSE, cells_per_dir, dir_min, dir_max, bcs, /* interpolate */ PETSC_TRUE, &dm_base));
    PetscCall(DMSetFromOptions(dm_base));
    PetscCall(DMViewFromOptions(dm_base, NULL, "-dm_base_view"));
    PetscCall(DMCopyFields(dm_base, forest));
    PetscCall(DMForestSetBaseDM(forest, dm_base));
    PetscCall(DMDestroy(&dm_base));
  }
  PetscCall(DMSetFromOptions(forest));
  PetscCall(DMSetUp(forest));

  PetscCall(DMViewFromOptions(forest, NULL, "-dm_forest_view"));
  DM plex;

  PetscCall(DMConvert(forest, DMPLEX, &plex));

  PetscInt numFields  = 2;
  PetscInt numComp[2] = {1, 1};
  PetscInt numDof[6]  = {0};
  for (PetscInt i = 0; i < numFields; i++) numDof[i * (dim + 1) + dim] = 1;

  PetscCall(DMSetNumFields(plex, numFields));
  PetscCall(DMSetNumFields(forest, numFields));

  PetscSection section;
  PetscCall(DMPlexCreateSection(plex, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, &section));

  const char *names[] = {"field 0", "field 1"};
  for (PetscInt i = 0; i < numFields; i++) PetscCall(PetscSectionSetFieldName(section, i, names[i]));
  PetscCall(DMSetLocalSection(forest, section));
  PetscCall(DMSetLocalSection(plex, section));
  PetscCall(PetscSectionDestroy(&section));

  PetscFE fe;
  PetscCall(PetscFECreateDefault(comm, dim, 1, PETSC_FALSE, NULL, PETSC_DEFAULT, &fe));
  for (PetscInt i = 0; i < numFields; i++) {
    PetscCall(DMSetField(plex, i, NULL, (PetscObject)fe));
    PetscCall(DMSetField(forest, i, NULL, (PetscObject)fe));
  }
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMCreateDS(plex));
  PetscCall(DMCreateDS(forest));

  /* Make another DM, based on the layout of the previous DM, but with a different number of fields */
  PetscCall(CloneDMWithNewSection(plex, &NewDM, 1));
  PetscCall(DMCreateDS(NewDM));
  PetscCall(DMCreateGlobalVector(NewDM, &NewDMVecGlobal));
  PetscCall(DMCreateLocalVector(NewDM, &NewDMVecLocal));
  PetscCall(VecSet(NewDMVecGlobal, 3.141592));
  PetscCall(DMGlobalToLocal(NewDM, NewDMVecGlobal, INSERT_VALUES, NewDMVecLocal));
  PetscCall(VecDestroy(&NewDMVecLocal));
  PetscCall(VecDestroy(&NewDMVecGlobal));
  PetscCall(DMClearDS(NewDM));
  PetscCall(DMDestroy(&NewDM));

  Vec g_vec, l_vec;
  PetscCall(DMCreateGlobalVector(plex, &g_vec));
  PetscCall(VecSet(g_vec, 1.0));
  PetscCall(DMCreateLocalVector(plex, &l_vec));
  PetscCall(DMGlobalToLocal(plex, g_vec, INSERT_VALUES, l_vec));
  PetscCall(VecViewFromOptions(l_vec, NULL, "-local_vec_view"));
  PetscCall(VecDestroy(&l_vec));
  PetscCall(VecDestroy(&g_vec));

  PetscCall(DMViewFromOptions(forest, NULL, "-dm_plex_view"));
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&forest));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: p4est
    args: -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash

TEST*/
