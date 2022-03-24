static char help[] = "Test for mesh reordering\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;               /* The topological mesh dimension */
  PetscReal refinementLimit;   /* Maximum volume of a refined cell */
  PetscInt  numFields;         /* The number of section fields */
  PetscInt *numComponents;     /* The number of field components */
  PetscInt *numDof;            /* The dof signature for the section */
  PetscInt  numGroups;         /* If greater than 1, use grouping in test */
} AppCtx;

PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscInt       len;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->numFields     = 1;
  options->numComponents = NULL;
  options->numDof        = NULL;
  options->numGroups     = 0;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-num_fields", "The number of section fields", "ex10.c", options->numFields, &options->numFields, NULL,1));
  if (options->numFields) {
    len  = options->numFields;
    CHKERRQ(PetscCalloc1(len, &options->numComponents));
    CHKERRQ(PetscOptionsIntArray("-num_components", "The number of components per field", "ex10.c", options->numComponents, &len, &flg));
    PetscCheckFalse(flg && (len != options->numFields),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %D should be %D", len, options->numFields);
  }
  CHKERRQ(PetscOptionsBoundedInt("-num_groups", "Group permutation by this many label values", "ex10.c", options->numGroups, &options->numGroups, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CleanupContext(AppCtx *user)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(user->numComponents));
  CHKERRQ(PetscFree(user->numDof));
  PetscFunctionReturn(0);
}

/* This mesh comes from~\cite{saad2003}, Fig. 2.10, p. 70. */
PetscErrorCode CreateTestMesh(MPI_Comm comm, DM *dm, AppCtx *options)
{
  const PetscInt    cells[16*3]  = {6, 7, 8,   7, 9, 10,  10, 11, 12,  11, 13, 14,   0,  6, 8,  6,  2,  7,   1, 8,  7,   1,  7, 10,
                                    2, 9, 7,  10, 9,  4,   1, 10, 12,  10,  4, 11,  12, 11, 3,  3, 11, 14,  11, 4, 13,  14, 13,  5};
  const PetscReal   coords[15*2] = {0, -3,  0, -1,  2, -1,  0,  1,  2, 1,
                                    0,  3,  1, -2,  1, -1,  0, -2,  2, 0,
                                    1,  0,  1,  1,  0,  0,  1,  2,  0, 2};

  PetscFunctionBegin;
  CHKERRQ(DMPlexCreateFromCellListPetsc(comm, 2, 16, 15, 3, PETSC_FALSE, cells, 2, coords, dm));
  PetscFunctionReturn(0);
}

PetscErrorCode TestReordering(DM dm, AppCtx *user)
{
  DM              pdm;
  IS              perm;
  Mat             A, pA;
  PetscInt        bw, pbw;
  MatOrderingType order = MATORDERINGRCM;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetOrdering(dm, order, NULL, &perm));
  CHKERRQ(DMPlexPermute(dm, perm, &pdm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) pdm, "perm_"));
  CHKERRQ(DMSetFromOptions(pdm));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(DMViewFromOptions(dm,  NULL, "-orig_dm_view"));
  CHKERRQ(DMViewFromOptions(pdm, NULL, "-dm_view"));
  CHKERRQ(DMCreateMatrix(dm, &A));
  CHKERRQ(DMCreateMatrix(pdm, &pA));
  CHKERRQ(MatComputeBandwidth(A, 0.0, &bw));
  CHKERRQ(MatComputeBandwidth(pA, 0.0, &pbw));
  CHKERRQ(MatViewFromOptions(A,  NULL, "-orig_mat_view"));
  CHKERRQ(MatViewFromOptions(pA, NULL, "-perm_mat_view"));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&pA));
  CHKERRQ(DMDestroy(&pdm));
  if (pbw > bw) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Ordering method %s increased bandwidth from %D to %D\n", order, bw, pbw));
  } else {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Ordering method %s reduced bandwidth from %D to %D\n", order, bw, pbw));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateGroupLabel(DM dm, PetscInt numGroups, DMLabel *label, AppCtx *options)
{
  const PetscInt groupA[10] = {15, 3, 13, 12, 2, 10, 7, 6, 0, 4};
  const PetscInt groupB[6]  = {14, 11, 9, 1, 8, 5};
  PetscInt       c;

  PetscFunctionBegin;
  if (numGroups < 2) {*label = NULL; PetscFunctionReturn(0);}
  PetscCheckFalse(numGroups != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Test only coded for 2 groups, not %D", numGroups);
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "groups", label));
  for (c = 0; c < 10; ++c) CHKERRQ(DMLabelSetValue(*label, groupA[c], 101));
  for (c = 0; c < 6;  ++c) CHKERRQ(DMLabelSetValue(*label, groupB[c], 1001));
  PetscFunctionReturn(0);
}

PetscErrorCode TestReorderingByGroup(DM dm, AppCtx *user)
{
  DM              pdm;
  DMLabel         label;
  Mat             A, pA;
  MatOrderingType order = MATORDERINGRCM;
  IS              perm;

  PetscFunctionBegin;
  CHKERRQ(CreateGroupLabel(dm, user->numGroups, &label, user));
  CHKERRQ(DMPlexGetOrdering(dm, order, label, &perm));
  CHKERRQ(DMLabelDestroy(&label));
  CHKERRQ(DMPlexPermute(dm, perm, &pdm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) pdm, "perm_"));
  CHKERRQ(DMSetFromOptions(pdm));
  CHKERRQ(DMViewFromOptions(dm,  NULL, "-orig_dm_view"));
  CHKERRQ(DMViewFromOptions(pdm, NULL, "-perm_dm_view"));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(DMCreateMatrix(dm, &A));
  CHKERRQ(DMCreateMatrix(pdm, &pA));
  CHKERRQ(MatViewFromOptions(A,  NULL, "-orig_mat_view"));
  CHKERRQ(MatViewFromOptions(pA, NULL, "-perm_mat_view"));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&pA));
  CHKERRQ(DMDestroy(&pdm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   s;
  AppCtx         user;
  PetscInt       dim;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(&user));
  if (user.numGroups < 1) {
    CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
    CHKERRQ(DMSetType(dm, DMPLEX));
  } else {
    CHKERRQ(CreateTestMesh(PETSC_COMM_WORLD, &dm, &user));
  }
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(dm, &dim));
  {
    PetscInt  len = (dim+1) * PetscMax(1, user.numFields);
    PetscBool flg;

    CHKERRQ(PetscCalloc1(len, &user.numDof));
    ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsIntArray("-num_dof", "The dof signature for the section", "ex10.c", user.numDof, &len, &flg));
    PetscCheckFalse(flg && (len != (dim+1) * PetscMax(1, user.numFields)),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of dof array is %D should be %D", len, (dim+1) * PetscMax(1, user.numFields));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (user.numGroups < 1) {
    CHKERRQ(DMSetNumFields(dm, user.numFields));
    CHKERRQ(DMCreateDS(dm));
    CHKERRQ(DMPlexCreateSection(dm, NULL, user.numComponents, user.numDof, 0, NULL, NULL, NULL, NULL, &s));
    CHKERRQ(DMSetLocalSection(dm, s));
    CHKERRQ(PetscSectionDestroy(&s));
    CHKERRQ(TestReordering(dm, &user));
  } else {
    CHKERRQ(DMSetNumFields(dm, user.numFields));
    CHKERRQ(DMCreateDS(dm));
    CHKERRQ(DMPlexCreateSection(dm, NULL, user.numComponents, user.numDof, 0, NULL, NULL, NULL, NULL, &s));
    CHKERRQ(DMSetLocalSection(dm, s));
    CHKERRQ(PetscSectionDestroy(&s));
    CHKERRQ(TestReorderingByGroup(dm, &user));
  }
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(CleanupContext(&user));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  # Two cell tests 0-3
  test:
    suffix: 0
    requires: triangle
    args: -dm_plex_simplex 1 -num_dof 1,0,0 -mat_view -dm_coord_space 0
  test:
    suffix: 1
    args: -dm_plex_simplex 0 -num_dof 1,0,0 -mat_view -dm_coord_space 0
  test:
    suffix: 2
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_simplex 1 -num_dof 1,0,0,0 -mat_view -dm_coord_space 0
  test:
    suffix: 3
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -num_dof 1,0,0,0 -mat_view -dm_coord_space 0
  # Refined tests 4-7
  test:
    suffix: 4
    requires: triangle
    args: -dm_plex_simplex 1 -dm_refine_volume_limit_pre 0.00625 -num_dof 1,0,0
  test:
    suffix: 5
    args: -dm_plex_simplex 0 -dm_refine 1 -num_dof 1,0,0
  test:
    suffix: 6
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_simplex 1 -dm_refine_volume_limit_pre 0.00625 -num_dof 1,0,0,0
  test:
    suffix: 7
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_refine 1 -num_dof 1,0,0,0
  # Parallel tests
  # Grouping tests
  test:
    suffix: group_1
    args: -num_groups 1 -num_dof 1,0,0 -is_view -orig_mat_view -perm_mat_view
  test:
    suffix: group_2
    args: -num_groups 2 -num_dof 1,0,0 -is_view -perm_mat_view

TEST*/
