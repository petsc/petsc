static char help[] = "Test for mesh reordering\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool cellSimplex;       /* Flag for simplices */
  PetscBool interpolate;       /* Flag for mesh interpolation */
  PetscBool refinementUniform; /* Uniformly refine the mesh */
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
  options->dim               = 2;
  options->cellSimplex       = PETSC_TRUE;
  options->interpolate       = PETSC_FALSE;
  options->refinementUniform = PETSC_FALSE;
  options->refinementLimit   = 0.0;
  options->numFields         = 1;
  options->numComponents     = NULL;
  options->numDof            = NULL;
  options->numGroups         = 0;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex10.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Flag for simplices", "ex10.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Flag for mesh interpolation", "ex10.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refinement_uniform", "Uniformly refine the mesh", "ex10.c", options->refinementUniform, &options->refinementUniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The maximum volume of a refined cell", "ex10.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_fields", "The number of section fields", "ex10.c", options->numFields, &options->numFields, NULL,1);CHKERRQ(ierr);
  if (options->numFields) {
    len  = options->numFields;
    ierr = PetscCalloc1(len, &options->numComponents);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-num_components", "The number of components per field", "ex10.c", options->numComponents, &len, &flg);CHKERRQ(ierr);
    if (flg && (len != options->numFields)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %D should be %D", len, options->numFields);
  }
  len  = (options->dim+1) * PetscMax(1, options->numFields);
  ierr = PetscMalloc1(len, &options->numDof);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-num_dof", "The dof signature for the section", "ex10.c", options->numDof, &len, &flg);CHKERRQ(ierr);
  if (flg && (len != (options->dim+1) * PetscMax(1, options->numFields))) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of dof array is %D should be %D", len, (options->dim+1) * PetscMax(1, options->numFields));
  ierr = PetscOptionsBoundedInt("-num_groups", "Group permutation by this many label values", "ex10.c", options->numGroups, &options->numGroups, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CleanupContext(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(user->numComponents);CHKERRQ(ierr);
  ierr = PetscFree(user->numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This mesh comes from~\cite{saad2003}, Fig. 2.10, p. 70. */
PetscErrorCode CreateTestMesh(MPI_Comm comm, DM *dm, AppCtx *options)
{
  const int         cells[16*3]  = {6, 7, 8,   7, 9, 10,  10, 11, 12,  11, 13, 14,   0,  6, 8,  6,  2,  7,   1, 8,  7,   1,  7, 10,
                                    2, 9, 7,  10, 9,  4,   1, 10, 12,  10,  4, 11,  12, 11, 3,  3, 11, 14,  11, 4, 13,  14, 13,  5};
  const double      coords[15*2] = {0, -3,  0, -1,  2, -1,  0,  1,  2, 1,
                                    0,  3,  1, -2,  1, -1,  0, -2,  2, 0,
                                    1,  0,  1,  1,  0,  0,  1,  2,  0, 2};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateFromCellList(comm, 2, 16, 15, 3, options->interpolate, cells, 2, coords, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestReordering(DM dm, AppCtx *user)
{
  DM              pdm;
  IS              perm;
  Mat             A, pA;
  PetscInt        bw, pbw;
  MatOrderingType order = MATORDERINGRCM;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetOrdering(dm, order, NULL, &perm);CHKERRQ(ierr);
  ierr = DMPlexPermute(dm, perm, &pdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(pdm);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &A);CHKERRQ(ierr);
  ierr = DMCreateMatrix(pdm, &pA);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(A, 0.0, &bw);CHKERRQ(ierr);
  ierr = MatComputeBandwidth(pA, 0.0, &pbw);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&pA);CHKERRQ(ierr);
  ierr = DMDestroy(&pdm);CHKERRQ(ierr);
  if (pbw > bw) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Ordering method %s increased bandwidth from %D to %D\n", order, bw, pbw);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Ordering method %s reduced bandwidth from %D to %D\n", order, bw, pbw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateGroupLabel(DM dm, PetscInt numGroups, DMLabel *label, AppCtx *options)
{
  const PetscInt groupA[10] = {15, 3, 13, 12, 2, 10, 7, 6, 0, 4};
  const PetscInt groupB[6]  = {14, 11, 9, 1, 8, 5};
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numGroups < 2) {*label = NULL; PetscFunctionReturn(0);}
  if (numGroups != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Test only coded for 2 groups, not %D", numGroups);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "groups", label);CHKERRQ(ierr);
  for (c = 0; c < 10; ++c) {ierr = DMLabelSetValue(*label, groupA[c], 101);CHKERRQ(ierr);}
  for (c = 0; c < 6;  ++c) {ierr = DMLabelSetValue(*label, groupB[c], 1001);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode TestReorderingByGroup(DM dm, AppCtx *user)
{
  DM              pdm;
  DMLabel         label;
  Mat             A, pA;
  MatOrderingType order = MATORDERINGRCM;
  IS              perm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = CreateGroupLabel(dm, user->numGroups, &label, user);CHKERRQ(ierr);
  ierr = DMPlexGetOrdering(dm, order, label, &perm);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  ierr = DMPlexPermute(dm, perm, &pdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(pdm);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &A);CHKERRQ(ierr);
  ierr = DMCreateMatrix(pdm, &pA);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,  NULL, "-orig_mat_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(pA, NULL, "-perm_mat_view");CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&pA);CHKERRQ(ierr);
  ierr = DMDestroy(&pdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   s;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(&user);CHKERRQ(ierr);
  if (user.numGroups < 1) {
    ierr = DMPlexCreateDoublet(PETSC_COMM_WORLD, user.dim, user.cellSimplex, user.interpolate, user.refinementUniform, user.refinementLimit, &dm);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    ierr = DMSetNumFields(dm, user.numFields);CHKERRQ(ierr);
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(dm, NULL, user.numComponents, user.numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
    ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
    ierr = TestReordering(dm, &user);CHKERRQ(ierr);
  } else {
    ierr = CreateTestMesh(PETSC_COMM_WORLD, &dm, &user);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    ierr = DMSetNumFields(dm, user.numFields);CHKERRQ(ierr);
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(dm, NULL, user.numComponents, user.numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
    ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
    ierr = TestReorderingByGroup(dm, &user);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = CleanupContext(&user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Two cell tests 0-3
  test:
    suffix: 0
    args: -dim 2 -interpolate 1 -cell_simplex 1 -num_dof 1,0,0 -mat_view
  test:
    suffix: 1
    args: -dim 2 -interpolate 1 -cell_simplex 0 -num_dof 1,0,0 -mat_view
  test:
    suffix: 2
    args: -dim 3 -interpolate 1 -cell_simplex 1 -num_dof 1,0,0,0 -mat_view
  test:
    suffix: 3
    args: -dim 3 -interpolate 1 -cell_simplex 0 -num_dof 1,0,0,0 -mat_view
  # Refined tests 4-7
  test:
    suffix: 4
    requires: triangle
    args: -dim 2 -interpolate 1 -cell_simplex 1 -refinement_limit 0.00625 -num_dof 1,0,0
  test:
    suffix: 5
    args: -dim 2 -interpolate 1 -cell_simplex 0 -refinement_uniform       -num_dof 1,0,0
  test:
    suffix: 6
    requires: ctetgen
    args: -dim 3 -interpolate 1 -cell_simplex 1 -refinement_limit 0.00625 -num_dof 1,0,0,0
  test:
    suffix: 7
    args: -dim 3 -interpolate 1 -cell_simplex 0 -refinement_uniform       -num_dof 1,0,0,0
  # Parallel tests
  # Grouping tests
  test:
    suffix: group_1
    args: -num_groups 1 -num_dof 1,0,0 -is_view -orig_mat_view -perm_mat_view
  test:
    suffix: group_2
    args: -num_groups 2 -num_dof 1,0,0 -is_view -perm_mat_view

TEST*/
