static char help[] = "Demonstration of flexible submesh assembly.\n\n";

#include <petscksp.h>

int main(int argc, char **argv)
{
  Mat                    G, A, B, C, D;
  ISLocalToGlobalMapping rowMap, colMap;
  IS                     row, col;
  PetscInt              *locRows, *locCols;
  PetscInt               m = 5, n = 5, M = PETSC_DETERMINE, N = PETSC_DETERMINE, rStart, cStart;
  PetscBool              isnest;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // Fill up blocks with local lexicographic numbering
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &G));
  PetscCall(MatSetSizes(G, m, n, M, N));
  PetscCall(MatSetFromOptions(G));
  PetscCall(PetscObjectTypeCompare((PetscObject)G, MATNEST, &isnest));
  PetscCall(MatGetOwnershipRange(G, &rStart, NULL));
  PetscCall(MatGetOwnershipRangeColumn(G, &cStart, NULL));
  if (isnest) {
    Mat                    submat[4];
    PetscLayout            layoutA, layoutD;
    PetscInt              *locRowsA, *locColsA, *locRowsD, *locColsD, rStartA, rStartD;
    ISLocalToGlobalMapping rowMapA, colMapA, rowMapD, colMapD;

    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutA));
    PetscCall(PetscLayoutSetLocalSize(layoutA, 3));
    PetscCall(PetscLayoutSetUp(layoutA));
    PetscCall(PetscLayoutGetRange(layoutA, &rStartA, NULL));
    PetscCall(PetscLayoutDestroy(&layoutA));
    PetscCall(PetscMalloc1(3, &locRowsA));
    for (PetscInt r = 0; r < 3; ++r) locRowsA[r] = r + rStartA;
    PetscCall(PetscMalloc1(3, &locColsA));
    for (PetscInt c = 0; c < 3; ++c) locColsA[c] = c + rStartA;
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locRowsA, PETSC_OWN_POINTER, &rowMapA));
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locColsA, PETSC_OWN_POINTER, &colMapA));
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutD));
    PetscCall(PetscLayoutSetLocalSize(layoutD, 2));
    PetscCall(PetscLayoutSetUp(layoutD));
    PetscCall(PetscLayoutGetRange(layoutD, &rStartD, NULL));
    PetscCall(PetscLayoutDestroy(&layoutD));
    PetscCall(PetscMalloc1(2, &locRowsD));
    for (PetscInt r = 0; r < 2; ++r) locRowsD[r] = r + rStartD;
    PetscCall(PetscMalloc1(2, &locColsD));
    for (PetscInt c = 0; c < 2; ++c) locColsD[c] = c + rStartD;
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locRowsD, PETSC_OWN_POINTER, &rowMapD));
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locColsD, PETSC_OWN_POINTER, &colMapD));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[0]));
    PetscCall(MatSetSizes(submat[0], 3, 3, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(submat[0], MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(submat[0], rowMapA, colMapA));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[1]));
    PetscCall(MatSetSizes(submat[1], 3, 2, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(submat[1], MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(submat[1], rowMapA, colMapD));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[2]));
    PetscCall(MatSetSizes(submat[2], 2, 3, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(submat[2], MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(submat[2], rowMapD, colMapA));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[3]));
    PetscCall(MatSetSizes(submat[3], 2, 2, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(submat[3], MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(submat[3], rowMapD, colMapD));
    for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetUp(submat[i]));
    PetscCall(MatNestSetSubMats(G, 2, NULL, 2, NULL, submat));
    for (PetscInt i = 0; i < 4; ++i) PetscCall(MatDestroy(&submat[i]));

    PetscCall(ISLocalToGlobalMappingDestroy(&rowMapA));
    PetscCall(ISLocalToGlobalMappingDestroy(&colMapA));
    PetscCall(ISLocalToGlobalMappingDestroy(&rowMapD));
    PetscCall(ISLocalToGlobalMappingDestroy(&colMapD));
  }
  PetscCall(PetscMalloc1(m, &locRows));
  for (PetscInt r = 0; r < m; ++r) locRows[r] = r + rStart;
  PetscCall(PetscMalloc1(n, &locCols));
  for (PetscInt c = 0; c < n; ++c) locCols[c] = c + cStart;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, m, locRows, PETSC_OWN_POINTER, &rowMap));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, n, locCols, PETSC_OWN_POINTER, &colMap));
  PetscCall(MatSetLocalToGlobalMapping(G, rowMap, colMap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rowMap));
  PetscCall(ISLocalToGlobalMappingDestroy(&colMap));

  // (0,0) Block A
  PetscInt A_row[] = {0, 1, 2};
  PetscInt A_col[] = {0, 1, 2};

  m = PETSC_STATIC_ARRAY_LENGTH(A_row);
  n = PETSC_STATIC_ARRAY_LENGTH(A_col);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, A_row, PETSC_COPY_VALUES, &row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, A_col, PETSC_COPY_VALUES, &col));
  PetscCall(MatGetLocalSubMatrix(G, row, col, &A));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  for (PetscInt i = 0; i < m; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      PetscScalar v = i * n + j;

      PetscCall(MatSetValuesLocal(A, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatDestroy(&A));

  // (0,1) Block B
  PetscInt B_row[] = {0, 1, 2};
  PetscInt B_col[] = {3, 4};

  m = PETSC_STATIC_ARRAY_LENGTH(B_row);
  n = PETSC_STATIC_ARRAY_LENGTH(B_col);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, B_row, PETSC_COPY_VALUES, &row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, B_col, PETSC_COPY_VALUES, &col));
  PetscCall(MatGetLocalSubMatrix(G, row, col, &B));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  for (PetscInt i = 0; i < m; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      PetscScalar v = i * n + j;

      PetscCall(MatSetValuesLocal(B, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatDestroy(&B));

  // (0,1) Block C
  PetscInt C_row[] = {3, 4};
  PetscInt C_col[] = {0, 1, 2};

  m = PETSC_STATIC_ARRAY_LENGTH(C_row);
  n = PETSC_STATIC_ARRAY_LENGTH(C_col);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, C_row, PETSC_COPY_VALUES, &row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, C_col, PETSC_COPY_VALUES, &col));
  PetscCall(MatGetLocalSubMatrix(G, row, col, &C));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  for (PetscInt i = 0; i < m; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      PetscScalar v = i * n + j;

      PetscCall(MatSetValuesLocal(C, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatDestroy(&C));

  // (0,0) Block D
  PetscInt D_row[] = {3, 4};
  PetscInt D_col[] = {3, 4};

  m = PETSC_STATIC_ARRAY_LENGTH(D_row);
  n = PETSC_STATIC_ARRAY_LENGTH(D_col);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, D_row, PETSC_COPY_VALUES, &row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, D_col, PETSC_COPY_VALUES, &col));
  PetscCall(MatGetLocalSubMatrix(G, row, col, &D));
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  for (PetscInt i = 0; i < m; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      PetscScalar v = i * n + j;

      PetscCall(MatSetValuesLocal(D, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatDestroy(&D));

  PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(G, NULL, "-G_view"));
  PetscCall(MatDestroy(&G));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
    args: -mat_type aij -G_view

  test:
    suffix: 1
    args: -mat_type nest -G_view -mat_view_nest_sub

  test:
    suffix: 2
    nsize: 2
    args: -mat_type aij -G_view

  test:
    suffix: 3
    nsize: 2
    args: -mat_type nest -G_view
TEST*/
