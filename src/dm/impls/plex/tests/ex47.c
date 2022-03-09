static char help[] = "The main goal of this code is to retrieve the original element numbers as found in the "
                     "initial partitions (sInitialPartition)... but after the call to DMPlexDistribute";

#include <petsc.h>

PetscReal sCoords2x5Mesh[18][2] = {
 {0.00000000000000000e+00, 0.00000000000000000e+00},
 {2.00000000000000000e+00, 0.00000000000000000e+00},
 {0.00000000000000000e+00, 1.00000000000000000e+00},
 {2.00000000000000000e+00, 1.00000000000000000e+00},
 {9.99999999997387978e-01, 0.00000000000000000e+00},
 {9.99999999997387978e-01, 1.00000000000000000e+00},
 {0.00000000000000000e+00, 2.00000000000000011e-01},
 {0.00000000000000000e+00, 4.00000000000000022e-01},
 {0.00000000000000000e+00, 5.99999999999999978e-01},
 {0.00000000000000000e+00, 8.00000000000000044e-01},
 {2.00000000000000000e+00, 2.00000000000000011e-01},
 {2.00000000000000000e+00, 4.00000000000000022e-01},
 {2.00000000000000000e+00, 5.99999999999999978e-01},
 {2.00000000000000000e+00, 8.00000000000000044e-01},
 {9.99999999997387756e-01, 2.00000000000000011e-01},
 {9.99999999997387978e-01, 4.00000000000000022e-01},
 {9.99999999997387978e-01, 6.00000000000000089e-01},
 {9.99999999997388089e-01, 8.00000000000000044e-01}};

//Connectivity of a 2x5 rectangular mesh of quads :
const PetscInt sConnectivity2x5Mesh[10][4] = {
  {0,4,14,6},
  {6,14,15,7},
  {7,15,16,8},
  {8,16,17,9},
  {9,17,5,2},
  {4,1,10,14},
  {14,10,11,15},
  {15,11,12,16},
  {16,12,13,17},
  {17,13,3,5}};

const PetscInt sInitialPartition2x5Mesh[2][5] = {
  {0,2,4,6,8},
  {1,3,5,7,9}
};

const PetscInt sNLoclCells2x5Mesh = 5;
const PetscInt sNGlobVerts2x5Mesh = 18;

int main(int argc, char **argv)
{
  const PetscInt   Nc                 = sNLoclCells2x5Mesh; //Same on each rank for this example...
  const PetscInt   Nv                 = sNGlobVerts2x5Mesh;
  const PetscInt*  InitPartForRank[2] = {&sInitialPartition2x5Mesh[0][0],
                                         &sInitialPartition2x5Mesh[1][0]};
  const PetscInt (*Conn)[4]           = sConnectivity2x5Mesh;

  const PetscInt   Ncor = 4;
  const PetscInt   dim  = 2;
  DM               dm, idm, ddm;
  PetscSF          sfVert, sfMig, sfPart;
  PetscPartitioner part;
  PetscSection     s;
  PetscInt        *cells, c;
  PetscMPIInt      size, rank;
  PetscBool        box = PETSC_FALSE, field = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a 2 processors example only");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-box", &box, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-field", &field, NULL));

  PetscCall(DMPlexCreate(PETSC_COMM_WORLD, &dm));
  if (box) {
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));
  } else {
    PetscCall(PetscMalloc1(Nc * Ncor, &cells));
    for (c = 0; c < Nc; ++c) {
      PetscInt cell = (InitPartForRank[rank])[c], cor;

      for (cor = 0; cor < Ncor; ++cor) {
        cells[c*Ncor + cor] = Conn[cell][cor];
      }
    }
    PetscCall(DMSetDimension(dm, dim));
    PetscCall(DMPlexBuildFromCellListParallel(dm, Nc, PETSC_DECIDE, Nv, Ncor, cells, &sfVert, NULL));
    PetscCall(PetscSFDestroy(&sfVert));
    PetscCall(PetscFree(cells));
    PetscCall(DMPlexInterpolate(dm, &idm));
    PetscCall(DMDestroy(&dm));
    dm = idm;
  }
  PetscCall(DMSetUseNatural(dm, PETSC_TRUE));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  if (field) {
   const PetscInt Nf         = 1;
   const PetscInt numComp[1] = {1};
   const PetscInt numDof[3]  = {0, 0, 1};
   const PetscInt numBC      = 0;

   PetscCall(DMSetNumFields(dm, Nf));
   PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &s));
   PetscCall(DMSetLocalSection(dm, s));
   PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));
   PetscCall(PetscSectionDestroy(&s));
  }

  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));

  PetscCall(DMPlexDistribute(dm, 0, &sfMig, &ddm));
  PetscCall(PetscSFView(sfMig, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFCreateInverseSF(sfMig, &sfPart));
  PetscCall(PetscObjectSetName((PetscObject) sfPart, "Inverse Migration SF"));
  PetscCall(PetscSFView(sfPart, PETSC_VIEWER_STDOUT_WORLD));

  Vec          lGlobalVec, lNatVec;
  PetscScalar *lNatVecArray;

  {
    PetscSection s;

    PetscCall(DMGetGlobalSection(dm, &s));
    PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(DMGetGlobalVector(dm, &lNatVec));
  PetscCall(PetscObjectSetName((PetscObject) lNatVec, "Natural Vector (initial partition)"));

  //Copying the initial partition into the "natural" vector:
  PetscCall(VecGetArray(lNatVec, &lNatVecArray));
  for (c = 0; c < Nc; ++c) lNatVecArray[c] = (InitPartForRank[rank])[c];
  PetscCall(VecRestoreArray(lNatVec, &lNatVecArray));

  PetscCall(DMGetGlobalVector(ddm,&lGlobalVec));
  PetscCall(PetscObjectSetName((PetscObject) lGlobalVec, "Global Vector (reordered element numbers in the petsc distributed order)"));
  PetscCall(VecZeroEntries(lGlobalVec));

  // The call to DMPlexNaturalToGlobalBegin/End does not produce our expected result...
  // In lGlobalVec, we expect to have:
  /*
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 3.
   * 9.
   * Process [1]
   * 1.
   * 5.
   * 7.
   * 0.
   * 6.
   *
   * but we obtained:
   *
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 0.
   * 0.
   * Process [1]
   * 0.
   * 0.
   * 0.
   * 0.
   * 0.
   */

   {
     PetscSF nsf;

     PetscCall(DMPlexGetGlobalToNaturalSF(ddm, &nsf));
     PetscCall(PetscSFView(nsf, NULL));
   }
  PetscCall(DMPlexNaturalToGlobalBegin(ddm, lNatVec, lGlobalVec));
  PetscCall(DMPlexNaturalToGlobalEnd  (ddm, lNatVec, lGlobalVec));

  PetscCall(VecView(lNatVec, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(lGlobalVec, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMRestoreGlobalVector(dm,&lNatVec));
  PetscCall(DMRestoreGlobalVector(ddm,&lGlobalVec));

  PetscCall(PetscSFDestroy(&sfMig));
  PetscCall(PetscSFDestroy(&sfPart));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&ddm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -field -petscpartitioner_type simple
    nsize: 2

    test:
      suffix: 0
      args:

    test:
      suffix: 1
      args: -box -dm_plex_simplex 0 -dm_plex_box_faces 2,5 -dm_distribute

TEST*/
