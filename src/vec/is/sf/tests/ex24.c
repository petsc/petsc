const char help[] = "Test overlapping PetscSF communication with empty roots and leaves";

#include <petscsf.h>

static PetscErrorCode testOverlappingCommunication(PetscSF sf)
{
  PetscInt  nroots, maxleaf;
  PetscInt *leafa, *leafb, *roota, *rootb;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  PetscCall(PetscSFGetLeafRange(sf, NULL, &maxleaf));
  PetscCall(PetscMalloc4(nroots, &roota, nroots, &rootb, maxleaf + 1, &leafa, maxleaf + 1, &leafb));

  // test reduce
  for (PetscInt i = 0; i < nroots; i++) roota[i] = 0;
  for (PetscInt i = 0; i < nroots; i++) rootb[i] = 0;
  for (PetscInt i = 0; i < maxleaf + 1; i++) leafa[i] = (i + 1);
  for (PetscInt i = 0; i < maxleaf + 1; i++) leafb[i] = -(i + 1);

  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafa, roota, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafb, rootb, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafa, roota, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafb, rootb, MPI_REPLACE));
  for (PetscInt i = 0; i < nroots; i++) PetscCheck(roota[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFReduce in (A,B,A,B) order crosses separate reductions");
  for (PetscInt i = 0; i < nroots; i++) PetscCheck(rootb[i] <= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFReduce in (A,B,A,B) order crosses separate reductions");

  for (PetscInt i = 0; i < nroots; i++) roota[i] = 0;
  for (PetscInt i = 0; i < nroots; i++) rootb[i] = 0;

  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafa, roota, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafb, rootb, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafb, rootb, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafa, roota, MPI_REPLACE));

  for (PetscInt i = 0; i < nroots; i++) PetscCheck(roota[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFReduce in (A,B,B,A) order crosses separate reductions");
  for (PetscInt i = 0; i < nroots; i++) PetscCheck(rootb[i] <= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFReduce in (A,B,B,A) order crosses separate reductions");

  // test bcast
  for (PetscInt i = 0; i < nroots; i++) roota[i] = (i + 1);
  for (PetscInt i = 0; i < nroots; i++) rootb[i] = -(i + 1);
  for (PetscInt i = 0; i < maxleaf + 1; i++) leafa[i] = 0;
  for (PetscInt i = 0; i < maxleaf + 1; i++) leafb[i] = 0;

  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, roota, leafa, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootb, leafb, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, roota, leafa, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootb, leafb, MPI_REPLACE));

  for (PetscInt i = 0; i < maxleaf + 1; i++) PetscCheck(leafa[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFBcast in (A,B,A,B) order crosses separate broadcasts");
  for (PetscInt i = 0; i < maxleaf + 1; i++) PetscCheck(leafb[i] <= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFBcast in (A,B,A,B) order crosses separate broadcasts");

  for (PetscInt i = 0; i < maxleaf + 1; i++) leafa[i] = 0;
  for (PetscInt i = 0; i < maxleaf + 1; i++) leafb[i] = 0;

  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, roota, leafa, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootb, leafb, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootb, leafb, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, roota, leafa, MPI_REPLACE));

  for (PetscInt i = 0; i < maxleaf + 1; i++) PetscCheck(leafa[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFBcast in (A,B,B,A) order crosses separate broadcasts");
  for (PetscInt i = 0; i < maxleaf + 1; i++) PetscCheck(leafb[i] <= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSFBcast in (A,B,B,A) order crosses separate broadcasts");

  PetscCall(PetscFree4(roota, rootb, leafa, leafb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createSparseSF(MPI_Comm comm, PetscSF *sf)
{
  PetscMPIInt rank;
  PetscInt    nroots, nleaves;
  PetscSFNode remote;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSFCreate(comm, sf));
  nroots       = (rank & 1) ? 1 : 0;
  nleaves      = (rank & 2) ? 1 : 0;
  remote.rank  = -1;
  remote.index = 0;
  if (nleaves == 1) remote.rank = (rank & 1) ? (rank ^ 2) : (rank ^ 1);
  PetscCall(PetscSFSetGraph(*sf, nroots, nleaves, NULL, PETSC_COPY_VALUES, &remote, PETSC_COPY_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscSF sf;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(createSparseSF(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(testOverlappingCommunication(sf));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 4
    suffix: 0
    output_file: output/empty.out

  test:
    TODO: frequent timeout with the CI job linux-hip-cmplx
    nsize: 4
    suffix: 0_window
    output_file: output/empty.out
    args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
    requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

TEST*/
