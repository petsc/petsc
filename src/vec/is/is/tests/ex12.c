
static char help[] = "Tests HDF5 ISView() / ISLoad(), and ISSetLayout()\n\n";

#include <petscis.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  const char      filename[] = "ex12.h5";
  const char      objname[]  = "is0";
  IS              is0, is1;
  PetscLayout     map;
  PetscViewer     viewer;
  PetscMPIInt     size, rank;
  MPI_Comm        comm;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  {
    PetscInt *idx, i, n, start, end;

    n = rank + 2;
    PetscCall(PetscCalloc1(n, &idx));
    PetscCall(ISCreateGeneral(comm, n, idx, PETSC_OWN_POINTER, &is0));
    PetscCall(PetscObjectSetName((PetscObject)is0, objname));
    PetscCall(ISGetLayout(is0, &map));
    PetscCall(PetscLayoutGetRange(map, &start, &end));
    PetscCheck(end - start == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "end - start == n");
    for (i=0; i<n; i++) idx[i] = i + start;
  }

  PetscCall(PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(ISView(is0, viewer));

  PetscCall(ISCreate(comm, &is1));
  PetscCall(PetscObjectSetName((PetscObject)is1, objname));
  PetscCall(ISSetLayout(is1, map));
  PetscCall(ISLoad(is1, viewer));

  {
    PetscBool flg;

    PetscCall(ISEqual(is0, is1, &flg));
    PetscCheck(flg, comm, PETSC_ERR_PLIB, "is0 and is1 differ");
  }

  PetscCall(ISDestroy(&is0));
  PetscCall(ISDestroy(&is1));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: hdf5
   test:
      nsize: {{1 3}}

TEST*/
