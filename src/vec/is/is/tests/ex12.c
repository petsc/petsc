
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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

  {
    PetscInt *idx, i, n, start, end;

    n = rank + 2;
    CHKERRQ(PetscCalloc1(n, &idx));
    CHKERRQ(ISCreateGeneral(comm, n, idx, PETSC_OWN_POINTER, &is0));
    CHKERRQ(PetscObjectSetName((PetscObject)is0, objname));
    CHKERRQ(ISGetLayout(is0, &map));
    CHKERRQ(PetscLayoutGetRange(map, &start, &end));
    PetscCheck(end - start == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "end - start == n");
    for (i=0; i<n; i++) idx[i] = i + start;
  }

  CHKERRQ(PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer));
  CHKERRQ(ISView(is0, viewer));

  CHKERRQ(ISCreate(comm, &is1));
  CHKERRQ(PetscObjectSetName((PetscObject)is1, objname));
  CHKERRQ(ISSetLayout(is1, map));
  CHKERRQ(ISLoad(is1, viewer));

  {
    PetscBool flg;

    CHKERRQ(ISEqual(is0, is1, &flg));
    PetscCheck(flg, comm, PETSC_ERR_PLIB, "is0 and is1 differ");
  }

  CHKERRQ(ISDestroy(&is0));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: hdf5
   test:
      nsize: {{1 3}}

TEST*/
