
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
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  {
    PetscInt *idx, i, n, start, end;

    n = rank + 2;
    ierr = PetscCalloc1(n, &idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, n, idx, PETSC_OWN_POINTER, &is0);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)is0, objname);CHKERRQ(ierr);
    ierr = ISGetLayout(is0, &map);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(map, &start, &end);CHKERRQ(ierr);
    PetscCheck(end - start == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "end - start == n");
    for (i=0; i<n; i++) idx[i] = i + start;
  }

  ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = ISView(is0, viewer);CHKERRQ(ierr);

  ierr = ISCreate(comm, &is1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is1, objname);CHKERRQ(ierr);
  ierr = ISSetLayout(is1, map);CHKERRQ(ierr);
  ierr = ISLoad(is1, viewer);CHKERRQ(ierr);

  {
    PetscBool flg;

    ierr = ISEqual(is0, is1, &flg);CHKERRQ(ierr);
    PetscCheck(flg, comm, PETSC_ERR_PLIB, "is0 and is1 differ");
  }

  ierr = ISDestroy(&is0);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: hdf5
   test:
      nsize: {{1 3}}

TEST*/
