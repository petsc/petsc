static char help[] = "Test integrity of subvector data, use \n\
use -hdf5 to specify HDF5 viewer format for subvector I/O \n\n";

/*
   Tests for transfer of data from subvectors to parent vectors after
   loading data into subvector. This routine does the following : creates
   a vector of size 50, sets it to 2 and saves it to disk. Creates a
   vector of size 100, set it to 1 and extracts the last 50 elements
   as a subvector. Loads the saved vector from disk into the subvector
   and restores the subvector. To verify that the data has been loaded
   into the parent vector, the sum of it's elements is calculated.
*/

#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  Vec            testvec;                 /* parent vector of size 100 */
  Vec            loadvec;                 /* subvector extracted from the parent vector */
  Vec            writevec;                /* vector used to save data to be loaded by loadvec */
  IS             loadis;                  /* index set to extract last 50 elements of testvec */
  PetscInt       low,high;                /* used to store vecownership output */
  PetscInt       issize, isstart;         /* index set params */
  PetscInt       skipuntil = 50;          /* parameter to slice the last N elements of parent vec */
  PetscViewer    viewer;                  /* viewer for I/O */
  PetscScalar    sum;                     /* used to test sum of parent vector elements */
  PetscBool      usehdf5 = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);if (ierr) return ierr;

  /* parse input options to determine I/O format */
  ierr = PetscOptionsGetBool(NULL,NULL,"-hdf5",&usehdf5,NULL);CHKERRQ(ierr);

  /* Create parent vector with 100 elements, set it to 1 */
  ierr = VecCreate(PETSC_COMM_WORLD, &testvec);CHKERRQ(ierr);
  ierr = VecSetSizes(testvec, PETSC_DECIDE,100);CHKERRQ(ierr);
  ierr = VecSetUp(testvec);CHKERRQ(ierr);
  ierr = VecSet(testvec, (PetscScalar) 1);CHKERRQ(ierr);

  /* Create a vector with 50 elements, set it to 2. */
  ierr = VecCreate(PETSC_COMM_WORLD, &writevec);CHKERRQ(ierr);
  ierr = VecSetSizes(writevec, PETSC_DECIDE,50);CHKERRQ(ierr);
  ierr = VecSetUp(writevec);CHKERRQ(ierr);
  ierr = VecSet(writevec, (PetscScalar) 2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)writevec,"temp");CHKERRQ(ierr);

  /* Save to disk in specified format, destroy vector & viewer */
  if (usehdf5) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing vector in hdf5 to vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  }
  ierr = VecView(writevec,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&writevec); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Create index sets on each mpi rank to select the last 50 elements of parent vec */
  ierr = VecGetOwnershipRange(testvec, &low, &high);CHKERRQ(ierr);
  if (low>=skipuntil) {
    isstart = low;
    issize = high - low;
  } else if (low<=skipuntil && high>=skipuntil ) {
    isstart = skipuntil;
    issize = high - skipuntil;
  } else {
    isstart = low;
    issize  = 0;
  }
  ierr = ISCreateStride(PETSC_COMM_WORLD, issize, isstart, 1, &loadis);CHKERRQ(ierr);

  /* Create subvector using the index set created above */
  ierr = VecGetSubVector(testvec, loadis, &loadvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)loadvec,"temp"); CHKERRQ(ierr);

  /* Load the previously saved vector into the subvector, destroy viewer */
  if (usehdf5) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in hdf5 from vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  }
  ierr = VecLoad(loadvec, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Restore subvector to transfer loaded data into parent vector */
  ierr = VecRestoreSubVector(testvec, loadis, &loadvec);CHKERRQ(ierr);

  /* Compute sum of parent vector elements */
  ierr = VecSum(testvec, &sum);CHKERRQ(ierr);

  /* to verify that the loaded data has been transferred */
  if (sum != 150) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB,"Data has not been transferred from subvector to parent vector");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecSum on parent vec is : %e\n",sum);CHKERRQ(ierr);

  /* destroy parent vector, index set and exit */
  ierr = VecDestroy(&testvec);CHKERRQ(ierr);
  ierr = ISDestroy(&loadis);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hdf5

  test:
    nsize: 4

  test:
    suffix: 2
    nsize: 4
    args: -hdf5

TEST*/
