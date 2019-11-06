static char help[] = "Test if VecLoad_HDF5 can correctly handle FFTW vectors\n\n";

/*
  fftw vectors alloate their data array through fftw_malloc() and have their specialized VecDestroy().
  When doing VecLoad on these vectors, we must take care of the v->array, v->array_allocated properly
  to avoid memory leaks and double-free.

  Contributed-by: Sajid Ali <sajidsyed2021@u.northwestern.edu>
*/

#include <petscmat.h>
#include <petscviewerhdf5.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       i,low,high,ldim,iglobal;
  PetscInt       m=64,dim[2]={8,8},DIM=2; /* FFT parameters */
  Vec            u,u_,H;    /* wave, work and transfer function vectors */
  Vec            slice_rid; /* vector to hold the refractive index */
  Mat            A;         /* FFT-matrix to call FFTW via interface */
  PetscViewer    viewer;    /* Load refractive index */
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Generate vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "ref_index");CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&ldim);CHKERRQ(ierr);

  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + low);
    ierr    = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ex50tmp.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Make FFT matrix (via interface) and create vecs aligned to it. */
  ierr   = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);

  /* Create vectors that are compatible with parallel layout of A - must call MatCreateVecs()! */
  ierr = MatCreateVecsFFTW(A,&u,&u_,&H);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&slice_rid);

  /* Load refractive index from file */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ex50tmp.h5",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)slice_rid,"ref_index");CHKERRQ(ierr);
  ierr = VecLoad(slice_rid,viewer);CHKERRQ(ierr); /* Test if VecLoad_HDF5 can correctly handle FFTW vectors */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&slice_rid);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&u_);CHKERRQ(ierr);
  ierr = VecDestroy(&H);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/*TEST

   build:
     requires: hdf5 fftw

   test:
     nsize: 2
     requires: complex
TEST*/
