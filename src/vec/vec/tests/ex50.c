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
  PetscInt       i,low,high,ldim,iglobal;
  PetscInt       m=64,dim[2]={8,8},DIM=2; /* FFT parameters */
  Vec            u,u_,H;    /* wave, work and transfer function vectors */
  Vec            slice_rid; /* vector to hold the refractive index */
  Mat            A;         /* FFT-matrix to call FFTW via interface */
  PetscViewer    viewer;    /* Load refractive index */
  PetscScalar    v;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

  /* Generate vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(PetscObjectSetName((PetscObject)u, "ref_index"));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecGetOwnershipRange(u,&low,&high));
  CHKERRQ(VecGetLocalSize(u,&ldim));

  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + low);
    CHKERRQ(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"ex50tmp.h5",FILE_MODE_WRITE,&viewer));
  CHKERRQ(VecView(u,viewer));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* Make FFT matrix (via interface) and create vecs aligned to it. */
  CHKERRQ(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));

  /* Create vectors that are compatible with parallel layout of A - must call MatCreateVecs()! */
  CHKERRQ(MatCreateVecsFFTW(A,&u,&u_,&H));
  CHKERRQ(VecDuplicate(u,&slice_rid));

  /* Load refractive index from file */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"ex50tmp.h5",FILE_MODE_READ,&viewer));
  CHKERRQ(PetscObjectSetName((PetscObject)slice_rid,"ref_index"));
  CHKERRQ(VecLoad(slice_rid,viewer)); /* Test if VecLoad_HDF5 can correctly handle FFTW vectors */
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&slice_rid));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&u_));
  CHKERRQ(VecDestroy(&H));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: hdf5 fftw

   test:
     nsize: 2
     requires: complex
TEST*/
