static char help[] = "Test of Kokkos matrix assemble with 1D Laplacian. Kokkos version of ex5cu \n\n";

#include <petscconf.h>
#include <petscmat.h>

/*
    Include Kokkos files
*/
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

#include <petscaijdevice.h>

int main(int argc,char **argv)
{
  PetscErrorCode               ierr;
  Mat                          A;
  PetscInt                     N=11, nz=3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure   d_mat;
  PetscLogEvent                event;
  Vec                          x,y;
  PetscMPIInt                  rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-nz_row", &nz, NULL)); // for debugging, will be wrong if nz<3
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-n", &N, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-num_threads", &num_threads, NULL));
  if (nz>N+1) {
    PetscPrintf(PETSC_COMM_WORLD,"warning decreasing nz\n");
    nz=N+1;
  }
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetType(A, MATAIJKOKKOS));
  CHKERRQ(MatSeqAIJSetPreallocation(A, nz, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A, nz,NULL,nz-1, NULL));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  // assemble end on CPU. We are not assembling redundant here, and ignoring off proc entries, but we could
  for (int i=Istart; i<Iend+1; i++) {
    PetscScalar values[] = {1,1,1,1};
    PetscInt    js[] = {i-1,i}, nn = (i==N) ? 1 : 2; // negative indices are ignored but >= N are not, so clip end
    CHKERRQ(MatSetValues(A,nn,js,nn,js,values,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  // test Kokkos
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecViewFromOptions(y,NULL,"-ex5_vec_view"));

  // assemble on GPU
  if (Iend<N) Iend++; // elements, ignore off processor entries so do redundant
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
  CHKERRQ(MatKokkosGetDeviceMatWrite(A,&d_mat));
  Kokkos::fence();
  Kokkos::parallel_for (Kokkos::RangePolicy<> (Istart,Iend+1), KOKKOS_LAMBDA (int i) {
      PetscScalar  values[] = {1,1,1,1};
      PetscInt     js[] = {i-1, i}, nn = (i==N) ? 1 : 2;
      MatSetValuesDevice(d_mat,nn,js,nn,js,values,ADD_VALUES);
    });
  Kokkos::fence();
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecViewFromOptions(y,NULL,"-ex5_vec_view"));
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Kokkos kernels required");
#endif
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: kokkos_kernels

   test:
     suffix: 0
     requires: kokkos_kernels double !complex !single
     args: -n 11 -ex5_vec_view
     nsize:  2

TEST*/
