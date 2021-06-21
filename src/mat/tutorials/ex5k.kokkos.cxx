static char help[] = "Test of Kokkos matrix assemble with 1D Laplacian. Kokkos version of ex5cu \n\n";

#include <petscconf.h>
#include <petscmat.h>
//#include <petsccublas.h>

/*
    Include Kokkos files
*/
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

#include <petscaijdevice.h>

void assemble_mat(Mat A, PetscInt start, PetscInt end, PetscInt Ne, PetscMPIInt rank)
{
  PetscInt        i;
  PetscScalar     values[] = {1,-1,-1,1.1};
  PetscErrorCode  ierr;
  for (i=start; i<end; i++) {
    PetscInt js[] = {i-1, i};
    ierr = MatSetValues(A,2,js,2,js,values,ADD_VALUES);
    if (ierr) return;
  }
}

int main(int argc,char **argv)
{
  PetscErrorCode               ierr;
  Mat                          A;
  PetscInt                     N=11, nz=3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure   d_mat;
  PetscLogEvent                event;
  Vec                          x,y;
  PetscMPIInt                  rank;
  PetscBool                    view_kokkos_configuration = PETSC_TRUE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_kokkos_configuration",&view_kokkos_configuration,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL, "-nz_row", &nz, NULL);CHKERRQ(ierr); // for debugging, will be wrong if nz<3
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-num_threads", &num_threads, NULL);CHKERRQ(ierr);
  if (nz>N+1) {
    PetscPrintf(PETSC_COMM_WORLD,"warning decreasing nz\n");
    nz=N+1;
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscLogEventRegister("GPU operator", MAT_CLASSID, &event);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, nz, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, nz,NULL,nz-1, NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  // assemble end on CPU. We are not doing it redudent here, and ignoring off proc entries, but we could
  assemble_mat(A, Istart, Iend, N, rank);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // test cusparse
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y,NULL,"-vec_view");CHKERRQ(ierr);

  // assemble on GPU
  if (Iend<N) Iend++; // elements, ignore off processor entries so do redundent
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = MatKokkosGetDeviceMatWrite(A,&d_mat);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr); // needed?
  Kokkos:: parallel_for (Kokkos::RangePolicy<> (Istart,Iend), KOKKOS_LAMBDA ( int i) {
      PetscScalar                  values[] = {1,-1,-1,1.1};
      PetscInt js[] = {i-1, i};
      MatSetValuesDevice(d_mat,2,js,2,js,values,ADD_VALUES);
    });
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Kokkos kernels required");
#endif
  ierr = PetscFinalize();
  return ierr;
}

/*
     The first test works for Kokkos wtih OpenMP and PThreads, the second with CUDA.

*/

/*TEST

   build:
     requires: kokkos_kernels

   test:
     suffix: 0
     requires: kokkos_kernels double !complex !single
     args: -view_kokkos_configuration -n 11 -vec_view
     nsize:  2

TEST*/
