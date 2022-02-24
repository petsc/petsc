static char help[] = "Test of CUDA matrix assemble with simple matrix.\n\n";

// This a minimal example of the use of the CUDA MatAIJ metadata for assembly.
//
// The matrix must be a type 'aijcusparse' and must first be assembled on the CPU to provide the nonzero pattern.
// Next, get a pointer to a simple CSR mirror (PetscSplitCSRDataStructure) of the matrix data on
//    the GPU with MatCUSPARSEGetDeviceMatWrite().
// Then use this object to populate the matrix on the GPU with MatSetValuesDevice().
// Finally call MatAssemblyBegin/End() and the matrix is ready to use on the GPU without matrix data movement between the
//    host and GPU.

#include <petscconf.h>
#include <petscmat.h>
#include <petscdevice.h>
#include <assert.h>

#include <petscaijdevice.h>
__global__
void assemble_on_gpu(PetscSplitCSRDataStructure d_mat, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank)
{
  const PetscInt  inc = blockDim.x, my0 = threadIdx.x;
  PetscInt        i;
  PetscErrorCode  ierr;

  for (i=start+my0; i<end+1; i+=inc) {
    PetscInt    js[] = {i-1, i}, nn = (i==N) ? 1 : 2; // negative indices are igored but >= N are not, so clip end
    PetscScalar values[] = {1,1,1,1};
    ierr = MatSetValuesDevice(d_mat,nn,js,nn,js,values,ADD_VALUES);if (ierr) assert(0);
  }
}

PetscErrorCode assemble_on_cpu(Mat A, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (i=start; i<end+1; i++) {
    PetscInt    js[] = {i-1, i}, nn = (i==N) ? 1 : 2;
    PetscScalar values[] = {1,1,1,1};
    CHKERRQ(MatSetValues(A,nn,js,nn,js,values,ADD_VALUES));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode             ierr;
  Mat                        A;
  PetscInt                   N=11, nz=3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure d_mat;
  PetscLogEvent              event;
  PetscMPIInt                rank,size;
  PetscBool                  testmpiseq = PETSC_FALSE;
  Vec                        x,y;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-n", &N, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-num_threads", &num_threads, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-nz_row", &nz, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-testmpiseq", &testmpiseq, NULL));
  if (nz<3)   nz=3;
  if (nz>N+1) nz=N+1;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));
  CHKERRQ(MatCreateAIJCUSPARSE(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,nz,NULL,nz-1,NULL,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  /* current GPU assembly code does not support offprocessor values insertion */
  CHKERRQ(assemble_on_cpu(A, Istart, Iend, N, rank));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  // test
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecViewFromOptions(y,NULL,"-ex5_vec_view"));

  if (testmpiseq && size == 1) {
    CHKERRQ(MatConvert(A,MATSEQAIJ,MAT_INPLACE_MATRIX,&A));
    CHKERRQ(MatConvert(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A));
  }
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
  CHKERRQ(MatCUSPARSEGetDeviceMatWrite(A,&d_mat));
  assemble_on_gpu<<<1,num_threads>>>(d_mat, Istart, Iend, N, rank);
  CHKERRCUDA(cudaDeviceSynchronize());
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));

  // test
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecViewFromOptions(y,NULL,"-ex5_vec_view"));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: cuda

   test:
      suffix: 0
      diff_args: -j
      args: -n 11 -ex5_vec_view
      nsize: 1

   test:
      suffix: 1
      diff_args: -j
      args: -n 11 -ex5_vec_view
      nsize: 2

   test:
      suffix: 2
      diff_args: -j
      args: -n 11 -testmpiseq -ex5_vec_view
      nsize: 1

TEST*/
