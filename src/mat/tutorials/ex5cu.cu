static char help[] = "Serial test of Cuda matrix assemble with 1D Laplacian.\n\n";

#include <petscconf.h>
#include <petscmat.h>
#include <petscaijdevice.h>
#include <petsccublas.h>


__global__
void assemble_device(PetscSplitCSRDataStructure *d_mat, PetscInt start, PetscInt end, PetscInt Ne, PetscMPIInt rank, PetscErrorCode *ierr)
{
  const PetscInt  inc = blockDim.x, my0 = threadIdx.x;
  PetscInt        i;
  PetscScalar     values[] = {1,-1,-1,1.1};
  for (i=start+my0; i<end; i+=inc) {
    PetscInt js[] = {i-1, i};
    MatSetValuesDevice(d_mat,2,js,2,js,values,ADD_VALUES,ierr);
    if (*ierr) return;
  }
}

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

int main(int argc,char **args)
{
  PetscErrorCode               ierr;
  Mat                          A;
  PetscInt                     N=11, nz=3, Istart, Iend, num_threads = 128;
  PetscSplitCSRDataStructure   *d_mat;
  PetscLogEvent                event;
  Vec                          x,y;
  cudaError_t                  cerr;
  PetscMPIInt                  rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-nz_row", &nz, NULL);CHKERRQ(ierr); // for debugging, will be wrong if nz<3
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-num_threads", &num_threads, NULL);CHKERRQ(ierr);
  if (nz>N+1) {
    PetscPrintf(PETSC_COMM_WORLD,"warning decreasing nz\n");
    nz=N+1;
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("GPU operator", MAT_CLASSID, &event);CHKERRQ(ierr);
  ierr = MatCreateAIJCUSPARSE(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,nz,NULL,nz-1,NULL,&A);CHKERRQ(ierr);
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
  ierr = MatCUSPARSEGetDeviceMatWrite(A,&d_mat);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr); // needed?
  assemble_device<<<1,num_threads>>>(d_mat, Istart, Iend, N, rank, &ierr);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  fflush(stdout);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: cuda !define(PETSC_USE_CTABLE)

   test:
      suffix: 0
      args: -n 11 -vec_view
      nsize:  2

TEST*/
