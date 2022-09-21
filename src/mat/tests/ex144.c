/* This program illustrates use of parallel real FFT */
static char help[] = "This program illustrates the use of parallel real 2D fft using fftw without PETSc interface";

#include <petscmat.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

int main(int argc, char **args)
{
  const ptrdiff_t N0 = 2056, N1 = 2056;
  fftw_plan       bplan, fplan;
  fftw_complex   *out;
  double         *in1, *in2;
  ptrdiff_t       alloc_local, local_n0, local_0_start;
  ptrdiff_t       local_n1, local_1_start;
  PetscInt        i, j;
  PetscMPIInt     size, rank;
  int             n, N, N_factor, NM;
  PetscScalar     one = 2.0, zero = 0.5;
  PetscScalar     two = 4.0, three = 8.0, four = 16.0;
  PetscScalar     a, *x_arr, *y_arr, *z_arr;
  PetscReal       enorm;
  Vec             fin, fout, fout1;
  Vec             ini, final;
  PetscRandom     rnd;
  PetscInt       *indx3, tempindx, low, *indx4, tempindx1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));

  alloc_local = fftw_mpi_local_size_2d_transposed(N0, N1 / 2 + 1, PETSC_COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
#if defined(DEBUGGING)
  printf("The value alloc_local is %ld from process %d\n", alloc_local, rank);
  printf("The value local_n0 is %ld from process %d\n", local_n0, rank);
  printf("The value local_0_start is  %ld from process %d\n", local_0_start, rank);
/*    printf("The value local_n1 is  %ld from process %d\n",local_n1,rank); */
/*    printf("The value local_1_start is  %ld from process %d\n",local_1_start,rank); */
/*    printf("The value local_n0 is  %ld from process %d\n",local_n0,rank); */
#endif

  /* Allocate space for input and output arrays  */
  in1 = (double *)fftw_malloc(sizeof(double) * alloc_local * 2);
  in2 = (double *)fftw_malloc(sizeof(double) * alloc_local * 2);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);

  N        = 2 * N0 * (N1 / 2 + 1);
  N_factor = N0 * N1;
  n        = 2 * local_n0 * (N1 / 2 + 1);

  /*    printf("The value N is  %d from process %d\n",N,rank);  */
  /*    printf("The value n is  %d from process %d\n",n,rank);  */
  /*    printf("The value n1 is  %d from process %d\n",n1,rank);*/
  /* Creating data vector and accompanying array with VeccreateMPIWithArray */
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, N, (PetscScalar *)in1, &fin));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, N, (PetscScalar *)out, &fout));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, N, (PetscScalar *)in2, &fout1));

  /* Set the vector with random data */
  PetscCall(VecSet(fin, zero));
  /*    for (i=0;i<N0*N1;i++) */
  /*       { */
  /*       VecSetValues(fin,1,&i,&one,INSERT_VALUES); */
  /*     } */

  /*    VecSet(fin,one); */
  i = 0;
  PetscCall(VecSetValues(fin, 1, &i, &one, INSERT_VALUES));
  i = 1;
  PetscCall(VecSetValues(fin, 1, &i, &two, INSERT_VALUES));
  i = 4;
  PetscCall(VecSetValues(fin, 1, &i, &three, INSERT_VALUES));
  i = 5;
  PetscCall(VecSetValues(fin, 1, &i, &four, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(fin));
  PetscCall(VecAssemblyEnd(fin));

  PetscCall(VecSet(fout, zero));
  PetscCall(VecSet(fout1, zero));

  /* Get the meaningful portion of array */
  PetscCall(VecGetArray(fin, &x_arr));
  PetscCall(VecGetArray(fout1, &z_arr));
  PetscCall(VecGetArray(fout, &y_arr));

  fplan = fftw_mpi_plan_dft_r2c_2d(N0, N1, (double *)x_arr, (fftw_complex *)y_arr, PETSC_COMM_WORLD, FFTW_ESTIMATE);
  bplan = fftw_mpi_plan_dft_c2r_2d(N0, N1, (fftw_complex *)y_arr, (double *)z_arr, PETSC_COMM_WORLD, FFTW_ESTIMATE);

  fftw_execute(fplan);
  fftw_execute(bplan);

  PetscCall(VecRestoreArray(fin, &x_arr));
  PetscCall(VecRestoreArray(fout1, &z_arr));
  PetscCall(VecRestoreArray(fout, &y_arr));

  /*    VecView(fin,PETSC_VIEWER_STDOUT_WORLD); */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ini));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &final));
  PetscCall(VecSetSizes(ini, local_n0 * N1, N0 * N1));
  PetscCall(VecSetSizes(final, local_n0 * N1, N0 * N1));
  PetscCall(VecSetFromOptions(ini));
  PetscCall(VecSetFromOptions(final));

  if (N1 % 2 == 0) {
    NM = N1 + 2;
  } else {
    NM = N1 + 1;
  }
  /*printf("The Value of NM is %d",NM); */
  PetscCall(VecGetOwnershipRange(fin, &low, NULL));
  /*printf("The local index is %d from %d\n",low,rank); */
  PetscCall(PetscMalloc1(local_n0 * N1, &indx3));
  PetscCall(PetscMalloc1(local_n0 * N1, &indx4));
  for (i = 0; i < local_n0; i++) {
    for (j = 0; j < N1; j++) {
      tempindx  = i * N1 + j;
      tempindx1 = i * NM + j;

      indx3[tempindx] = local_0_start * N1 + tempindx;
      indx4[tempindx] = low + tempindx1;
      /*          printf("index3 %d from proc %d is \n",indx3[tempindx],rank); */
      /*          printf("index4 %d from proc %d is \n",indx4[tempindx],rank); */
    }
  }

  PetscCall(PetscMalloc2(local_n0 * N1, &x_arr, local_n0 * N1, &y_arr)); /* arr must be allocated for VecGetValues() */
  PetscCall(VecGetValues(fin, local_n0 * N1, indx4, (PetscScalar *)x_arr));
  PetscCall(VecSetValues(ini, local_n0 * N1, indx3, x_arr, INSERT_VALUES));

  PetscCall(VecAssemblyBegin(ini));
  PetscCall(VecAssemblyEnd(ini));

  PetscCall(VecGetValues(fout1, local_n0 * N1, indx4, y_arr));
  PetscCall(VecSetValues(final, local_n0 * N1, indx3, y_arr, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(final));
  PetscCall(VecAssemblyEnd(final));
  PetscCall(PetscFree2(x_arr, y_arr));

  /*
    VecScatter      vecscat;
    IS              indx1,indx2;
    for (i=0;i<N0;i++) {
       indx = i*NM;
       ISCreateStride(PETSC_COMM_WORLD,N1,indx,1,&indx1);
       indx = i*N1;
       ISCreateStride(PETSC_COMM_WORLD,N1,indx,1,&indx2);
       VecScatterCreate(fin,indx1,ini,indx2,&vecscat);
       VecScatterBegin(vecscat,fin,ini,INSERT_VALUES,SCATTER_FORWARD);
       VecScatterEnd(vecscat,fin,ini,INSERT_VALUES,SCATTER_FORWARD);
       VecScatterBegin(vecscat,fout1,final,INSERT_VALUES,SCATTER_FORWARD);
       VecScatterEnd(vecscat,fout1,final,INSERT_VALUES,SCATTER_FORWARD);
    }
*/

  a = 1.0 / (PetscReal)N_factor;
  PetscCall(VecScale(fout1, a));
  PetscCall(VecScale(final, a));

  /*    VecView(ini,PETSC_VIEWER_STDOUT_WORLD);   */
  /*    VecView(final,PETSC_VIEWER_STDOUT_WORLD); */
  PetscCall(VecAXPY(final, -1.0, ini));

  PetscCall(VecNorm(final, NORM_1, &enorm));
  if (enorm > 1.e-10) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Error norm of |x - z|  = %e\n", enorm));

  /* Execute fftw with function fftw_execute and destroy it after execution */
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(bplan);
  fftw_free(in1);
  PetscCall(VecDestroy(&fin));
  fftw_free(out);
  PetscCall(VecDestroy(&fout));
  fftw_free(in2);
  PetscCall(VecDestroy(&fout1));

  PetscCall(VecDestroy(&ini));
  PetscCall(VecDestroy(&final));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscFree(indx3));
  PetscCall(PetscFree(indx4));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !mpiuni fftw !complex

   test:
      output_file: output/ex144.out

   test:
      suffix: 2
      nsize: 3
      output_file: output/ex144.out

TEST*/
