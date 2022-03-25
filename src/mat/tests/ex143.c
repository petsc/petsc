static char help[] = "Illustrate how to use mpi FFTW and PETSc-FFTW interface \n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this

 Usage:
   mpiexec -n <np> ./ex143 -use_FFTW_interface NO
   mpiexec -n <np> ./ex143 -use_FFTW_interface YES
*/

#include <petscmat.h>
#include <fftw3-mpi.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=50,N1=20,N=N0*N1,DIM;
  PetscRandom    rdm;
  PetscScalar    a;
  PetscReal      enorm;
  Vec            x,y,z;
  PetscBool      view=PETSC_FALSE,use_interface=PETSC_TRUE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex143");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-vec_view draw", "View the vectors", "ex143", view, &view, NULL));
  PetscCall(PetscOptionsBool("-use_FFTW_interface", "Use PETSc-FFTW interface", "ex143",use_interface, &use_interface, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_FFTW_interface",&use_interface,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  if (!use_interface) {
    /* Use mpi FFTW without PETSc-FFTW interface, 2D case only */
    /*---------------------------------------------------------*/
    fftw_plan    fplan,bplan;
    fftw_complex *data_in,*data_out,*data_out2;
    ptrdiff_t    alloc_local,local_n0,local_0_start;

    DIM = 2;
    if (rank == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Use FFTW without PETSc-FFTW interface, DIM %" PetscInt_FMT "\n",DIM));
    }
    fftw_mpi_init();
    N           = N0*N1;
    alloc_local = fftw_mpi_local_size_2d(N0,N1,PETSC_COMM_WORLD,&local_n0,&local_0_start);

    data_in   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
    data_out  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
    data_out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_in,&x));
    PetscCall(PetscObjectSetName((PetscObject) x, "Real Space vector"));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_out,&y));
    PetscCall(PetscObjectSetName((PetscObject) y, "Frequency space vector"));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*N1,(PetscInt)N,(const PetscScalar*)data_out2,&z));
    PetscCall(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

    fplan = fftw_mpi_plan_dft_2d(N0,N1,data_in,data_out,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
    bplan = fftw_mpi_plan_dft_2d(N0,N1,data_out,data_out2,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);

    PetscCall(VecSetRandom(x, rdm));
    if (view) PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

    fftw_execute(fplan);
    if (view) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

    fftw_execute(bplan);

    /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    a    = 1.0/(PetscReal)N;
    PetscCall(VecScale(z,a));
    if (view) PetscCall(VecView(z, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecAXPY(z,-1.0,x));
    PetscCall(VecNorm(z,NORM_1,&enorm));
    if (enorm > 1.e-11 && rank == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm));
    }

    /* Free spaces */
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    fftw_free(data_in);  PetscCall(VecDestroy(&x));
    fftw_free(data_out); PetscCall(VecDestroy(&y));
    fftw_free(data_out2);PetscCall(VecDestroy(&z));

  } else {
    /* Use PETSc-FFTW interface                  */
    /*-------------------------------------------*/
    PetscInt i,*dim,k;
    Mat      A;

    N=1;
    for (i=1; i<5; i++) {
      DIM  = i;
      PetscCall(PetscMalloc1(i,&dim));
      for (k=0; k<i; k++) {
        dim[k]=30;
      }
      N *= dim[i-1];

      /* Create FFTW object */
      if (rank == 0) printf("Use PETSc-FFTW interface...%d-DIM: %d\n",(int)DIM,(int)N);

      PetscCall(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));

      /* Create vectors that are compatible with parallel layout of A - must call MatCreateVecs()! */

      PetscCall(MatCreateVecsFFTW(A,&x,&y,&z));
      PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector"));
      PetscCall(PetscObjectSetName((PetscObject) y, "Frequency space vector"));
      PetscCall(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

      /* Set values of space vector x */
      PetscCall(VecSetRandom(x,rdm));

      if (view) PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

      /* Apply FFTW_FORWARD and FFTW_BACKWARD */
      PetscCall(MatMult(A,x,y));
      if (view) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

      PetscCall(MatMultTranspose(A,y,z));

      /* Compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      a    = 1.0/(PetscReal)N;
      PetscCall(VecScale(z,a));
      if (view) PetscCall(VecView(z,PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecAXPY(z,-1.0,x));
      PetscCall(VecNorm(z,NORM_1,&enorm));
      if (enorm > 1.e-9 && rank == 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
      }

      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&z));
      PetscCall(MatDestroy(&A));

      PetscCall(PetscFree(dim));
    }
  }

  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !mpiuni fftw complex

   test:
      output_file: output/ex143.out

   test:
      suffix: 2
      nsize: 3
      output_file: output/ex143.out

TEST*/
