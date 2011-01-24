static char help[] = "Test mpi FFTW interface \n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this

*/

#include "petscmat.h"
#include "fftw3-mpi.h"

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  const ptrdiff_t N0=10,N1=4,N=N0*N1;
  fftw_plan       fplan,bplan;
  fftw_complex    *data_in,*data_out,*data_out2;
  ptrdiff_t       alloc_local,local_n0,local_0_start,i,j;
  PetscRandom     rdm;
  PetscScalar     a;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires complex numbers");
#endif
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  fftw_mpi_init();

  alloc_local = fftw_mpi_local_size_2d(N0,N1,PETSC_COMM_WORLD,&local_n0,&local_0_start);
  printf("[%d] local_n0, local_0_start %d %d\n",rank,(PetscInt)local_n0,(PetscInt)local_0_start);

  data_in   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
  data_out  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
  data_out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
  
  fplan = fftw_mpi_plan_dft_2d(N0,N1,data_in,data_out,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
  bplan = fftw_mpi_plan_dft_2d(N0,N1,data_out,data_out2,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);

  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  for (i=0; i<local_n0; ++i){
    for (j=0; j<N1; ++j){
      ierr = PetscRandomGetValue(rdm,&a);CHKERRQ(ierr);
      data_in[i*N1 + j][0] = PetscRealPart(a);
      data_in[i*N1 + j][1] = PetscImaginaryPart(a);
      //printf("%g %g\n",data_in[i*N1 + j][0],data_in[i*N1 + j][1] );
    }
  }

  fftw_execute(fplan);
  if (!rank) printf( "y: \n");
  for (i=0; i<local_n0; ++i){
    for (j=0; j<N1; ++j){
      //printf("%g %g\n",data_out[i*N1 + j][0],data_out[i*N1 + j][1] );
    }
  }


  fftw_execute(bplan);

  /* Compare */
  if (!rank) printf( "z: \n");
  for (i=0; i<local_n0; ++i){
    for (j=0; j<N1; ++j){
      //printf("%g %g\n",data_out2[i*N1 + j][0]/N,data_out2[i*N1 + j][1]/N );
      PetscRealPart(a)      = (PetscReal)(data_in[i*N1 + j][0] - data_out2[i*N1 + j][0]/N);
      PetscImaginaryPart(a) = (PetscReal)(data_in[i*N1 + j][1] - data_out2[i*N1 + j][1]/N);
      if (PetscAbsScalar(a) > 1.e-12 ) SETERRQ4(PETSC_COMM_WORLD,1,"(%g %g) != (%g %g)",data_in[i*N1 + j][0],data_in[i*N1 + j][1],data_out2[i*N1 + j][0]/N,data_out2[i*N1 + j][1]/N);
    }
  }

  /* free spaces */
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(bplan);
  fftw_free(data_in);
  fftw_free(data_out);
  fftw_free(data_out2);
  
  ierr = PetscFinalize();
  return 0;
}
