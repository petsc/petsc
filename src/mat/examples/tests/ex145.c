/* This program illustrates use of paralllel real FFT*/
static char help[]="This program illustrates the use of parallel i1D complex fftw";
#include <petscmat.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
    const ptrdiff_t    N0=50;
    fftw_plan          bplan,fplan;
    fftw_complex       *out,*in1,*in2;
    ptrdiff_t          f_alloc_local,f_local_n0,f_local_0_start;
    ptrdiff_t          f_local_n1,f_local_1_end;
    ptrdiff_t          b_alloc_local,b_local_n0,b_local_0_start;
    ptrdiff_t          b_local_n1,b_local_1_end;

//    PetscInt i,j,indx[100],n1;
    PetscInt  size,rank;//n,N,*in,N_factor;
//    PetscScalar *data_fin,value1,one=1.0,zero=0.0;
//    PetscScalar a,*x_arr,*y_arr,*z_arr,enorm;
//    Vec fin,fout,fout1,x,y;
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
    fftw_mpi_init();
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    if (size==1){
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Works only for parallel FFTW 1D");}
    else {
        f_alloc_local = fftw_mpi_local_size_1d(N0,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE,&f_local_n0,&f_local_0_start,&f_local_n1,&f_local_1_end);
        b_alloc_local = fftw_mpi_local_size_1d(N0,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE,&b_local_n0,&b_local_0_start,&b_local_n1,&b_local_1_end);
    } 
    printf("Forward n0 and forward n1 are %ld and %ld\n",f_local_n0,f_local_n1);
    printf("Backward n0 and backward n1 are %ld and %ld\n",b_local_n0,b_local_n1);

}


