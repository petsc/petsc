/* This program illustrates use of parallel real FFT */
static char help[]="This program illustrates the use of parallel real fftw (without PETSc interface)";
#include <petscmat.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
    ptrdiff_t N0=2,N1=2,N2=2,N3=2,dim[4],N,D;
    fftw_plan bplan,fplan;
    fftw_complex *out;
    double *in1,*in2;
    ptrdiff_t alloc_local,local_n0,local_0_start;
    ptrdiff_t local_n1,local_1_start;
    PetscInt i,j,indx[100],n1;
    PetscInt  size,rank,n,*in,N_factor;
    PetscScalar *data_fin,value1,one=1.0,zero=0.0;
    PetscScalar a,*x_arr,*y_arr,*z_arr,enorm;
    Vec fin,fout,fout1,x,y;
    PetscRandom    rnd;
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

    PetscRandomCreate(PETSC_COMM_WORLD,&rnd);
    D=4;
    dim[0]=N0;dim[1]=N1;dim[2]=N2;dim[3]=N3/2+1;


    alloc_local = fftw_mpi_local_size_transposed(D,dim,PETSC_COMM_WORLD,&local_n0,&local_0_start,&local_n1,&local_1_start);

    printf("The value alloc_local is %ld from process %d\n",alloc_local,rank);
    printf("The value local_n0 is %ld from process %d\n",local_n0,rank);
    printf("The value local_0_start is  %ld from process %d\n",local_0_start,rank);
    printf("The value local_n1 is  %ld from process %d\n",local_n1,rank);
    printf("The value local_1_start is  %ld from process %d\n",local_1_start,rank);

    /* Allocate space for input and output arrays  */

    in1=(double *)fftw_malloc(sizeof(double)*alloc_local*2);
    in2=(double *)fftw_malloc(sizeof(double)*alloc_local*2);
    out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*alloc_local);


    N=2*N0*N1*N2*(N3/2+1);N_factor=N0*N1*N2*N3;
    n=2*local_n0*N1*N2*(N3/2+1);n1=local_n1*N0*2*N1*N2;

//    printf("The value N is  %d from process %d\n",N,rank);
//    printf("The value n is  %d from process %d\n",n,rank);
//    printf("The value n1 is  %d from process %d\n",n1,rank);
    /* Creating data vector and accompanying array with VeccreateMPIWithArray */
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar *)in1,&fin);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)out,&fout);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)in2,&fout1);CHKERRQ(ierr);

//    VecGetSize(fin,&size);
//    printf("The size is %d\n",size);

    VecSet(fin,one);
//    VecAssemblyBegin(fin);
//    VecAssemblyEnd(fin);
//    VecView(fin,PETSC_VIEWER_STDOUT_WORLD);


    VecGetArray(fin,&x_arr);
    VecGetArray(fout1,&z_arr);
    VecGetArray(fout,&y_arr);

    dim[3]=N3;

    fplan=fftw_mpi_plan_dft_r2c(D,dim,(double *)x_arr,(fftw_complex *)y_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);
    bplan=fftw_mpi_plan_dft_c2r(D,dim,(fftw_complex *)y_arr,(double *)z_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);

    fftw_execute(fplan);
    fftw_execute(bplan);

    VecRestoreArray(fin,&x_arr);
    VecRestoreArray(fout1,&z_arr);
    VecRestoreArray(fout,&y_arr);

//    a = 1.0/(PetscReal)N_factor;
//    ierr = VecScale(fout1,a);CHKERRQ(ierr);

    VecAssemblyBegin(fout1);
    VecAssemblyEnd(fout1);

    VecView(fout1,PETSC_VIEWER_STDOUT_WORLD);

    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    fftw_free(in1); ierr = VecDestroy(&fin) ;  CHKERRQ(ierr);
    fftw_free(out); ierr = VecDestroy(&fout);  CHKERRQ(ierr);
    fftw_free(in2); ierr = VecDestroy(&fout1); CHKERRQ(ierr);

    ierr = PetscFinalize();
    return 0;
}
