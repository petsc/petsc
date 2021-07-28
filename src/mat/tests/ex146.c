/* This program illustrates use of paralllel real FFT*/
static char help[]="This program illustrates the use of parallel real 3D fftw (without PETSc interface)";
#include <petscmat.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

int main(int argc,char **args)
{
  ptrdiff_t      N0=256,N1=256,N2=256,N3=2,dim[4];
  fftw_plan      bplan,fplan;
  fftw_complex   *out;
  double         *in1,*in2;
  ptrdiff_t      alloc_local,local_n0,local_0_start;
  ptrdiff_t      local_n1,local_1_start;
  PetscInt       i,j,indx,n1;
  PetscInt       size,rank,n,N,*in,N_factor,NM;
  PetscScalar    *data_fin,value1,one=1.57,zero=0.0;
  PetscScalar    a,*x_arr,*y_arr,*z_arr,enorm;
  Vec            fin,fout,fout1,ini,final;
  PetscRandom    rnd;
  PetscErrorCode ierr;
  VecScatter     vecscat,vecscat1;
  IS             indx1,indx2;
  PetscInt       *indx3,k,l,*indx4;
  PetscInt       low,tempindx,tempindx1;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers. Your current scalar type is complex");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);

  PetscRandomCreate(PETSC_COMM_WORLD,&rnd);

  alloc_local = fftw_mpi_local_size_3d_transposed(N0,N1,N2/2+1,PETSC_COMM_WORLD,&local_n0,&local_0_start,&local_n1,&local_1_start);

/*    printf("The value alloc_local is %ld from process %d\n",alloc_local,rank);     */
  printf("The value local_n0 is %ld from process %d\n",local_n0,rank);
/*    printf("The value local_0_start is  %ld from process %d\n",local_0_start,rank);*/
/*    printf("The value local_n1 is  %ld from process %d\n",local_n1,rank);          */
/*    printf("The value local_1_start is  %ld from process %d\n",local_1_start,rank);*/

  /* Allocate space for input and output arrays  */

  in1=(double*)fftw_malloc(sizeof(double)*alloc_local*2);
  in2=(double*)fftw_malloc(sizeof(double)*alloc_local*2);
  out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

  N=2*N0*N1*(N2/2+1);N_factor=N0*N1*N2;
  n=2*local_n0*N1*(N2/2+1);n1=local_n1*N0*2*N1;

/*    printf("The value N is  %d from process %d\n",N,rank);   */
/*    printf("The value n is  %d from process %d\n",n,rank);   */
/*    printf("The value n1 is  %d from process %d\n",n1,rank); */
  /* Creating data vector and accompanying array with VeccreateMPIWithArray */
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)in1,&fin);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)out,&fout);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)in2,&fout1);CHKERRQ(ierr);

/*    VecGetSize(fin,&size); */
/*    printf("The size is %d\n",size); */

  VecSet(fin,one);
  VecSet(fout,zero);
  VecSet(fout1,zero);

  VecAssemblyBegin(fin);
  VecAssemblyEnd(fin);
/*    VecView(fin,PETSC_VIEWER_STDOUT_WORLD); */

  VecGetArray(fin,&x_arr);
  VecGetArray(fout1,&z_arr);
  VecGetArray(fout,&y_arr);

  fplan=fftw_mpi_plan_dft_r2c_3d(N0,N1,N2,(double*)x_arr,(fftw_complex*)y_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);
  bplan=fftw_mpi_plan_dft_c2r_3d(N0,N1,N2,(fftw_complex*)y_arr,(double*)z_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);

  fftw_execute(fplan);
  fftw_execute(bplan);

  VecRestoreArray(fin,&x_arr);
  VecRestoreArray(fout1,&z_arr);
  VecRestoreArray(fout,&y_arr);

/*    a = 1.0/(PetscReal)N_factor; */
/*    ierr = VecScale(fout1,a);CHKERRQ(ierr); */
  VecCreate(PETSC_COMM_WORLD,&ini);
  VecCreate(PETSC_COMM_WORLD,&final);
  VecSetSizes(ini,local_n0*N1*N2,N_factor);
  VecSetSizes(final,local_n0*N1*N2,N_factor);
/*    VecSetSizes(ini,PETSC_DECIDE,N_factor); */
/*    VecSetSizes(final,PETSC_DECIDE,N_factor); */
  VecSetFromOptions(ini);
  VecSetFromOptions(final);

  if (N2%2==0) NM=N2+2;
  else NM=N2+1;

  ierr = VecGetOwnershipRange(fin,&low,NULL);CHKERRQ(ierr);
  printf("The local index is %d from %d\n",low,rank);
  ierr = PetscMalloc1(local_n0*N1*N2,&indx3);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_n0*N1*N2,&indx4);CHKERRQ(ierr);
  for (i=0; i<local_n0; i++) {
    for (j=0;j<N1;j++) {
      for (k=0;k<N2;k++) {
        tempindx  = i*N1*N2 + j*N2 + k;
        tempindx1 = i*N1*NM + j*NM + k;

        indx3[tempindx]=local_0_start*N1*N2+tempindx;
        indx4[tempindx]=low+tempindx1;
      }
      /*          printf("index3 %d from proc %d is \n",indx3[tempindx],rank); */
      /*          printf("index4 %d from proc %d is \n",indx4[tempindx],rank); */
    }
  }
  VecGetValues(fin,local_n0*N1*N2,indx4,x_arr);
  VecSetValues(ini,local_n0*N1*N2,indx3,x_arr,INSERT_VALUES);
  VecAssemblyBegin(ini);
  VecAssemblyEnd(ini);

  VecGetValues(fout1,local_n0*N1*N2,indx4,y_arr);
  VecSetValues(final,local_n0*N1*N2,indx3,y_arr,INSERT_VALUES);
  VecAssemblyBegin(final);
  VecAssemblyEnd(final);

  printf("The local index value is %ld from %d",local_n0*N1*N2,rank);
/*
  for (i=0;i<N0;i++) {
     for (j=0;j<N1;j++) {
        indx=i*N1*NM+j*NM;
        ISCreateStride(PETSC_COMM_WORLD,N2,indx,1,&indx1);
        indx=i*N1*N2+j*N2;
        ISCreateStride(PETSC_COMM_WORLD,N2,indx,1,&indx2);
        VecScatterCreate(fin,indx1,ini,indx2,&vecscat);
        VecScatterBegin(vecscat,fin,ini,INSERT_VALUES,SCATTER_FORWARD);
        VecScatterEnd(vecscat,fin,ini,INSERT_VALUES,SCATTER_FORWARD);
        VecScatterCreate(fout1,indx1,final,indx2,&vecscat1);
        VecScatterBegin(vecscat1,fout1,final,INSERT_VALUES,SCATTER_FORWARD);
        VecScatterEnd(vecscat1,fout1,final,INSERT_VALUES,SCATTER_FORWARD);
     }
  }
*/
  a    = 1.0/(PetscReal)N_factor;
  ierr = VecScale(fout1,a);CHKERRQ(ierr);
  ierr = VecScale(final,a);CHKERRQ(ierr);

  VecAssemblyBegin(ini);
  VecAssemblyEnd(ini);

  VecAssemblyBegin(final);
  VecAssemblyEnd(final);

/*    VecView(final,PETSC_VIEWER_STDOUT_WORLD); */
  ierr = VecAXPY(final,-1.0,ini);CHKERRQ(ierr);
  ierr = VecNorm(final,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm of |x - z|  = %e\n",enorm);CHKERRQ(ierr);
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(bplan);
  fftw_free(in1); ierr = VecDestroy(&fin);CHKERRQ(ierr);
  fftw_free(out); ierr = VecDestroy(&fout);CHKERRQ(ierr);
  fftw_free(in2); ierr = VecDestroy(&fout1);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
