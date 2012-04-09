/* This program illustrates use of parallel real FFT */
static char help[]="This program illustrates the use of parallel real 2D fft using fftw (without PETSc interface)";
#include <petscmat.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
    const ptrdiff_t N0=2056,N1=2056;
    fftw_plan       bplan,fplan;
    fftw_complex    *out;
    double          *in1,*in2;
    ptrdiff_t       alloc_local,local_n0,local_0_start;
    ptrdiff_t       local_n1,local_1_start;
    PetscInt        i,j,n1;
    PetscInt        size,rank,n,N,N_factor,NM;
    PetscScalar     one=2.0,zero=0.5;
    PetscScalar     two=4.0,three=8.0,four=16.0;
    PetscScalar     a,*x_arr,*y_arr,*z_arr,enorm;
    Vec             fin,fout,fout1;
    Vec             ini,final;
    PetscRandom     rnd;
    PetscErrorCode  ierr; 
    PetscInt        *indx3,tempindx,low,*indx4,tempindx1;
    
    ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);

    alloc_local = fftw_mpi_local_size_2d_transposed(N0,N1/2+1,PETSC_COMM_WORLD,&local_n0,&local_0_start,&local_n1,&local_1_start);
#if defined(DEBUGGING)
    printf("The value alloc_local is %ld from process %d\n",alloc_local,rank);  
    printf("The value local_n0 is %ld from process %d\n",local_n0,rank);  
    printf("The value local_0_start is  %ld from process %d\n",local_0_start,rank);  
//    printf("The value local_n1 is  %ld from process %d\n",local_n1,rank);  
//    printf("The value local_1_start is  %ld from process %d\n",local_1_start,rank);  
//    printf("The value local_n0 is  %ld from process %d\n",local_n0,rank);  
#endif

    /* Allocate space for input and output arrays  */
    in1=(double *)fftw_malloc(sizeof(double)*alloc_local*2);
    in2=(double *)fftw_malloc(sizeof(double)*alloc_local*2);
    out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*alloc_local);
    
    N=2*N0*(N1/2+1);N_factor=N0*N1;
    n=2*local_n0*(N1/2+1);n1=local_n1*N0*2;

//    printf("The value N is  %d from process %d\n",N,rank);  
//    printf("The value n is  %d from process %d\n",n,rank);  
//    printf("The value n1 is  %d from process %d\n",n1,rank);  
    /* Creating data vector and accompanying array with VeccreateMPIWithArray */
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,( PetscScalar*)in1,&fin);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)out,&fout);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,(PetscScalar*)in2,&fout1);CHKERRQ(ierr);

    /* Set the vector with random data */
    ierr = VecSet(fin,zero);CHKERRQ(ierr);
//    for(i=0;i<N0*N1;i++)
//       {
//       VecSetValues(fin,1,&i,&one,INSERT_VALUES);
//     }
 
//    VecSet(fin,one);
    i=0;
    ierr = VecSetValues(fin,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    i=1;
    ierr = VecSetValues(fin,1,&i,&two,INSERT_VALUES);CHKERRQ(ierr);
    i=4;
    ierr = VecSetValues(fin,1,&i,&three,INSERT_VALUES);CHKERRQ(ierr);
    i=5;
    ierr = VecSetValues(fin,1,&i,&four,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(fin);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(fin);CHKERRQ(ierr);
    
    ierr = VecSet(fout,zero);CHKERRQ(ierr);
    ierr = VecSet(fout1,zero);CHKERRQ(ierr);
        
    // Get the meaningful portion of array 
    ierr = VecGetArray(fin,&x_arr);CHKERRQ(ierr);
    ierr = VecGetArray(fout1,&z_arr);CHKERRQ(ierr);
    ierr = VecGetArray(fout,&y_arr);CHKERRQ(ierr);

    fplan=fftw_mpi_plan_dft_r2c_2d(N0,N1,(double *)x_arr,(fftw_complex *)y_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);
    bplan=fftw_mpi_plan_dft_c2r_2d(N0,N1,(fftw_complex *)y_arr,(double *)z_arr,PETSC_COMM_WORLD,FFTW_ESTIMATE);
   
    fftw_execute(fplan);
    fftw_execute(bplan);  
  
    ierr = VecRestoreArray(fin,&x_arr);
    ierr = VecRestoreArray(fout1,&z_arr);
    ierr = VecRestoreArray(fout,&y_arr);

//    VecView(fin,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecCreate(PETSC_COMM_WORLD,&ini);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&final);CHKERRQ(ierr);
    ierr = VecSetSizes(ini,local_n0*N1,N0*N1);CHKERRQ(ierr);
    ierr = VecSetSizes(final,local_n0*N1,N0*N1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(ini);CHKERRQ(ierr);   
    ierr = VecSetFromOptions(final);CHKERRQ(ierr);   
 
    if (N1%2==0){
      NM = N1+2;
    } else {
      NM = N1+1;
    }
    //printf("The Value of NM is %d",NM);
    ierr = VecGetOwnershipRange(fin,&low,PETSC_NULL);  
    //printf("The local index is %d from %d\n",low,rank);
    ierr = PetscMalloc(sizeof(PetscInt)*local_n0*N1,&indx3);
    ierr = PetscMalloc(sizeof(PetscInt)*local_n0*N1,&indx4);
    for (i=0;i<local_n0;i++){
      for (j=0;j<N1;j++){
        tempindx = i*N1 + j;
        tempindx1 = i*NM + j;
        indx3[tempindx]=local_0_start*N1+tempindx;
        indx4[tempindx]=low+tempindx1;
        //          printf("index3 %d from proc %d is \n",indx3[tempindx],rank);
        //          printf("index4 %d from proc %d is \n",indx4[tempindx],rank);
      }
    }

    ierr = VecGetValues(fin,local_n0*N1,indx4,x_arr);CHKERRQ(ierr);   
    ierr = VecSetValues(ini,local_n0*N1,indx3,x_arr,INSERT_VALUES);CHKERRQ(ierr);   
    ierr = VecAssemblyBegin(ini);CHKERRQ(ierr);   
    ierr = VecAssemblyEnd(ini);CHKERRQ(ierr);   

    ierr = VecGetValues(fout1,local_n0*N1,indx4,y_arr);
    ierr = VecSetValues(final,local_n0*N1,indx3,y_arr,INSERT_VALUES);
    ierr = VecAssemblyBegin(final);
    ierr = VecAssemblyEnd(final);

/*    
    VecScatter      vecscat;
    IS              indx1,indx2;
    for (i=0;i<N0;i++){
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

    a = 1.0/(PetscReal)N_factor;
    ierr = VecScale(fout1,a);CHKERRQ(ierr);
    ierr = VecScale(final,a);CHKERRQ(ierr);
 

//    VecView(ini,PETSC_VIEWER_STDOUT_WORLD);
//    VecView(final,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecAXPY(final,-1.0,ini);CHKERRQ(ierr);
     
    ierr = VecNorm(final,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-10){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm of |x - z|  = %e\n",enorm);CHKERRQ(ierr);
    }
 
    // Execute fftw with function fftw_execute and destory it after execution
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);
    fftw_free(in1);  ierr = VecDestroy(&fin);CHKERRQ(ierr);
    fftw_free(out);  ierr = VecDestroy(&fout);CHKERRQ(ierr);
    fftw_free(in2);  ierr = VecDestroy(&fout1);CHKERRQ(ierr);

    ierr = VecDestroy(&ini);CHKERRQ(ierr);
    ierr = VecDestroy(&final);CHKERRQ(ierr);

    ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
    ierr = PetscFree(indx3);CHKERRQ(ierr);
    ierr = PetscFree(indx4);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}


