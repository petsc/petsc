static char help[]= "Test SF cuda stream synchronization in device to host communication\n\n";
/*
  SF uses asynchronous operations internally. When destination data is on GPU, it does asynchronous
  operations in the default stream and does not sync these operations since it assumes routines consume
  the destination data are also on the default stream. However, when destination data in on CPU,
  SF must guarentee the data is ready to use on CPU after PetscSFXxxEnd().
 */

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,n=100000; /* Big enough to make the asynchronous copy meaningful */
  PetscScalar        *val;
  const PetscScalar  *yval;
  Vec                x,y;
  PetscMPIInt        size;
  IS                 ix,iy;
  VecScatter         vscat;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uni-processor test");

  /* Create two CUDA vectors x, y. Though we only care y's memory on host, we make y a CUDA vector,
     since we want to have y's memory on host pinned (i.e.,non-pagable), to really trigger asynchronous
     cudaMemcpyDeviceToHost.
   */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,n,&y);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  /* Init x, y, and push them to GPU (their offloadmask = PETSC_OFFLOAD_GPU) */
  ierr = VecGetArray(x,&val);CHKERRQ(ierr);
  for (i=0; i<n; i++) val[i] = i/2.0;
  ierr = VecRestoreArray(x,&val);CHKERRQ(ierr);
  ierr = VecScale(x,2.0);CHKERRQ(ierr);
  ierr = VecSet(y,314);CHKERRQ(ierr);

  /* Pull y to CPU (make its offloadmask = PETSC_OFFLOAD_CPU) */
  ierr = VecGetArray(y,&val);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&val);CHKERRQ(ierr);

  /* The vscat is simply a vector copy */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&ix);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&iy);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix,y,iy,&vscat);CHKERRQ(ierr);

  /* Do device to host vecscatter and then immediately use y on host. VecScat/SF may use asynchronous
     cudaMemcpy or kernels, but it must guarentee y is ready to use on host. Otherwise, wrong data will be displayed.
   */
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&yval);CHKERRQ(ierr);
  /* Display the first and the last entries of y to see if it is valid on host */
  ierr = PetscPrintf(PETSC_COMM_SELF,"y[0]=%.f, y[%" PetscInt_FMT "] = %.f\n",(float)PetscRealPart(yval[0]),n-1,(float)PetscRealPart(yval[n-1]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y,&yval);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&ix);CHKERRQ(ierr);
  ierr = ISDestroy(&iy);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
    requires: cuda
    diff_args: -j
    #make sure the host memory is pinned
    # sf_backend cuda is not needed if compiling only with cuda
    args: -vec_type cuda -sf_backend cuda -vec_pinned_memory_min 0

   test:
    suffix: hip
    requires: hip
    diff_args: -j
    output_file: output/ex2_1.out
    #make sure the host memory is pinned
    # sf_backend hip is not needed if compiling only with hip
    args:  -vec_type hip -sf_backend hip -vec_pinned_memory_min 0

TEST*/
