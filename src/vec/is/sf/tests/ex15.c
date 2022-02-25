static char help[]= "  Test VecScatterRemap() on various vecscatter. \n\
We may do optimization based on index patterns. After index remapping by VecScatterRemap(), we need to \n\
make sure the vecscatter works as expected with the optimizaiton. \n\
  VecScatterRemap() does not support all kinds of vecscatters. In addition, it only supports remapping \n\
entries where we read the data (i.e., todata in paralle scatter, fromdata in sequential scatter). This test \n\
tests VecScatterRemap on parallel to paralle (PtoP) vecscatter, sequential general to sequential \n\
general (SGToSG) vecscatter and sequential general to sequential stride 1 (SGToSS_Stride1) vecscatter.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,n,*ix,*iy,*tomap,start;
  Vec                x,y;
  PetscMPIInt        nproc,rank;
  IS                 isx,isy;
  const PetscInt     *ranges;
  VecScatter         vscat;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  PetscCheckFalse(nproc != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test must run with exactly two MPI ranks");

  /* ====================================================================
     (1) test VecScatterRemap on a parallel to parallel (PtoP) vecscatter
     ====================================================================
   */

  n = 64;  /* long enough to trigger memcpy optimizations both in local scatter and remote scatter */

  /* create two MPI vectors x, y of length n=64, N=128 */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Initialize x as {0~127} */
  ierr = VecGetOwnershipRanges(x,&ranges);CHKERRQ(ierr);
  for (i=ranges[rank]; i<ranges[rank+1]; i++) { ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr); }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* create two general index sets isx = {0~127} and isy = {32~63,64~95,96~127,0~31}. isx is sequential, but we use
     it as general and let PETSc detect the pattern and optimize it. indices in isy are set to make the vecscatter
     have both local scatter and remote scatter (i.e., MPI communication)
   */
  ierr = PetscMalloc2(n,&ix,n,&iy);CHKERRQ(ierr);
  start = ranges[rank];
  for (i=ranges[rank]; i<ranges[rank+1]; i++) ix[i-start] = i;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);

  if (rank == 0) { for (i=0; i<n; i++) iy[i] = i+32; }
  else for (i=0; i<n/2; i++) { iy[i] = i+96; iy[i+n/2] = i; }

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,iy,PETSC_COPY_VALUES,&isy);CHKERRQ(ierr);

  /* create a vecscatter that shifts x to the tail by quater periodically and puts the results in y */
  ierr = VecScatterCreate(x,isx,y,isy,&vscat);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {Q3,Q0,Q1,Q2} of x, that is {96~127,0~31,32~63,64~95} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Before VecScatterRemap on PtoP, MPI vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter. It changes where we read vector
     x entries to send out, but does not change the communication pattern (i.e., send/recv pairs and msg lengths).

     We create tomap as {32~63,0~31}. Originaly, we read from indices {0~64} of the local x to send out. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices {32~63,0~31} of the local x.
     isy is unchanged. So, we will shift x to {Q2,Q1,Q0,Q3}, that is {64~95,32~63,0~31,96~127}
  */
  ierr = PetscMalloc1(n,&tomap);CHKERRQ(ierr);
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  ierr = VecScatterRemap(vscat,tomap,NULL);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {64~95,32~63,0~31,96~127} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on PtoP, MPI vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* destroy everything before we recreate them in different types */
  ierr = PetscFree2(ix,iy);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = PetscFree(tomap);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  /* ==========================================================================================
     (2) test VecScatterRemap on a sequential general to sequential general (SGToSG) vecscatter
     ==========================================================================================
   */
  n = 64; /* long enough to trigger memcpy optimizations in local scatter */

  /* create two seq vectors x, y of length n */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Initialize x as {0~63} */
  for (i=0; i<n; i++) { ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr); }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* create two general index sets isx = isy = {0~63}, which are sequential, but we use them as
     general and let PETSc detect the pattern and optimize it */
  ierr = PetscMalloc2(n,&ix,n,&iy);CHKERRQ(ierr);
  for (i=0; i<n; i++) ix[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = ISDuplicate(isx,&isy);CHKERRQ(ierr);

  /* create a vecscatter that just copies x to y */
  ierr = VecScatterCreate(x,isx,y,isy,&vscat);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {0~63} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBefore VecScatterRemap on SGToSG, SEQ vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter.

     Create tomap as {32~63,0~31}. Originaly, we read from indices {0~64} of seq x to write to y. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices{32~63,0~31} of seq x.
   */
  ierr = PetscMalloc1(n,&tomap);CHKERRQ(ierr);
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  ierr = VecScatterRemap(vscat,tomap,NULL);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {32~63,0~31} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on SGToSG, SEQ vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* destroy everything before we recreate them in different types */
  ierr = PetscFree2(ix,iy);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = PetscFree(tomap);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  /* ===================================================================================================
     (3) test VecScatterRemap on a sequential general to sequential stride 1 (SGToSS_Stride1) vecscatter
     ===================================================================================================
   */
  n = 64; /* long enough to trigger memcpy optimizations in local scatter */

  /* create two seq vectors x of length n, and y of length n/2 */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n/2,&y);CHKERRQ(ierr);

  /* Initialize x as {0~63} */
  for (i=0; i<n; i++) { ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr); }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* create a general index set isx = {0:63:2}, which actually is a stride IS with first=0, n=32, step=2,
     but we use it as general and let PETSc detect the pattern and optimize it. */
  ierr = PetscMalloc2(n/2,&ix,n/2,&iy);CHKERRQ(ierr);
  for (i=0; i<n/2; i++) ix[i] = i*2;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n/2,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);

  /* create a stride1 index set isy = {0~31}. We intentionally set the step to 1 to trigger optimizations */
  ierr = ISCreateStride(PETSC_COMM_SELF,32,0,1,&isy);CHKERRQ(ierr);

  /* create a vecscatter that just copies even entries of x to y */
  ierr = VecScatterCreate(x,isx,y,isy,&vscat);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {0:63:2} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBefore VecScatterRemap on SGToSS_Stride1, SEQ vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter.

     Create tomap as {32~63,0~31}. Originaly, we read from indices{0:63:2} of seq x to write to y. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices{32:63:2,0:31:2} of seq x.
   */
  ierr = PetscMalloc1(n,&tomap);CHKERRQ(ierr);
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  ierr = VecScatterRemap(vscat,tomap,NULL);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* view y to check the result. y should be {32:63:2,0:31:2} */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on SGToSS_Stride1, SEQ vector y is:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* destroy everything before PetscFinalize */
  ierr = PetscFree2(ix,iy);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = PetscFree(tomap);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      diff_args: -j
      requires: double
TEST*/

