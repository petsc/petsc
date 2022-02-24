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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(nproc != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test must run with exactly two MPI ranks");

  /* ====================================================================
     (1) test VecScatterRemap on a parallel to parallel (PtoP) vecscatter
     ====================================================================
   */

  n = 64;  /* long enough to trigger memcpy optimizations both in local scatter and remote scatter */

  /* create two MPI vectors x, y of length n=64, N=128 */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* Initialize x as {0~127} */
  CHKERRQ(VecGetOwnershipRanges(x,&ranges));
  for (i=ranges[rank]; i<ranges[rank+1]; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* create two general index sets isx = {0~127} and isy = {32~63,64~95,96~127,0~31}. isx is sequential, but we use
     it as general and let PETSc detect the pattern and optimize it. indices in isy are set to make the vecscatter
     have both local scatter and remote scatter (i.e., MPI communication)
   */
  CHKERRQ(PetscMalloc2(n,&ix,n,&iy));
  start = ranges[rank];
  for (i=ranges[rank]; i<ranges[rank+1]; i++) ix[i-start] = i;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,ix,PETSC_COPY_VALUES,&isx));

  if (rank == 0) { for (i=0; i<n; i++) iy[i] = i+32; }
  else for (i=0; i<n/2; i++) { iy[i] = i+96; iy[i+n/2] = i; }

  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,iy,PETSC_COPY_VALUES,&isy));

  /* create a vecscatter that shifts x to the tail by quater periodically and puts the results in y */
  CHKERRQ(VecScatterCreate(x,isx,y,isy,&vscat));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {Q3,Q0,Q1,Q2} of x, that is {96~127,0~31,32~63,64~95} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Before VecScatterRemap on PtoP, MPI vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter. It changes where we read vector
     x entries to send out, but does not change the communication pattern (i.e., send/recv pairs and msg lengths).

     We create tomap as {32~63,0~31}. Originaly, we read from indices {0~64} of the local x to send out. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices {32~63,0~31} of the local x.
     isy is unchanged. So, we will shift x to {Q2,Q1,Q0,Q3}, that is {64~95,32~63,0~31,96~127}
  */
  CHKERRQ(PetscMalloc1(n,&tomap));
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  CHKERRQ(VecScatterRemap(vscat,tomap,NULL));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {64~95,32~63,0~31,96~127} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on PtoP, MPI vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* destroy everything before we recreate them in different types */
  CHKERRQ(PetscFree2(ix,iy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(PetscFree(tomap));
  CHKERRQ(VecScatterDestroy(&vscat));

  /* ==========================================================================================
     (2) test VecScatterRemap on a sequential general to sequential general (SGToSG) vecscatter
     ==========================================================================================
   */
  n = 64; /* long enough to trigger memcpy optimizations in local scatter */

  /* create two seq vectors x, y of length n */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* Initialize x as {0~63} */
  for (i=0; i<n; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* create two general index sets isx = isy = {0~63}, which are sequential, but we use them as
     general and let PETSc detect the pattern and optimize it */
  CHKERRQ(PetscMalloc2(n,&ix,n,&iy));
  for (i=0; i<n; i++) ix[i] = i;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,ix,PETSC_COPY_VALUES,&isx));
  CHKERRQ(ISDuplicate(isx,&isy));

  /* create a vecscatter that just copies x to y */
  CHKERRQ(VecScatterCreate(x,isx,y,isy,&vscat));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {0~63} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nBefore VecScatterRemap on SGToSG, SEQ vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter.

     Create tomap as {32~63,0~31}. Originaly, we read from indices {0~64} of seq x to write to y. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices{32~63,0~31} of seq x.
   */
  CHKERRQ(PetscMalloc1(n,&tomap));
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  CHKERRQ(VecScatterRemap(vscat,tomap,NULL));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {32~63,0~31} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on SGToSG, SEQ vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* destroy everything before we recreate them in different types */
  CHKERRQ(PetscFree2(ix,iy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(PetscFree(tomap));
  CHKERRQ(VecScatterDestroy(&vscat));

  /* ===================================================================================================
     (3) test VecScatterRemap on a sequential general to sequential stride 1 (SGToSS_Stride1) vecscatter
     ===================================================================================================
   */
  n = 64; /* long enough to trigger memcpy optimizations in local scatter */

  /* create two seq vectors x of length n, and y of length n/2 */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n/2,&y));

  /* Initialize x as {0~63} */
  for (i=0; i<n; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* create a general index set isx = {0:63:2}, which actually is a stride IS with first=0, n=32, step=2,
     but we use it as general and let PETSc detect the pattern and optimize it. */
  CHKERRQ(PetscMalloc2(n/2,&ix,n/2,&iy));
  for (i=0; i<n/2; i++) ix[i] = i*2;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n/2,ix,PETSC_COPY_VALUES,&isx));

  /* create a stride1 index set isy = {0~31}. We intentionally set the step to 1 to trigger optimizations */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,32,0,1,&isy));

  /* create a vecscatter that just copies even entries of x to y */
  CHKERRQ(VecScatterCreate(x,isx,y,isy,&vscat));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {0:63:2} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nBefore VecScatterRemap on SGToSS_Stride1, SEQ vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* now call the weird subroutine VecScatterRemap to slightly change the vecscatter.

     Create tomap as {32~63,0~31}. Originaly, we read from indices{0:63:2} of seq x to write to y. The remap
     does indices[i] = tomap[indices[i]]. Therefore, after the remap, we read from indices{32:63:2,0:31:2} of seq x.
   */
  CHKERRQ(PetscMalloc1(n,&tomap));
  for (i=0; i<n/2; i++) { tomap[i] = i+n/2; tomap[i+n/2] = i; };
  CHKERRQ(VecScatterRemap(vscat,tomap,NULL));
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* view y to check the result. y should be {32:63:2,0:31:2} */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After VecScatterRemap on SGToSS_Stride1, SEQ vector y is:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* destroy everything before PetscFinalize */
  CHKERRQ(PetscFree2(ix,iy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(PetscFree(tomap));
  CHKERRQ(VecScatterDestroy(&vscat));

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
