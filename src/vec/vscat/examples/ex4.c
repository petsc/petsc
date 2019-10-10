
static char help[]= "Test event log of VecScatter with various block sizes\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,low,high,n=256,N,errors,tot_errors,tot_msg,tot_len,avg_len;
  PetscInt           bs=1,ix[2],iy[2];
  PetscMPIInt        nproc,rank;
  PetscScalar       *xval;
  const PetscScalar *yval;
  Vec                x,y;
  IS                 isx,isy;
  VecScatter         ctx;
  const PetscInt     niter = 10;
  PetscLogStage      stage1,stage2;
  PetscLogEvent      event1,event2;
  PetscLogDouble     numMessages,messageLength;
  PetscEventPerfInfo eventInfo;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Scatter(bs=1)", &stage1);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatter(bs=1)", PETSC_OBJECT_CLASSID, &event1);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Scatter(bs=4)", &stage2);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatter(bs=4)", PETSC_OBJECT_CLASSID, &event2);CHKERRQ(ierr);

  /* Create a parallel vector x and a sequential vector y */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRQ(ierr);

  /*=======================================
     test VecScatter with bs = 1
    ======================================*/

  /* the code works as if we are going to do 3-point stencil computations on a 1D domain x,
     which has periodic boundary conditions but the two ghosts are scatterred to beginning of y.
   */
  bs    = 1;
  ix[0] = rank? low-1 : N-1; /* ix[] contains global indices of the two ghost points */
  ix[1] = (rank != nproc-1)? high : 0;
  iy[0] = 0;
  iy[1] = 1;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,iy,PETSC_COPY_VALUES,&isy);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,isx,y,isy,&ctx);CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event1,0,0,0,0);CHKERRQ(ierr);
  errors = 0;
  for (i=0; i<niter; i++) {
    /* set x = 0+i, 1+i, 2+i, ..., N-1+i */
    ierr = VecGetArray(x,&xval);CHKERRQ(ierr);
    for (j=0; j<n; j++) xval[j] = (PetscScalar)(low+j+i);
    ierr = VecRestoreArray(x,&xval);CHKERRQ(ierr);
    /* scatter the ghosts to y */
    ierr = VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* check if y has correct values */
    ierr = VecGetArrayRead(y,&yval);CHKERRQ(ierr);
    if ((PetscInt)PetscRealPart(yval[0]) != ix[0]+i) errors++;
    if ((PetscInt)PetscRealPart(yval[1]) != ix[1]+i) errors++;
    ierr = VecRestoreArrayRead(y,&yval);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(event1,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* check if we found wrong values on any processors */
  ierr = MPI_Allreduce(&errors,&tot_errors,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (tot_errors) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: wrong values were scatterred in vecscatter with bs = %d\n",bs);CHKERRQ(ierr); }

  /* print out event log of VecScatter(bs=1) */
  ierr    = PetscLogEventGetPerfInfo(stage1,event1,&eventInfo);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&eventInfo.numMessages,  &numMessages,  1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&eventInfo.messageLength,&messageLength,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  tot_msg = (PetscInt)numMessages*0.5; /* two MPI calls (Send & Recv) per message */
  tot_len = (PetscInt)messageLength*0.5;
  avg_len = tot_msg? (PetscInt)(messageLength/numMessages) : 0;
  /* when nproc > 2, tot_msg = 2*nproc*niter, tot_len = tot_msg*sizeof(PetscScalar)*bs */
  ierr    = PetscPrintf(PETSC_COMM_WORLD,"VecScatter(bs=%d) has sent out %d messages, total %d bytes, with average length %d bytes\n",bs,tot_msg,tot_len,avg_len);CHKERRQ(ierr);

  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  /*=======================================
     test VecScatter with bs = 4
    ======================================*/

  /* similar to the 3-point stencil above, except that this time a ghost is a block */
  bs    = 4; /* n needs to be a multiple of bs to make the following code work */
  ix[0] = rank? low/bs-1 : N/bs-1; /* ix[] contains global indices of the two ghost blocks */
  ix[1] = (rank != nproc-1)? high/bs : 0;
  iy[0] = 0;
  iy[1] = 1;

  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,ix,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,iy,PETSC_COPY_VALUES,&isy);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,isx,y,isy,&ctx);CHKERRQ(ierr);

  ierr = PetscLogStagePush(stage2);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event2,0,0,0,0);CHKERRQ(ierr);
  errors = 0;
  for (i=0; i<niter; i++) {
    /* set x = 0+i, 1+i, 2+i, ..., N-1+i */
    ierr = VecGetArray(x,&xval);CHKERRQ(ierr);
    for (j=0; j<n; j++) xval[j] = (PetscScalar)(low+j+i);
    ierr = VecRestoreArray(x,&xval);CHKERRQ(ierr);
    /* scatter the ghost blocks to y */
    ierr = VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* check if y has correct values */
    ierr = VecGetArrayRead(y,&yval);CHKERRQ(ierr);
    if ((PetscInt)PetscRealPart(yval[0])  != ix[0]*bs+i) errors++;
    if ((PetscInt)PetscRealPart(yval[bs]) != ix[1]*bs+i) errors++;
    ierr = VecRestoreArrayRead(y,&yval);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(event2,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* check if we found wrong values on any processors */
  ierr = MPI_Allreduce(&errors,&tot_errors,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (tot_errors) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: wrong values were scatterred in vecscatter with bs = %d\n",bs);CHKERRQ(ierr); }

  /* print out event log of VecScatter(bs=4) */
  ierr    = PetscLogEventGetPerfInfo(stage2,event2,&eventInfo);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&eventInfo.numMessages,  &numMessages,  1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr    = MPI_Allreduce(&eventInfo.messageLength,&messageLength,1,MPIU_PETSCLOGDOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  tot_msg = (PetscInt)numMessages*0.5; /* two MPI calls (Send & Recv) per message */
  tot_len = (PetscInt)messageLength*0.5;
  avg_len = tot_msg? (PetscInt)(messageLength/numMessages) : 0;
  /* when nproc > 2, tot_msg = 2*nproc*niter, tot_len = tot_msg*sizeof(PetscScalar)*bs */
  ierr    = PetscPrintf(PETSC_COMM_WORLD,"VecScatter(bs=%d) has sent out %d messages, total %d bytes, with average length %d bytes\n",bs,tot_msg,tot_len,avg_len);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Program finished\n");CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      args:
      requires: double define(PETSC_USE_LOG)

   test:
      suffix: 2
      nsize: 4
      args: -vecscatter_type mpi3
      # have this filter since some messages might go through shared memory and PETSc have not yet
      # implemented message logging for them. Add this test to just test mpi3 VecScatter type works.
      filter: grep -v "VecScatter(bs="
      requires: double define(PETSC_USE_LOG) define(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
TEST*/
