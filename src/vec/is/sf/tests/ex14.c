
static char help[] = "Test event log of VecScatter with various block sizes\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscInt           i, j, low, high, n = 256, N, errors, tot_errors;
  PetscInt           bs = 1, ix[2], iy[2];
  PetscMPIInt        nproc, rank;
  PetscScalar       *xval;
  const PetscScalar *yval;
  Vec                x, y;
  IS                 isx, isy;
  VecScatter         ctx;
  const PetscInt     niter = 10;
#if defined(PETSC_USE_LOG)
  PetscLogStage      stage1, stage2;
  PetscLogEvent      event1, event2;
  PetscLogDouble     numMessages, messageLength;
  PetscEventPerfInfo eventInfo;
  PetscInt           tot_msg, tot_len, avg_len;
#endif

  PetscFunctionBegin;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nproc));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscLogStageRegister("Scatter(bs=1)", &stage1));
  PetscCall(PetscLogEventRegister("VecScatter(bs=1)", PETSC_OBJECT_CLASSID, &event1));
  PetscCall(PetscLogStageRegister("Scatter(bs=4)", &stage2));
  PetscCall(PetscLogEventRegister("VecScatter(bs=4)", PETSC_OBJECT_CLASSID, &event2));

  /* Create a parallel vector x and a sequential vector y */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecGetOwnershipRange(x, &low, &high));
  PetscCall(VecGetSize(x, &N));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &y));

  /*=======================================
     test VecScatter with bs = 1
    ======================================*/

  /* the code works as if we are going to do 3-point stencil computations on a 1D domain x,
     which has periodic boundary conditions but the two ghosts are scatterred to beginning of y.
   */
  bs    = 1;
  ix[0] = rank ? low - 1 : N - 1; /* ix[] contains global indices of the two ghost points */
  ix[1] = (rank != nproc - 1) ? high : 0;
  iy[0] = 0;
  iy[1] = 1;

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, ix, PETSC_COPY_VALUES, &isx));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2, iy, PETSC_COPY_VALUES, &isy));
  PetscCall(VecScatterCreate(x, isx, y, isy, &ctx));
  PetscCall(VecScatterSetUp(ctx));

  PetscCall(PetscLogStagePush(stage1));
  PetscCall(PetscLogEventBegin(event1, 0, 0, 0, 0));
  errors = 0;
  for (i = 0; i < niter; i++) {
    /* set x = 0+i, 1+i, 2+i, ..., N-1+i */
    PetscCall(VecGetArray(x, &xval));
    for (j = 0; j < n; j++) xval[j] = (PetscScalar)(low + j + i);
    PetscCall(VecRestoreArray(x, &xval));
    /* scatter the ghosts to y */
    PetscCall(VecScatterBegin(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
    /* check if y has correct values */
    PetscCall(VecGetArrayRead(y, &yval));
    if ((PetscInt)PetscRealPart(yval[0]) != ix[0] + i) errors++;
    if ((PetscInt)PetscRealPart(yval[1]) != ix[1] + i) errors++;
    PetscCall(VecRestoreArrayRead(y, &yval));
  }
  PetscCall(PetscLogEventEnd(event1, 0, 0, 0, 0));
  PetscCall(PetscLogStagePop());

  /* check if we found wrong values on any processors */
  PetscCallMPI(MPI_Allreduce(&errors, &tot_errors, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  if (tot_errors) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: wrong values were scatterred in vecscatter with bs = %" PetscInt_FMT "\n", bs));

    /* print out event log of VecScatter(bs=1) */
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogEventGetPerfInfo(stage1, event1, &eventInfo));
  PetscCallMPI(MPI_Allreduce(&eventInfo.numMessages, &numMessages, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&eventInfo.messageLength, &messageLength, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  tot_msg = (PetscInt)numMessages * 0.5; /* two MPI calls (Send & Recv) per message */
  tot_len = (PetscInt)messageLength * 0.5;
  avg_len = tot_msg ? (PetscInt)(messageLength / numMessages) : 0;
  /* when nproc > 2, tot_msg = 2*nproc*niter, tot_len = tot_msg*sizeof(PetscScalar)*bs */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecScatter(bs=%" PetscInt_FMT ") has sent out %" PetscInt_FMT " messages, total %" PetscInt_FMT " bytes, with average length %" PetscInt_FMT " bytes\n", bs, tot_msg, tot_len, avg_len));
#endif

  PetscCall(ISDestroy(&isx));
  PetscCall(ISDestroy(&isy));
  PetscCall(VecScatterDestroy(&ctx));

  /*=======================================
     test VecScatter with bs = 4
    ======================================*/

  /* similar to the 3-point stencil above, except that this time a ghost is a block */
  bs    = 4;                                /* n needs to be a multiple of bs to make the following code work */
  ix[0] = rank ? low / bs - 1 : N / bs - 1; /* ix[] contains global indices of the two ghost blocks */
  ix[1] = (rank != nproc - 1) ? high / bs : 0;
  iy[0] = 0;
  iy[1] = 1;

  PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 2, ix, PETSC_COPY_VALUES, &isx));
  PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 2, iy, PETSC_COPY_VALUES, &isy));

  PetscCall(VecScatterCreate(x, isx, y, isy, &ctx));
  /* Call SetUp explicitly, otherwise messages in implicit SetUp will be counted in events below */
  PetscCall(VecScatterSetUp(ctx));

  PetscCall(PetscLogStagePush(stage2));
  PetscCall(PetscLogEventBegin(event2, 0, 0, 0, 0));
  errors = 0;
  for (i = 0; i < niter; i++) {
    /* set x = 0+i, 1+i, 2+i, ..., N-1+i */
    PetscCall(VecGetArray(x, &xval));
    for (j = 0; j < n; j++) xval[j] = (PetscScalar)(low + j + i);
    PetscCall(VecRestoreArray(x, &xval));
    /* scatter the ghost blocks to y */
    PetscCall(VecScatterBegin(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
    /* check if y has correct values */
    PetscCall(VecGetArrayRead(y, &yval));
    if ((PetscInt)PetscRealPart(yval[0]) != ix[0] * bs + i) errors++;
    if ((PetscInt)PetscRealPart(yval[bs]) != ix[1] * bs + i) errors++;
    PetscCall(VecRestoreArrayRead(y, &yval));
  }
  PetscCall(PetscLogEventEnd(event2, 0, 0, 0, 0));
  PetscCall(PetscLogStagePop());

  /* check if we found wrong values on any processors */
  PetscCallMPI(MPI_Allreduce(&errors, &tot_errors, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  if (tot_errors) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: wrong values were scatterred in vecscatter with bs = %" PetscInt_FMT "\n", bs));

    /* print out event log of VecScatter(bs=4) */
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogEventGetPerfInfo(stage2, event2, &eventInfo));
  PetscCallMPI(MPI_Allreduce(&eventInfo.numMessages, &numMessages, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&eventInfo.messageLength, &messageLength, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
  tot_msg = (PetscInt)numMessages * 0.5; /* two MPI calls (Send & Recv) per message */
  tot_len = (PetscInt)messageLength * 0.5;
  avg_len = tot_msg ? (PetscInt)(messageLength / numMessages) : 0;
  /* when nproc > 2, tot_msg = 2*nproc*niter, tot_len = tot_msg*sizeof(PetscScalar)*bs */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecScatter(bs=%" PetscInt_FMT ") has sent out %" PetscInt_FMT " messages, total %" PetscInt_FMT " bytes, with average length %" PetscInt_FMT " bytes\n", bs, tot_msg, tot_len, avg_len));
#endif

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Program finished\n"));
  PetscCall(ISDestroy(&isx));
  PetscCall(ISDestroy(&isy));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args:
      requires: double defined(PETSC_USE_LOG)

TEST*/
