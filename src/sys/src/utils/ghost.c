
#include "petscsys.h"

#undef  __FUNCT__
#define __FUNCT__ "PetscGhostExchange"
/*@C
  PetscGhostExchange - This functions transfers data between local and ghost storage without a predefined mapping.

  Collective on MPI_Comm

  Input Parameters:
+ comm         - The communicator
. numGhosts    - The number of ghosts in this domain
. ghostProcs   - The processor from which to obtain each ghost
. ghostIndices - The global index for each ghost
. dataType     - The type of the variables
. firstVar     - The first variable on each processor
. addv         - The insert mode, INSERT_VALUES or ADD_VALUES
- mode         - The direction of the transfer, SCATTER_FORWARD or SCATTER_REVERSE

  Output Parameters:
+ locVars      - The local variable array
- ghostVars    - The ghost variables

  Note:
  The data in ghostVars is assumed contiguous and implicitly indexed by the order of
  ghostProcs and ghostIndices. The SCATTER_FORWARD mode will take the requested data
  from locVars and copy it to ghostVars in the order specified by ghostIndices. The
  SCATTER_REVERSE mode will take data from ghostVars and copy it to locVars.

  Level: advanced

.keywords: ghost, exchange
.seealso: GridGlobalToLocal(), GridLocalToGlobal()
@*/
int PetscGhostExchange(MPI_Comm comm, int numGhosts, int *ghostProcs, int *ghostIndices, PetscDataType dataType,
                      int *firstVar, InsertMode addv, ScatterMode mode, void *locVars, void *ghostVars)
{
  int         *numSendGhosts; /* The number of ghosts from each domain */
  int         *numRecvGhosts; /* The number of local variables which are ghosts in each domain */
  int         *sumSendGhosts; /* The prefix sums of numSendGhosts */
  int         *sumRecvGhosts; /* The prefix sums of numRecvGhosts */
  int         *offsets;       /* The offset into the send array for each domain */
  int          totSendGhosts; /* The number of ghosts to request variables for */
  int          totRecvGhosts; /* The number of nodes to provide class info about */
  int         *sendIndices;   /* The canonical indices of ghosts in this domain */
  int         *recvIndices;   /* The canonical indices of ghosts to return variables for */
  char        *tempVars;      /* The variables of the requested or submitted ghosts */
  char        *locBytes   = (char *) locVars;
  MPI_Datatype MPIType;
  int          typeSize;
#ifdef PETSC_USE_BOPT_g
  int          numLocVars;
#endif
  int          size, rank;
  int          proc, ghost, locIndex, byte;
  int          ierr;

  PetscFunctionBegin;
  /* Initialize communication */
  ierr = MPI_Comm_size(comm, &size);                                                                  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);                                                                      CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(int), &numSendGhosts);                                             CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(int), &numRecvGhosts);                                             CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(int), &sumSendGhosts);                                             CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(int), &sumRecvGhosts);                                             CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(int), &offsets);                                                   CHKERRQ(ierr);
  ierr = PetscMemzero(numSendGhosts,  size * sizeof(int));                                            CHKERRQ(ierr);
  ierr = PetscMemzero(numRecvGhosts,  size * sizeof(int));                                            CHKERRQ(ierr);
  ierr = PetscMemzero(sumSendGhosts,  size * sizeof(int));                                            CHKERRQ(ierr);
  ierr = PetscMemzero(sumRecvGhosts,  size * sizeof(int));                                            CHKERRQ(ierr);
  ierr = PetscMemzero(offsets,        size * sizeof(int));                                            CHKERRQ(ierr);
#ifdef PETSC_USE_BOPT_g
  numLocVars = firstVar[rank+1] - firstVar[rank];
#endif

  /* Get number of ghosts needed from each processor */
  for(ghost = 0; ghost < numGhosts; ghost++) {
    numSendGhosts[ghostProcs[ghost]]++;
  }

  /* Get number of ghosts to provide variables for */
  ierr = MPI_Alltoall(numSendGhosts, 1, MPI_INT, numRecvGhosts, 1, MPI_INT, comm);                        CHKERRQ(ierr);
  for(proc = 1; proc < size; proc++) {
    sumSendGhosts[proc] = sumSendGhosts[proc-1] + numSendGhosts[proc-1];
    sumRecvGhosts[proc] = sumRecvGhosts[proc-1] + numRecvGhosts[proc-1];
    offsets[proc]       = sumSendGhosts[proc];
  }
  totSendGhosts = sumSendGhosts[size-1] + numSendGhosts[size-1];
  totRecvGhosts = sumRecvGhosts[size-1] + numRecvGhosts[size-1];
  if (numGhosts != totSendGhosts) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of ghosts %d in send, should be %d", totSendGhosts, numGhosts);
  }

  ierr = PetscDataTypeGetSize(dataType, &typeSize);                                                       CHKERRQ(ierr);
  if (totSendGhosts) {
    ierr = PetscMalloc(totSendGhosts * sizeof(int), &sendIndices);                                        CHKERRQ(ierr);
  }
  if (totRecvGhosts) {
    ierr = PetscMalloc(totRecvGhosts * sizeof(int), &recvIndices);                                        CHKERRQ(ierr);
    ierr = PetscMalloc(totRecvGhosts * typeSize,    &tempVars);                                           CHKERRQ(ierr);
  }

  /* Must order ghosts by processor */
  for(ghost = 0; ghost < numGhosts; ghost++) {
    sendIndices[offsets[ghostProcs[ghost]]++] = ghostIndices[ghost];
  }

  /* Get canonical indices of ghosts to provide variables for */
  ierr = MPI_Alltoallv(sendIndices, numSendGhosts, sumSendGhosts, MPI_INT,
                       recvIndices, numRecvGhosts, sumRecvGhosts, MPI_INT, comm);
  CHKERRQ(ierr);

  switch(mode)
  {
  case SCATTER_FORWARD:
    /* Get ghost variables */
    if (addv == INSERT_VALUES) {
      for(ghost = 0; ghost < totRecvGhosts; ghost++) {
        locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
        if ((locIndex < 0) || (locIndex >= numLocVars)) {
          SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
        }
#endif
        for(byte = 0; byte < typeSize; byte++) {
          tempVars[ghost*typeSize+byte] = locBytes[locIndex*typeSize+byte];
        }
      }
    } else {
      for(ghost = 0; ghost < totRecvGhosts; ghost++) {
        locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
        if ((locIndex < 0) || (locIndex >= numLocVars)) {
          SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
        }
#endif
        for(byte = 0; byte < typeSize; byte++) {
          tempVars[ghost*typeSize+byte] += locBytes[locIndex*typeSize+byte];
        }
      }
    }

    /* Communicate local variables to ghost storage */
    ierr = PetscDataTypeToMPIDataType(dataType, &MPIType);                                                CHKERRQ(ierr);
    ierr = MPI_Alltoallv(tempVars,  numRecvGhosts, sumRecvGhosts, MPIType,
                         ghostVars, numSendGhosts, sumSendGhosts, MPIType, comm);
    CHKERRQ(ierr);
    break;
  case SCATTER_REVERSE:
    /* Communicate ghost variables to local storage */
    ierr = PetscDataTypeToMPIDataType(dataType, &MPIType);                                                CHKERRQ(ierr);
    ierr = MPI_Alltoallv(ghostVars, numSendGhosts, sumSendGhosts, MPIType,
                         tempVars,  numRecvGhosts, sumRecvGhosts, MPIType, comm);
    CHKERRQ(ierr);

    /* Get ghost variables */
    if (addv == INSERT_VALUES) {
      for(ghost = 0; ghost < totRecvGhosts; ghost++) {
        locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
        if ((locIndex < 0) || (locIndex >= numLocVars)) {
          SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
        }
#endif
        for(byte = 0; byte < typeSize; byte++) {
          locBytes[locIndex*typeSize+byte] = tempVars[ghost*typeSize+byte];
        }
      }
    } else {
      /* There must be a better way to do this -- Ask Bill */
      if (typeSize == sizeof(int)) {
        int *tempInt = (int *) tempVars;
        int *locInt  = (int *) locVars;

        for(ghost = 0; ghost < totRecvGhosts; ghost++) {
          locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
          if ((locIndex < 0) || (locIndex >= numLocVars)) {
            SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
          }
#endif
          locInt[locIndex] += tempInt[ghost];
        }
      } else if (typeSize == sizeof(long int)) {
        long int *tempLongInt = (long int *) tempVars;
        long int *locLongInt  = (long int *) locVars;

        for(ghost = 0; ghost < totRecvGhosts; ghost++) {
          locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
          if ((locIndex < 0) || (locIndex >= numLocVars)) {
            SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
          }
#endif
          locLongInt[locIndex] += tempLongInt[ghost];
        }
      } else {
        for(ghost = 0; ghost < totRecvGhosts; ghost++) {
          locIndex = recvIndices[ghost] - firstVar[rank];
#ifdef PETSC_USE_BOPT_g
          if ((locIndex < 0) || (locIndex >= numLocVars)) {
            SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid ghost index %d, not in [0,%d)", locIndex, numLocVars);
          }
#endif
          for(byte = 0; byte < typeSize; byte++) {
            locBytes[locIndex*typeSize+byte] += tempVars[ghost*typeSize+byte];
          }
        }
      }
    }
    break;
  default:
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid scatter mode %d", mode);
  }

  /* Cleanup */
  ierr = PetscFree(numSendGhosts);                                                                        CHKERRQ(ierr);
  ierr = PetscFree(numRecvGhosts);                                                                        CHKERRQ(ierr);
  ierr = PetscFree(sumSendGhosts);                                                                        CHKERRQ(ierr);
  ierr = PetscFree(sumRecvGhosts);                                                                        CHKERRQ(ierr);
  ierr = PetscFree(offsets);                                                                              CHKERRQ(ierr);
  if (totSendGhosts) {
    ierr = PetscFree(sendIndices);                                                                        CHKERRQ(ierr);
  }
  if (totRecvGhosts) {
    ierr = PetscFree(recvIndices);                                                                        CHKERRQ(ierr);
    ierr = PetscFree(tempVars);                                                                           CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
