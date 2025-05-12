static char help[] = "Tests ISLocalToGlobalMappingGetInfo() and ISLocalToGlobalMappingGetNodeInfo().\n\n";

#include <petscis.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  ISLocalToGlobalMapping ltog = NULL;
  PetscSF                mlsf;
  PetscInt              *p, *ns, **ids, *leaves, *mleaves;
  PetscInt               nl, mnl, mnr, i, j, k, n, np, bs = 1, test = 0;
  PetscViewer            viewer;
  PetscMPIInt            rank, size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-test", &test, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  switch (test) {
  case 1: /* quads */
    if (size > 1) {
      if (size == 4) {
        if (rank == 0) {
          PetscInt id[4] = {0, 1, 2, 3};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 1) {
          PetscInt id[4] = {2, 3, 6, 7};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 2) {
          PetscInt id[4] = {1, 4, 3, 5};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 3) {
          PetscInt id[8] = {3, 5, 7, 8};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        }
      } else {
        if (rank == 0) {
          PetscInt id[8] = {0, 1, 2, 3, 1, 4, 3, 5};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 8, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == size - 1) {
          PetscInt id[8] = {2, 3, 6, 7, 3, 5, 7, 8};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 8, id, PETSC_COPY_VALUES, &ltog));
        } else {
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 0, NULL, PETSC_COPY_VALUES, &ltog));
        }
      }
    } else {
      PetscInt id[16] = {0, 1, 2, 3, 1, 4, 3, 5, 2, 3, 6, 7, 3, 5, 7, 8};
      PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 16, id, PETSC_COPY_VALUES, &ltog));
    }
    break;
  case 2: /* mix quads and tets with holes */
    if (size > 1) {
      if (size == 4) {
        if (rank == 0) {
          PetscInt id[3] = {1, 2, 3};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 3, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 1) {
          PetscInt id[4] = {1, 4, 5, 3};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 2) {
          PetscInt id[3] = {3, 6, 2};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 3, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == 3) {
          PetscInt id[3] = {3, 5, 8};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 3, id, PETSC_COPY_VALUES, &ltog));
        }
      } else {
        if (rank == 0) {
          PetscInt id[9] = {1, 2, 3, 3, 5, 8, 3, 6, 2};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 9, id, PETSC_COPY_VALUES, &ltog));
        } else if (rank == size - 1) {
          PetscInt id[4] = {5, 3, 1, 4};
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 4, id, PETSC_COPY_VALUES, &ltog));
        } else {
          PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 0, NULL, PETSC_COPY_VALUES, &ltog));
        }
      }
    } else {
      PetscInt id[13] = {1, 2, 3, 1, 4, 5, 3, 6, 3, 2, 5, 3, 8};
      PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 13, id, PETSC_COPY_VALUES, &ltog));
    }
    break;
  default:
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, 0, NULL, PETSC_COPY_VALUES, &ltog));
    break;
  }
  PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ltog), &viewer));
  PetscCall(ISLocalToGlobalMappingView(ltog, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "GETINFO OUTPUT\n"));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(ISLocalToGlobalMappingGetInfo(ltog, &np, &p, &ns, &ids));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local NP %" PetscInt_FMT "\n", rank, np));
  for (i = 0; i < np; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   procs[%" PetscInt_FMT "] = %" PetscInt_FMT ", shared %" PetscInt_FMT "\n", rank, i, p[i], ns[i]));
    for (j = 0; j < ns[i]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]     ids[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, j, ids[i][j]));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(ISLocalToGlobalMappingRestoreInfo(ltog, &np, &p, &ns, &ids));
  PetscCall(PetscViewerASCIIPrintf(viewer, "GETNODEINFO OUTPUT\n"));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(ISLocalToGlobalMappingGetNodeInfo(ltog, &n, &ns, &ids));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local N %" PetscInt_FMT "\n", rank, n));
  for (i = 0; i < n; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   sharedby[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, ns[i]));
    for (j = 0; j < ns[i]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]     ids[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, j, ids[i][j]));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(ltog, &n, &ns, &ids));

  /* Test block */
  PetscCall(PetscViewerASCIIPrintf(viewer, "GETBLOCKINFO OUTPUT\n"));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(ISLocalToGlobalMappingGetBlockInfo(ltog, &np, &p, &ns, &ids));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local NP %" PetscInt_FMT "\n", rank, np));
  for (i = 0; i < np; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   procs[%" PetscInt_FMT "] = %" PetscInt_FMT ", shared %" PetscInt_FMT "\n", rank, i, p[i], ns[i]));
    for (j = 0; j < ns[i]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]     ids[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, j, ids[i][j]));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(ISLocalToGlobalMappingRestoreBlockInfo(ltog, &np, &p, &ns, &ids));
  PetscCall(PetscViewerASCIIPrintf(viewer, "GETBLOCKNODEINFO OUTPUT\n"));
  PetscCall(ISLocalToGlobalMappingGetBlockNodeInfo(ltog, &n, &ns, &ids));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local N %" PetscInt_FMT "\n", rank, n));
  for (i = 0; i < n; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   sharedby[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, ns[i]));
    for (j = 0; j < ns[i]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]     ids[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, j, ids[i][j]));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(ISLocalToGlobalMappingGetBlockMultiLeavesSF(ltog, &mlsf));
  PetscCall(PetscSFGetGraph(mlsf, &mnr, &mnl, NULL, NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(ltog, &nl));
  nl /= bs;
  for (i = 0, j = 0; i < n; i++) j += ns[i];
  PetscCheck(mnr == nl, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of roots in multi-leaves SF %" PetscInt_FMT " != %" PetscInt_FMT, mnr, nl);
  PetscCheck(mnl == j, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of leaves in multi-leaves SF %" PetscInt_FMT " != %" PetscInt_FMT, mnl, j);
  PetscCall(PetscMalloc2(2 * nl, &leaves, 2 * mnl, &mleaves));
  for (i = 0; i < nl; i++) {
    leaves[2 * i]     = -(rank + 1);
    leaves[2 * i + 1] = i;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "BLOCK MULTI-LEAVES INPUT\n"));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  for (i = 0; i < nl; i++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   input[%" PetscInt_FMT "] = (%" PetscInt_FMT ", %" PetscInt_FMT ")\n", rank, i, leaves[2 * i], leaves[2 * i + 1]));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscSFBcastBegin(mlsf, MPIU_2INT, leaves, mleaves, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(mlsf, MPIU_2INT, leaves, mleaves, MPI_REPLACE));
  PetscCall(PetscViewerASCIIPrintf(viewer, "BLOCK MULTI-LEAVES OUTPUT\n"));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local N %" PetscInt_FMT "\n", rank, n));
  for (i = 0, k = 0; i < n; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]   sharedby[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, ns[i]));
    for (j = 0; j < ns[i]; j++) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]     recv[%" PetscInt_FMT "] = (%" PetscInt_FMT ", %" PetscInt_FMT ") from %" PetscInt_FMT "\n", rank, j, mleaves[2 * (k + j)], mleaves[2 * (k + j) + 1], ids[i][j]));
    }
    k += ns[i];
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscFree2(leaves, mleaves));
  PetscCall(ISLocalToGlobalMappingRestoreBlockNodeInfo(ltog, &n, &ns, &ids));

  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: ltog_info
     nsize: {{1 2 3 4 5}separate output}
     args: -bs {{1 3}separate output} -test {{0 1 2}separate output}

TEST*/
