
static char help[] = "Test ISLocalToGlobalMappingCreateSF().\n\n";

#include <petscis.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  MPI_Comm               comm;
  PetscViewer            viewer;
  PetscViewerFormat      format;
  PetscMPIInt            rank,size;
  PetscInt               i,nLocal = 3,nGlobal;
  PetscInt              *indices;
  PetscBool              flg, auto_offset = PETSC_FALSE;
  ISLocalToGlobalMapping l2g0, l2g1;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&nLocal,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-auto_offset",&auto_offset,NULL));
  PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-viewer", &viewer, &format, NULL));
  PetscCall(PetscMalloc1(nLocal,&indices));
  for (i=0; i<nLocal; i++) {
    indices[i] = i + rank;
  }
  nGlobal = size-1+nLocal;
  if (viewer) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(PetscViewerASCIIPrintf(viewer, "nGlobal: %" PetscInt_FMT "\n", nGlobal));
  }

  /* Create a local-to-global mapping using ISLocalToGlobalMappingCreate() */
  {
    PetscCall(ISLocalToGlobalMappingCreate(comm,1,nLocal,indices,PETSC_USE_POINTER,&l2g0));
    PetscCall(ISLocalToGlobalMappingSetFromOptions(l2g0));
    if (viewer) {
      PetscCall(PetscObjectSetName((PetscObject)l2g0, "l2g0"));
      PetscCall(ISLocalToGlobalMappingView(l2g0,viewer));
    }
  }

  /* Create the same local-to-global mapping using ISLocalToGlobalMappingCreateSF() */
  {
    PetscSF     sf;
    PetscLayout rootLayout;

    PetscCall(PetscSFCreate(comm, &sf));
    PetscCall(PetscLayoutCreateFromSizes(comm,PETSC_DECIDE,nGlobal,1,&rootLayout));
    PetscCall(PetscSFSetGraphLayout(sf,rootLayout,nLocal,NULL,PETSC_USE_POINTER,indices));
    PetscCall(PetscSFSetFromOptions(sf));
    PetscCall(ISLocalToGlobalMappingCreateSF(sf,auto_offset ? PETSC_DECIDE : rootLayout->rstart,&l2g1));
    if (viewer) {
      PetscCall(PetscObjectSetName((PetscObject)sf, "sf1"));
      PetscCall(PetscObjectSetName((PetscObject)l2g1, "l2g1"));
      PetscCall(PetscSFView(sf,viewer));
      PetscCall(ISLocalToGlobalMappingView(l2g1,viewer));
    }
    PetscCall(PetscLayoutDestroy(&rootLayout));
    PetscCall(PetscSFDestroy(&sf));
  }

  /* Compare the two local-to-global mappings by comparing results of apply for the same input */
  {
    IS input, output0, output1;

    PetscCall(ISCreateStride(comm,nLocal,0,1,&input));
    PetscCall(ISLocalToGlobalMappingApplyIS(l2g0, input, &output0));
    PetscCall(ISLocalToGlobalMappingApplyIS(l2g1, input, &output1));
    if (viewer) {
      PetscCall(PetscObjectSetName((PetscObject)input,   "input"));
      PetscCall(PetscObjectSetName((PetscObject)output0, "output0"));
      PetscCall(PetscObjectSetName((PetscObject)output1, "output1"));
      PetscCall(ISView(input,   viewer));
      PetscCall(ISView(output0, viewer));
      PetscCall(ISView(output1, viewer));
    }
    PetscCall(ISEqual(output0, output1, &flg));
    PetscCheck(flg,comm, PETSC_ERR_PLIB, "output0 != output1");
    PetscCall(ISDestroy(&input));
    PetscCall(ISDestroy(&output0));
    PetscCall(ISDestroy(&output1));
  }

  if (viewer) {
     PetscCall(PetscViewerPopFormat(viewer));
     PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g0));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g1));
  PetscCall(PetscFree(indices));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: {{1 2 3}separate output}
      args: -auto_offset {{true false}} -viewer

   test:
      suffix: 2
      nsize: {{1 2 3}}
      args: -n 33 -auto_offset {{true false}}

TEST*/
