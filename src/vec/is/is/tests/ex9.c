
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
  PetscErrorCode         ierr;
  PetscInt               i,nLocal = 3,nGlobal;
  PetscInt              *indices;
  PetscBool              flg, auto_offset = PETSC_FALSE;
  ISLocalToGlobalMapping l2g0, l2g1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&nLocal,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-auto_offset",&auto_offset,NULL));
  CHKERRQ(PetscOptionsGetViewer(comm, NULL, NULL, "-viewer", &viewer, &format, NULL));
  CHKERRQ(PetscMalloc1(nLocal,&indices));
  for (i=0; i<nLocal; i++) {
    indices[i] = i + rank;
  }
  nGlobal = size-1+nLocal;
  if (viewer) {
    CHKERRQ(PetscViewerPushFormat(viewer, format));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "nGlobal: %" PetscInt_FMT "\n", nGlobal));
  }

  /* Create a local-to-global mapping using ISLocalToGlobalMappingCreate() */
  {
    CHKERRQ(ISLocalToGlobalMappingCreate(comm,1,nLocal,indices,PETSC_USE_POINTER,&l2g0));
    CHKERRQ(ISLocalToGlobalMappingSetFromOptions(l2g0));
    if (viewer) {
      CHKERRQ(PetscObjectSetName((PetscObject)l2g0, "l2g0"));
      CHKERRQ(ISLocalToGlobalMappingView(l2g0,viewer));
    }
  }

  /* Create the same local-to-global mapping using ISLocalToGlobalMappingCreateSF() */
  {
    PetscSF     sf;
    PetscLayout rootLayout;

    CHKERRQ(PetscSFCreate(comm, &sf));
    CHKERRQ(PetscLayoutCreateFromSizes(comm,PETSC_DECIDE,nGlobal,1,&rootLayout));
    CHKERRQ(PetscSFSetGraphLayout(sf,rootLayout,nLocal,NULL,PETSC_USE_POINTER,indices));
    CHKERRQ(PetscSFSetFromOptions(sf));
    CHKERRQ(ISLocalToGlobalMappingCreateSF(sf,auto_offset ? PETSC_DECIDE : rootLayout->rstart,&l2g1));
    if (viewer) {
      CHKERRQ(PetscObjectSetName((PetscObject)sf, "sf1"));
      CHKERRQ(PetscObjectSetName((PetscObject)l2g1, "l2g1"));
      CHKERRQ(PetscSFView(sf,viewer));
      CHKERRQ(ISLocalToGlobalMappingView(l2g1,viewer));
    }
    CHKERRQ(PetscLayoutDestroy(&rootLayout));
    CHKERRQ(PetscSFDestroy(&sf));
  }

  /* Compare the two local-to-global mappings by comparing results of apply for the same input */
  {
    IS input, output0, output1;

    CHKERRQ(ISCreateStride(comm,nLocal,0,1,&input));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(l2g0, input, &output0));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(l2g1, input, &output1));
    if (viewer) {
      CHKERRQ(PetscObjectSetName((PetscObject)input,   "input"));
      CHKERRQ(PetscObjectSetName((PetscObject)output0, "output0"));
      CHKERRQ(PetscObjectSetName((PetscObject)output1, "output1"));
      CHKERRQ(ISView(input,   viewer));
      CHKERRQ(ISView(output0, viewer));
      CHKERRQ(ISView(output1, viewer));
    }
    CHKERRQ(ISEqual(output0, output1, &flg));
    PetscCheck(flg,comm, PETSC_ERR_PLIB, "output0 != output1");
    CHKERRQ(ISDestroy(&input));
    CHKERRQ(ISDestroy(&output0));
    CHKERRQ(ISDestroy(&output1));
  }

  if (viewer) {
     CHKERRQ(PetscViewerPopFormat(viewer));
     CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g0));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g1));
  CHKERRQ(PetscFree(indices));
  ierr = PetscFinalize();
  return ierr;
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
