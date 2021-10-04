
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&nLocal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-auto_offset",&auto_offset,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, NULL, NULL, "-viewer", &viewer, &format, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nLocal,&indices);CHKERRQ(ierr);
  for (i=0; i<nLocal; i++) {
    indices[i] = i + rank;
  }
  nGlobal = size-1+nLocal;
  if (viewer) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nGlobal: %D\n", nGlobal);CHKERRQ(ierr);
  }

  /* Create a local-to-global mapping using ISLocalToGlobalMappingCreate() */
  {
    ierr = ISLocalToGlobalMappingCreate(comm,1,nLocal,indices,PETSC_USE_POINTER,&l2g0);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingSetFromOptions(l2g0);CHKERRQ(ierr);
    if (viewer) {
      ierr = PetscObjectSetName((PetscObject)l2g0, "l2g0");CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingView(l2g0,viewer);CHKERRQ(ierr);
    }
  }

  /* Create the same local-to-global mapping using ISLocalToGlobalMappingCreateSF() */
  {
    PetscSF     sf;
    PetscLayout rootLayout;

    ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
    ierr = PetscLayoutCreateFromSizes(comm,PETSC_DECIDE,nGlobal,1,&rootLayout);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,rootLayout,nLocal,NULL,PETSC_USE_POINTER,indices);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateSF(sf,auto_offset ? PETSC_DECIDE : rootLayout->rstart,&l2g1);CHKERRQ(ierr);
    if (viewer) {
      ierr = PetscObjectSetName((PetscObject)sf, "sf1");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)l2g1, "l2g1");CHKERRQ(ierr);
      ierr = PetscSFView(sf,viewer);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingView(l2g1,viewer);CHKERRQ(ierr);
    }
    ierr = PetscLayoutDestroy(&rootLayout);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  }

  /* Compare the two local-to-global mappings by comparing results of apply for the same input */
  {
    IS input, output0, output1;

    ierr = ISCreateStride(comm,nLocal,0,1,&input);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(l2g0, input, &output0);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(l2g1, input, &output1);CHKERRQ(ierr);
    if (viewer) {
      ierr = PetscObjectSetName((PetscObject)input,   "input");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)output0, "output0");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)output1, "output1");CHKERRQ(ierr);
      ierr = ISView(input,   viewer);CHKERRQ(ierr);
      ierr = ISView(output0, viewer);CHKERRQ(ierr);
      ierr = ISView(output1, viewer);CHKERRQ(ierr);
    }
    ierr = ISEqual(output0, output1, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(comm, PETSC_ERR_PLIB, "output0 != output1");
    ierr = ISDestroy(&input);CHKERRQ(ierr);
    ierr = ISDestroy(&output0);CHKERRQ(ierr);
    ierr = ISDestroy(&output1);CHKERRQ(ierr);
  }

  if (viewer) {
     ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
     ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&l2g0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2g1);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
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
