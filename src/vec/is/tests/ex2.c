static char help[]= "Tests ISView() and ISLoad() \n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               n = 3, *izero, j, i;
  PetscInt               ix[3][3][3] = {{{3,5,4},{1,7,9},{0,2,8}},
                                        {{0,2,8},{3,5,4},{1,7,9}},
                                        {{1,7,9},{0,2,8},{3,5,4}}};
  IS                     isx[3],il;
  PetscMPIInt            size,rank;
  PetscViewer            vx,vl;
  PetscBool              equal;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size > 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Example only works with up to three processes");

  ierr = PetscCalloc1(size*n,&izero);CHKERRQ(ierr);
  for (i = 0; i < 3; i++) {
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]);CHKERRQ(ierr);
  }

  for (j = 0; j < 3; j++) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_WRITE,&vx);CHKERRQ(ierr);
    ierr = ISView(isx[0],vx);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vx);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl);CHKERRQ(ierr);
    ierr = ISCreate(PETSC_COMM_WORLD,&il);CHKERRQ(ierr);
    ierr = ISLoad(il,vl);CHKERRQ(ierr);
    ierr = ISEqual(il,isx[0],&equal);CHKERRQ(ierr);
    PetscAssertFalse(!equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set loaded from file does not match",j);
    ierr = ISDestroy(&il);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_APPEND,&vx);CHKERRQ(ierr);
    ierr = ISView(isx[1],vx);CHKERRQ(ierr);
    ierr = ISView(isx[2],vx);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vx);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISCreate(PETSC_COMM_WORLD,&il);CHKERRQ(ierr);
      ierr = ISLoad(il,vl);CHKERRQ(ierr);
      ierr = ISEqual(il,isx[i],&equal);CHKERRQ(ierr);
      PetscAssertFalse(!equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      ierr = ISDestroy(&il);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il);CHKERRQ(ierr);
      ierr = ISLoad(il,vl);CHKERRQ(ierr);
      ierr = ISEqual(il,isx[i],&equal);CHKERRQ(ierr);
      PetscAssertFalse(!equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      ierr = ISDestroy(&il);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);
  }

  for (j = 0; j < 3; j++) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_WRITE,&vx);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetSkipHeader(vx,PETSC_TRUE);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISView(isx[i],vx);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vx);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_READ,&vl);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetSkipHeader(vl,PETSC_TRUE);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il);CHKERRQ(ierr);
      ierr = ISLoad(il,vl);CHKERRQ(ierr);
      ierr = ISEqual(il,isx[i],&equal);CHKERRQ(ierr);
      PetscAssertFalse(!equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      ierr = ISDestroy(&il);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);
  }

  for (i = 0; i < 3; i++) {
    ierr = ISDestroy(&isx[i]);CHKERRQ(ierr);
  }

  for (j = 0; j < 2; j++) {
    const char *filename  = (j == 0) ? "testfile_isstride" : "testfile_isblock";
    PetscInt    blocksize = (j == 0) ? 1 : size;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&vx);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      if (j == 0) {
        ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,rank+1,&isx[i]);CHKERRQ(ierr);
      } else {
        ierr = ISCreateBlock(PETSC_COMM_WORLD,blocksize,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]);CHKERRQ(ierr);
      }
      ierr = ISView(isx[i],vx);CHKERRQ(ierr);
      ierr = ISToGeneral(isx[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vx);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&vl);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISCreateGeneral(PETSC_COMM_WORLD,blocksize*n,izero,PETSC_COPY_VALUES,&il);CHKERRQ(ierr);
      ierr = ISLoad(il,vl);CHKERRQ(ierr);
      ierr = ISEqual(il,isx[i],&equal);CHKERRQ(ierr);
      PetscAssertFalse(!equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      ierr = ISDestroy(&il);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);
    for (i = 0; i < 3; i++) {
      ierr = ISDestroy(&isx[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(izero);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      output_file: output/ex2_1.out
      test:
        suffix: stdio_1
        nsize: 1
      test:
        suffix: stdio_2
        nsize: 2
      test:
        suffix: stdio_3
        nsize: 3

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      output_file: output/ex2_1.out
      test:
        suffix: mpiio_1
        nsize: 1
      test:
        suffix: mpiio_2
        nsize: 2
      test:
        suffix: mpiio_3
        nsize: 3

TEST*/
