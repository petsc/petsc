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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Example only works with up to three processes");

  CHKERRQ(PetscCalloc1(size*n,&izero));
  for (i = 0; i < 3; i++) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]));
  }

  for (j = 0; j < 3; j++) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_WRITE,&vx));
    CHKERRQ(ISView(isx[0],vx));
    CHKERRQ(PetscViewerDestroy(&vx));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    CHKERRQ(ISCreate(PETSC_COMM_WORLD,&il));
    CHKERRQ(ISLoad(il,vl));
    CHKERRQ(ISEqual(il,isx[0],&equal));
    PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set loaded from file does not match",j);
    CHKERRQ(ISDestroy(&il));
    CHKERRQ(PetscViewerDestroy(&vl));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_APPEND,&vx));
    CHKERRQ(ISView(isx[1],vx));
    CHKERRQ(ISView(isx[2],vx));
    CHKERRQ(PetscViewerDestroy(&vx));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISCreate(PETSC_COMM_WORLD,&il));
      CHKERRQ(ISLoad(il,vl));
      CHKERRQ(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      CHKERRQ(ISDestroy(&il));
    }
    CHKERRQ(PetscViewerDestroy(&vl));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il));
      CHKERRQ(ISLoad(il,vl));
      CHKERRQ(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      CHKERRQ(ISDestroy(&il));
    }
    CHKERRQ(PetscViewerDestroy(&vl));
  }

  for (j = 0; j < 3; j++) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_WRITE,&vx));
    CHKERRQ(PetscViewerBinarySetSkipHeader(vx,PETSC_TRUE));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISView(isx[i],vx));
    }
    CHKERRQ(PetscViewerDestroy(&vx));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_READ,&vl));
    CHKERRQ(PetscViewerBinarySetSkipHeader(vl,PETSC_TRUE));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il));
      CHKERRQ(ISLoad(il,vl));
      CHKERRQ(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      CHKERRQ(ISDestroy(&il));
    }
    CHKERRQ(PetscViewerDestroy(&vl));
  }

  for (i = 0; i < 3; i++) {
    CHKERRQ(ISDestroy(&isx[i]));
  }

  for (j = 0; j < 2; j++) {
    const char *filename  = (j == 0) ? "testfile_isstride" : "testfile_isblock";
    PetscInt    blocksize = (j == 0) ? 1 : size;
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&vx));
    for (i = 0; i < 3; i++) {
      if (j == 0) {
        CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,rank,rank+1,&isx[i]));
      } else {
        CHKERRQ(ISCreateBlock(PETSC_COMM_WORLD,blocksize,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]));
      }
      CHKERRQ(ISView(isx[i],vx));
      CHKERRQ(ISToGeneral(isx[i]));
    }
    CHKERRQ(PetscViewerDestroy(&vx));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,blocksize*n,izero,PETSC_COPY_VALUES,&il));
      CHKERRQ(ISLoad(il,vl));
      CHKERRQ(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      CHKERRQ(ISDestroy(&il));
    }
    CHKERRQ(PetscViewerDestroy(&vl));
    for (i = 0; i < 3; i++) {
      CHKERRQ(ISDestroy(&isx[i]));
    }
  }
  CHKERRQ(PetscFree(izero));

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
