static char help[]= "Tests ISView() and ISLoad() \n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt               n = 3, *izero, j, i;
  PetscInt               ix[3][3][3] = {{{3,5,4},{1,7,9},{0,2,8}},
                                        {{0,2,8},{3,5,4},{1,7,9}},
                                        {{1,7,9},{0,2,8},{3,5,4}}};
  IS                     isx[3],il;
  PetscMPIInt            size,rank;
  PetscViewer            vx,vl;
  PetscBool              equal;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size < 4,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Example only works with up to three processes");

  PetscCall(PetscCalloc1(size*n,&izero));
  for (i = 0; i < 3; i++) {
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]));
  }

  for (j = 0; j < 3; j++) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_WRITE,&vx));
    PetscCall(ISView(isx[0],vx));
    PetscCall(PetscViewerDestroy(&vx));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    PetscCall(ISCreate(PETSC_COMM_WORLD,&il));
    PetscCall(ISLoad(il,vl));
    PetscCall(ISEqual(il,isx[0],&equal));
    PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set loaded from file does not match",j);
    PetscCall(ISDestroy(&il));
    PetscCall(PetscViewerDestroy(&vl));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_APPEND,&vx));
    PetscCall(ISView(isx[1],vx));
    PetscCall(ISView(isx[2],vx));
    PetscCall(PetscViewerDestroy(&vx));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      PetscCall(ISCreate(PETSC_COMM_WORLD,&il));
      PetscCall(ISLoad(il,vl));
      PetscCall(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      PetscCall(ISDestroy(&il));
    }
    PetscCall(PetscViewerDestroy(&vl));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il));
      PetscCall(ISLoad(il,vl));
      PetscCall(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      PetscCall(ISDestroy(&il));
    }
    PetscCall(PetscViewerDestroy(&vl));
  }

  for (j = 0; j < 3; j++) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_WRITE,&vx));
    PetscCall(PetscViewerBinarySetSkipHeader(vx,PETSC_TRUE));
    for (i = 0; i < 3; i++) {
      PetscCall(ISView(isx[i],vx));
    }
    PetscCall(PetscViewerDestroy(&vx));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile_noheader",FILE_MODE_READ,&vl));
    PetscCall(PetscViewerBinarySetSkipHeader(vl,PETSC_TRUE));
    for (i = 0; i < 3; i++) {
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,izero,PETSC_COPY_VALUES,&il));
      PetscCall(ISLoad(il,vl));
      PetscCall(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      PetscCall(ISDestroy(&il));
    }
    PetscCall(PetscViewerDestroy(&vl));
  }

  for (i = 0; i < 3; i++) {
    PetscCall(ISDestroy(&isx[i]));
  }

  for (j = 0; j < 2; j++) {
    const char *filename  = (j == 0) ? "testfile_isstride" : "testfile_isblock";
    PetscInt    blocksize = (j == 0) ? 1 : size;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&vx));
    for (i = 0; i < 3; i++) {
      if (j == 0) {
        PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,rank,rank+1,&isx[i]));
      } else {
        PetscCall(ISCreateBlock(PETSC_COMM_WORLD,blocksize,n,ix[i][rank],PETSC_COPY_VALUES,&isx[i]));
      }
      PetscCall(ISView(isx[i],vx));
      PetscCall(ISToGeneral(isx[i]));
    }
    PetscCall(PetscViewerDestroy(&vx));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&vl));
    for (i = 0; i < 3; i++) {
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,blocksize*n,izero,PETSC_COPY_VALUES,&il));
      PetscCall(ISLoad(il,vl));
      PetscCall(ISEqual(il,isx[i],&equal));
      PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Iteration %" PetscInt_FMT " - Index set %" PetscInt_FMT " loaded from file does not match",j,i);
      PetscCall(ISDestroy(&il));
    }
    PetscCall(PetscViewerDestroy(&vl));
    for (i = 0; i < 3; i++) {
      PetscCall(ISDestroy(&isx[i]));
    }
  }
  PetscCall(PetscFree(izero));

  PetscCall(PetscFinalize());
  return 0;
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
