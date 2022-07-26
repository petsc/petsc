
static char help[] = "Tests MatCreateSubMatrices() for SBAIJ matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            BAIJ,SBAIJ,*subBAIJ,*subSBAIJ;
  PetscViewer    viewer;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscInt       n = 2,issize,M,N;
  PetscMPIInt    rank;
  IS             isrow,iscol,irow[n],icol[n];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&BAIJ));
  PetscCall(MatSetType(BAIJ,MATMPIBAIJ));
  PetscCall(MatLoad(BAIJ,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&SBAIJ));
  PetscCall(MatSetType(SBAIJ,MATMPISBAIJ));
  PetscCall(MatLoad(SBAIJ,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatGetSize(BAIJ,&M,&N));
  issize = M/4;
  PetscCall(ISCreateStride(PETSC_COMM_SELF,issize,0,1,&isrow));
  irow[0] = irow[1] = isrow;
  issize = N/2;
  PetscCall(ISCreateStride(PETSC_COMM_SELF,issize,0,1,&iscol));
  icol[0] = icol[1] = iscol;
  PetscCall(MatCreateSubMatrices(BAIJ,n,irow,icol,MAT_INITIAL_MATRIX,&subBAIJ));
  PetscCall(MatCreateSubMatrices(BAIJ,n,irow,icol,MAT_REUSE_MATRIX,&subBAIJ));

  /* irow and icol must be same for SBAIJ matrices! */
  icol[0] = icol[1] = isrow;
  PetscCall(MatCreateSubMatrices(SBAIJ,n,irow,icol,MAT_INITIAL_MATRIX,&subSBAIJ));
  PetscCall(MatCreateSubMatrices(SBAIJ,n,irow,icol,MAT_REUSE_MATRIX,&subSBAIJ));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    PetscCall(MatView(subBAIJ[0],PETSC_VIEWER_STDOUT_SELF));
    PetscCall(MatView(subSBAIJ[0],PETSC_VIEWER_STDOUT_SELF));
  }

  /* Free data structures */
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroySubMatrices(n,&subBAIJ));
  PetscCall(MatDestroySubMatrices(n,&subSBAIJ));
  PetscCall(MatDestroy(&BAIJ));
  PetscCall(MatDestroy(&SBAIJ));

  PetscCall(PetscFinalize());
  return 0;
}
