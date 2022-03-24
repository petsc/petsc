
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&BAIJ));
  CHKERRQ(MatSetType(BAIJ,MATMPIBAIJ));
  CHKERRQ(MatLoad(BAIJ,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&SBAIJ));
  CHKERRQ(MatSetType(SBAIJ,MATMPISBAIJ));
  CHKERRQ(MatLoad(SBAIJ,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatGetSize(BAIJ,&M,&N));
  issize = M/4;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,issize,0,1,&isrow));
  irow[0] = irow[1] = isrow;
  issize = N/2;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,issize,0,1,&iscol));
  icol[0] = icol[1] = iscol;
  CHKERRQ(MatCreateSubMatrices(BAIJ,n,irow,icol,MAT_INITIAL_MATRIX,&subBAIJ));
  CHKERRQ(MatCreateSubMatrices(BAIJ,n,irow,icol,MAT_REUSE_MATRIX,&subBAIJ));

  /* irow and icol must be same for SBAIJ matrices! */
  icol[0] = icol[1] = isrow;
  CHKERRQ(MatCreateSubMatrices(SBAIJ,n,irow,icol,MAT_INITIAL_MATRIX,&subSBAIJ));
  CHKERRQ(MatCreateSubMatrices(SBAIJ,n,irow,icol,MAT_REUSE_MATRIX,&subSBAIJ));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    CHKERRQ(MatView(subBAIJ[0],PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(MatView(subSBAIJ[0],PETSC_VIEWER_STDOUT_SELF));
  }

  /* Free data structures */
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroySubMatrices(n,&subBAIJ));
  CHKERRQ(MatDestroySubMatrices(n,&subSBAIJ));
  CHKERRQ(MatDestroy(&BAIJ));
  CHKERRQ(MatDestroy(&SBAIJ));

  CHKERRQ(PetscFinalize());
  return 0;
}
