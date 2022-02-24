static char help[] = "Basic test of various routines with SBAIJ matrices\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       ia[3]={0,2,4};
  PetscInt       ja[4]={0,1,0,1};
  PetscScalar    c[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  PetscMPIInt    size;
  Mat            ssbaij,msbaij;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is an example with two processors only!");
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&ssbaij));
  CHKERRQ(MatSetType(ssbaij,MATSEQSBAIJ));
  CHKERRQ(MatSetBlockSize(ssbaij,2));
  CHKERRQ(MatSetSizes(ssbaij,4,8,4,8));
  CHKERRQ(MatSeqSBAIJSetPreallocationCSR(ssbaij,2,ia,ja,c));
  CHKERRQ(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,ssbaij,PETSC_DECIDE,MAT_INITIAL_MATRIX,&msbaij));
  CHKERRQ(MatView(msbaij,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&msbaij));
  CHKERRQ(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,ssbaij,4,MAT_INITIAL_MATRIX,&msbaij));
  CHKERRQ(MatView(msbaij,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatCreateVecs(msbaij,&x,&y));
  CHKERRQ(VecSet(x,1));
  CHKERRQ(MatMult(msbaij,x,y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatMultAdd(msbaij,x,x,y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatGetDiagonal(msbaij,y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&msbaij));
  CHKERRQ(MatDestroy(&ssbaij));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     nsize: 2
     filter: sed "s?\.??g"

TEST*/
