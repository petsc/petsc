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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is an example with two processors only!");
  ierr = MatCreate(PETSC_COMM_SELF,&ssbaij);CHKERRQ(ierr);
  ierr = MatSetType(ssbaij,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(ssbaij,2);CHKERRQ(ierr);
  ierr = MatSetSizes(ssbaij,4,8,4,8);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocationCSR(ssbaij,2,ia,ja,c);CHKERRQ(ierr);
  ierr = MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,ssbaij,PETSC_DECIDE,MAT_INITIAL_MATRIX,&msbaij);CHKERRQ(ierr);
  ierr = MatView(msbaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&msbaij);CHKERRQ(ierr);
  ierr = MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,ssbaij,4,MAT_INITIAL_MATRIX,&msbaij);CHKERRQ(ierr);
  ierr = MatView(msbaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatCreateVecs(msbaij,&x,&y);CHKERRQ(ierr);
  ierr = VecSet(x,1);CHKERRQ(ierr);
  ierr = MatMult(msbaij,x,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMultAdd(msbaij,x,x,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatGetDiagonal(msbaij,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&msbaij);CHKERRQ(ierr);
  ierr = MatDestroy(&ssbaij);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     nsize: 2
     filter: sed "s?\.??g"

TEST*/
