static const char help[] = "Test MatGetLocalSubMatrix() with multiple levels of nesting.\n\n";

#include <petscmat.h>

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  IS is0a,is0b,is0,is1,isl0a,isl0b,isl0,isl1;
  Mat A,Aexplicit;
  PetscBool usenest;
  PetscMPIInt rank,size;
  PetscInt i,j;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  {
    const PetscInt ix0a[] = {rank*2+0},ix0b[] = {rank*2+1},ix0[] = {rank*3+0,rank*3+1},ix1[] = {rank*3+2};
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,ix0a,PETSC_COPY_VALUES,&is0a);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,ix0b,PETSC_COPY_VALUES,&is0b);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,2,ix0,PETSC_COPY_VALUES,&is0);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,ix1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  }
  {
    ierr = ISCreateStride(PETSC_COMM_SELF,6,0,1,&isl0);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,0,1,&isl0a);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,3,1,&isl0b);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,6,1,&isl1);CHKERRQ(ierr);
  }

  usenest = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-nest",&usenest,PETSC_NULL);CHKERRQ(ierr);
  if (usenest) {
    ISLocalToGlobalMapping l2g;
    const PetscInt l2gind[3] = {(rank-1+size)%size,rank,(rank+1)%size};
    Mat B[9];
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,3,l2gind,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
    for (i=0; i<9; i++) {
      ierr = MatCreateAIJ(PETSC_COMM_WORLD,1,1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL,&B[i]);CHKERRQ(ierr);
      ierr = MatSetUp(B[i]);CHKERRQ(ierr);
      ierr = MatSetLocalToGlobalMapping(B[i],l2g,l2g);CHKERRQ(ierr);
    }
    {
      const IS isx[] = {is0a,is0b};
      const Mat Bx00[] = {B[0],B[1],B[3],B[4]},Bx01[] = {B[2],B[5]},Bx10[] = {B[6],B[7]};
      Mat B00,B01,B10;
      ierr = MatCreateNest(PETSC_COMM_WORLD,2,isx,2,isx,Bx00,&B00);CHKERRQ(ierr);
      ierr = MatSetUp(B00);CHKERRQ(ierr);
      ierr = MatCreateNest(PETSC_COMM_WORLD,2,isx,1,PETSC_NULL,Bx01,&B01);CHKERRQ(ierr);
      ierr = MatSetUp(B01);CHKERRQ(ierr);
      ierr = MatCreateNest(PETSC_COMM_WORLD,1,PETSC_NULL,2,isx,Bx10,&B10);CHKERRQ(ierr);
      ierr = MatSetUp(B10);CHKERRQ(ierr);
      {
        Mat By[] = {B00,B01,B10,B[8]};
        IS isy[] = {is0,is1};
        ierr = MatCreateNest(PETSC_COMM_WORLD,2,isy,2,isy,By,&A);CHKERRQ(ierr);
        ierr = MatSetUp(A);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&B00);CHKERRQ(ierr);
      ierr = MatDestroy(&B01);CHKERRQ(ierr);
      ierr = MatDestroy(&B10);CHKERRQ(ierr);
    }
    for (i=0; i<9; i++) {ierr = MatDestroy(&B[i]);CHKERRQ(ierr);}
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
  } else {
    ISLocalToGlobalMapping l2g;
    PetscInt l2gind[9];
    for (i=0; i<3; i++) for (j=0; j<3; j++) l2gind[3*i+j] = ((rank-1+j+size) % size)*3 + i;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,9,l2gind,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL,&A);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(A,l2g,l2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
  }

  {
    Mat A00,A11,A0a0a,A0a0b;
    ierr = MatGetLocalSubMatrix(A,isl0,isl0,&A00);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(A,isl1,isl1,&A11);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(A00,isl0a,isl0a,&A0a0a);CHKERRQ(ierr);
    ierr = MatGetLocalSubMatrix(A00,isl0a,isl0b,&A0a0b);CHKERRQ(ierr);

    ierr = MatSetValueLocal(A0a0a,0,0,100*rank+1,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValueLocal(A0a0a,0,1,100*rank+2,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValueLocal(A0a0a,2,2,100*rank+9,ADD_VALUES);CHKERRQ(ierr);

    ierr = MatSetValueLocal(A0a0b,1,1,100*rank+50+5,ADD_VALUES);CHKERRQ(ierr);

    ierr = MatSetValueLocal(A11,0,0,1000*(rank+1)+1,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValueLocal(A11,1,2,1000*(rank+1)+6,ADD_VALUES);CHKERRQ(ierr);

    ierr = MatRestoreLocalSubMatrix(A00,isl0a,isl0a,&A0a0a);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(A00,isl0a,isl0b,&A0a0b);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(A,isl0,isl0,&A00);CHKERRQ(ierr);
    ierr = MatRestoreLocalSubMatrix(A,isl1,isl1,&A11);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatComputeExplicitOperator(A,&Aexplicit);CHKERRQ(ierr);
  ierr = MatView(Aexplicit,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aexplicit);CHKERRQ(ierr);
  ierr = ISDestroy(&is0a);CHKERRQ(ierr);
  ierr = ISDestroy(&is0b);CHKERRQ(ierr);
  ierr = ISDestroy(&is0);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&isl0a);CHKERRQ(ierr);
  ierr = ISDestroy(&isl0b);CHKERRQ(ierr);
  ierr = ISDestroy(&isl0);CHKERRQ(ierr);
  ierr = ISDestroy(&isl1);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
