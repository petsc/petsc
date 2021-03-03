static const char help[] = "Test MatGetLocalSubMatrix() with multiple levels of nesting.\n\n";

#include <petscmat.h>

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  IS             is0a,is0b,is0,is1,isl0a,isl0b,isl0,isl1;
  Mat            A,Aexplicit;
  PetscBool      usenest;
  PetscMPIInt    rank,size;
  PetscInt       i,j;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  {
    PetscInt ix0a[1],ix0b[1],ix0[2],ix1[1];

    ix0a[0] = rank*2+0;
    ix0b[0] = rank*2+1;
    ix0[0]  = rank*3+0; ix0[1] = rank*3+1;
    ix1[0]  = rank*3+2;
    ierr    = ISCreateGeneral(PETSC_COMM_WORLD,1,ix0a,PETSC_COPY_VALUES,&is0a);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(PETSC_COMM_WORLD,1,ix0b,PETSC_COPY_VALUES,&is0b);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(PETSC_COMM_WORLD,2,ix0,PETSC_COPY_VALUES,&is0);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(PETSC_COMM_WORLD,1,ix1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  }
  {
    ierr = ISCreateStride(PETSC_COMM_SELF,6,0,1,&isl0);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,0,1,&isl0a);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,3,1,&isl0b);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,3,6,1,&isl1);CHKERRQ(ierr);
  }

  usenest = PETSC_FALSE;
  ierr    = PetscOptionsGetBool(NULL,NULL,"-nest",&usenest,NULL);CHKERRQ(ierr);
  if (usenest) {
    ISLocalToGlobalMapping l2g;
    PetscInt               l2gind[3];
    Mat                    B[9];

    l2gind[0] = (rank-1+size)%size; l2gind[1] = rank; l2gind[2] = (rank+1)%size;
    ierr      = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,3,l2gind,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
    for (i=0; i<9; i++) {
      ierr = MatCreateAIJ(PETSC_COMM_WORLD,1,1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,NULL,PETSC_DECIDE,NULL,&B[i]);CHKERRQ(ierr);
      ierr = MatSetUp(B[i]);CHKERRQ(ierr);
      ierr = MatSetLocalToGlobalMapping(B[i],l2g,l2g);CHKERRQ(ierr);
    }
    {
      IS  isx[2];
      Mat Bx00[4],Bx01[2],Bx10[2];
      Mat B00,B01,B10;

      isx[0]  = is0a; isx[1] = is0b;
      Bx00[0] = B[0]; Bx00[1] = B[1]; Bx00[2] = B[3]; Bx00[3] = B[4];
      Bx01[0] = B[2]; Bx01[1] = B[5];
      Bx10[0] = B[6]; Bx10[1] = B[7];

      ierr = MatCreateNest(PETSC_COMM_WORLD,2,isx,2,isx,Bx00,&B00);CHKERRQ(ierr);
      ierr = MatSetUp(B00);CHKERRQ(ierr);
      ierr = MatCreateNest(PETSC_COMM_WORLD,2,isx,1,NULL,Bx01,&B01);CHKERRQ(ierr);
      ierr = MatSetUp(B01);CHKERRQ(ierr);
      ierr = MatCreateNest(PETSC_COMM_WORLD,1,NULL,2,isx,Bx10,&B10);CHKERRQ(ierr);
      ierr = MatSetUp(B10);CHKERRQ(ierr);
      {
        Mat By[4];
        IS  isy[2];

        By[0]  = B00; By[1] = B01; By[2] = B10; By[3] = B[8];
        isy[0] = is0; isy[1] = is1;

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
    PetscInt               l2gind[9];
    for (i=0; i<3; i++) for (j=0; j<3; j++) l2gind[3*i+j] = ((rank-1+j+size) % size)*3 + i;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,9,l2gind,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,NULL,PETSC_DECIDE,NULL,&A);CHKERRQ(ierr);
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

  ierr = MatComputeOperator(A,MATAIJ,&Aexplicit);CHKERRQ(ierr);
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
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 3

   test:
      suffix: nest
      nsize: 3
      args: -nest

TEST*/
