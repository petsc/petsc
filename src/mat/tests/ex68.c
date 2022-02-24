
static char help[] = "Tests MatReorderForNonzeroDiagonal().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat,B,C;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscMPIInt    size;
  PetscScalar    v;
  IS             isrow,iscol,identity;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* ------- Assemble matrix, --------- */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,4,4));
  CHKERRQ(MatSetFromOptions(mat));
  CHKERRQ(MatSetUp(mat));

  /* set anti-diagonal of matrix */
  v    = 1.0;
  i    = 0; j = 3;
  CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
  v    = 2.0;
  i    = 1; j = 2;
  CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
  v    = 3.0;
  i    = 2; j = 1;
  CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
  v    = 4.0;
  i    = 3; j = 0;
  CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_DENSE));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix\n"));
  CHKERRQ(MatView(mat,viewer));

  CHKERRQ(MatGetOrdering(mat,MATORDERINGNATURAL,&isrow,&iscol));

  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by identity\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol));
  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by identity + NonzeroDiagonal()\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));

  CHKERRQ(MatGetOrdering(mat,MATORDERINGND,&isrow,&iscol));
  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by ND\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"ND row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"ND column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));

  CHKERRQ(MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol));
  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by ND + NonzeroDiagonal()\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"ND + NonzeroDiagonal() row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"ND + NonzeroDiagonal() column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));

  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));

  CHKERRQ(MatGetOrdering(mat,MATORDERINGRCM,&isrow,&iscol));
  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by RCM\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"RCM row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"RCM column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));

  CHKERRQ(MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol));
  CHKERRQ(MatPermute(mat,isrow,iscol,&B));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Original matrix permuted by RCM + NonzeroDiagonal()\n"));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"RCM + NonzeroDiagonal() row permutation\n"));
  CHKERRQ(ISView(isrow,viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"RCM + NonzeroDiagonal() column permutation\n"));
  CHKERRQ(ISView(iscol,viewer));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size == 1) {
    CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,4,0,1,&identity));
    CHKERRQ(MatPermute(B,identity,identity,&C));
    CHKERRQ(MatConvert(C,MATSEQSBAIJ,MAT_INPLACE_MATRIX,&C));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(ISDestroy(&identity));
  }
  CHKERRQ(MatDestroy(&B));
  /* Test MatLUFactor(); set diagonal as zeros as requested by PETSc matrix factorization */
  for (i=0; i<4; i++) {
    v = 0.0;
    CHKERRQ(MatSetValues(mat,1,&i,1,&i,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatLUFactor(mat,isrow,iscol,NULL));

  /* Free data structures */
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&mat));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
