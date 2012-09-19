
static char help[] = "Tests MatReorderForNonzeroDiagonal().\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat,B;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    v;
  IS             isrow,iscol;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);


  /* ------- Assemble matrix, --------- */

  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,4,4);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);

  /* set anti-diagonal of matrix */
  v = 1.0;
  i = 0; j = 3;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 2.0;
  i = 1; j = 2;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 3.0;
  i = 2; j = 1;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 4.0;
  i = 3; j = 0;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_DENSE);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,viewer);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERINGNATURAL,&isrow,&iscol);CHKERRQ(ierr);

  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by identity\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by identity + NonzeroDiagonal()\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Column permutation\n");CHKERRQ(ierr);
  ierr = ISView(iscol,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERINGND,&isrow,&iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by ND\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ND row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ND column permutation\n");CHKERRQ(ierr);
  ierr = ISView(iscol,viewer);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by ND + NonzeroDiagonal()\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ND + NonzeroDiagonal() row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ND + NonzeroDiagonal() column permutation\n");CHKERRQ(ierr);
  ierr = ISView(iscol,viewer);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERINGRCM,&isrow,&iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by RCM\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"RCM row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"RCM column permutation\n");CHKERRQ(ierr);
  ierr = ISView(iscol,viewer);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Original matrix permuted by RCM + NonzeroDiagonal()\n");CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"RCM + NonzeroDiagonal() row permutation\n");CHKERRQ(ierr);
  ierr = ISView(isrow,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"RCM + NonzeroDiagonal() column permutation\n");CHKERRQ(ierr);
  ierr = ISView(iscol,viewer);CHKERRQ(ierr);

  ierr = MatLUFactor(mat,isrow,iscol,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Factored matrix permuted by RCM + NonzeroDiagonal()\n");CHKERRQ(ierr);
  ierr = MatView(mat,viewer);CHKERRQ(ierr);

  /* Free data structures */
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

