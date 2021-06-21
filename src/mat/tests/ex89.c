static char help[] ="Tests MatPtAP() for MPIMAIJ and MPIAIJ \n ";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             coarsedm,finedm;
  PetscMPIInt    size,rank;
  PetscInt       M,N,Z,i,nrows;
  PetscScalar    one = 1.0;
  PetscReal      fill=2.0;
  Mat            A,P,C;
  PetscScalar    *array,alpha;
  PetscBool      Test_3D=PETSC_FALSE,flg;
  const PetscInt *ia,*ja;
  PetscInt       dof;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  M = 10; N = 10; Z = 10;
  dof  = 10;

  ierr = PetscOptionsGetBool(NULL,NULL,"-test_3D",&Test_3D,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL);CHKERRQ(ierr);
  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&coarsedm);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&coarsedm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(coarsedm);CHKERRQ(ierr);
  ierr = DMSetUp(coarsedm);CHKERRQ(ierr);

  /* This makes sure the coarse DMDA has the same partition as the fine DMDA */
  ierr = DMRefine(coarsedm,PetscObjectComm((PetscObject)coarsedm),&finedm);CHKERRQ(ierr);

  /*------------------------------------------------------------*/
  ierr = DMSetMatType(finedm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(finedm,&A);CHKERRQ(ierr);

  /* set val=one to A */
  if (size == 1) {
    ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(A,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(A,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
  } else {
    Mat AA,AB;
    ierr = MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL);CHKERRQ(ierr);
    ierr = MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(AA,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(AA,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    ierr = MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSeqAIJGetArray(AB,&array);CHKERRQ(ierr);
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      ierr = MatSeqAIJRestoreArray(AB,&array);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg);CHKERRQ(ierr);
  }
  /* Create interpolation between the fine and coarse grids */
  ierr = DMCreateInterpolation(coarsedm,finedm,&P,NULL);CHKERRQ(ierr);

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  /* (1) Developer API */
  ierr = MatProductCreate(A,P,NULL,&C);CHKERRQ(ierr);
  ierr = MatProductSetType(C,MATPRODUCT_PtAP);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,"allatonce");CHKERRQ(ierr);
  ierr = MatProductSetFill(C,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatProductSymbolic(C);CHKERRQ(ierr);
  ierr = MatProductNumeric(C);CHKERRQ(ierr);
  ierr = MatProductNumeric(C);CHKERRQ(ierr); /* Test reuse of symbolic C */

  { /* Test MatProductView() */
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(comm,NULL, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatProductView(C,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = MatPtAPMultEqual(A,P,C,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatProduct_PtAP");
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  /* (2) User API */
  ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  /* Test MAT_REUSE_MATRIX - reuse symbolic C */
  alpha=1.0;
  for (i=0; i<1; i++) {
    alpha -= 0.1;
    ierr   = MatScale(A,alpha);CHKERRQ(ierr);
    ierr   = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  }

  /* Free intermediate data structures created for reuse of C=Pt*A*P */
  ierr = MatProductClear(C);CHKERRQ(ierr);

  ierr = MatPtAPMultEqual(A,P,C,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP");

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = DMDestroy(&finedm);CHKERRQ(ierr);
  ierr = DMDestroy(&coarsedm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -M 10 -N 10 -Z 10
      output_file: output/ex89_1.out

   test:
      suffix: allatonce
      nsize: 4
      args: -M 10 -N 10 -Z 10
      output_file: output/ex89_2.out

   test:
      suffix: allatonce_merged
      nsize: 4
      args: -M 10 -M 5 -M 10 -matproduct_ptap_via allatonce_merged
      output_file: output/ex89_3.out

   test:
      suffix: nonscalable_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -matproduct_ptap_via nonscalable
      output_file: output/ex89_4.out

   test:
      suffix: allatonce_merged_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -matproduct_ptap_via allatonce_merged
      output_file: output/ex89_3.out

   test:
      suffix: nonscalable
      nsize: 4
      args: -M 10 -N 10 -Z 10 -matproduct_ptap_via nonscalable
      output_file: output/ex89_5.out

TEST*/
