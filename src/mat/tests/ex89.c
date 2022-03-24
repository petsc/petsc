static char help[] ="Tests MatPtAP() for MPIMAIJ and MPIAIJ \n ";

#include <petscdmda.h>

int main(int argc,char **argv)
{
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

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  M = 10; N = 10; Z = 10;
  dof  = 10;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_3D",&Test_3D,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL));
  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    CHKERRQ(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&coarsedm));
  } else {
    CHKERRQ(DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&coarsedm));
  }
  CHKERRQ(DMSetFromOptions(coarsedm));
  CHKERRQ(DMSetUp(coarsedm));

  /* This makes sure the coarse DMDA has the same partition as the fine DMDA */
  CHKERRQ(DMRefine(coarsedm,PetscObjectComm((PetscObject)coarsedm),&finedm));

  /*------------------------------------------------------------*/
  CHKERRQ(DMSetMatType(finedm,MATAIJ));
  CHKERRQ(DMCreateMatrix(finedm,&A));

  /* set val=one to A */
  if (size == 1) {
    CHKERRQ(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(A,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(A,&array));
    }
    CHKERRQ(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  } else {
    Mat AA,AB;
    CHKERRQ(MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL));
    CHKERRQ(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(AA,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(AA,&array));
    }
    CHKERRQ(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    CHKERRQ(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      CHKERRQ(MatSeqAIJGetArray(AB,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      CHKERRQ(MatSeqAIJRestoreArray(AB,&array));
    }
    CHKERRQ(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  }
  /* Create interpolation between the fine and coarse grids */
  CHKERRQ(DMCreateInterpolation(coarsedm,finedm,&P,NULL));

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  /* (1) Developer API */
  CHKERRQ(MatProductCreate(A,P,NULL,&C));
  CHKERRQ(MatProductSetType(C,MATPRODUCT_PtAP));
  CHKERRQ(MatProductSetAlgorithm(C,"allatonce"));
  CHKERRQ(MatProductSetFill(C,PETSC_DEFAULT));
  CHKERRQ(MatProductSetFromOptions(C));
  CHKERRQ(MatProductSymbolic(C));
  CHKERRQ(MatProductNumeric(C));
  CHKERRQ(MatProductNumeric(C)); /* Test reuse of symbolic C */

  { /* Test MatProductView() */
    PetscViewer viewer;
    CHKERRQ(PetscViewerASCIIOpen(comm,NULL, &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatProductView(C,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  CHKERRQ(MatPtAPMultEqual(A,P,C,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatProduct_PtAP");
  CHKERRQ(MatDestroy(&C));

  /* (2) User API */
  CHKERRQ(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C));
  /* Test MAT_REUSE_MATRIX - reuse symbolic C */
  alpha=1.0;
  for (i=0; i<1; i++) {
    alpha -= 0.1;
    CHKERRQ(MatScale(A,alpha));
    CHKERRQ(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C));
  }

  /* Free intermediate data structures created for reuse of C=Pt*A*P */
  CHKERRQ(MatProductClear(C));

  CHKERRQ(MatPtAPMultEqual(A,P,C,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP");

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(DMDestroy(&finedm));
  CHKERRQ(DMDestroy(&coarsedm));
  CHKERRQ(PetscFinalize());
  return 0;
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
      args: -M 10 -M 5 -M 10 -mat_product_algorithm allatonce_merged
      output_file: output/ex89_3.out

   test:
      suffix: nonscalable_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -mat_product_algorithm nonscalable
      output_file: output/ex89_4.out

   test:
      suffix: allatonce_merged_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -mat_product_algorithm allatonce_merged
      output_file: output/ex89_3.out

   test:
      suffix: nonscalable
      nsize: 4
      args: -M 10 -N 10 -Z 10 -mat_product_algorithm nonscalable
      output_file: output/ex89_5.out

TEST*/
