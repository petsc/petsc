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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  M = 10; N = 10; Z = 10;
  dof  = 10;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_3D",&Test_3D,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL));
  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    PetscCall(DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&coarsedm));
  } else {
    PetscCall(DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,NULL,&coarsedm));
  }
  PetscCall(DMSetFromOptions(coarsedm));
  PetscCall(DMSetUp(coarsedm));

  /* This makes sure the coarse DMDA has the same partition as the fine DMDA */
  PetscCall(DMRefine(coarsedm,PetscObjectComm((PetscObject)coarsedm),&finedm));

  /*------------------------------------------------------------*/
  PetscCall(DMSetMatType(finedm,MATAIJ));
  PetscCall(DMCreateMatrix(finedm,&A));

  /* set val=one to A */
  if (size == 1) {
    PetscCall(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(A,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(A,&array));
    }
    PetscCall(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  } else {
    Mat AA,AB;
    PetscCall(MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL));
    PetscCall(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AA,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AA,&array));
    }
    PetscCall(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    PetscCall(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AB,&array));
      for (i=0; i<ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AB,&array));
    }
    PetscCall(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nrows,&ia,&ja,&flg));
  }
  /* Create interpolation between the fine and coarse grids */
  PetscCall(DMCreateInterpolation(coarsedm,finedm,&P,NULL));

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  /* (1) Developer API */
  PetscCall(MatProductCreate(A,P,NULL,&C));
  PetscCall(MatProductSetType(C,MATPRODUCT_PtAP));
  PetscCall(MatProductSetAlgorithm(C,"allatonce"));
  PetscCall(MatProductSetFill(C,PETSC_DEFAULT));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscCall(MatProductNumeric(C));
  PetscCall(MatProductNumeric(C)); /* Test reuse of symbolic C */

  { /* Test MatProductView() */
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(comm,NULL, &viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatProductView(C,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(MatPtAPMultEqual(A,P,C,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatProduct_PtAP");
  PetscCall(MatDestroy(&C));

  /* (2) User API */
  PetscCall(MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C));
  /* Test MAT_REUSE_MATRIX - reuse symbolic C */
  alpha=1.0;
  for (i=0; i<1; i++) {
    alpha -= 0.1;
    PetscCall(MatScale(A,alpha));
    PetscCall(MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C));
  }

  /* Free intermediate data structures created for reuse of C=Pt*A*P */
  PetscCall(MatProductClear(C));

  PetscCall(MatPtAPMultEqual(A,P,C,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatPtAP");

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&P));
  PetscCall(DMDestroy(&finedm));
  PetscCall(DMDestroy(&coarsedm));
  PetscCall(PetscFinalize());
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
