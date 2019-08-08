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
  PetscScalar    *array,none = -1.0,alpha;
  Vec            x,v1,v2,v3,v4;
  PetscReal      norm,norm_tmp,norm_tmp1,tol=100.*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscBool      Test_3D=PETSC_FALSE,flg;
  const PetscInt *ia,*ja;
  PetscInt       dof;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
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
  /* Create vectors v1 and v2 that are compatible with A */
  ierr = MatCreateVecs(A,&v1,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(v1,&v2);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  /* Test MAT_REUSE_MATRIX - reuse symbolic C */
  alpha=1.0;
  for (i=0; i<1; i++) {
    alpha -= 0.1;
    ierr   = MatScale(A,alpha);CHKERRQ(ierr);
    ierr   = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  }

  /* Free intermediate data structures created for reuse of C=Pt*A*P */
  ierr = MatFreeIntermediateDataStructures(C);CHKERRQ(ierr);

  /* Create vector x that is compatible with P */
  ierr = MatCreateVecs(P,&x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&v3);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&v4);CHKERRQ(ierr);

  norm = 0.0;
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = MatMult(P,x,v1);CHKERRQ(ierr);
    ierr = MatMult(A,v1,v2);CHKERRQ(ierr);  /* v2 = A*P*x */

    ierr = MatMultTranspose(P,v2,v3);CHKERRQ(ierr); /* v3 = Pt*A*P*x */
    ierr = MatMult(C,x,v4);CHKERRQ(ierr);           /* v3 = C*x   */
    ierr = VecAXPY(v4,none,v3);CHKERRQ(ierr);
    ierr = VecNorm(v4,NORM_1,&norm_tmp);CHKERRQ(ierr);
    ierr = VecNorm(v3,NORM_1,&norm_tmp1);CHKERRQ(ierr);

    norm_tmp /= norm_tmp1;
    if (norm_tmp > norm) norm = norm_tmp;
  }
  if (norm >= tol && !rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatPtAP(), |v3 - v4|/|v3|: %g\n",(double)norm);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = VecDestroy(&v3);CHKERRQ(ierr);
  ierr = VecDestroy(&v4);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&v1);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = DMDestroy(&finedm);CHKERRQ(ierr);
  ierr = DMDestroy(&coarsedm);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
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
      args: -M 10 -N 10 -Z 10 -matmaijptap_via allatonce
      output_file: output/ex89_1.out

   test:
      suffix: allatonce_merged
      nsize: 4
      args: -M 10 -M 5 -M 10 -matmaijptap_via allatonce_merged
      output_file: output/ex96_1.out

   test:
      suffix: allatonce_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -matmaijptap_via allatonce
      output_file: output/ex96_1.out

   test:
      suffix: allatonce_merged_3D
      nsize: 4
      args: -M 10 -M 5 -M 10 -test_3D 1 -matmaijptap_via allatonce_merged
      output_file: output/ex96_1.out

TEST*/
