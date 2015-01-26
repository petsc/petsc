
static char help[] = "Tests MatConvert(), MatElementalHermitianGenDefiniteEig() for MATELEMENTAL interface.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex173 -fA $Deig/graphene_xxs_A_aij -fB $Deig/graphene_xxs_B_aij -vl -0.8 -vu -0.75
*/

#include <petscmat.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,Aelem,B,Belem,X,Xe,C1,C2,EVAL;
  Vec            eval;
  PetscErrorCode ierr;
  PetscViewer    view;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg,flgB;
  PetscScalar    one = 1.0,*Earray;
  PetscMPIInt    rank;
  PetscReal      vl,vu,norm;
  Mat            We;
  PetscInt       M,N,m;

  PetscInitialize(&argc,&args,(char*)0,help);
#if !defined(PETSC_HAVE_ELEMENTAL)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires ELEMENTAL");
#endif
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Load PETSc matrices */
  ierr = PetscOptionsGetString(NULL,"-fA",file[0],PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"orig_");CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  PetscOptionsGetString(NULL,"-fB",file[1],PETSC_MAX_PATH_LEN,&flgB);
  if (flgB) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&view);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(B,"orig_");CHKERRQ(ierr);
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatLoad(B,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  } else {
    /* Create matrix B = I */
    PetscInt rstart,rend,i;
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(B,"orig_");CHKERRQ(ierr);
    ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(B,1,&i,1,&i,&one,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Convert AIJ/DENSE matrices into Elemental matrices */
  if (!rank) printf(" Convert AIJ/DENSE matrix A into Elemental matrix... \n");
  ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Aelem);CHKERRQ(ierr);
  if (!rank) printf(" Convert AIJ/DENSE matrix B into Elemental matrix... \n");
  ierr = MatConvert(B, MATELEMENTAL, MAT_INITIAL_MATRIX, &Belem);CHKERRQ(ierr);

  /* Test accuracy */
  ierr = MatMultEqual(A,Aelem,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");
  ierr = MatMultEqual(B,Belem,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"B != B_elemental.");

  /* Test MatElementalHermitianGenDefiniteEig() */
  if (!rank) printf(" Compute Ax = lambda Bx... \n");
  vl = -0.8, vu = -0.7;
  ierr = PetscOptionsGetReal(NULL,"-vl",&vl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-vu",&vu,NULL);CHKERRQ(ierr);
  PetscInt eigtype = 1; /* elem::AXBX */
  PetscInt uplo = 1;    /* = elem::UPPER */
  ierr = MatElementalHermitianGenDefiniteEig(eigtype,uplo,Aelem,Belem,&We,&Xe,vl,vu);CHKERRQ(ierr);
  //ierr = MatView(We,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = MatView(Xe,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (!rank) printf(" Convert Elemental matrices We and Xe into MATDENSE matrices... \n");
  ierr = MatConvert(We,MATDENSE,MAT_INITIAL_MATRIX,&EVAL);CHKERRQ(ierr); /* EVAL is a Mx1 matrix */
  ierr = MatConvert(Xe,MATDENSE,MAT_INITIAL_MATRIX,&X);CHKERRQ(ierr);
  //ierr = MatView(EVAL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Check || A*X - B*X*We || */
  ierr = MatMatMult(A,X,MAT_INITIAL_MATRIX,1.0,&C1);CHKERRQ(ierr); /* C1 = A*X */
  ierr = MatMatMult(B,X,MAT_INITIAL_MATRIX,1.0,&C2);CHKERRQ(ierr); /* C2 = B*X */
 
  /* Get vector eval from matrix EVAL for MatDiagonalScale() */
  ierr = MatGetLocalSize(EVAL,&m,NULL);CHKERRQ(ierr); 
  ierr = MatDenseGetArray(EVAL,&Earray);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)EVAL),1,m,PETSC_DECIDE,Earray,&eval);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(EVAL,&Earray);CHKERRQ(ierr);
  //ierr = VecView(eval,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  ierr = MatDiagonalScale(C2,NULL,eval);CHKERRQ(ierr); /* C2 = B*X*eval */
  ierr = MatAXPY(C1,-1.0,C2,SAME_NONZERO_PATTERN);CHKERRQ(ierr); /* C1 = - C2 + C1 */
  ierr = MatNorm(C1,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (!rank) printf(" || A*X - B*X*We || = %g\n",norm);
  ierr = MatDestroy(&C1);CHKERRQ(ierr);
  ierr = MatDestroy(&C2);CHKERRQ(ierr);

  ierr = VecDestroy(&eval);CHKERRQ(ierr);
  ierr = MatDestroy(&We);CHKERRQ(ierr);
  ierr = MatDestroy(&EVAL);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&Xe);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&Aelem);CHKERRQ(ierr);
  ierr = MatDestroy(&Belem);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
