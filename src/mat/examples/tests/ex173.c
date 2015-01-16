
static char help[] = "Tests MatConvert() from MATAIJ and MATDENSE to MATELEMENTAL.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,Aelem,Belem;
  PetscErrorCode ierr;
  PetscViewer    view;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  //HermitianGenDefiniteEigType eigType;

  PetscInitialize(&argc,&args,(char*)0,help);

  /* Now reload PETSc matrix and view it */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  /* Convert A into a Elemental matrix */
  ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Aelem);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
  ierr = MatConvert(A, MATELEMENTAL, MAT_REUSE_MATRIX, &A);CHKERRQ(ierr);

  /* Test accuracy */
  ierr = MatMultEqual(A,Aelem,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");

  /* Create matrix Belem */
  PetscInt M,N;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&Belem);CHKERRQ(ierr);
  ierr = MatSetSizes(Belem,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(Belem,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Belem);CHKERRQ(ierr);
  ierr = MatSetUp(Belem);CHKERRQ(ierr);

  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscInt       nrows,ncols,i,j;
  ierr = MatGetOwnershipIS(Belem,&isrows,&iscols);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-Cexp_view_ownership",&flg);CHKERRQ(ierr);
  if (flg) { /* View ownership of explicit C */
    IS tmp;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ownership of explicit C:\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row index set:\n");CHKERRQ(ierr);
    ierr = ISOnComm(isrows,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp);CHKERRQ(ierr);
    ierr = ISView(tmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Column index set:\n");CHKERRQ(ierr);
    ierr = ISOnComm(iscols,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp);CHKERRQ(ierr);
    ierr = ISView(tmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }

  /* Set local matrix entries */
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);

  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      if (rows[i] == cols[j]) {
        PetscScalar v = 1.0;
        ierr = MatSetValues(Belem,1,&rows[i],1,&cols[j],&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Belem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Belem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(Belem,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);

  /* Test MatElementalComputeEigenvalues() */
  ierr = MatElementalComputeEigenvalues(Aelem);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aelem);CHKERRQ(ierr);
  ierr = MatDestroy(&Belem);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
