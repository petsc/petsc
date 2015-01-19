
static char help[] = "Tests MatConvert() from MATAIJ and MATDENSE to MATELEMENTAL.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,Aelem,B,Belem;
  PetscErrorCode ierr;
  PetscViewer    view;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg,flgB;
  PetscScalar    one = 1.0;
  PetscMPIInt    rank;
  PetscReal      vl,vu;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Now reload PETSc matrix and view it */
  ierr = PetscOptionsGetString(NULL,"-fA",file[0],PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  PetscOptionsGetString(NULL,"-fB",file[1],PETSC_MAX_PATH_LEN,&flgB);
  if (flgB) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&view);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
    ierr = MatLoad(B,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  } else {
    /* Create matrix B = I */
    PetscInt M,N,rstart,rend,i;
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(B,1,&i,1,&i,&one,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Convert AIJ matrices into Elemental matrices */
  if (!rank) printf(" Convert AIJ matrix A into Elemental matrix... \n");
  ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Aelem);CHKERRQ(ierr);
  if (!rank) printf(" Convert AIJ matrix B into Elemental matrix... \n");
  ierr = MatConvert(B, MATELEMENTAL, MAT_INITIAL_MATRIX, &Belem);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
  //ierr = MatConvert(A, MATELEMENTAL, MAT_REUSE_MATRIX, &A);CHKERRQ(ierr);

  /* Test accuracy */
  ierr = MatMultEqual(A,Aelem,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");
   ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatMultEqual(B,Belem,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"B != B_elemental.");
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Create matrix Belem */
#if defined(CreateBelem) 
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
        ierr = MatSetValues(Belem,1,&rows[i],1,&cols[j],&one,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Belem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Belem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(Belem,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
#endif

  /* Test MatElementalComputeEigenvalues() */
  if (!rank) printf(" Compute Ax = lambda Bx... \n");
  vl=-0.8, vu=1.2;
  ierr = PetscOptionsGetReal(NULL,"-vl",&vl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-vu",&vu,NULL);CHKERRQ(ierr);
  ierr = MatElementalHermitianGenDefiniteEig(Aelem,Belem,vl,vu);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aelem);CHKERRQ(ierr);
  ierr = MatDestroy(&Belem);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
