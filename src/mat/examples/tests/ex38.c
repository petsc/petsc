
static char help[] = "Tests MatSetValues().\n\n"; 

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,Cexp;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    *v;
  PetscMPIInt    rank,size;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  // Get local block or element size
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-row_oriented",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = MatGetOwnershipIS(C,&isrows,&iscols);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-Cexp_view_ownership",&flg);CHKERRQ(ierr);
  if (flg) { // View ownership of explicit C
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

  // Set local matrix entries
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc(nrows*ncols*sizeof *v,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      //v[i*ncols+j] = (PetscReal)(rank); 
      v[i*ncols+j] = (PetscReal)(rank*10000+100*rows[i]+cols[j]); 
      if (rank==-1) {ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] set (%d, %d, %g)\n",rank,rows[i],cols[j],v[i*ncols+j]);CHKERRQ(ierr);}
    }
  }
  ierr = MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Test MatView() 
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(C,&Cexp);CHKERRQ(ierr);
  ierr = MatView(Cexp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&Cexp);CHKERRQ(ierr);

  // Set unowned matrix entries - add subdiagonals and diagonals from proc[0]
  if (rank == 0) { 
    PetscInt M,N,cols[2];
    ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
    for (i=0; i<M; i++){
      cols[0] = i;   v[0] = i + 0.5;
      cols[1] = i-1; v[1] = 0.5;
      if (i) {
        ierr = MatSetValues(C,1,&i,2,cols,v,ADD_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValues(C,1,&i,1,&i,v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(C,&Cexp);CHKERRQ(ierr);
  ierr = MatView(Cexp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // Test MatMult() 
  ierr = MatMultEqual(C,Cexp,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Cexp. MatMultEqual() fails");

  // Test MatMultAdd() 
  ierr = MatMultAddEqual(C,Cexp,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Cexp. MatMultAddEqual() fails");

  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Cexp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

 
