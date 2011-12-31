static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

#undef __FUNCT__  
#define __FUNCT__ "Assemble"
PetscErrorCode Assemble(MPI_Comm comm,PetscInt bs,const MatType mtype)
{
  const PetscInt rc[] = {0,1,2,3};
  const PetscScalar vals[] = {1, 2, 3, 4, 5, 6, 7, 8,
                              9,10,11,12,13,14,15,16,
                              17,18,19,20,21,22,23,24,
                              25,26,27,28,29,30,31,32,
                              33,34,35,36,37,38,39,40,
                              41,42,43,44,45,46,47,48,
                              49,50,51,52,53,54,55,56,
                              57,58,49,60,61,62,63,64};
  Mat A;
  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4*bs,4*bs);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,bs,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,bs,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* All processes contribute a global matrix */
  ierr = MatSetValuesBlocked(A,4,rc,4,rc,vals,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Matrix %s(%D)\n",mtype,bs);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size != 2) SETERRQ(comm,PETSC_ERR_USER,"This example must be run with exactly two processes");
  ierr = Assemble(comm,2,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,2,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,1,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,1,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
