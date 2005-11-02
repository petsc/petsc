/*
    Reads in individual PETSc matrix files for each processor and concatinates them
  together into a single file containing the entire matrix

    Do NOT use this, use ../ex5.c instead, it is MUCH more memory efficient
*/
#include "petscmat.h"
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscViewer    in,out;
  Mat            inmat,outmat;
  const char     *infile = "split", *outfile = "together";

  PetscInitialize(&argc,&argv,(char*) 0,0);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,infile,FILE_MODE_READ,&in);CHKERRQ(ierr);
  ierr = MatLoad(in,MATSEQAIJ,&inmat);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(in);CHKERRQ(ierr);

  ierr = MatMerge(PETSC_COMM_WORLD,inmat,PETSC_DECIDE,MAT_INITIAL_MATRIX,&outmat);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfile,FILE_MODE_WRITE,&out);CHKERRQ(ierr);
  ierr = MatView(outmat,out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(out);CHKERRQ(ierr);
  ierr = MatDestroy(outmat);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
