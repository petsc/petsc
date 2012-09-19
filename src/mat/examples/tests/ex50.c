
static char help[] = "Reads in a matrix and vector in ASCII format. Writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            b;
  char           filein[PETSC_MAX_PATH_LEN],finname[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN];
  PetscInt       n,col,row,rowin;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    val,*array;
  FILE*          file;
  PetscViewer    view;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",filein,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate file for reading");
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",fileout,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate file for writing");

  ierr = PetscFixFilename(filein,finname);CHKERRQ(ierr);
  if (!(file = fopen(finname,"r"))) {
    SETERRQ(PETSC_COMM_SELF,1,"cannot open input file\n");
  }
  fscanf(file,"%d\n",&n);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);

  for (row=0; row<n; row++) {
    fscanf(file,"row %d:",&rowin);
    if (rowin != row) SETERRQ(PETSC_COMM_SELF,1,"Bad file");
    while (fscanf(file," %d %le",&col,(double*)&val)) {
      ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);
  for (row=0; row<n; row++) {
    fscanf(file," ii= %d %le",&col,(double*)(array+row));
  }
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);

  fclose(file);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix complete.\n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = VecView(b,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

