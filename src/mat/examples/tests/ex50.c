
#include <petscmat.h>

#if !defined(PETSC_USE_64BIT_INDICES)
static char help[] = "Reads in a matrix and vector in ASCII format. Writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";
#endif


int main(int argc,char **args)
{
#if !defined(PETSC_USE_64BIT_INDICES)
  Mat            A;
  Vec            b;
  char           filein[PETSC_MAX_PATH_LEN],finname[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN];
  PetscInt       n,col,row,rowin;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    val,*array;
  FILE           *file;
  PetscViewer    view;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate file for reading");
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate file for writing");

  ierr = PetscFixFilename(filein,finname);CHKERRQ(ierr);
  if (!(file = fopen(finname,"r"))) SETERRQ(PETSC_COMM_SELF,1,"Cannot open input file\n");
  if (fscanf(file,"%d\n",&n) != 1) SETERRQ(PETSC_COMM_SELF,1,"Badly formatted input file\n");

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);

  for (row=0; row<n; row++) {
    if (fscanf(file,"row %d:",&rowin) != 1) SETERRQ(PETSC_COMM_SELF,1,"Badly formatted input file\n");
    if (rowin != row) SETERRQ(PETSC_COMM_SELF,1,"Bad file");
    while (fscanf(file," %d %le",&col,(double*)&val)) {
      ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);
  for (row=0; row<n; row++) {
    if (fscanf(file," ii= %d %le",&col,(double*)(array+row)) != 2)  SETERRQ(PETSC_COMM_SELF,1,"Badly formatted input file\n");
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
  return ierr;
#else
  return 0;
#endif
}

