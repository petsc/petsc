#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex50.c,v 1.12 1999/04/16 16:07:27 bsmith Exp balay $";
#endif

static char help[] = "Reads in a matrix and vector in ASCII format and writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  Vec    b;
  char   filein[256],finname[256],fileout[256];
  int    n,ierr,col,row;
  int    flg,rowin;
  Scalar val,*array;
  FILE*  file;
  Viewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-fin",filein,255,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate file for reading");
  ierr = OptionsGetString(PETSC_NULL,"-fout",fileout,255,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate file for writing");

  ierr = PetscFixFilename(filein,finname);CHKERRA(ierr);
  if ((file = fopen(finname,"r")) == 0) {
    SETERRA(1,0,"cannot open input file\n");
  }
  fscanf(file,"%d\n",&n);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);CHKERRA(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&b);CHKERRA(ierr);
  ierr = VecSetFromOptions(b);CHKERRA(ierr);

  for ( row=0; row<n; row++ ) {
    fscanf(file,"row %d:",&rowin);
    if (rowin != row) SETERRA(1,0,"Bad file");
    while (fscanf(file," %d %le",&col,&val)) {
      MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);
    }  
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = VecGetArray(b,&array);CHKERRA(ierr);
  for ( row=0; row<n; row++ ) {
    fscanf(file," ii= %d %le",&col,array+row);
  }
  ierr = VecRestoreArray(b,&array);CHKERRA(ierr);

  fclose(file);

  PetscPrintf(PETSC_COMM_SELF,"Reading matrix complete.\n");
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,fileout,BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view);CHKERRA(ierr);
  ierr = VecView(b,view);CHKERRA(ierr);
  ierr = ViewerDestroy(view);CHKERRA(ierr);

  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

