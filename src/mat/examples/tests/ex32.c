#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex32.c,v 1.12 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Reads in a matrix and vector in ASCII slap format and writes\n\
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
  char   filein[128],fileout[128];
  int    i, j, m, n, nnz, ierr, rank, size, start, end, *col, *row, *brow, length;
  int    flg;
  Scalar *val, *bval;
  FILE*  file;
  Viewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-fin",filein,127,&flg); CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  if ((file = PetscFOpen(PETSC_COMM_SELF,filein,"r")) == 0) {
    SETERRA(1,0,"cannot open file\n");
  }
  fscanf(file,"  NUNKNS =%d  NCOEFF =%d\n",&n,&nnz);
  fscanf(file,"  JA POINTER IN SLAPSV\n");

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,20,0,&A); CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&b); CHKERRA(ierr);

  col = (int *) PetscMalloc((n+1)*sizeof(int)); CHKPTRA(col);
  for (i=0; i<n+1; i++)
    fscanf(file,"     I=%d%d\n",&j,&col[i]);
  fscanf(file,"  EOD JA\n");

  val = (Scalar *) PetscMalloc(nnz*sizeof(Scalar)); CHKPTRA(val);
  row = (int *) PetscMalloc(nnz*sizeof(int)); CHKPTRA(row);
  fscanf(file,"  COEFFICIENT MATRIX IN SLAPSV: I, IA, A\n");
  for (i=0; i<nnz; i++) {
    fscanf(file,"    %d%d%le\n",&j,&row[i],&val[i]);
    row[i]--;
  }
  fscanf(file,"  EOD IA\n");

  bval = (Scalar *) PetscMalloc(n*sizeof(Scalar)); CHKPTRA(bval);
  brow = (int *) PetscMalloc(n*sizeof(int)); CHKPTRA(brow);
  fscanf(file,"  RESIDUAL IN SLAPSV ;IRHS=%d\n",&j);
  for (i=0; i<n; i++) {
    fscanf(file,"      %d%le%d\n",&j,bval+i,&j);
    brow[i] = i;
  }
  fscanf(file,"  EOD RESIDUAL");
  fclose(file);

  m = n/size+1;
  start = rank*m;
  end = (rank+1)*m; if (end > n) end = n;
  for (j=start; j<end; j++) {
    length = col[j+1]-col[j];
    MatSetValues(A,length,&row[col[j]-1],1,&j,&val[col[j]-1],INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = VecGetOwnershipRange(b,&start,&end); CHKERRA(ierr);
  ierr = VecSetValues(b,end-start,brow+start,bval+start,INSERT_VALUES); CHKERRA(ierr);
  ierr = VecAssemblyBegin(b); CHKERRA(ierr);
  ierr = VecAssemblyEnd(b); CHKERRA(ierr);

  PetscFree(col); PetscFree(val); PetscFree(row);
  PetscFree(bval); PetscFree(brow);

  PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");
  ierr = OptionsGetString(PETSC_NULL,"-fout",fileout,127,&flg); CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,fileout,BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view); CHKERRA(ierr);
  ierr = VecView(b,view); CHKERRA(ierr);
  ierr = ViewerDestroy(view); CHKERRA(ierr);

  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

