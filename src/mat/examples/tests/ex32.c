/*$Id: ex32.c,v 1.21 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Reads in a matrix and vector in ASCII slap format and writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  Vec    b;
  char   filein[128],fileout[128];
  int    i,j,m,n,nnz,ierr,rank,size,start,end,*col,*row,*brow,length;
  Scalar *val,*bval;
  FILE*  file;
  PetscViewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",filein,127,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRA(ierr);

  fscanf(file,"  NUNKNS =%d  NCOEFF =%d\n",&n,&nnz);
  fscanf(file,"  JA POINTER IN SLAPSV\n");

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,20,0,&A);CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&b);CHKERRA(ierr);

ierr = PetscMalloc((n+1)*sizeof(int),&  col );CHKPTRA(col);
  for (i=0; i<n+1; i++)
    fscanf(file,"     I=%d%d\n",&j,&col[i]);
  fscanf(file,"  EOD JA\n");

ierr = PetscMalloc(nnz*sizeof(Scalar),&(  val ));CHKPTRA(val);
ierr = PetscMalloc(nnz*sizeof(int),&(  row ));CHKPTRA(row);
  fscanf(file,"  COEFFICIENT MATRIX IN SLAPSV: I, IA, A\n");
  for (i=0; i<nnz; i++) {
    fscanf(file,"    %d%d%le\n",&j,&row[i],&val[i]);
    row[i]--;
  }
  fscanf(file,"  EOD IA\n");

ierr = PetscMalloc(n*sizeof(Scalar),&(  bval ));CHKPTRA(bval);
ierr = PetscMalloc(n*sizeof(int),&(  brow ));CHKPTRA(brow);
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
    ierr = MatSetValues(A,length,&row[col[j]-1],1,&j,&val[col[j]-1],INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = VecGetOwnershipRange(b,&start,&end);CHKERRA(ierr);
  ierr = VecSetValues(b,end-start,brow+start,bval+start,INSERT_VALUES);CHKERRA(ierr);
  ierr = VecAssemblyBegin(b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);

  ierr = PetscFree(col);CHKERRA(ierr);
  ierr = PetscFree(val);CHKERRA(ierr);
  ierr = PetscFree(row);CHKERRA(ierr);
  ierr = PetscFree(bval);CHKERRA(ierr);
  ierr = PetscFree(brow);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");CHKERRA(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",fileout,127,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,PETSC_BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view);CHKERRA(ierr);
  ierr = VecView(b,view);CHKERRA(ierr);
  ierr = PetscViewerDestroy(view);CHKERRA(ierr);

  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

