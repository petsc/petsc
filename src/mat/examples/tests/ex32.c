
static char help[] = "Reads in a matrix and vector in ASCII slap format. Writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            b;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN];
  PetscInt       i,j,m,n,nnz,start,end,*col,*row,*brow,length;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    *val,*bval;
  FILE*          file;
  PetscViewer    view;
  PetscTruth     opt;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",filein,PETSC_MAX_PATH_LEN-1,&opt);CHKERRQ(ierr);
  if (!opt) {
    SETERRQ(PETSC_ERR_ARG_WRONG, "No filename was specified for this test");
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  fscanf(file,"  NUNKNS =%d  NCOEFF =%d\n",&n,&nnz);
  fscanf(file,"  JA POINTER IN SLAPSV\n");

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,20,0,&A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);

  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&col);CHKERRQ(ierr);
  for (i=0; i<n+1; i++)
    fscanf(file,"     I=%d%d\n",&j,&col[i]);
  fscanf(file,"  EOD JA\n");

  ierr = PetscMalloc(nnz*sizeof(PetscScalar),&val);CHKERRQ(ierr);
  ierr = PetscMalloc(nnz*sizeof(PetscInt),&row);CHKERRQ(ierr);
  fscanf(file,"  COEFFICIENT MATRIX IN SLAPSV: I, IA, A\n");
  for (i=0; i<nnz; i++) {
    fscanf(file,"    %d%d%le\n",&j,&row[i],(double*)&val[i]);
    row[i]--;
  }
  fscanf(file,"  EOD IA\n");

  ierr = PetscMalloc(n*sizeof(PetscScalar),&bval);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&brow);CHKERRQ(ierr);
  fscanf(file,"  RESIDUAL IN SLAPSV ;IRHS=%d\n",&j);
  for (i=0; i<n; i++) {
    fscanf(file,"      %d%le%d\n",&j,(double*)(bval+i),&j);
    brow[i] = i;
  }
  fscanf(file,"  EOD RESIDUAL");
  fclose(file);

  m = n/size+1;
  start = rank*m;
  end = (rank+1)*m; if (end > n) end = n;
  for (j=start; j<end; j++) {
    length = col[j+1]-col[j];
    ierr = MatSetValues(A,length,&row[col[j]-1],1,&j,&val[col[j]-1],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
  ierr = VecSetValues(b,end-start,brow+start,bval+start,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = PetscFree(col);CHKERRQ(ierr);
  ierr = PetscFree(val);CHKERRQ(ierr);
  ierr = PetscFree(row);CHKERRQ(ierr);
  ierr = PetscFree(bval);CHKERRQ(ierr);
  ierr = PetscFree(brow);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",fileout,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = VecView(b,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(view);CHKERRQ(ierr);

  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

