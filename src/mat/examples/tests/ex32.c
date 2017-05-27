
#include <petscmat.h>

#if !defined(PETSC_USE_64BIT_INDICES)
static char help[] = "Reads in a matrix and vector in ASCII slap format. Writes\n\
them using the PETSc sparse format. Input parameters are:\n\
  -fin <filename> : input file\n\
  -fout <filename> : output file\n\n";
#endif

int main(int argc,char **args)
{
#if !defined(PETSC_USE_64BIT_INDICES)
  Mat            A;
  Vec            b;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN];
  PetscInt       i,j,m,n,nnz,start,end,*col,*row,*brow,length;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    *val,*bval;
  FILE           *file;
  PetscViewer    view;
  PetscBool      opt;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Read in matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,PETSC_MAX_PATH_LEN,&opt);CHKERRQ(ierr);
  if (!opt) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "No filename was specified for this test");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  if (fscanf(file,"  NUNKNS =%d  NCOEFF =%d\n",&n,&nnz) != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
  if (fscanf(file,"  JA POINTER IN SLAPSV\n")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,20,0,&A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);

  ierr = PetscMalloc1(n+1,&col);CHKERRQ(ierr);
  for (i=0; i<n+1; i++) {
    if (fscanf(file,"     I=%d%d\n",&j,&col[i]) != 2)  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
  }
  if (fscanf(file,"  EOD JA\n")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");

  ierr = PetscMalloc1(nnz,&val);CHKERRQ(ierr);
  ierr = PetscMalloc1(nnz,&row);CHKERRQ(ierr);
  if (fscanf(file,"  COEFFICIENT MATRIX IN SLAPSV: I, IA, A\n")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
  for (i=0; i<nnz; i++) {
    if (fscanf(file,"    %d%d%le\n",&j,&row[i],(double*)&val[i]) != 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
    row[i]--;
  }
  if (fscanf(file,"  EOD IA\n")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");

  ierr = PetscMalloc1(n,&bval);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&brow);CHKERRQ(ierr);
  if (fscanf(file,"  RESIDUAL IN SLAPSV ;IRHS=%d\n",&j) != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
  for (i=0; i<n; i++) {
    if (fscanf(file,"      %d%le%d\n",&j,(double*)(bval+i),&j) != 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
    brow[i] = i;
  }
  if (fscanf(file,"  EOD RESIDUAL")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG, "Incorrectly formatted file");
  fclose(file);

  m     = n/size+1;
  start = rank*m;
  end   = (rank+1)*m; if (end > n) end = n;
  for (j=start; j<end; j++) {
    length = col[j+1]-col[j];
    ierr   = MatSetValues(A,length,&row[col[j]-1],1,&j,&val[col[j]-1],INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
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

