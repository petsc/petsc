
static char help[] = "Reads a PETSc matrix from a file partitions it\n\n";

/*T
   Concepts: partitioning
   Processors: n
T*/

/* 
  Include "petscmat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            
     petscviewer.h - viewers    

  Example of usage:  
    mpiexec -n 3 ex73 -f <matfile> -mat_partitioning_type parmetis/scotch -viewer_binary_skip_info -nox
*/
#include "petscksp.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  const MatType   mtype = MATMPIAIJ; /* matrix format */
  Mat             A,B;               /* matrix */
  PetscViewer     fd;                /* viewer */
  char            file[PETSC_MAX_PATH_LEN];         /* input file name */
  PetscTruth      flg;
  PetscInt        ierr,*nlocal;
  PetscMPIInt     rank,size;
  MatPartitioning part;
  IS              is,isn;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* 
     Determine file from which we read the matrix
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);

  /* 
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
      Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,mtype,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /*
       Partition the graph of the matrix 
  */
  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscInt),&nlocal);CHKERRQ(ierr);
  /* get number of new vertices for each processor */
  ierr = ISPartitioningCount(is,size,nlocal);CHKERRQ(ierr); 
  ierr = ISDestroy(is);CHKERRQ(ierr);

  /* get old global number of each new global number */
  ierr = ISInvertPermutation(isn,nlocal[rank],&is);CHKERRQ(ierr);
  ierr = PetscFree(nlocal);CHKERRQ(ierr);
  ierr = ISDestroy(isn);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(part);CHKERRQ(ierr);

  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = MatView(B,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = MatDestroy(B);CHKERRQ(ierr); 


  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

