/*$Id: ex73.c,v 1.4 2000/09/22 20:44:16 bsmith Exp bsmith $*/

static char help[] = 
"Reads a PETSc matrix from a file partitions it\n\n";

/*T
   Concepts: partitioning
   Processors: n
T*/

/* 
  Include "petscmat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            
     petscviewer.h - viewers               
*/
#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  MatType         mtype = MATSEQSBAIJ;            /* matrix format */
  Mat             A,B;                /* matrix */
  Viewer          fd;               /* viewer */
  char            file[128];        /* input file name */
  PetscTruth      flg;
  int             ierr,*nlocal,rank,size;
  MatPartitioning part;
  IS              is,isn;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /* 
     Determine file from which we read the matrix
  */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRA(ierr);

  /* 
       Open binary file.  Note that we use BINARY_RDONLY to indicate
       reading from this file.
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);

  /*
      Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,mtype,&A);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  ierr = MatView(A,VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /*
       Partition the graph of the matrix 
  */
  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRA(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRA(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRA(ierr);
  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRA(ierr);
  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRA(ierr);
  nlocal = (int*)PetscMalloc(size*sizeof(int));CHKPTRA(nlocal);
  /* get number of new vertices for each processor */
  ierr = ISPartitioningCount(is,nlocal);CHKERRA(ierr); 
  ierr = ISDestroy(is);CHKERRA(ierr);

  /* get old global number of each new global number */
  ierr = ISInvertPermutation(isn,nlocal[rank],&is);CHKERRA(ierr);
  ierr = PetscFree(nlocal);CHKERRA(ierr);
  ierr = ISDestroy(isn);CHKERRA(ierr);
  ierr = MatPartitioningDestroy(part);CHKERRA(ierr);

  ierr = ISSort(is);CHKERRA(ierr);
  ierr = ISAllGather(is,&isn);CHKERRA(ierr);


  ierr = MatGetSubMatrix(A,is,isn,PETSC_DECIDE,MAT_INITIAL_MATRIX,&B);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);
  ierr = ISDestroy(isn);CHKERRA(ierr);

  ierr = MatView(B,VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = MatDestroy(A);CHKERRA(ierr); 
  ierr = MatDestroy(B);CHKERRA(ierr); 


  PetscFinalize();
  return 0;
}

