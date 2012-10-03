
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
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  MatType         mtype = MATMPIAIJ; /* matrix format */
  Mat             A,B;               /* matrix */
  PetscViewer     fd;                /* viewer */
  char            file[PETSC_MAX_PATH_LEN];         /* input file name */
  PetscBool       flg,viewMats,viewIS,viewVecs;
  PetscInt        ierr,*nlocal,m,n;
  PetscMPIInt     rank,size;
  MatPartitioning part;
  IS              is,isn;
  Vec             xin, xout;
  VecScatter      scat;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-view_mats", &viewMats);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-view_is", &viewIS);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-view_vecs", &viewVecs);CHKERRQ(ierr);

  /*
     Determine file from which we read the matrix
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);

  /*
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
      Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xin);CHKERRQ(ierr);
  ierr = VecLoad(xin,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  if (viewMats){
    if (!rank) printf("Original matrix:\n");
    ierr = MatView(A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (viewVecs){
    if (!rank) printf("Original vector:\n");
    ierr = VecView(xin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Partition the graph of the matrix */
  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);

  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  if (viewIS){
    if (!rank) printf("IS1 - new processor ownership:\n");
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  if (viewIS){
    if (!rank) printf("IS2 - new global numbering:\n");
    ierr = ISView(isn,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* get number of new vertices for each processor */
  ierr = PetscMalloc(size*sizeof(PetscInt),&nlocal);CHKERRQ(ierr);
  ierr = ISPartitioningCount(is,size,nlocal);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* get old global number of each new global number */
  ierr = ISInvertPermutation(isn,nlocal[rank],&is);CHKERRQ(ierr);
  ierr = PetscFree(nlocal);CHKERRQ(ierr);
  ierr = ISDestroy(&isn);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  if (viewIS){
    if (!rank) printf("IS3=inv(IS2) - old global number of each new global number:\n");
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* move the matrix rows to the new processes they have been assigned to by the permutation */
  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* move the vector rows to the new processes they have been assigned to */
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&xout);CHKERRQ(ierr);
  ierr = VecScatterCreate(xin,is,xout,PETSC_NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  if (viewMats){
    if (!rank) printf("Partitioned matrix:\n");
    ierr = MatView(B,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (viewVecs){
    if (!rank) printf("Mapped vector:\n");
    ierr = VecView(xout,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  {
    PetscInt          rstart,i,*nzd,*nzo,nzl,nzmax = 0,*ncols,nrow,j;
    Mat               J;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscScalar       *nvals;

    ierr = MatGetOwnershipRange(B,&rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(2*m*sizeof(PetscInt),&nzd);CHKERRQ(ierr);
    ierr = PetscMemzero(nzd,2*m*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc(2*m*sizeof(PetscInt),&nzo);CHKERRQ(ierr);
    ierr = PetscMemzero(nzo,2*m*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = MatGetRow(B,i+rstart,&nzl,&cols,PETSC_NULL);CHKERRQ(ierr);
      for (j=0; j<nzl; j++) {
        if (cols[j] >= rstart && cols[j] < rstart+n) {nzd[2*i] += 2; nzd[2*i+1] += 2;}
        else {nzo[2*i] += 2; nzo[2*i+1] += 2;}
      }
      nzmax = PetscMax(nzmax,nzd[2*i]+nzo[2*i]);
      ierr = MatRestoreRow(B,i+rstart,&nzl,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,2*m,2*m,PETSC_DECIDE,PETSC_DECIDE,0,nzd,0,nzo,&J);CHKERRQ(ierr);
    ierr = PetscInfo(0,"Created empty Jacobian matrix\n");CHKERRQ(ierr);
    ierr = PetscFree(nzd);CHKERRQ(ierr);
    ierr = PetscFree(nzo);CHKERRQ(ierr);
    ierr = PetscMalloc2(nzmax,PetscInt,&ncols,nzmax,PetscScalar,&nvals);CHKERRQ(ierr);
    ierr = PetscMemzero(nvals,nzmax*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = MatGetRow(B,i+rstart,&nzl,&cols,&vals);CHKERRQ(ierr);
      for (j=0; j<nzl; j++) {
        ncols[2*j]   = 2*cols[j];
        ncols[2*j+1] = 2*cols[j]+1;
      }
      nrow = 2*(i+rstart);
      ierr = MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES);CHKERRQ(ierr);
      nrow = 2*(i+rstart) + 1;
      ierr = MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i+rstart,&nzl,&cols,&vals);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (viewMats){
      if (!rank) printf("Jacobian matrix structure:\n");
      ierr = MatView(J,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = PetscFree2(ncols,nvals);CHKERRQ(ierr);
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&xin);CHKERRQ(ierr);
  ierr = VecDestroy(&xout);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

