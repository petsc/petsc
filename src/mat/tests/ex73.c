
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
#include <petscmat.h>

int main(int argc,char **args)
{
  MatType         mtype = MATMPIAIJ; /* matrix format */
  Mat             A,B;               /* matrix */
  PetscViewer     fd;                /* viewer */
  char            file[PETSC_MAX_PATH_LEN];         /* input file name */
  PetscBool       flg,viewMats,viewIS,viewVecs,useND,noVecLoad = PETSC_FALSE;
  PetscInt        ierr,*nlocal,m,n;
  PetscMPIInt     rank,size;
  MatPartitioning part;
  IS              is,isn;
  Vec             xin, xout;
  VecScatter      scat;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-view_mats", &viewMats));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-view_is", &viewIS));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-view_vecs", &viewVecs));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-use_nd", &useND));
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-novec_load", &noVecLoad));

  /*
     Determine file from which we read the matrix
  */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));

  /*
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /*
      Load the matrix and vector; then destroy the viewer.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,mtype));
  CHKERRQ(MatLoad(A,fd));
  if (!noVecLoad) {
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&xin));
    CHKERRQ(VecLoad(xin,fd));
  } else {
    CHKERRQ(MatCreateVecs(A,&xin,NULL));
    CHKERRQ(VecSetRandom(xin,NULL));
  }
  CHKERRQ(PetscViewerDestroy(&fd));
  if (viewMats) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Original matrix:\n"));
    CHKERRQ(MatView(A,PETSC_VIEWER_DRAW_WORLD));
  }
  if (viewVecs) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Original vector:\n"));
    CHKERRQ(VecView(xin,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Partition the graph of the matrix */
  CHKERRQ(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part,A));
  CHKERRQ(MatPartitioningSetFromOptions(part));

  /* get new processor owner number of each vertex */
  if (useND) {
    CHKERRQ(MatPartitioningApplyND(part,&is));
  } else {
    CHKERRQ(MatPartitioningApply(part,&is));
  }
  if (viewIS) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"IS1 - new processor ownership:\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* get new global number of each old global number */
  CHKERRQ(ISPartitioningToNumbering(is,&isn));
  if (viewIS) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"IS2 - new global numbering:\n"));
    CHKERRQ(ISView(isn,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* get number of new vertices for each processor */
  CHKERRQ(PetscMalloc1(size,&nlocal));
  CHKERRQ(ISPartitioningCount(is,size,nlocal));
  CHKERRQ(ISDestroy(&is));

  /* get old global number of each new global number */
  CHKERRQ(ISInvertPermutation(isn,useND ? PETSC_DECIDE : nlocal[rank],&is));
  if (viewIS) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"IS3=inv(IS2) - old global number of each new global number:\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* move the matrix rows to the new processes they have been assigned to by the permutation */
  CHKERRQ(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(PetscFree(nlocal));
  CHKERRQ(ISDestroy(&isn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatPartitioningDestroy(&part));
  if (viewMats) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Partitioned matrix:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_DRAW_WORLD));
  }

  /* move the vector rows to the new processes they have been assigned to */
  CHKERRQ(MatGetLocalSize(B,&m,&n));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&xout));
  CHKERRQ(VecScatterCreate(xin,is,xout,NULL,&scat));
  CHKERRQ(VecScatterBegin(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&scat));
  if (viewVecs) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Mapped vector:\n"));
    CHKERRQ(VecView(xout,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(VecDestroy(&xout));
  CHKERRQ(ISDestroy(&is));

  {
    PetscInt          rstart,i,*nzd,*nzo,nzl,nzmax = 0,*ncols,nrow,j;
    Mat               J;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscScalar       *nvals;

    CHKERRQ(MatGetOwnershipRange(B,&rstart,NULL));
    CHKERRQ(PetscCalloc2(2*m,&nzd,2*m,&nzo));
    for (i=0; i<m; i++) {
      CHKERRQ(MatGetRow(B,i+rstart,&nzl,&cols,NULL));
      for (j=0; j<nzl; j++) {
        if (cols[j] >= rstart && cols[j] < rstart+n) {
          nzd[2*i] += 2;
          nzd[2*i+1] += 2;
        } else {
          nzo[2*i] += 2;
          nzo[2*i+1] += 2;
        }
      }
      nzmax = PetscMax(nzmax,nzd[2*i]+nzo[2*i]);
      CHKERRQ(MatRestoreRow(B,i+rstart,&nzl,&cols,NULL));
    }
    CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,2*m,2*m,PETSC_DECIDE,PETSC_DECIDE,0,nzd,0,nzo,&J));
    CHKERRQ(PetscInfo(0,"Created empty Jacobian matrix\n"));
    CHKERRQ(PetscFree2(nzd,nzo));
    CHKERRQ(PetscMalloc2(nzmax,&ncols,nzmax,&nvals));
    CHKERRQ(PetscArrayzero(nvals,nzmax));
    for (i=0; i<m; i++) {
      CHKERRQ(MatGetRow(B,i+rstart,&nzl,&cols,&vals));
      for (j=0; j<nzl; j++) {
        ncols[2*j]   = 2*cols[j];
        ncols[2*j+1] = 2*cols[j]+1;
      }
      nrow = 2*(i+rstart);
      CHKERRQ(MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES));
      nrow = 2*(i+rstart) + 1;
      CHKERRQ(MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(B,i+rstart,&nzl,&cols,&vals));
    }
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    if (viewMats) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Jacobian matrix structure:\n"));
      CHKERRQ(MatView(J,PETSC_VIEWER_DRAW_WORLD));
    }
    CHKERRQ(MatDestroy(&J));
    CHKERRQ(PetscFree2(ncols,nvals));
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&xin));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      requires: parmetis datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -nox -f ${DATAFILESPATH}/matrices/arco1 -mat_partitioning_type parmetis -viewer_binary_skip_info -novec_load

   test:
      requires: parmetis !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex73_1.out
      suffix: parmetis_nd_32
      nsize: 3
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -mat_partitioning_type parmetis -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: parmetis !complex double defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex73_1.out
      suffix: parmetis_nd_64
      nsize: 3
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int64-float64 -mat_partitioning_type parmetis -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: ptscotch !complex double !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex73_1.out
      suffix: ptscotch_nd_32
      nsize: 4
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -mat_partitioning_type ptscotch -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: ptscotch !complex double defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex73_1.out
      suffix: ptscotch_nd_64
      nsize: 4
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int64-float64 -mat_partitioning_type ptscotch -viewer_binary_skip_info -use_nd -novec_load

TEST*/
