
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
  PetscInt        *nlocal,m,n;
  PetscMPIInt     rank,size;
  MatPartitioning part;
  IS              is,isn;
  Vec             xin, xout;
  VecScatter      scat;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-view_mats", &viewMats));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-view_is", &viewIS));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-view_vecs", &viewVecs));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-use_nd", &useND));
  PetscCall(PetscOptionsHasName(NULL,NULL, "-novec_load", &noVecLoad));

  /*
     Determine file from which we read the matrix
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));

  /*
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /*
      Load the matrix and vector; then destroy the viewer.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,mtype));
  PetscCall(MatLoad(A,fd));
  if (!noVecLoad) {
    PetscCall(VecCreate(PETSC_COMM_WORLD,&xin));
    PetscCall(VecLoad(xin,fd));
  } else {
    PetscCall(MatCreateVecs(A,&xin,NULL));
    PetscCall(VecSetRandom(xin,NULL));
  }
  PetscCall(PetscViewerDestroy(&fd));
  if (viewMats) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Original matrix:\n"));
    PetscCall(MatView(A,PETSC_VIEWER_DRAW_WORLD));
  }
  if (viewVecs) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Original vector:\n"));
    PetscCall(VecView(xin,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Partition the graph of the matrix */
  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
  PetscCall(MatPartitioningSetAdjacency(part,A));
  PetscCall(MatPartitioningSetFromOptions(part));

  /* get new processor owner number of each vertex */
  if (useND) {
    PetscCall(MatPartitioningApplyND(part,&is));
  } else {
    PetscCall(MatPartitioningApply(part,&is));
  }
  if (viewIS) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"IS1 - new processor ownership:\n"));
    PetscCall(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* get new global number of each old global number */
  PetscCall(ISPartitioningToNumbering(is,&isn));
  if (viewIS) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"IS2 - new global numbering:\n"));
    PetscCall(ISView(isn,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* get number of new vertices for each processor */
  PetscCall(PetscMalloc1(size,&nlocal));
  PetscCall(ISPartitioningCount(is,size,nlocal));
  PetscCall(ISDestroy(&is));

  /* get old global number of each new global number */
  PetscCall(ISInvertPermutation(isn,useND ? PETSC_DECIDE : nlocal[rank],&is));
  if (viewIS) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"IS3=inv(IS2) - old global number of each new global number:\n"));
    PetscCall(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* move the matrix rows to the new processes they have been assigned to by the permutation */
  PetscCall(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&B));
  PetscCall(PetscFree(nlocal));
  PetscCall(ISDestroy(&isn));
  PetscCall(MatDestroy(&A));
  PetscCall(MatPartitioningDestroy(&part));
  if (viewMats) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Partitioned matrix:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_DRAW_WORLD));
  }

  /* move the vector rows to the new processes they have been assigned to */
  PetscCall(MatGetLocalSize(B,&m,&n));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&xout));
  PetscCall(VecScatterCreate(xin,is,xout,NULL,&scat));
  PetscCall(VecScatterBegin(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scat));
  if (viewVecs) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Mapped vector:\n"));
    PetscCall(VecView(xout,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecDestroy(&xout));
  PetscCall(ISDestroy(&is));

  {
    PetscInt          rstart,i,*nzd,*nzo,nzl,nzmax = 0,*ncols,nrow,j;
    Mat               J;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscScalar       *nvals;

    PetscCall(MatGetOwnershipRange(B,&rstart,NULL));
    PetscCall(PetscCalloc2(2*m,&nzd,2*m,&nzo));
    for (i=0; i<m; i++) {
      PetscCall(MatGetRow(B,i+rstart,&nzl,&cols,NULL));
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
      PetscCall(MatRestoreRow(B,i+rstart,&nzl,&cols,NULL));
    }
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,2*m,2*m,PETSC_DECIDE,PETSC_DECIDE,0,nzd,0,nzo,&J));
    PetscCall(PetscInfo(0,"Created empty Jacobian matrix\n"));
    PetscCall(PetscFree2(nzd,nzo));
    PetscCall(PetscMalloc2(nzmax,&ncols,nzmax,&nvals));
    PetscCall(PetscArrayzero(nvals,nzmax));
    for (i=0; i<m; i++) {
      PetscCall(MatGetRow(B,i+rstart,&nzl,&cols,&vals));
      for (j=0; j<nzl; j++) {
        ncols[2*j]   = 2*cols[j];
        ncols[2*j+1] = 2*cols[j]+1;
      }
      nrow = 2*(i+rstart);
      PetscCall(MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES));
      nrow = 2*(i+rstart) + 1;
      PetscCall(MatSetValues(J,1,&nrow,2*nzl,ncols,nvals,INSERT_VALUES));
      PetscCall(MatRestoreRow(B,i+rstart,&nzl,&cols,&vals));
    }
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    if (viewMats) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Jacobian matrix structure:\n"));
      PetscCall(MatView(J,PETSC_VIEWER_DRAW_WORLD));
    }
    PetscCall(MatDestroy(&J));
    PetscCall(PetscFree2(ncols,nvals));
  }

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&xin));
  PetscCall(PetscFinalize());
  return 0;
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
