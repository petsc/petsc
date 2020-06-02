
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-view_mats", &viewMats);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-view_is", &viewIS);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-view_vecs", &viewVecs);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-use_nd", &useND);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL, "-novec_load", &noVecLoad);CHKERRQ(ierr);

  /*
     Determine file from which we read the matrix
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);

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
  if (!noVecLoad) {
    ierr = VecCreate(PETSC_COMM_WORLD,&xin);CHKERRQ(ierr);
    ierr = VecLoad(xin,fd);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(A,&xin,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(xin,NULL);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  if (viewMats) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Original matrix:\n");CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  if (viewVecs) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Original vector:\n");CHKERRQ(ierr);
    ierr = VecView(xin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Partition the graph of the matrix */
  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);

  /* get new processor owner number of each vertex */
  if (useND) {
    ierr = MatPartitioningApplyND(part,&is);CHKERRQ(ierr);
  } else {
    ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  }
  if (viewIS) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"IS1 - new processor ownership:\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  if (viewIS) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"IS2 - new global numbering:\n");CHKERRQ(ierr);
    ierr = ISView(isn,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* get number of new vertices for each processor */
  ierr = PetscMalloc1(size,&nlocal);CHKERRQ(ierr);
  ierr = ISPartitioningCount(is,size,nlocal);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* get old global number of each new global number */
  ierr = ISInvertPermutation(isn,useND ? PETSC_DECIDE : nlocal[rank],&is);CHKERRQ(ierr);
  if (viewIS) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"IS3=inv(IS2) - old global number of each new global number:\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* move the matrix rows to the new processes they have been assigned to by the permutation */
  ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = PetscFree(nlocal);CHKERRQ(ierr);
  ierr = ISDestroy(&isn);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  if (viewMats) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Partitioned matrix:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  /* move the vector rows to the new processes they have been assigned to */
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&xout);CHKERRQ(ierr);
  ierr = VecScatterCreate(xin,is,xout,NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,xin,xout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);
  if (viewVecs) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Mapped vector:\n");CHKERRQ(ierr);
    ierr = VecView(xout,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&xout);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  {
    PetscInt          rstart,i,*nzd,*nzo,nzl,nzmax = 0,*ncols,nrow,j;
    Mat               J;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscScalar       *nvals;

    ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
    ierr = PetscCalloc2(2*m,&nzd,2*m,&nzo);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = MatGetRow(B,i+rstart,&nzl,&cols,NULL);CHKERRQ(ierr);
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
      ierr  = MatRestoreRow(B,i+rstart,&nzl,&cols,NULL);CHKERRQ(ierr);
    }
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,2*m,2*m,PETSC_DECIDE,PETSC_DECIDE,0,nzd,0,nzo,&J);CHKERRQ(ierr);
    ierr = PetscInfo(0,"Created empty Jacobian matrix\n");CHKERRQ(ierr);
    ierr = PetscFree2(nzd,nzo);CHKERRQ(ierr);
    ierr = PetscMalloc2(nzmax,&ncols,nzmax,&nvals);CHKERRQ(ierr);
    ierr = PetscArrayzero(nvals,nzmax);CHKERRQ(ierr);
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
    if (viewMats) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Jacobian matrix structure:\n");CHKERRQ(ierr);
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      requires: parmetis datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -nox -f ${DATAFILESPATH}/matrices/arco1 -mat_partitioning_type parmetis -viewer_binary_skip_info -novec_load

   test:
      requires: parmetis !complex double !define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex73_1.out
      suffix: parmetis_nd_32
      nsize: 3
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -mat_partitioning_type parmetis -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: parmetis !complex double define(PETSC_USE_64BIT_INDICES)
      output_file: output/ex73_1.out
      suffix: parmetis_nd_64
      nsize: 3
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int64-float64 -mat_partitioning_type parmetis -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: ptscotch !complex double !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex73_1.out
      suffix: ptscotch_nd_32
      nsize: 4
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -mat_partitioning_type ptscotch -viewer_binary_skip_info -use_nd -novec_load

   test:
      requires: ptscotch !complex double define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex73_1.out
      suffix: ptscotch_nd_64
      nsize: 4
      args: -nox -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int64-float64 -mat_partitioning_type ptscotch -viewer_binary_skip_info -use_nd -novec_load

TEST*/
