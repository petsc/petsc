
#define USE_FAST_MAT_SET_VALUES

#include <petscsys.h>
#include <petscviewer.h>

#if defined(USE_FAST_MAT_SET_VALUES)
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#define MatSetValues MatSetValues_MPIAIJ
#else 
#include <petscmat.h>
#endif


/*
   Opens a separate file for each process and reads in ITS portion
  of a large parallel matrix. Only requires enough memory to store
  the processes portion of the matrix ONCE.

    petsc-maint@mcs.anl.gov
*/
#undef __FUNCT__  
#define __FUNCT__ "Mat_Parallel_Load"
int Mat_Parallel_Load(MPI_Comm comm,const char *name,Mat *newmat)
{
  Mat            A;
  PetscScalar    *vals;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,j,rstart,rend;
  PetscInt       header[4],M,N,m;
  PetscInt       *ourlens,*offlens,jj,*mycols,maxnz;
  PetscInt       cend,cstart,n,*rowners;
  int            fd1,fd2;
  PetscViewer    viewer1,viewer2;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Open the files; each process opens its own file */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer1,&fd1);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd1,(char *)header,4,PETSC_INT);CHKERRQ(ierr);

  /* open the file twice so that later we can read entries from two different parts of the
     file at the same time. Note that due to file caching this should not impact performance */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer2);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer2,&fd2);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd2,(char *)header,4,PETSC_INT);CHKERRQ(ierr);

  /* error checking on files */
  if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
  ierr = MPI_Allreduce(header+2,&N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  if (N != size*header[2]) SETERRQ(PETSC_COMM_SELF,1,"All files must have matrices with the same number of total columns");
    
  /* number of rows in matrix is sum of rows in all files */
  m = header[1]; N = header[2];
  ierr = MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);

  /* determine rows of matrices owned by each process */
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 
  ierr = PetscFree(rowners);CHKERRQ(ierr);

  /* determine column ownership if matrix is not square */
  if (N != M) {
    n      = N/size + ((N % size) > rank);
    ierr   = MPI_Scan(&n,&cend,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    cstart = cend - n;
  } else {
    cstart = rstart;
    cend   = rend;
    n      = cend - cstart;
  }

  /* read in local row lengths */
  ierr = PetscMalloc(m*sizeof(PetscInt),&ourlens);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&offlens);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd1,ourlens,m,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd2,ourlens,m,PETSC_INT);CHKERRQ(ierr);

  /* determine buffer space needed for column indices of any one row*/
  maxnz = 0;
  for (i=0; i<m; i++) {
    maxnz = PetscMax(maxnz,ourlens[i]);
  }

  /* allocate enough memory to hold a single row of column indices */
  ierr = PetscMalloc(maxnz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = PetscBinaryRead(fd1,mycols,ourlens[i],PETSC_INT);CHKERRQ(ierr);
    for (j=0; j<ourlens[i]; j++) {
      if (mycols[j] < cstart || mycols[j] >= cend) offlens[i]++;
    }
  }

  /* on diagonal entries are all that were not counted as off-diagonal */
  for (i=0; i<m; i++) {
    ourlens[i] -= offlens[i];
  }

  /* create our matrix */
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,0,ourlens,0,offlens);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }
  ierr = PetscFree(offlens);CHKERRQ(ierr);

  /* allocate enough memory to hold a single row of matrix values */
  ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

  /* read in my part of the matrix numerical values and columns 1 row at a time and put in matrix  */
  jj = rstart;
  for (i=0; i<m; i++) {
    ierr = PetscBinaryRead(fd1,vals,ourlens[i],PETSC_SCALAR);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd2,mycols,ourlens[i],PETSC_INT);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&jj,ourlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
    jj++;
  }
  ierr = PetscFree(ourlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = A;
  ierr = PetscViewerDestroy(&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A;
  char           name[1024];
  PetscBool      flg;

  PetscInitialize(&argc,&args,0,0);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",name,1024,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"Must pass in filename with -f option");
  ierr = Mat_Parallel_Load(PETSC_COMM_WORLD,name,&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
