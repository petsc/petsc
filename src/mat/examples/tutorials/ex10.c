
static char help[] = "Reads a PETSc matrix and computes the 2 norm of the columns\n\n"; 

/*T
   Concepts: Mat^loading a binary matrix;
   Processors: n
T*/

/* 
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h    - vectors
     petscsys.h    - system routines       petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers               
*/
#include "petscmat.h"

extern PetscErrorCode MatGetColumnNorms(Mat,PetscReal *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A;                /* matrix */
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscReal             *norms;
  PetscInt              n,cstart,cend;
  PetscTruth            flg;

  PetscInitialize(&argc,&args,(char *)0,help);


  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate binary file with the -f option");

  /* 
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
    Load the matrix; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&cstart,&cend);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&norms);CHKERRQ(ierr);
  ierr = MatGetColumnNorms(A,norms);CHKERRQ(ierr);
  ierr = PetscRealView(cend-cstart,norms+cstart,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(norms);CHKERRQ(ierr);

  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

#include "../src/mat/impls/aij/seq/aij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_SeqAIJ"
PetscErrorCode MatGetColumnNorms_SeqAIJ(Mat A,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,m,n;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscMemzero(norms,n*sizeof(PetscReal));CHKERRQ(ierr);  
  for (i=0; i<aij->i[m]; i++) {
    norms[aij->j[i]] += PetscAbsScalar(aij->a[i]*aij->a[i]);
  }
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/aij/mpi/mpiaij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_MPIAIJ"
PetscErrorCode MatGetColumnNorms_MPIAIJ(Mat A,PetscReal *norms)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)A->data;
  PetscInt       i,n,*garray = aij->garray;
  Mat_SeqAIJ     *a_aij = (Mat_SeqAIJ*) aij->A->data;
  Mat_SeqAIJ     *b_aij = (Mat_SeqAIJ*) aij->B->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMemzero(work,n*sizeof(PetscReal));CHKERRQ(ierr);  
  for (i=0; i<a_aij->i[aij->A->rmap->n]; i++) {
    work[A->rmap->rstart + a_aij->j[i]] += PetscAbsScalar(a_aij->a[i]*a_aij->a[i]);
  }
  for (i=0; i<b_aij->i[aij->B->rmap->n]; i++) {
    work[garray[b_aij->j[i]]] += PetscAbsScalar(b_aij->a[i]*b_aij->a[i]);
  }
  ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_SUM,A->hdr.comm);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_SeqDense"
PetscErrorCode MatGetColumnNorms_SeqDense(Mat A,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m,n;
  PetscScalar    *a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscMemzero(norms,n*sizeof(PetscReal));CHKERRQ(ierr);  
  ierr = MatGetArray(A,&a);CHKERRQ(ierr);

  for (i=0; i<n; i++ ){ 
    for (j=0; j<m; j++) {
      norms[i] += PetscAbsScalar(a[j]*a[j]);
    }
    a += m;
  }
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/dense/mpi/mpidense.h"
#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_MPIDense"
PetscErrorCode MatGetColumnNorms_MPIDense(Mat A,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = MatGetColumnNorms_SeqDense(a->A,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPI_SUM,A->hdr.comm);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms"
/*
    MatGetColumnNorms - Gets the 2 norms of each column of a sparse or dense matrix.

  Input Parameter:
.  A - the matrix

  Output Parameter:
.  norms - an array as large as the TOTAL number of columns in the matrix

   Notes: Each process has ALL the column norms after the call.
*/
PetscErrorCode MatGetColumnNorms(Mat A,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscInt       i,n;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatGetColumnNorms_SeqAIJ(A,norms);CHKERRQ(ierr);
  } else {
    ierr = PetscTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatGetColumnNorms_SeqDense(A,norms);CHKERRQ(ierr);
    } else {
      ierr = PetscTypeCompare((PetscObject)A,MATMPIDENSE,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = MatGetColumnNorms_MPIDense(A,norms);CHKERRQ(ierr);
      } else {
        ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
        if (flg) {
          ierr = MatGetColumnNorms_MPIAIJ(A,norms);CHKERRQ(ierr);
        } else SETERRQ(PETSC_ERR_SUP,"Not coded for this matrix type");
      }
    } 
  }
  for (i=0; i<n; i++) norms[i] = sqrt(norms[i]);
  PetscFunctionReturn(0);
}
  
