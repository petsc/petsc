#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiadj.c,v 1.19 1998/12/17 22:10:54 bsmith Exp bsmith $";
#endif

/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure.
*/
#include "sys.h"
#include "src/mat/impls/adj/mpi/mpiadj.h"

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAdj_ASCII"
extern int MatView_MPIAdj_ASCII(Mat A,Viewer viewer)
{
  Mat_MPIAdj  *a = (Mat_MPIAdj *) A->data;
  int         ierr, i,j, m = a->m,  format;
  FILE        *fd;
  char        *outputname;
  MPI_Comm    comm = A->comm;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetOutputname(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else {
    for ( i=0; i<m; i++ ) {
      PetscSynchronizedFPrintf(comm,fd,"row %d:",i+a->rstart);
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
        PetscSynchronizedFPrintf(comm,fd," %d ",a->j[j]);
      }
      PetscSynchronizedFPrintf(comm,fd,"\n");
    }
  } 
  PetscSynchronizedFlush(comm);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAdj"
int MatView_MPIAdj(Mat A,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)){
    ierr = MatView_MPIAdj_ASCII(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIAdj"
int MatDestroy_MPIAdj(Mat mat)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) mat->data;
  int        ierr;

  PetscFunctionBegin;
  if (--mat->refct > 0) PetscFunctionReturn(0);

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping); CHKERRQ(ierr);
  }
  if (mat->rmap) {
    ierr = MapDestroy(mat->rmap);CHKERRQ(ierr);
  }
  if (mat->cmap) {
    ierr = MapDestroy(mat->cmap);CHKERRQ(ierr);
  }

#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d, NZ=%d",mat->m,mat->n,a->nz);
#endif
  if (a->diag) PetscFree(a->diag);
  PetscFree(a->i);
  PetscFree(a->j);
  PetscFree(a->rowners);
  PetscFree(a); 

  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIAdj"
int MatSetOption_MPIAdj(Mat A,MatOption op)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) A->data;

  if (op == MAT_STRUCTURALLY_SYMMETRIC) {
    a->symmetric = PETSC_TRUE;
  } else {
    PLogInfo(A,"MatSetOption_MPIAdj:Option ignored\n");
  }
  PetscFunctionReturn(0);
}


/*
     Adds diagonal pointers to sparse matrix structure.
*/

#undef __FUNC__  
#define __FUNC__ "MatMarkDiag_MPIAdj"
int MatMarkDiag_MPIAdj(Mat A)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) A->data; 
  int        i,j, *diag, m = a->m;

  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<a->m; i++ ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      if (a->j[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
  a->diag = diag;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIAdj"
int MatGetSize_MPIAdj(Mat A,int *m,int *n)
{
  if (m) *m = A->M;
  if (n) *n = A->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIAdj"
int MatGetLocalSize_MPIAdj(Mat A,int *m,int *n)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) A->data; 
  if (m) *m = a->m; 
  if (n) *n = A->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIAdj"
int MatGetOwnershipRange_MPIAdj(Mat A,int *m,int *n)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) A->data;
  *m = a->rstart; *n = a->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIAdj"
int MatGetRow_MPIAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *) A->data;
  int        *itmp;

  row -= a->rstart;

  if (row < 0 || row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = PETSC_NULL;
  if (idx) {
    itmp = a->j + a->i[row];
    if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIAdj"
int MatRestoreRow_MPIAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIAdj"
int MatGetBlockSize_MPIAdj(Mat A, int *bs)
{
  *bs = 1;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatEqual_MPIAdj"
int MatEqual_MPIAdj(Mat A,Mat B, PetscTruth* flg)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *)A->data, *b = (Mat_MPIAdj *)B->data;
 int         flag = 1,ierr;

  if (B->type != MATMPIADJ) SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");

  /* If the  matrix dimensions are not equal, or no of nonzeros */
  if ((a->m != b->m ) ||( a->nz != b->nz)) {
    flag = 0;
  }
  
  /* if the a->i are the same */
  if (PetscMemcmp(a->i,b->i,(a->m+1)*sizeof(int))) { 
    flag = 0;
  }
  
  /* if a->j are the same */
  if (PetscMemcmp(a->j, b->j, (a->nz)*sizeof(int))) { 
    flag = 0;
  }

  ierr = MPI_Allreduce(&flag,flg,1,MPI_INT,MPI_LAND,A->comm);CHKERRQ(ierr);
  

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {0,
       MatGetRow_MPIAdj,
       MatRestoreRow_MPIAdj,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatEqual_MPIAdj,
       0,
       0,
       0,
       0,
       0,
       0,
       MatSetOption_MPIAdj,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetSize_MPIAdj,
       MatGetLocalSize_MPIAdj,
       MatGetOwnershipRange_MPIAdj,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetBlockSize_MPIAdj,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetMaps_Petsc};


#undef __FUNC__  
#define __FUNC__ "MatCreateMPIAdj"
/*@C
   MatCreateMPIAdj - Creates a sparse matrix representing an adjacency list.
   The matrix does not have numerical values associated with it, but is
   intended for ordering (to reduce bandwidth etc) and partitioning.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of local rows
.  n - number of columns
.  i - the indices into j for the start of each row
-  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).

   Output Parameter:
.  A - the matrix 

   Notes: This matrix object does not support most matrix operations, include
   MatSetValues().
   You must NOT free the ii and jj arrays yourself. PETSc will free them
   when the matrix is destroyed.

   Possible values for MatSetOption() - MAT_STRUCTURALLY_SYMMETRIC

.seealso: MatCreate(), MatCreateSeqAdj(), MatGetOrdering()
@*/
int MatCreateMPIAdj(MPI_Comm comm,int m,int n,int *i,int *j, Mat *A)
{
  Mat        B;
  Mat_MPIAdj *b;
  int        ii,ierr, flg,size,rank;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  *A                  = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIADJ,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data             = (void *) (b = PetscNew(Mat_MPIAdj)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIAdj));
  PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));
  B->ops->destroy          = MatDestroy_MPIAdj;
  B->ops->view             = MatView_MPIAdj;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  B->assembled        = PETSC_FALSE;
  
  b->m = m; B->m = m; 
  ierr = MPI_Allreduce(&m,&B->M,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  B->n = n; B->N = n;

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = MapCreateMPI(comm,m,B->M,&B->rmap);CHKERRQ(ierr);
  /* we don't know the "local columns" so just use the row information :-( */
  ierr = MapCreateMPI(comm,m,B->M,&B->cmap);CHKERRQ(ierr);

  b->rowners = (int *) PetscMalloc((size+1)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,(size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAdj));
  ierr = MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for ( ii=2; ii<=size; ii++ ) {
    b->rowners[ii] += b->rowners[ii-1];
  }
  b->rstart = b->rowners[rank]; 
  b->rend   = b->rowners[rank+1]; 

  b->j  = j;
  b->i  = i;

  b->nz               = i[m];
  b->diag             = 0;
  b->symmetric        = PETSC_FALSE;

  *A = B;

  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



