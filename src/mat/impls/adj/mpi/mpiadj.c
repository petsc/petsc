/*$Id: mpiadj.c,v 1.37 2000/04/09 03:10:00 bsmith Exp bsmith $*/

/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure.
*/
#include "sys.h"
#include "src/mat/impls/adj/mpi/mpiadj.h"

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatView_MPIAdj_ASCII"
extern int MatView_MPIAdj_ASCII(Mat A,Viewer viewer)
{
  Mat_MPIAdj  *a = (Mat_MPIAdj*)A->data;
  int         ierr,i,j,m = a->m, format;
  char        *outputname;

  PetscFunctionBegin;
  ierr = ViewerGetOutputname(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else {
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = ViewerASCIISynchronizedPrintf(viewer,"row %d:",i+a->rstart);CHKERRQ(ierr);
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        ierr = ViewerASCIISynchronizedPrintf(viewer," %d ",a->j[j]);CHKERRQ(ierr);
      }
      ierr = ViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } 
  ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatView_MPIAdj"
int MatView_MPIAdj(Mat A,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MatView_MPIAdj_ASCII(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by MPIAdj",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatDestroy_MPIAdj"
int MatDestroy_MPIAdj(Mat mat)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)mat->data;
  int        ierr;

  PetscFunctionBegin;

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping);CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping);CHKERRQ(ierr);
  }
  if (mat->rmap) {
    ierr = MapDestroy(mat->rmap);CHKERRQ(ierr);
  }
  if (mat->cmap) {
    ierr = MapDestroy(mat->cmap);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d, NZ=%d",mat->m,mat->n,a->nz);
#endif
  if (a->diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  if (a->values) {ierr = PetscFree(a->values);CHKERRQ(ierr);}
  ierr = PetscFree(a->i);CHKERRQ(ierr);
  ierr = PetscFree(a->j);CHKERRQ(ierr);
  ierr = PetscFree(a->rowners);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);

  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatSetOption_MPIAdj"
int MatSetOption_MPIAdj(Mat A,MatOption op)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;

  PetscFunctionBegin;
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
#define  __FUNC__ /*<a name=""></a>*/"MatMarkDiagonal_MPIAdj"
int MatMarkDiagonal_MPIAdj(Mat A)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data; 
  int        i,j,*diag,m = a->m;

  PetscFunctionBegin;
  diag = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for (i=0; i<a->m; i++) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
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
#define  __FUNC__ /*<a name=""></a>*/"MatGetSize_MPIAdj"
int MatGetSize_MPIAdj(Mat A,int *m,int *n)
{
  PetscFunctionBegin;
  if (m) *m = A->M;
  if (n) *n = A->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatGetSize_MPIAdj"
int MatGetLocalSize_MPIAdj(Mat A,int *m,int *n)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data; 
  PetscFunctionBegin;
  if (m) *m = a->m; 
  if (n) *n = A->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatGetOwnershipRange_MPIAdj"
int MatGetOwnershipRange_MPIAdj(Mat A,int *m,int *n)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;
  PetscFunctionBegin;
  if (m) *m = a->rstart;
  if (n) *n = a->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatGetRow_MPIAdj"
int MatGetRow_MPIAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAdj *a = (Mat_MPIAdj*)A->data;
  int        *itmp;

  PetscFunctionBegin;
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
#define  __FUNC__ /*<a name=""></a>*/"MatRestoreRow_MPIAdj"
int MatRestoreRow_MPIAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatGetBlockSize_MPIAdj"
int MatGetBlockSize_MPIAdj(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatEqual_MPIAdj"
int MatEqual_MPIAdj(Mat A,Mat B,PetscTruth* flg)
{
  Mat_MPIAdj *a = (Mat_MPIAdj *)A->data,*b = (Mat_MPIAdj *)B->data;
  int         ierr;
  PetscTruth  flag;

  PetscFunctionBegin;
  if (B->type != MATMPIADJ) SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");

  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((a->m != b->m) ||(a->nz != b->nz)) {
    flag = PETSC_FALSE;
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(a->m+1)*sizeof(int),&flag);CHKERRQ(ierr);
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int),&flag);CHKERRQ(ierr);

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
#define  __FUNC__ /*<a name=""></a>*/"MatCreateMPIAdj"
/*@C
   MatCreateMPIAdj - Creates a sparse matrix representing an adjacency list.
   The matrix does not have numerical values associated with it, but is
   intended for ordering (to reduce bandwidth etc) and partitioning.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows
.  n - number of columns
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row).
       The indices in i and j start with zero (NOT with one).
-  values -[optional] edge weights

   Output Parameter:
.  A - the matrix 

   Level: intermediate

   Notes: This matrix object does not support most matrix operations, include
   MatSetValues().
   You must NOT free the ii, values and jj arrays yourself. PETSc will free them
   when the matrix is destroyed. And you must allocate them with PetscMalloc()

   Possible values for MatSetOption() - MAT_STRUCTURALLY_SYMMETRIC

.seealso: MatCreate(), MatCreateSeqAdj(), MatGetOrdering()
@*/
int MatCreateMPIAdj(MPI_Comm comm,int m,int n,int *i,int *j,int *values,Mat *A)
{
  Mat        B;
  Mat_MPIAdj *b;
  int        ii,ierr,size,rank;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  *A                  = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIADJ,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data             = (void*)(b = PetscNew(Mat_MPIAdj));CHKPTRQ(b);
  ierr                = PetscMemzero(b,sizeof(Mat_MPIAdj));CHKERRQ(ierr);
  ierr                = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy     = MatDestroy_MPIAdj;
  B->ops->view        = MatView_MPIAdj;
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
  /* we don't know the "local columns" so just use the row information :-(*/
  ierr = MapCreateMPI(comm,m,B->M,&B->cmap);CHKERRQ(ierr);

  b->rowners = (int*)PetscMalloc((size+1)*sizeof(int));CHKPTRQ(b->rowners);
  PLogObjectMemory(B,(size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAdj));
  ierr = MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for (ii=2; ii<=size; ii++) {
    b->rowners[ii] += b->rowners[ii-1];
  }
  b->rstart = b->rowners[rank]; 
  b->rend   = b->rowners[rank+1]; 

#if defined(PETSC_BOPT_g)
  if (i[0] != 0) SETERR1(1,1,"First i[] index must be zero, instead it is %d\n",i[0]);
  for (ii=1; ii<m; ii++) {
    if (i[ii] < 0 || i[ii] > i[ii-1]) {
      SETERRQ4(1,1,"i[%d] index is out of range: i[%d]",ii,i[ii],ii-1,i[ii-1]);
    }
  }
  for (ii=0; ii<i[m]; ii++) {
    if (i[ii] < 0 || i[ii] >= N) {
      SETERRQ2(1,1,"Column index %d out of range %d\n",ii,i[ii]);
    }
  } 
#endif

  b->j  = j;
  b->i  = i;
  b->values = values;

  b->nz               = i[m];
  b->diag             = 0;
  b->symmetric        = PETSC_FALSE;

  *A = B;

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatConvert_MPIAdj"
int MatConvert_MPIAdj(Mat A,MatType type,Mat *B)
{
  int      i,ierr,m,N,nzeros = 0,*ia,*ja,*rj,len,rstart,cnt,j,*a;
  Scalar   *ra;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,PETSC_NULL);CHKERRQ(ierr);
  
  /* count the number of nonzeros per row */
  for (i=0; i<m; i++) {
    ierr   = MatGetRow(A,i+rstart,&len,&rj,PETSC_NULL);CHKERRQ(ierr);
    for (j=0; j<len; j++) {
      if (rj[j] == i+rstart) {len--; break;}    /* don't count diagonal */
    }
    ierr   = MatRestoreRow(A,i+rstart,&len,&rj,PETSC_NULL);CHKERRQ(ierr);
    nzeros += len;
  }

  /* malloc space for nonzeros */
  a  = (int*)PetscMalloc((nzeros+1)*sizeof(int));CHKPTRQ(a);
  ia = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(ia);
  ja = (int*)PetscMalloc((nzeros+1)*sizeof(int));CHKPTRQ(ja);

  nzeros = 0;
  ia[0]  = 0;
  for (i=0; i<m; i++) {
    ierr    = MatGetRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    cnt     = 0;
    for (j=0; j<len; j++) {
      if (rj[j] != i+rstart) { /* if not diagonal */
        a[nzeros+cnt]    = (int) PetscAbsScalar(ra[j]);
        ja[nzeros+cnt++] = rj[j];
      } 
    }
    ierr    = MatRestoreRow(A,i+rstart,&len,&rj,&ra);CHKERRQ(ierr);
    nzeros += cnt;
    ia[i+1] = nzeros; 
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreateMPIAdj(comm,m,N,ia,ja,a,B);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}






