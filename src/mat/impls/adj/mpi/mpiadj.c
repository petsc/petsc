#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: adj.c,v 1.5 1997/08/22 15:14:50 bsmith Exp $";
#endif

/*
    Defines the basic matrix operations for the ADJ adjacency list matrix data-structure.
*/

#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/mat/impls/adj/seq/adj.h"
#include "src/inline/bitarray.h"

extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNC__  
#define __FUNC__ "MatGetRowIJ_SeqAdj"
int MatGetRowIJ_SeqAdj(Mat A,int oshift,PetscTruth symmetric,int *m,int **ia,int **ja,
                           PetscTruth *done)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  int        ierr,i;
 
  *m     = A->m;
  if (!ia) return 0;
  if (symmetric && !a->symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->m,a->i,a->j,0,oshift,ia,ja); CHKERRQ(ierr);
  } else if (oshift == 1) {
    int nz = a->i[a->m] + 1; 
    /* malloc space and  add 1 to i and j indices */
    *ia = (int *) PetscMalloc( (a->m+1)*sizeof(int) ); CHKPTRQ(*ia);
    *ja = (int *) PetscMalloc( (nz+1)*sizeof(int) ); CHKPTRQ(*ja);
    for ( i=0; i<nz; i++ )     (*ja)[i] = a->j[i] + 1;
    for ( i=0; i<a->m+1; i++ ) (*ia)[i] = a->i[i] + 1;
  } else {
    *ia = a->i; *ja = a->j;
  }
  
  return 0; 
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRowIJ_SeqAdj"
int MatRestoreRowIJ_SeqAdj(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,int **ja,
                               PetscTruth *done)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;

  if (!ia) return 0;
  if ((symmetric && !a->symmetric) || oshift == 1) {
    PetscFree(*ia);
    PetscFree(*ja);
  }
  return 0; 
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAdj_Binary"
extern int MatView_SeqAdj_Binary(Mat A,Viewer viewer)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  int        i, fd, *col_lens, ierr;
  Scalar     *values;

  ierr        = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  col_lens    = (int *) PetscMalloc( (4+a->m)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for ( i=0; i<a->m; i++ ) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,BINARY_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  ierr = PetscBinaryWrite(fd,a->j,a->nz,BINARY_INT,0); CHKERRQ(ierr);

  /* store nonzero values */
  values = (Scalar *) PetscMalloc( a->nz*sizeof(Scalar) );CHKPTRQ(values);
  PetscMemzero(values,a->nz*sizeof(Scalar) );
  ierr = PetscBinaryWrite(fd,values,a->nz,BINARY_SCALAR,0); CHKERRQ(ierr);
  PetscFree(values);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAdj_ASCII"
extern int MatView_SeqAdj_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqAdj  *a = (Mat_SeqAdj *) A->data;
  int         ierr, i,j, m = a->m,  format;
  FILE        *fd;
  char        *outputname;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    return 0;
  } 
  else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    fprintf(fd,"%% Size = %d %d \n",m,a->n);
    fprintf(fd,"%% Nonzeros = %d \n",a->nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",a->nz);
    fprintf(fd,"zzz = [\n");

    for (i=0; i<m; i++) {
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
#if defined(PETSC_COMPLEX)
        fprintf(fd,"%d %d  %18.16e + %18.16e i \n",i+1,a->j[j],0.0,0.0);
#else
        fprintf(fd,"%d %d  %18.16e\n", i+1, a->j[j], 0.0);
#endif
      }
    }
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } 
  else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
#if defined(PETSC_COMPLEX)
        fprintf(fd," %d %g + %g i",a->j[j],0.0,0.0);
#else
        fprintf(fd," %d %g ",a->j[j],0.0);
#endif
      }
      fprintf(fd,"\n");
    }
  } 
  else {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
#if defined(PETSC_COMPLEX)
        fprintf(fd," %d %g + %g i",a->j[j],0.0,0.0);
#else
        fprintf(fd," %d %g ",a->j[j],0.0);
#endif
      }
      fprintf(fd,"\n");
    }
  }
  fflush(fd);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAdj_Draw"
extern int MatView_SeqAdj_Draw(Mat A,Viewer viewer)
{
  Mat_SeqAdj  *a = (Mat_SeqAdj *) A->data;
  int         ierr, i,j, m = a->m,color;
  int         format;
  double      xl,yl,xr,yr,w,h,x_l,x_r,y_l,y_r;
  Draw        draw;
  PetscTruth  isnull;

  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(draw); CHKERRQ(ierr);
  ierr = DrawClear(draw); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;

  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */

  if (format != VIEWER_FORMAT_DRAW_CONTOUR) {
    color = DRAW_BLUE;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
        x_l = a->j[j]; x_r = x_l + 1.0;
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
  }
  ierr = DrawPause(draw); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAdj"
int MatView_SeqAdj(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  Mat_SeqAdj  *a = (Mat_SeqAdj*) A->data;
  ViewerType  vtype;
  int         ierr;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    Scalar *values;
    values = (Scalar *) PetscMalloc(a->nz*sizeof(Scalar));CHKPTRQ(values);
    PetscMemzero(values,a->nz*sizeof(Scalar));
    ierr = ViewerMatlabPutSparse_Private(viewer,a->m,a->n,a->nz,values,a->i,a->j);CHKERRQ(ierr);
    PetscFree(values);
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    return MatView_SeqAdj_ASCII(A,viewer);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_SeqAdj_Binary(A,viewer);
  }
  else if (vtype == DRAW_VIEWER) {
    return MatView_SeqAdj_Draw(A,viewer);
  }
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAdj"
int MatDestroy_SeqAdj(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  if (a->diag) PetscFree(a->diag);
  PetscFree(a->i);
  PetscFree(a->j);
  PetscFree(a); 

  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqAdj"
int MatSetOption_SeqAdj(Mat A,MatOption op)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;

  if (op == MAT_STRUCTURALLY_SYMMETRIC) {
    a->symmetric = PETSC_TRUE;
  } else {
    PLogInfo(A,"Info:MatSetOption_SeqAdj:Option ignored\n");
  }
  return 0;
}


/*
     Adds diagonal pointers to sparse matrix structure.
*/

#undef __FUNC__  
#define __FUNC__ "MatMarkDiag_SeqAdj"
int MatMarkDiag_SeqAdj(Mat A)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data; 
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
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqAdj"
int MatGetInfo_SeqAdj(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;

  info->rows_global    = (double)a->m;
  info->columns_global = (double)a->n;
  info->rows_local     = (double)a->m;
  info->columns_local  = (double)a->n;
  info->block_size     = 1.0;
  info->nz_allocated   = (double)a->nz;
  info->nz_used        = (double)a->nz;
  info->nz_unneeded    = 0.0;
  /*  if (info->nz_unneeded != A->info.nz_unneeded) 
    printf("space descrepancy: maxnz-nz = %d, nz_unneeded = %d\n",(int)info->nz_unneeded,(int)A->info.nz_unneeded); */
  info->assemblies     = 0.0;
  info->mallocs        = 0.0;
  info->memory         = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqAdj"
int MatGetSize_SeqAdj(Mat A,int *m,int *n)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  *m = a->m; *n = a->n;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqAdj"
int MatGetOwnershipRange_SeqAdj(Mat A,int *m,int *n)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  *m = 0; *n = a->m;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqAdj"
int MatGetRow_SeqAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  int        *itmp;

  if (row < 0 || row >= a->m) SETERRQ(1,0,"Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = PETSC_NULL;
  if (idx) {
    itmp = a->j + a->i[row];
    if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqAdj"
int MatRestoreRow_SeqAdj(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqAdj"
int MatGetBlockSize_SeqAdj(Mat A, int *bs)
{
  *bs = 1;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatIncreaseOverlap_SeqAdj"
int MatIncreaseOverlap_SeqAdj(Mat A, int is_max, IS *is, int ov)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  int        row, i,j,k,l,m,n, *idx,ierr, *nidx, isz, val;
  int        start, end, *ai, *aj;
  char       *table;

  m     = a->m;
  ai    = a->i;
  aj    = a->j;

  if (ov < 0)  SETERRQ(1,0,"illegal overlap value used");

  table = (char *) PetscMalloc((m/BITSPERBYTE +1)*sizeof(char)); CHKPTRQ(table); 
  nidx  = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(nidx); 

  for ( i=0; i<is_max; i++ ) {
    /* Initialize the two local arrays */
    isz  = 0;
    PetscMemzero(table,(m/BITSPERBYTE +1)*sizeof(char));
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);  CHKERRQ(ierr);
    
    /* Enter these into the temp arrays. I.e., mark table[row], enter row into new index */
    for ( j=0; j<n ; ++j){
      if(!BT_LOOKUP(table, idx[j])) { nidx[isz++] = idx[j];}
    }
    ierr = ISRestoreIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
    
    k = 0;
    for ( j=0; j<ov; j++){ /* for each overlap */
      n = isz;
      for ( ; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for ( l = start; l<end ; l++){
          val = aj[l];
          if (!BT_LOOKUP(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, isz, nidx, (is+i)); CHKERRQ(ierr);
  }
  PetscFree(table);
  PetscFree(nidx);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatEqual_SeqAdj"
int MatEqual_SeqAdj(Mat A,Mat B, PetscTruth* flg)
{
  Mat_SeqAdj *a = (Mat_SeqAdj *)A->data, *b = (Mat_SeqAdj *)B->data;

  if (B->type != MATSEQADJ) SETERRQ(1,0,"Matrices must be same type");

  /* If the  matrix dimensions are not equal, or no of nonzeros */
  if ((a->m != b->m ) || (a->n !=b->n) ||( a->nz != b->nz)) {
    *flg = PETSC_FALSE; return 0; 
  }
  
  /* if the a->i are the same */
  if (PetscMemcmp(a->i,b->i,(a->m+1)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  /* if a->j are the same */
  if (PetscMemcmp(a->j, b->j, (a->nz)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  *flg = PETSC_TRUE; 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatPermute_SeqAdj"
int MatPermute_SeqAdj(Mat A, IS rowp, IS colp, Mat *B)
{ 
  Mat_SeqAdj *a = (Mat_SeqAdj *) A->data;
  Scalar     *vwork;
  int        i, ierr, nz = a->nz, m = a->m, n = a->n, *cwork,*ii,*jj;
  int        *row,*col,j,*lens;
  IS         icolp,irowp;

  ierr = ISInvertPermutation(rowp,&irowp); CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&row); CHKERRQ(ierr);
  ierr = ISInvertPermutation(colp,&icolp); CHKERRQ(ierr);
  ierr = ISGetIndices(icolp,&col); CHKERRQ(ierr);
  
  /* determine lengths of permuted rows */
  lens = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(lens);
  for (i=0; i<m; i++ ) {
    lens[row[i]] = a->i[i+1] - a->i[i];
  }

  ii = (int *) PetscMalloc((m+1)*sizeof(int));CHKPTRQ(ii);
  jj = (int *) PetscMalloc((nz+1)*sizeof(int));CHKPTRQ(jj);
  ii[0] = 0;
  for (i=1; i<=m; i++ ) {
    ii[i] = ii[i-1] + lens[i-1];
  }
  PetscFree(lens);

  for (i=0; i<m; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    for (j=0; j<nz; j++ ) { jj[j+ii[row[i]]] = col[cwork[j]];}
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }

  ierr = MatCreateSeqAdj(A->comm,m,n,ii,jj,B);CHKERRQ(ierr);

  ierr = ISRestoreIndices(irowp,&row); CHKERRQ(ierr);
  ierr = ISRestoreIndices(icolp,&col); CHKERRQ(ierr);
  ierr = ISDestroy(irowp); CHKERRQ(ierr);
  ierr = ISDestroy(icolp); CHKERRQ(ierr);
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {0,
       MatGetRow_SeqAdj,MatRestoreRow_SeqAdj,
       0,0,
       0,0,
       0,0,
       0,0,
       0,0,
       0,
       0,
       MatGetInfo_SeqAdj,MatEqual_SeqAdj,
       0,0,0,
       0,0,
       0,
       MatSetOption_SeqAdj,0,0,
       0,0,0,0,
       MatGetSize_SeqAdj,MatGetSize_SeqAdj,MatGetOwnershipRange_SeqAdj,
       0,0,
       0,0,
       0,0,0,
       0,0,0,
       0,MatIncreaseOverlap_SeqAdj,
       0,0,
       0,
       0,0,0,
       0,
       MatGetBlockSize_SeqAdj,
       MatGetRowIJ_SeqAdj,
       MatRestoreRowIJ_SeqAdj,
       0,
       0,
       0,
       0,
       0,
       MatPermute_SeqAdj};


#undef __FUNC__  
#define __FUNC__ "MatCreateSeqAdj"
/*@C
   MatCreateSeqAdj - Creates a sparse matrix representing an adjacency list.
     The matrix does not have numerical values associated with it, but is
     intended for ordering (to reduce bandwidth etc) and partitioning.

   Input Parameters:
.  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  i - the indices into j for the start of each row
.  j - the column indices for each row (sorted for each row)
       The indices in i and j start with zero NOT one.

   Output Parameter:
.  A - the matrix 

   Notes: You must free the ii and jj arrays yourself. PETSc will free them
   when the matrix is destroyed.

.  MatSetOptions() possible values - MAT_STRUCTURALLY_SYMMETRIC

.seealso: MatCreate(), MatCreateMPIADJ(), MatGetReordering()
@*/
int MatCreateSeqAdj(MPI_Comm comm,int m,int n,int *i,int *j, Mat *A)
{
  Mat        B;
  Mat_SeqAdj *b;
  int        ierr, flg,size;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,0,"Comm must be of size 1");

  *A                  = 0;
  PetscHeaderCreate(B,_p_Mat,MAT_COOKIE,MATSEQADJ,comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data             = (void *) (b = PetscNew(Mat_SeqAdj)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqAdj));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy          = MatDestroy_SeqAdj;
  B->view             = MatView_SeqAdj;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  B->assembled        = PETSC_FALSE;
  
  b->m = m; B->m = m; B->M = m;
  b->n = n; B->n = n; B->N = n;

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
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqAdj"
int MatLoad_SeqAdj(Viewer viewer,MatType type,Mat *A)
{
  int          i, nz, ierr, fd, header[4],size,*rowlengths = 0,M,N,*ii,*jj;
  MPI_Comm     comm;
  
  PetscObjectGetComm((PetscObject) viewer,&comm);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,BINARY_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,0,"not matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];

  /* read in row lengths */
  rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);

  /* create our matrix */
  ii = (int *) PetscMalloc( (M+1)*sizeof(int) );CHKPTRQ(ii);
  jj = (int *) PetscMalloc( (nz+1)*sizeof(int) ); CHKPTRQ(jj);

  /* read in column indices and adjust for Fortran indexing*/
  ierr = PetscBinaryRead(fd,jj,nz,BINARY_INT); CHKERRQ(ierr);

  /* set matrix "i" values */
  ii[0] = 0;
  for ( i=1; i<= M; i++ ) {
    ii[i]      = ii[i-1] + rowlengths[i-1];
  }
  PetscFree(rowlengths);   

  ierr = MatCreateSeqAdj(comm,M,N,ii,jj,A); CHKERRQ(ierr);
  return 0;
}


