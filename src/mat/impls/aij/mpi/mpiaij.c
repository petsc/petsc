#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiaij.c,v 1.301 1999/09/27 21:29:44 bsmith Exp bsmith $";
#endif

#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"

extern int MatSetUpMultiply_MPIAIJ(Mat);
extern int DisAssemble_MPIAIJ(Mat);
extern int MatSetValues_SeqAIJ(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatGetRow_SeqAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqAIJ(Mat,int,int*,int**,Scalar**);
extern int MatPrintHelp_SeqAIJ(Mat);

/* local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  This is done in a non scable way since the 
length of colmap equals the global matrix length. 
*/
#undef __FUNC__  
#define __FUNC__ "CreateColmap_MPIAIJ_Private"
int CreateColmap_MPIAIJ_Private(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *B = (Mat_SeqAIJ*) aij->B->data;
  int        n = B->n,i,ierr;

  PetscFunctionBegin;
#if defined (PETSC_USE_CTABLE)
  ierr = TableCreate(aij->n/5,&aij->colmap);CHKERRQ(ierr); 
  for ( i=0; i<n; i++ ){
    ierr = TableAdd(aij->colmap,aij->garray[i]+1,i+1);CHKERRQ(ierr);
  }
#else
  aij->colmap = (int *) PetscMalloc((aij->N+1)*sizeof(int));CHKPTRQ(aij->colmap);
  PLogObjectMemory(mat,aij->N*sizeof(int));
  ierr = PetscMemzero(aij->colmap,aij->N*sizeof(int));CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) aij->colmap[aij->garray[i]] = i+1;
#endif
  PetscFunctionReturn(0);
}

#define CHUNKSIZE   15
#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv) \
{ \
 \
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift; \
    rmax = aimax[row]; nrow = ailen[row];  \
    col1 = col - shift; \
     \
    low = 0; high = nrow; \
    while (high-low > 5) { \
      t = (low+high)/2; \
      if (rp[t] > col) high = t; \
      else             low  = t; \
    } \
      for ( _i=0; _i<nrow; _i++ ) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto a_noinsert; \
        } \
      }  \
      if (nonew == 1) goto a_noinsert; \
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = ai[a->m] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(a->m+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len );CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];} \
        for ( ii=row+1; ii<a->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - ai[row] - nrow - shift); \
        ierr = PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow, \
                                                           len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow, \
                                                           len*sizeof(Scalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
 \
        ierr = PetscFree(a->a);CHKERRQ(ierr);  \
        if (!a->singlemalloc) { \
           ierr = PetscFree(a->i);CHKERRQ(ierr); \
           ierr = PetscFree(a->j);CHKERRQ(ierr); \
        } \
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = 1; \
 \
        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift; \
        rmax = aimax[row] = aimax[row] + CHUNKSIZE; \
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar))); \
        a->maxnz += CHUNKSIZE; \
        a->reallocs++; \
      } \
      N = nrow++ - 1; a->nz++; \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        ap[ii+1] = ap[ii]; \
      } \
      rp[_i] = col1;  \
      ap[_i] = value;  \
      a_noinsert: ; \
      ailen[row] = nrow; \
} 

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv) \
{ \
 \
    rp   = bj + bi[row] + shift; ap = ba + bi[row] + shift; \
    rmax = bimax[row]; nrow = bilen[row];  \
    col1 = col - shift; \
     \
    low = 0; high = nrow; \
    while (high-low > 5) { \
      t = (low+high)/2; \
      if (rp[t] > col) high = t; \
      else             low  = t; \
    } \
       for ( _i=0; _i<nrow; _i++ ) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto b_noinsert; \
        } \
      }  \
      if (nonew == 1) goto b_noinsert; \
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = bi[b->m] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(b->m+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len );CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = bi[ii];} \
        for ( ii=row+1; ii<b->m+1; ii++ ) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,bj,(bi[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - bi[row] - nrow - shift); \
        ierr = PetscMemcpy(new_j+bi[row]+shift+nrow+CHUNKSIZE,bj+bi[row]+shift+nrow, \
                                                           len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,ba,(bi[row]+nrow+shift)*sizeof(Scalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bi[row]+shift+nrow+CHUNKSIZE,ba+bi[row]+shift+nrow, \
                                                           len*sizeof(Scalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
 \
        ierr = PetscFree(b->a);CHKERRQ(ierr);  \
        if (!b->singlemalloc) { \
          ierr = PetscFree(b->i);CHKERRQ(ierr); \
          ierr = PetscFree(b->j);CHKERRQ(ierr); \
        } \
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = 1; \
 \
        rp   = bj + bi[row] + shift; ap = ba + bi[row] + shift; \
        rmax = bimax[row] = bimax[row] + CHUNKSIZE; \
        PLogObjectMemory(B,CHUNKSIZE*(sizeof(int) + sizeof(Scalar))); \
        b->maxnz += CHUNKSIZE; \
        b->reallocs++; \
      } \
      N = nrow++ - 1; b->nz++; \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        ap[ii+1] = ap[ii]; \
      } \
      rp[_i] = col1;  \
      ap[_i] = value;  \
      b_noinsert: ; \
      bilen[row] = nrow; \
}

#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIAIJ"
int MatSetValues_MPIAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Scalar     value;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;
  int        roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat        A = aij->A;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  int        *aimax = a->imax, *ai = a->i, *ailen = a->ilen,*aj = a->j;
  Scalar     *aa = a->a;

  Mat        B = aij->B;
  Mat_SeqAIJ *b = (Mat_SeqAIJ *) B->data; 
  int        *bimax = b->imax, *bi = b->i, *bilen = b->ilen,*bj = b->j;
  Scalar     *ba = b->a;

  int        *rp,ii,nrow,_i,rmax, N, col1,low,high,t; 
  int        nonew = a->nonew,shift = a->indexshift; 
  Scalar     *ap;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
    if (im[i] >= aij->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
        else if (in[j] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        else if (in[j] >= aij->N) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");}
#endif
        else {
          if (mat->was_assembled) {
            if (!aij->colmap) {
              ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined (PETSC_USE_CTABLE)
            ierr = TableFind(aij->colmap,in[j]+1,&col);CHKERRQ(ierr);
	    col--;
#else
            col = aij->colmap[in[j]] - 1;
#endif
            if (col < 0 && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
              ierr = DisAssemble_MPIAIJ(mat);CHKERRQ(ierr);
              col =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private() */
              B = aij->B;
              b = (Mat_SeqAIJ *) B->data; 
              bimax = b->imax; bi = b->i; bilen = b->ilen; bj = b->j;
              ba = b->a;
            }
          } else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {
      if (!aij->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIAIJ"
int MatGetValues_MPIAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (idxm[i] >= aij->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
        if (idxn[j] >= aij->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatGetValues(aij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!aij->colmap) {
            ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
          }
#if defined (PETSC_USE_CTABLE)
          ierr = TableFind(aij->colmap,idxn[j]+1,&col);CHKERRQ(ierr);
          col --;
#else
          col = aij->colmap[idxn[j]] - 1;
#endif
          if ((col < 0) || (aij->garray[col] != idxn[j])) *(v+i*n+j) = 0.0;
          else {
            ierr = MatGetValues(aij->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          }
        }
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIAIJ"
int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr,nstash,reallocs;
  InsertMode  addv;

  PetscFunctionBegin;
  if (aij->donotstash) {
    PetscFunctionReturn(0);
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,mat->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,aij->rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(aij->A,"MatAssemblyBegin_MPIAIJ:Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIAIJ"
int MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int         i,j,rstart,ncols,n,ierr,flg;
  int         *row,*col,other_disassembled;
  Scalar      *val;
  InsertMode  addv = mat->insertmode;

  PetscFunctionBegin;
  if (!aij->donotstash) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for ( i=0; i<n; ) {
        /* Now identify the consecutive vals belonging to the same row */
        for ( j=i,rstart=row[j]; j<n; j++ ) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPIAIJ(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  }
 
  ierr = MatAssemblyBegin(aij->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->A,mode);CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqAIJ*) aij->B->data)->nonew)  {
    ierr = MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);CHKERRQ(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = DisAssemble_MPIAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIAIJ(mat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(aij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->B,mode);CHKERRQ(ierr);

  if (aij->rowvalues) {
    ierr = PetscFree(aij->rowvalues);CHKERRQ(ierr);
    aij->rowvalues = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIAIJ"
int MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIAIJ"
int MatZeroRows_MPIAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work,row;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values,rstart=l->rstart;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) );CHKPTRQ(nprocs);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  procs  = nprocs + size;
  owner  = (int *) PetscMalloc((N+1)*sizeof(int));CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) );CHKPTRQ(work);
  ierr = MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nrecvs = work[rank]; 
  ierr = MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  nmax = work[rank];
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) );CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) );CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<N; i++ ) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ISRestoreIndices(is,&rows);

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) );CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  lrows = (int *) PetscMalloc( (slen+1)*sizeof(int) );CHKPTRQ(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);

  /*
        Zero the required rows. If the "diagonal block" of the matrix
     is square and the user wishes to set the diagonal we use seperate
     code so that MatSetValues() is not called for each diagonal allocating
     new memory, thus calling lots of mallocs and slowing things down.

       Contributed by: Mathew Knepley
  */
  /* must zero l->B before l->A because the (diag) case below may put values into l->B*/
  ierr = MatZeroRows(l->B,istmp,0);CHKERRQ(ierr); 
  if (diag && (l->A->M == l->A->N)) {
    ierr      = MatZeroRows(l->A,istmp,diag);CHKERRQ(ierr);
  } else if (diag) {
    ierr = MatZeroRows(l->A,istmp,0);CHKERRQ(ierr);
    if (((Mat_SeqAIJ*)l->A->data)->nonew) {
      SETERRQ(PETSC_ERR_SUP,0,"MatZeroRows() on rectangular matrices cannot be used with the Mat options\n\
MAT_NO_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    }
    for ( i = 0; i < slen; i++ ) {
      row  = lrows[i] + rstart;
      ierr = MatSetValues(A,1,&row,1,&row,diag,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows(l->A,istmp,0);CHKERRQ(ierr);
  }
  ierr = ISDestroy(istmp);CHKERRQ(ierr);
  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_MPIAIJ"
int MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr,nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != a->n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"Incompatible partition of A and xx");
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIAIJ"
int MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIAIJ"
int MatMultTrans_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtrans)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtrans)(a->A,xx,yy);CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIAIJ"
int MatMultTransAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtrans)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIAIJ"
int MatGetDiagonal_MPIAIJ(Mat A,Vec v)
{
  int        ierr;
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;

  PetscFunctionBegin;
  if (a->M != a->N) SETERRQ(PETSC_ERR_SUP,0,"Supports only square matrix where A->A is diag block");
  if (a->rstart != a->cstart || a->rend != a->cend) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"row partition must equal col partition");  
  }
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatScale_MPIAIJ"
int MatScale_MPIAIJ(Scalar *aa,Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatScale(aa,a->A);CHKERRQ(ierr);
  ierr = MatScale(aa,a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIAIJ"
int MatDestroy_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
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
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",aij->M,aij->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = PetscFree(aij->rowners);CHKERRQ(ierr);
  ierr = MatDestroy(aij->A);CHKERRQ(ierr);
  ierr = MatDestroy(aij->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
  if (aij->colmap) TableDelete(aij->colmap);
#else
  if (aij->colmap) {ierr = PetscFree(aij->colmap);CHKERRQ(ierr);}
#endif
  if (aij->garray) {ierr = PetscFree(aij->garray);CHKERRQ(ierr);}
  if (aij->lvec)   VecDestroy(aij->lvec);
  if (aij->Mvctx)  VecScatterDestroy(aij->Mvctx);
  if (aij->rowvalues) {ierr = PetscFree(aij->rowvalues);CHKERRQ(ierr);}
  ierr = PetscFree(aij);CHKERRQ(ierr);
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ_Binary"
int MatView_MPIAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr;

  PetscFunctionBegin;
  if (aij->size == 1) {
    ierr = MatView(aij->A,viewer);CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_ERR_SUP,0,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ_ASCIIorDraworSocket"
int MatView_MPIAIJ_ASCIIorDraworSocket(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ* C = (Mat_SeqAIJ*)aij->A->data;
  int         ierr, format,shift = C->indexshift,rank = aij->rank ;
  int         size = aij->size;
  FILE        *fd;

  PetscFunctionBegin;
  if (PetscTypeCompare(viewer,ASCII_VIEWER)) { 
    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      MatInfo info;
      int     flg;
      ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
      ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg);CHKERRQ(ierr);
      PetscSequentialPhaseBegin(mat->comm,1);
      if (flg) fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, not using I-node routines\n",
         rank,aij->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);
      else fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, using I-node routines\n",
         rank,aij->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      fprintf(fd,"[%d] on-diagonal part: nz %d \n",rank,(int)info.nz_used);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      fprintf(fd,"[%d] off-diagonal part: nz %d \n",rank,(int)info.nz_used); 
      fflush(fd);
      PetscSequentialPhaseEnd(mat->comm,1);
      ierr = VecScatterView(aij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == VIEWER_FORMAT_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (PetscTypeCompare(viewer,DRAW_VIEWER)) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = MatView(aij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqAIJ *Aloc;
    int         M = aij->M, N = aij->N,m,*ai,*aj,row,*cols,i,*ct;
    Scalar      *a;

    if (!rank) {
      ierr = MatCreateMPIAIJ(mat->comm,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreateMPIAIJ(mat->comm,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
    }
    PLogObjectParent(mat,A);

    /* copy over the A part */
    Aloc = (Mat_SeqAIJ*) aij->A->data;
    m = Aloc->m; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row = aij->rstart;
    for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += aij->cstart + shift;}
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERT_VALUES);CHKERRQ(ierr);
      row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
    } 
    aj = Aloc->j;
    for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= aij->cstart + shift;}

    /* copy over the B part */
    Aloc = (Mat_SeqAIJ*) aij->B->data;
    m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row = aij->rstart;
    ct = cols = (int *) PetscMalloc( (ai[m]+1)*sizeof(int) );CHKPTRQ(cols);
    for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = aij->garray[aj[i]+shift];}
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERT_VALUES);CHKERRQ(ierr);
      row++; a += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
    } 
    ierr = PetscFree(ct);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* 
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the Draw object
    */
    if (!rank || PetscTypeCompare(viewer,DRAW_VIEWER)) {
      ierr = MatView(((Mat_MPIAIJ*)(A->data))->A,viewer);CHKERRQ(ierr);
    }
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIAIJ"
int MatView_MPIAIJ(Mat mat,Viewer viewer)
{
  int         ierr;
 
  PetscFunctionBegin;
  if (PetscTypeCompare(viewer,ASCII_VIEWER) || PetscTypeCompare(viewer,DRAW_VIEWER) || 
      PetscTypeCompare(viewer,SOCKET_VIEWER) || PetscTypeCompare(viewer,BINARY_VIEWER)) { 
    ierr = MatView_MPIAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
    /*
  } else if (PetscTypeCompare(viewer,BINARY_VIEWER)) {
    ierr = MatView_MPIAIJ_Binary(mat,viewer);CHKERRQ(ierr);
    */
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

/*
    This has to provide several versions.

     2) a) use only local smoothing updating outer values only once.
        b) local smoothing updating outer values each inner iteration
     3) color updating out values betwen colors.
*/
#undef __FUNC__  
#define __FUNC__ "MatRelax_MPIAIJ"
int MatRelax_MPIAIJ(Mat matin,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        AA = mat->A, BB = mat->B;
  Mat_SeqAIJ *A = (Mat_SeqAIJ *) AA->data, *B = (Mat_SeqAIJ *)BB->data;
  Scalar     *b,*x,*xs,*ls,d,*v,sum;
  int        ierr,*idx, *diag;
  int        n = mat->n, m = mat->m, i,shift = A->indexshift;

  PetscFunctionBegin;
  if (!A->diag) {ierr = MatMarkDiag_SeqAIJ(AA);CHKERRQ(ierr);}
  diag = A->diag;
  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
    if (xx != bb) {
      ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
    } else {
      b = x;
    }
    ierr = VecGetArray(mat->lvec,&ls);CHKERRQ(ierr);
    xs = x + shift; /* shift by one for index start of 1 */
    ls = ls + shift;
    while (its--) {
      /* go down through the rows */
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
	PLogFlops(4*n+3);
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
      /* come up through the rows */
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
	PLogFlops(4*n+3)
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    }    
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); }
    ierr = VecRestoreArray(mat->lvec,&ls);CHKERRQ(ierr);
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
    if (xx != bb) {
      ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
    } else {
      b = x;
    }
    ierr = VecGetArray(mat->lvec,&ls);CHKERRQ(ierr);
    xs = x + shift; /* shift by one for index start of 1 */
    ls = ls + shift;
    while (its--) {
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
	PLogFlops(4*n+3);
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    } 
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); }
    ierr = VecRestoreArray(mat->lvec,&ls);CHKERRQ(ierr);
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,its,xx);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
    if (xx != bb) {
      ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
    } else {
      b = x;
    }
    ierr = VecGetArray(mat->lvec,&ls);CHKERRQ(ierr);
    xs = x + shift; /* shift by one for index start of 1 */
    ls = ls + shift;
    while (its--) {
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
	PLogFlops(4*n+3);
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + A->a[diag[i]+shift]*x[i])/d;
      }
    } 
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); }
    ierr = VecRestoreArray(mat->lvec,&ls);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Parallel SOR not supported");
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIAIJ"
int MatGetInfo_MPIAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        A = mat->A, B = mat->B;
  int        ierr;
  double     isend[5], irecv[5];

  PetscFunctionBegin;
  info->block_size     = 1.0;
  ierr = MatGetInfo(A,MAT_LOCAL,info);CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  ierr = MatGetInfo(B,MAT_LOCAL,info);CHKERRQ(ierr);
  isend[0] += info->nz_used; isend[1] += info->nz_allocated; isend[2] += info->nz_unneeded;
  isend[3] += info->memory;  isend[4] += info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  info->rows_global       = (double)mat->M;
  info->columns_global    = (double)mat->N;
  info->rows_local        = (double)mat->m;
  info->columns_local     = (double)mat->N;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIAIJ"
int MatSetOption_MPIAIJ(Mat A,MatOption op)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_UNSORTED ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERR ||
      op == MAT_NEW_NONZERO_LOCATION_ERR) {
        ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
        ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_ROW_ORIENTED) {
    a->roworiented = 1; 
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"MatSetOption_MPIAIJ:Option ignored\n");
  else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = 0;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_IGNORE_OFF_PROC_ENTRIES) {
    a->donotstash = 1;
  } else if (op == MAT_NO_NEW_DIAGONALS){
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIAIJ"
int MatGetSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->M;
  if (n) *n = mat->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIAIJ"
int MatGetLocalSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->m;
  if (n) *n = mat->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIAIJ"
int MatGetOwnershipRange_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;

  PetscFunctionBegin;
  *m = mat->rstart; *n = mat->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIAIJ"
int MatGetRow_MPIAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB,*v_p;
  int        i, ierr, *cworkA, *cworkB, **pcA, **pcB, cstart = mat->cstart;
  int        nztot, nzA, nzB, lrow, rstart = mat->rstart, rend = mat->rend;
  int        *cmap, *idx_p;

  PetscFunctionBegin;
  if (mat->getrowactive == PETSC_TRUE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ *) mat->A->data,*Ba = (Mat_SeqAIJ *) mat->B->data; 
    int     max = 1,m = mat->m,tmp;
    for ( i=0; i<m; i++ ) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues  = (Scalar *) PetscMalloc(max*(sizeof(int)+sizeof(Scalar)));CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int *) (mat->rowvalues + max);
  }

  if (row < rstart || row >= rend) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Only local rows")
  lrow = row - rstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = (*mat->A->ops->getrow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr = (*mat->B->ops->getrow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for ( i=0; i<nzB; i++ ) {
          if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
        for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for ( i=0; i<imark; i++ ) {
            idx_p[i] = cmap[cworkB[i]];
          }
        } else {
          for ( i=0; i<nzB; i++ ) {
            if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
            else break;
          }
          imark = i;
        }
        for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
        for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
      } 
    } else {
      if (idx) *idx = 0; 
      if (v)   *v   = 0;
    }
  }
  *nz = nztot;
  ierr = (*mat->A->ops->restorerow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr = (*mat->B->ops->restorerow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIAIJ"
int MatRestoreRow_MPIAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;

  PetscFunctionBegin;
  if (aij->getrowactive == PETSC_FALSE) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"MatGetRow not called");
  }
  aij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIAIJ"
int MatNorm_MPIAIJ(Mat mat,NormType type,double *norm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *amat = (Mat_SeqAIJ*) aij->A->data, *bmat = (Mat_SeqAIJ*) aij->B->data;
  int        ierr, i, j, cstart = aij->cstart,shift = amat->indexshift;
  double     sum = 0.0;
  Scalar     *v;

  PetscFunctionBegin;
  if (aij->size == 1) {
    ierr =  MatNorm(aij->A,type,norm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz; i++ ) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz; i++ ) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = sqrt(*norm);
    } else if (type == NORM_1) { /* max column norm */
      double *tmp, *tmp2;
      int    *jj, *garray = aij->garray;
      tmp  = (double *) PetscMalloc( (aij->N+1)*sizeof(double) );CHKPTRQ(tmp);
      tmp2 = (double *) PetscMalloc( (aij->N+1)*sizeof(double) );CHKPTRQ(tmp2);
      ierr = PetscMemzero(tmp,aij->N*sizeof(double));CHKERRQ(ierr);
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for ( j=0; j<amat->nz; j++ ) {
        tmp[cstart + *jj++ + shift] += PetscAbsScalar(*v);  v++;
      }
      v = bmat->a; jj = bmat->j;
      for ( j=0; j<bmat->nz; j++ ) {
        tmp[garray[*jj++ + shift]] += PetscAbsScalar(*v); v++;
      }
      ierr = MPI_Allreduce(tmp,tmp2,aij->N,MPI_DOUBLE,MPI_SUM,mat->comm);CHKERRQ(ierr);
      for ( j=0; j<aij->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscFree(tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp = 0.0;
      for ( j=0; j<amat->m; j++ ) {
        v = amat->a + amat->i[j] + shift;
        sum = 0.0;
        for ( i=0; i<amat->i[j+1]-amat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        v = bmat->a + bmat->i[j] + shift;
        for ( i=0; i<bmat->i[j+1]-bmat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      ierr = MPI_Allreduce(&ntemp,norm,1,MPI_DOUBLE,MPI_MAX,mat->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_MPIAIJ"
int MatTranspose_MPIAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ *Aloc = (Mat_SeqAIJ *) a->A->data;
  int        ierr,shift = Aloc->indexshift;
  int        M = a->M, N = a->N,m,*ai,*aj,row,*cols,i,*ct;
  Mat        B;
  Scalar     *array;

  PetscFunctionBegin;
  if (matout == PETSC_NULL && M != N) {
    SETERRQ(PETSC_ERR_ARG_SIZ,0,"Square matrix only for in-place");
  }

  ierr = MatCreateMPIAIJ(A->comm,a->n,a->m,N,M,0,PETSC_NULL,0,PETSC_NULL,&B);CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_SeqAIJ*) a->A->data;
  m = Aloc->m; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += a->cstart + shift;}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= a->cstart + shift;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*) a->B->data;
  m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  ct = cols = (int *) PetscMalloc( (1+ai[m]-shift)*sizeof(int) );CHKPTRQ(cols);
  for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = a->garray[aj[i]+shift];}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  ierr = PetscFree(ct);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    PetscOps       *Abops;
    struct _MatOps *Aops;

    /* This isn't really an in-place transpose .... but free data structures from a */
    ierr = PetscFree(a->rowners);CHKERRQ(ierr);
    ierr = MatDestroy(a->A);CHKERRQ(ierr);
    ierr = MatDestroy(a->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
    if (a->colmap) TableDelete(a->colmap);
#else
    if (a->colmap) {ierr = PetscFree(a->colmap);CHKERRQ(ierr);}
#endif
    if (a->garray) {ierr = PetscFree(a->garray);CHKERRQ(ierr);}
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    ierr = PetscFree(a);CHKERRQ(ierr);

    /*
       This is horrible, horrible code. We need to keep the 
      A pointers for the bops and ops but copy everything 
      else from C.
    */
    Abops   = A->bops;
    Aops    = A->ops;
    ierr    = PetscMemcpy(A,B,sizeof(struct _p_Mat));CHKERRQ(ierr);
    A->bops = Abops;
    A->ops  = Aops; 
    PetscHeaderDestroy(B);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_MPIAIJ"
int MatDiagonalScale_MPIAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat        a = aij->A, b = aij->B;
  int        ierr,s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_ERR_ARG_SIZ,0,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(rr,aij->lvec,INSERT_VALUES,SCATTER_FORWARD,aij->Mvctx);CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_ERR_ARG_SIZ,0,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,0);CHKERRQ(ierr);
  }
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(rr,aij->lvec,INSERT_VALUES,SCATTER_FORWARD,aij->Mvctx);CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,0,aij->lvec);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_MPIAIJ"
int MatPrintHelp_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*) A->data;
  int        ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqAIJ(a->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIAIJ"
int MatGetBlockSize_MPIAIJ(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "MatSetUnfactored_MPIAIJ"
int MatSetUnfactored_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*) A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatEqual_MPIAIJ"
int MatEqual_MPIAIJ(Mat A, Mat B, PetscTruth *flag)
{
  Mat_MPIAIJ *matB = (Mat_MPIAIJ *) B->data,*matA = (Mat_MPIAIJ *) A->data;
  Mat        a, b, c, d;
  PetscTruth flg;
  int        ierr;

  PetscFunctionBegin;
  if (B->type != MATMPIAIJ) SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a, c, &flg);CHKERRQ(ierr);
  if (flg == PETSC_TRUE) {
    ierr = MatEqual(b, d, &flg);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg, flag, 1, MPI_INT, MPI_LAND, A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCopy_MPIAIJ"
int MatCopy_MPIAIJ(Mat A,Mat B,MatStructure str)
{
  int        ierr;
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  if (str != SAME_NONZERO_PATTERN || B->type != MATMPIAIJ) {
    /* because of the column compression in the off-processor part of the matrix a->B,
       the number of columns in a->B and b->B may be different, hence we cannot call
       the MatCopy() directly on the two parts. If need be, we can provide a more 
       efficient copy than the MatCopy_Basic() by first uncompressing the a->B matrices
       then copying the submatrices */
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
    ierr = MatCopy(a->B,b->B,str);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}

extern int MatDuplicate_MPIAIJ(Mat,MatDuplicateOption,Mat *);
extern int MatIncreaseOverlap_MPIAIJ(Mat , int, IS *, int);
extern int MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
extern int MatGetSubMatrices_MPIAIJ (Mat ,int , IS *,IS *,MatReuse,Mat **);
extern int MatGetSubMatrix_MPIAIJ (Mat ,IS,IS,int,MatReuse,Mat *);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,
       MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,
       MatMultAdd_MPIAIJ,
       MatMultTrans_MPIAIJ,
       MatMultTransAdd_MPIAIJ,
       0,
       0,
       0,
       0,
       0,
       0,
       MatRelax_MPIAIJ,
       MatTranspose_MPIAIJ,
       MatGetInfo_MPIAIJ,
       MatEqual_MPIAIJ,
       MatGetDiagonal_MPIAIJ,
       MatDiagonalScale_MPIAIJ,
       MatNorm_MPIAIJ,
       MatAssemblyBegin_MPIAIJ,
       MatAssemblyEnd_MPIAIJ,
       0,
       MatSetOption_MPIAIJ,
       MatZeroEntries_MPIAIJ,
       MatZeroRows_MPIAIJ,
       0,
       0,
       0,
       0,
       MatGetSize_MPIAIJ,
       MatGetLocalSize_MPIAIJ,
       MatGetOwnershipRange_MPIAIJ,
       0,
       0,
       0,
       0,
       MatDuplicate_MPIAIJ,
       0,
       0,
       0,
       0,
       0,
       MatGetSubMatrices_MPIAIJ,
       MatIncreaseOverlap_MPIAIJ,
       MatGetValues_MPIAIJ,
       MatCopy_MPIAIJ,
       MatPrintHelp_MPIAIJ,
       MatScale_MPIAIJ,
       0,
       0,
       0,
       MatGetBlockSize_MPIAIJ,
       0,
       0,
       0,
       0, 
       MatFDColoringCreate_MPIAIJ,
       0,
       MatSetUnfactored_MPIAIJ,
       0,
       0,
       MatGetSubMatrix_MPIAIJ,
       0,
       0,
       MatGetMaps_Petsc};

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatStoreValues_MPIAIJ"
int MatStoreValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatRetrieveValues_MPIAIJ"
int MatRetrieveValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include "pc.h"
EXTERN_C_BEGIN
extern int MatGetDiagonalBlock_MPIAIJ(Mat,PetscTruth *,MatReuse,Mat *);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatCreateMPIAIJ"
/*@C
   MatCreateMPIAIJ - Creates a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the 
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the 
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the 
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or PETSC_NULL, if d_nz is used to specify the nonzero structure. 
           The size of this array is equal to the number of local rows, i.e 'm'. 
           You must leave room for the diagonal entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or PETSC_NULL, if o_nz is used to specify the nonzero 
           structure. The size of this array is equal to the number 
           of local rows, i.e 'm'. 

   Output Parameter:
.  A - the matrix 

   Notes:
   m,n,M,N parameters specify the size of the matrix, and its partitioning across
   processors, while d_nz,d_nnz,o_nz,o_nnz parameters specify the approximate
   storage requirements for this matrix.

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one 
   processor than it must be used on all processors that share the object for 
   that argument.

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users manual for details.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   The parallel matrix is partitioned such that the first m0 rows belong to 
   process 0, the next m1 rows belong to process 1, the next m2 rows belong 
   to process 2 etc.. where m0,m1,m2... are the input parameter 'm'.

   The DIAGONAL portion of the local submatrix of a processor can be defined 
   as the submatrix which is obtained by extraction the part corresponding 
   to the rows r1-r2 and columns r1-r2 of the global matrix, where r1 is the 
   first row that belongs to the processor, and r2 is the last row belonging 
   to the this processor. This is a square mxm matrix. The remaining portion 
   of the local submatrix (mxN) constitute the OFF-DIAGONAL portion.

   If o_nnz, d_nnz are specified, then o_nz, and d_nz are ignored.

   By default, this format uses inodes (identical nodes) when possible.
   We search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_aij_no_inode  - Do not use inodes
.  -mat_aij_inode_limit <limit> - Sets inode limit (max limit=5)
-  -mat_aij_oneindex - Internally use indexing starting at 1
        rather than 0.  Note that when calling MatSetValues(),
        the user still MUST index entries starting at 0!
.   -mat_mpi - use the parallel matrix data structures even on one processor 
               (defaults to using SeqBAIJ format on one processor)
.   -mat_mpi - use the parallel matrix data structures even on one processor 
               (defaults to using SeqAIJ format on one processor)


   Example usage:
  
   Consider the following 8x8 matrix with 34 non-zero values, that is 
   assembled across 3 processors. Lets assume that proc0 owns 3 rows,
   proc1 owns 3 rows, proc2 owns 2 rows. This division can be shown 
   as follows:

.vb
            1  2  0  |  0  3  0  |  0  4
    Proc0   0  5  6  |  7  0  0  |  8  0
            9  0 10  | 11  0  0  | 12  0
    -------------------------------------
           13  0 14  | 15 16 17  |  0  0
    Proc1   0 18  0  | 19 20 21  |  0  0 
            0  0  0  | 22 23  0  | 24  0
    -------------------------------------
    Proc2  25 26 27  |  0  0 28  | 29  0
           30  0  0  | 31 32 33  |  0 34
.ve

   This can be represented as a collection of submatrices as:

.vb
      A B C
      D E F
      G H I
.ve

   Where the submatrices A,B,C are owned by proc0, D,E,F are
   owned by proc1, G,H,I are owned by proc2.

   The 'm' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
   The 'n' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
   The 'M','N' parameters are 8,8, and have the same values on all procs.

   The DIAGONAL submatrices corresponding to proc0,proc1,proc2 are
   submatrices [A], [E], [I] respectively. The OFF-DIAGONAL submatrices
   corresponding to proc0,proc1,proc2 are [BC], [DF], [GH] respectively.
   Internally, each processor stores the DIAGONAL part, and the OFF-DIAGONAL
   part as SeqAIJ matrices. for eg: proc1 will store [E] as a SeqAIJ
   matrix, ans [DF] as another SeqAIJ matrix.

   When d_nz, o_nz parameters are specified, d_nz storage elements are
   allocated for every row of the local diagonal submatrix, and o_nz
   storage locations are allocated for every row of the OFF-DIAGONAL submat.
   One way to choose d_nz and o_nz is to use the max nonzerors per local 
   rows for each of the local DIAGONAL, and the OFF-DIAGONAL submatrices. 
   In this case, the values of d_nz,o_nz are:
.vb
     proc0 : dnz = 2, o_nz = 2
     proc1 : dnz = 3, o_nz = 2
     proc2 : dnz = 1, o_nz = 4
.ve
   We are allocating m*(d_nz+o_nz) storage locations for every proc. This
   translates to 3*(2+2)=12 for proc0, 3*(3+2)=15 for proc1, 2*(1+4)=10
   for proc3. i.e we are using 12+15+10=37 storage locations to store 
   34 values.

   When d_nnz, o_nnz parameters are specified, the storage is specified
   for every row, coresponding to both DIAGONAL and OFF-DIAGONAL submatrices.
   In the above case the values for d_nnz,o_nnz are:
.vb
     proc0: d_nnz = [2,2,2] and o_nnz = [2,2,2]
     proc1: d_nnz = [3,3,2] and o_nnz = [2,1,1]
     proc2: d_nnz = [1,1]   and o_nnz = [4,4]
.ve
   Here the space allocated is sum of all the above values i.e 34, and
   hence pre-allocation is perfect.

   Level: intermediate

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues()
@*/
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIAIJ   *b;
  int          ierr, i,size,flag1 = 0, flag2 = 0;

  PetscFunctionBegin;

  if (d_nz < -2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"d_nz cannot be less than -2: value %d",d_nz);
  if (o_nz < -2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"o_nz cannot be less than -2: value %d",o_nz);
  if (d_nnz) {
    for (i=0; i<m; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"d_nnz cannot be less than 0: local row %d value %d",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<m; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"o_nnz cannot be less than 0: local row %d value %d",i,o_nnz[i]);
    }
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_mpiaij",&flag1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_mpi",&flag2);CHKERRQ(ierr);
  if (!flag1 && !flag2 && size == 1) {
    if (M == PETSC_DECIDE) M = m;
    if (N == PETSC_DECIDE) N = n;
    ierr = MatCreateSeqAIJ(comm,M,N,d_nz,d_nnz,A);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  *A = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIAIJ,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data         = (void *) (b = PetscNew(Mat_MPIAIJ));CHKPTRQ(b);
  ierr            = PetscMemzero(b,sizeof(Mat_MPIAIJ));CHKERRQ(ierr);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy = MatDestroy_MPIAIJ;
  B->ops->view    = MatView_MPIAIJ;
  B->factor       = 0;
  B->assembled    = PETSC_FALSE;
  B->mapping      = 0;

  B->insertmode      = NOT_SET_VALUES;
  b->size            = size;
  ierr = MPI_Comm_rank(comm,&b->rank);CHKERRQ(ierr);

  if (m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");
  }

  ierr = PetscSplitOwnership(comm,&m,&M);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = MapCreateMPI(comm,m,M,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,n,N,&B->cmap);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int));CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAIJ));
  b->cowners = b->rowners + b->size + 2;
  ierr = MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  ierr = MPI_Allgather(&n,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 

  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,n,d_nz,d_nnz,&b->A);CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m,N,o_nz,o_nnz,&b->B);CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash  = 0;
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used for matrix vector multiply */
  b->lvec      = 0;
  b->Mvctx     = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_MPIAIJ",
                                     (void*)MatStoreValues_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_MPIAIJ",
                                     (void*)MatRetrieveValues_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPIAIJ",
                                     (void*)MatGetDiagonalBlock_MPIAIJ);CHKERRQ(ierr);
  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_MPIAIJ"
int MatDuplicate_MPIAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat        mat;
  Mat_MPIAIJ *a,*oldmat = (Mat_MPIAIJ *) matin->data;
  int        ierr, len=0, flg;

  PetscFunctionBegin;
  *newmat       = 0;
  PetscHeaderCreate(mat,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIAIJ,"Mat",matin->comm,MatDestroy,MatView);
  PLogObjectCreate(mat);
  mat->data         = (void *) (a = PetscNew(Mat_MPIAIJ));CHKPTRQ(a);
  ierr              = PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  mat->ops->destroy = MatDestroy_MPIAIJ;
  mat->ops->view    = MatView_MPIAIJ;
  mat->factor       = matin->factor;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a->m = mat->m   = oldmat->m;
  a->n = mat->n   = oldmat->n;
  a->M = mat->M   = oldmat->M;
  a->N = mat->N   = oldmat->N;

  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->cstart       = oldmat->cstart;
  a->cend         = oldmat->cend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = 0;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;

  a->rowners = (int *) PetscMalloc(2*(a->size+2)*sizeof(int));CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,2*(a->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAIJ));
  a->cowners = a->rowners + a->size + 2;
  ierr       = PetscMemcpy(a->rowners,oldmat->rowners,2*(a->size+2)*sizeof(int));CHKERRQ(ierr);
  ierr       = MatStashCreate_Private(matin->comm,1,&mat->stash);CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = TableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    a->colmap = (int *) PetscMalloc((a->N)*sizeof(int));CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->N)*sizeof(int));
    ierr      = PetscMemcpy(a->colmap,oldmat->colmap,(a->N)*sizeof(int));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;
  if (oldmat->garray) {
    len = ((Mat_SeqAIJ *) (oldmat->B->data))->n;
    a->garray = (int *) PetscMalloc((len+1)*sizeof(int));CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    if (len) { ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));CHKERRQ(ierr); }
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat);CHKERRQ(ierr);
  }
  ierr = FListDuplicate(matin->qlist,&mat->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIAIJ"
int MatLoad_MPIAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          i, nz, ierr, j,rstart, rend, fd;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag,cend,cstart,n;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix in special format on disk, cannot load as MPIAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PetscMalloc((size+2)*sizeof(int));CHKPTRQ(rowners);
  ierr = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PetscMalloc( 2*(rend-rstart)*sizeof(int) );CHKPTRQ(ourlens);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( M*sizeof(int) );CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) );CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);CHKERRQ(ierr);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PetscMalloc( size*sizeof(int) );CHKPTRQ(procsnz);
    ierr    = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]; j< rowners[i+1]; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int *) PetscMalloc( maxnz*sizeof(int) );CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz     = procsnz[0];
    mycols = (int *) PetscMalloc( nz*sizeof(int) );CHKPTRQ(mycols);
    ierr   = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PetscMalloc( nz*sizeof(int) );CHKPTRQ(mycols);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");
  }

  /* determine column ownership if matrix is not square */
  if (N != M) {
    n      = N/size + ((N % size) > rank);
    ierr   = MPI_Scan(&n,&cend,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    cstart = cend - n;
  } else {
    cstart = rstart;
    cend   = rend;
    n      = cend - cstart;
  }

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(int));CHKERRQ(ierr);
  jj = 0;
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<ourlens[i]; j++ ) {
      if (mycols[jj] < cstart || mycols[jj] >= cend) offlens[i]++;
      jj++;
    }
  }

  /* create our matrix */
  for ( i=0; i<m; i++ ) {
    ourlens[i] -= offlens[i];
  }
  ierr = MatCreateMPIAIJ(comm,m,n,M,N,0,ourlens,0,offlens,newmat);CHKERRQ(ierr);
  A = *newmat;
  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) );CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz   = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for ( i=1; i<size; i++ ) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) );CHKPTRQ(vals);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr     = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  ierr = PetscFree(ourlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_MPIAIJ"
/*
    Not great since it makes two copies of the submatrix, first an SeqAIJ 
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatGetSubMatrices_MPIAIJ()
*/
int MatGetSubMatrix_MPIAIJ(Mat mat,IS isrow,IS iscol,int csize,MatReuse call,Mat *newmat)
{
  int        ierr, i, m,n,rstart,row,rend,nz,*cwork,size,rank,j;
  Mat        *local,M, Mreuse;
  Scalar     *vwork,*aa;
  MPI_Comm   comm = mat->comm;
  Mat_SeqAIJ *aij;
  int        *ii, *jj,nlocal,*dlens,*olens,dlen,olen,jend;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject *)&Mreuse);CHKERRQ(ierr);
    if (!Mreuse) SETERRQ(1,1,"Submatrix passed in was not used before, cannot reuse");
    local = &Mreuse;
    ierr  = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&local);CHKERRQ(ierr);
  } else {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    Mreuse = *local;
    ierr = PetscFree(local);CHKERRQ(ierr);
  }

  /* 
      m - number of local rows
      n - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  ierr = MatGetSize(Mreuse,&m,&n);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    aij = (Mat_SeqAIJ *) (Mreuse)->data;
    if (aij->indexshift) SETERRQ(PETSC_ERR_SUP,1,"No support for index shifted matrix");
    ii  = aij->i;
    jj  = aij->j;

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal 
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    if (csize == PETSC_DECIDE) {
      nlocal = n/size + ((n % size) > rank);
    } else {
      nlocal = csize;
    }
    ierr   = MPI_Scan(&nlocal,&rend,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart = rend - nlocal;
    if (rank == size - 1 && rend != n) {
      SETERRQ(1,1,"Local column sizes do not add up to total number of columns");
    }

    /* next, compute all the lengths */
    dlens = (int *) PetscMalloc( (2*m+1)*sizeof(int) );CHKPTRQ(dlens);
    olens = dlens + m;
    for ( i=0; i<m; i++ ) {
      jend = ii[i+1] - ii[i];
      olen = 0;
      dlen = 0;
      for ( j=0; j<jend; j++ ) {
        if ( *jj < rstart || *jj >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }
    ierr = MatCreateMPIAIJ(comm,m,nlocal,PETSC_DECIDE,n,0,dlens,0,olens,&M);CHKERRQ(ierr);
    ierr = PetscFree(dlens);CHKERRQ(ierr);
  } else {
    int ml,nl;

    M = *newmat;
    ierr = MatGetLocalSize(M,&ml,&nl);CHKERRQ(ierr);
    if (ml != m) SETERRQ(PETSC_ERR_ARG_SIZ,1,"Previous matrix must be same size/layout as request");
    ierr = MatZeroEntries(M);CHKERRQ(ierr);
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE; 
    M->assembled     = PETSC_FALSE;
  }
  ierr = MatGetOwnershipRange(M,&rstart,&rend);CHKERRQ(ierr);
  aij = (Mat_SeqAIJ *) (Mreuse)->data;
  if (aij->indexshift) SETERRQ(PETSC_ERR_SUP,1,"No support for index shifted matrix");
  ii  = aij->i;
  jj  = aij->j;
  aa  = aij->a;
  for (i=0; i<m; i++) {
    row   = rstart + i;
    nz    = ii[i+1] - ii[i];
    cwork = jj;     jj += nz;
    vwork = aa;     aa += nz;
    ierr = MatSetValues_MPIAIJ(M,1,&row,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = M;

  /* save submatrix used in processor for next request */
  if (call ==  MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)M,"SubMatrix",(PetscObject)Mreuse);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Mreuse);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
