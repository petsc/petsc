
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpibaij.c,v 1.77 1997/08/07 14:40:00 bsmith Exp balay $";
#endif

#include "pinclude/pviewer.h"
#include "src/mat/impls/baij/mpi/mpibaij.h"
#include "src/vec/vecimpl.h"


extern int MatSetUpMultiply_MPIBAIJ(Mat); 
extern int DisAssemble_MPIBAIJ(Mat);
extern int MatIncreaseOverlap_MPIBAIJ(Mat,int,IS *,int);
extern int MatGetSubMatrices_MPIBAIJ(Mat,int,IS *,IS *,MatGetSubMatrixCall,Mat **);

/* 
     Local utility routine that creates a mapping from the global column 
   number to the local number in the off-diagonal part of the local 
   storage of the matrix.  This is done in a non scable way since the 
   length of colmap equals the global matrix length. 
*/
#undef __FUNC__  
#define __FUNC__ "CreateColmap_MPIBAIJ_Private"
static int CreateColmap_MPIBAIJ_Private(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *B = (Mat_SeqBAIJ*) baij->B->data;
  int         nbs = B->nbs,i,bs=B->bs;;

  baij->colmap = (int *) PetscMalloc((baij->Nbs+1)*sizeof(int));CHKPTRQ(baij->colmap);
  PLogObjectMemory(mat,baij->Nbs*sizeof(int));
  PetscMemzero(baij->colmap,baij->Nbs*sizeof(int));
  for ( i=0; i<nbs; i++ ) baij->colmap[baij->garray[i]] = i*bs+1;
  return 0;
}

#define CHUNKSIZE  10

#define  MatSetValues_SeqBAIJ_A_Private(row,col,value,addv) \
{ \
 \
    brow = row/bs;  \
    rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
    rmax = aimax[brow]; nrow = ailen[brow]; \
      bcol = col/bs; \
      ridx = row % bs; cidx = col % bs; \
      low = 0; high = nrow; \
      while (high-low > 3) { \
        t = (low+high)/2; \
        if (rp[t] > bcol) high = t; \
        else              low  = t; \
      } \
      for ( _i=low; _i<high; _i++ ) { \
        if (rp[_i] > bcol) break; \
        if (rp[_i] == bcol) { \
          bap  = ap +  bs2*_i + bs*cidx + ridx; \
          if (addv == ADD_VALUES) *bap += value;  \
          else                    *bap  = value;  \
          goto a_noinsert; \
        } \
      } \
      if (a->nonew == 1) goto a_noinsert; \
      else if (a->nonew == -1) SETERRQ(1,0,"Inserting a new nonzero in the matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (a->nonew == -2) SETERRQ(1,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+bs2*sizeof(Scalar))+(a->mbs+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + bs2*new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<brow+1; ii++ ) {new_i[ii] = ai[ii];} \
        for ( ii=brow+1; ii<a->mbs+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        PetscMemcpy(new_j,aj,(ai[brow]+nrow)*sizeof(int)); \
        len = (new_nz - CHUNKSIZE - ai[brow] - nrow); \
        PetscMemcpy(new_j+ai[brow]+nrow+CHUNKSIZE,aj+ai[brow]+nrow, \
                                                           len*sizeof(int)); \
        PetscMemcpy(new_a,aa,(ai[brow]+nrow)*bs2*sizeof(Scalar)); \
        PetscMemzero(new_a+bs2*(ai[brow]+nrow),bs2*CHUNKSIZE*sizeof(Scalar)); \
        PetscMemcpy(new_a+bs2*(ai[brow]+nrow+CHUNKSIZE), \
                    aa+bs2*(ai[brow]+nrow),bs2*len*sizeof(Scalar));  \
        /* free up old matrix storage */ \
        PetscFree(a->a);  \
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);} \
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = 1; \
 \
        rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
        rmax = aimax[brow] = aimax[brow] + CHUNKSIZE; \
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(Scalar))); \
        a->maxnz += bs2*CHUNKSIZE; \
        a->reallocs++; \
        a->nz++; \
      } \
      N = nrow++ - 1;  \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(Scalar)); \
      } \
      if (N>=_i) PetscMemzero(ap+bs2*_i,bs2*sizeof(Scalar));  \
      rp[_i]                      = bcol;  \
      ap[bs2*_i + bs*cidx + ridx] = value;  \
      a_noinsert:; \
    ailen[brow] = nrow; \
} 

#define  MatSetValues_SeqBAIJ_B_Private(row,col,value,addv) \
{ \
 \
    brow = row/bs;  \
    rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
    rmax = bimax[brow]; nrow = bilen[brow]; \
      bcol = col/bs; \
      ridx = row % bs; cidx = col % bs; \
      low = 0; high = nrow; \
      while (high-low > 3) { \
        t = (low+high)/2; \
        if (rp[t] > bcol) high = t; \
        else              low  = t; \
      } \
      for ( _i=low; _i<high; _i++ ) { \
        if (rp[_i] > bcol) break; \
        if (rp[_i] == bcol) { \
          bap  = ap +  bs2*_i + bs*cidx + ridx; \
          if (addv == ADD_VALUES) *bap += value;  \
          else                    *bap  = value;  \
          goto b_noinsert; \
        } \
      } \
      if (b->nonew == 1) goto b_noinsert; \
      else if (b->nonew == -1) SETERRQ(1,0,"Inserting a new nonzero in the matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = bi[b->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        Scalar *new_a; \
 \
        if (b->nonew == -2) SETERRQ(1,0,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+bs2*sizeof(Scalar))+(b->mbs+1)*sizeof(int); \
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a); \
        new_j   = (int *) (new_a + bs2*new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for ( ii=0; ii<brow+1; ii++ ) {new_i[ii] = bi[ii];} \
        for ( ii=brow+1; ii<b->mbs+1; ii++ ) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        PetscMemcpy(new_j,bj,(bi[brow]+nrow)*sizeof(int)); \
        len = (new_nz - CHUNKSIZE - bi[brow] - nrow); \
        PetscMemcpy(new_j+bi[brow]+nrow+CHUNKSIZE,bj+bi[brow]+nrow, \
                                                           len*sizeof(int)); \
        PetscMemcpy(new_a,ba,(bi[brow]+nrow)*bs2*sizeof(Scalar)); \
        PetscMemzero(new_a+bs2*(bi[brow]+nrow),bs2*CHUNKSIZE*sizeof(Scalar)); \
        PetscMemcpy(new_a+bs2*(bi[brow]+nrow+CHUNKSIZE), \
                    ba+bs2*(bi[brow]+nrow),bs2*len*sizeof(Scalar));  \
        /* free up old matrix storage */ \
        PetscFree(b->a);  \
        if (!b->singlemalloc) {PetscFree(b->i);PetscFree(b->j);} \
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = 1; \
 \
        rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
        rmax = bimax[brow] = bimax[brow] + CHUNKSIZE; \
        PLogObjectMemory(B,CHUNKSIZE*(sizeof(int) + bs2*sizeof(Scalar))); \
        b->maxnz += bs2*CHUNKSIZE; \
        b->reallocs++; \
        b->nz++; \
      } \
      N = nrow++ - 1;  \
      /* shift up all the later entries in this row */ \
      for ( ii=N; ii>=_i; ii-- ) { \
        rp[ii+1] = rp[ii]; \
        PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(Scalar)); \
      } \
      if (N>=_i) PetscMemzero(ap+bs2*_i,bs2*sizeof(Scalar));  \
      rp[_i]                      = bcol;  \
      ap[bs2*_i + bs*cidx + ridx] = value;  \
      b_noinsert:; \
    bilen[brow] = nrow; \
} 

#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIBAIJ"
int MatSetValues_MPIBAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Scalar      value;
  int         ierr,i,j,row,col;
  int         roworiented = baij->roworiented,rstart_orig=baij->rstart_bs ;
  int         rend_orig=baij->rend_bs,cstart_orig=baij->cstart_bs;
  int         cend_orig=baij->cend_bs,bs=baij->bs;

  /* Some Variables required in the macro */
  Mat         A = baij->A;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) (A)->data; 
  int         *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j; 
  Scalar      *aa=a->a;

  Mat         B = baij->B;
  Mat_SeqBAIJ *b = (Mat_SeqBAIJ *) (B)->data; 
  int         *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j; 
  Scalar      *ba=b->a;

  int         *rp,ii,nrow,_i,rmax,N,brow,bcol; 
  int         low,high,t,ridx,cidx,bs2=a->bs2; 
  Scalar      *ap,*bap;

  for ( i=0; i<m; i++ ) {
#if defined(PETSC_BOPT_g)
    if (im[i] < 0) SETERRQ(1,0,"Negative row");
    if (im[i] >= baij->M) SETERRQ(1,0,"Row too large");
#endif
    if (im[i] >= rstart_orig && im[i] < rend_orig) {
      row = im[i] - rstart_orig;
      for ( j=0; j<n; j++ ) {
        if (in[j] >= cstart_orig && in[j] < cend_orig){
          col = in[j] - cstart_orig;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqBAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
#if defined(PETSC_BOPT_g)
        else if (in[j] < 0) {SETERRQ(1,0,"Negative column");}
        else if (in[j] >= baij->N) {SETERRQ(1,0,"Col too large");}
#endif
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }
            col = baij->colmap[in[j]/bs] - 1 + in[j]%bs;
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr); 
              col =  in[j]; 
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B = baij->B;
              b = (Mat_SeqBAIJ *) (B)->data; 
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j; 
              ba=b->a;
            }
          }
          else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqBAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } 
    else {
      if (roworiented && !baij->donotstash) {
        ierr = StashValues_Private(&baij->stash,im[i],n,in,v+i*n,addv);CHKERRQ(ierr);
      }
      else {
        if (!baij->donotstash) {
          row = im[i];
	  for ( j=0; j<n; j++ ) {
	    ierr = StashValues_Private(&baij->stash,row,1,in+j,v+i+j*m,addv);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  return 0;
}

extern int MatSetValuesBlocked_SeqBAIJ(Mat,int,int*,int,int*,Scalar*,InsertMode);
#undef __FUNC__  
#define __FUNC__ "MatSetValuesBlocked_MPIBAIJ"
int MatSetValuesBlocked_MPIBAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Scalar      *value,*barray=baij->barray;
  int         ierr,i,j,ii,jj,row,col,k,l;
  int         roworiented = baij->roworiented,rstart=baij->rstart ;
  int         rend=baij->rend,cstart=baij->cstart,stepval;
  int         cend=baij->cend,bs=baij->bs,bs2=baij->bs2;

  
  if(!barray) {
    baij->barray = barray = (Scalar*) PetscMalloc(bs2*sizeof(Scalar)); CHKPTRQ(barray);
  }

  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for ( i=0; i<m; i++ ) {
#if defined(PETSC_BOPT_g)
    if (im[i] < 0) SETERRQ(1,0,"Negative row");
    if (im[i] >= baij->Mbs) SETERRQ(1,0,"Row too large");
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for ( j=0; j<n; j++ ) {
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = v + i*bs2;
        } else if((!roworiented) && (m == 1)) {
          barray = v + j*bs2;
        } else { /* Here a copy is required */
          if (roworiented) { 
            value = v + i*(stepval+bs)*bs + j*bs;
          } else {
            value = v + j*(stepval+bs)*bs + i*bs;
          }
          for ( ii=0; ii<bs; ii++,value+=stepval ) {
            for (jj=0; jj<bs; jj++ ) {
              *barray++  = *value++; 
            }
          }
          barray -=bs2;
        }
          
        if (in[j] >= cstart && in[j] < cend){
          col  = in[j] - cstart;
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->A,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
        }
#if defined(PETSC_BOPT_g)
        else if (in[j] < 0) {SETERRQ(1,0,"Negative column");}
        else if (in[j] >= baij->Nbs) {SETERRQ(1,0,"Column too large");}
#endif
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_BOPT_g)
            if ((baij->colmap[in[j]] - 1) % bs) {SETERRQ(1,0,"Incorrect colmap");}
#endif
            col = (baij->colmap[in[j]] - 1)/bs;
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr); 
              col =  in[j];              
            }
          }
          else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->B,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
        }
      }
    } 
    else {
      if (!baij->donotstash) {
        if (roworiented ) {
          row   = im[i]*bs;
          value = v + i*(stepval+bs)*bs;
          for ( j=0; j<bs; j++,row++ ) {
            for ( k=0; k<n; k++ ) {
              for ( col=in[k]*bs,l=0; l<bs; l++,col++) {
                ierr = StashValues_Private(&baij->stash,row,1,&col,value++,addv);CHKERRQ(ierr);
              }
            }
          }
        }
        else {
          for ( j=0; j<n; j++ ) {
            value = v + j*(stepval+bs)*bs + i*bs;
            col   = in[j]*bs;
            for ( k=0; k<bs; k++,col++,value+=stepval) {
              for ( row = im[i]*bs,l=0; l<bs; l++,row++) {
                ierr = StashValues_Private(&baij->stash,row,1,&col,value++,addv);CHKERRQ(ierr);
              }
            }
          }
        }
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIBAIJ"
int MatGetValues_MPIBAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int        bs=baij->bs,ierr,i,j, bsrstart = baij->rstart*bs, bsrend = baij->rend*bs;
  int        bscstart = baij->cstart*bs, bscend = baij->cend*bs,row,col;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,0,"Negative row");
    if (idxm[i] >= baij->M) SETERRQ(1,0,"Row too large");
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,0,"Negative column");
        if (idxn[j] >= baij->N) SETERRQ(1,0,"Col too large");
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues(baij->A,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
        }
        else {
          if (!baij->colmap) {
            ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
          } 
          if((baij->colmap[idxn[j]/bs]-1 < 0) || 
             (baij->garray[(baij->colmap[idxn[j]/bs]-1)/bs] != idxn[j]/bs)) *(v+i*n+j) = 0.0;
          else {
            col  = (baij->colmap[idxn[j]/bs]-1) + idxn[j]%bs;
            ierr = MatGetValues(baij->B,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
          } 
        }
      }
    } 
    else {
      SETERRQ(1,0,"Only local values currently supported");
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIBAIJ"
int MatNorm_MPIBAIJ(Mat mat,NormType type,double *norm)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *amat = (Mat_SeqBAIJ*) baij->A->data, *bmat = (Mat_SeqBAIJ*) baij->B->data;
  int        ierr, i,bs2=baij->bs2;
  double     sum = 0.0;
  Scalar     *v;

  if (baij->size == 1) {
    ierr =  MatNorm(baij->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz*bs2; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz*bs2; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);
      *norm = sqrt(*norm);
    }
    else
      SETERRQ(PETSC_ERR_SUP,0,"No support for this norm yet");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIBAIJ"
int MatAssemblyBegin_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         size = baij->size, *owners = baij->rowners,bs=baij->bs;
  int         rank = baij->rank,tag = mat->tag, *owner,*starts,count,ierr;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(1,0,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (baij->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<baij->stash.n; i++ ) {
    idx = baij->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j]*bs && idx < owners[j+1]*bs) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.


       This could be done better.
  */
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc(3*(baij->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<baij->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  baij->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  baij->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  baij->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(mat,"MatAssemblyBegin_MPIBAIJ:Number of off-processor values %d\n",baij->stash.n);
  ierr = StashDestroy_Private(&baij->stash); CHKERRQ(ierr);

  baij->svalues    = svalues;    baij->rvalues    = rvalues;
  baij->nsends     = nsends;     baij->nrecvs     = nreceives;
  baij->send_waits = send_waits; baij->recv_waits = recv_waits;
  baij->rmax       = nmax;

  return 0;
}
#include <math.h>
#define HASH_KEY 0.6180339887
#define HASH1(size,key) ((int)((size)*fmod(((key)*HASH_KEY),1))+1)

int CreateHashTable(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat         A = baij->A, B=baij->B;
  Mat_SeqBAIJ *a=(Mat_SeqBAIJ *)A->data, *b=(Mat_SeqBAIJ *)B->data;
  int         i,j,k,nz=a->nz+b->nz,h1,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  int         size=(int)(1.5*nz),ct=0,max=0;
  /* Scalar      *aa=a->a,*ba=b->a; */
  double      key;
  static double *HT;
  static      int flag=1;

  
  /* Allocate Memory for Hash Table */
  if (flag) {
    HT = (double*)PetscMalloc(size*sizeof(double));
    flag = 0;
  }
  PetscMemzero(HT,size*sizeof(double));

  /* Loop Over A */
  for ( i=0; i<a->n; i++ ) {
    for ( j=ai[i]; j<ai[i+1]; j++ ) {
      key = i*baij->n+aj[j]+1;
      h1  = HASH1(size,key);

      for ( k=1; k<size; k++ ){
        if (HT[(h1*k)%size] == 0.0) {
          HT[(h1*k)%size] = key;
          break;
        } else ct++;
      }
      if (k> max) max =k;
    }
  }
   printf("***max1 = %d\n",max);
  /* Loop Over B */
  for ( i=0; i<b->n; i++ ) {
    for ( j=bi[i]; j<bi[i+1]; j++ ) {
      key = i*b->n+bj[j]+1;
      h1  = HASH1(size,key);
      for ( k=1; k<size; k++ ){
        if (HT[(h1*k)%size] == 0.0) {
          HT[(h1*k)%size] = key;
          break;
        } else ct++;
      }
      if (k> max) max =k;
    }
  }

  printf("***max2 = %d\n",max);
  /* Print Summary */
  for ( i=0,key=0.0,j=0; i<size; i++) 
    if (HT[i]) {j++;}

  printf("Size %d Average Buckets %d no of Keys %d\n",size,ct,j);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIBAIJ"
int MatAssemblyEnd_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = baij->nrecvs, count = nrecvs, i, n, ierr;
  int         bs=baij->bs,row,col,other_disassembled,flg;
  Scalar      *values,val;
  InsertMode  addv = mat->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,baij->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = baij->rvalues + 3*imdex*baij->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - baij->rstart*bs;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= baij->cstart*bs && col < baij->cend*bs) {
        col -= baij->cstart*bs;
        ierr = MatSetValues(baij->A,1,&row,1,&col,&val,addv); CHKERRQ(ierr)
      } 
      else {
        if (mat->was_assembled) {
          if (!baij->colmap) {
            ierr = CreateColmap_MPIBAIJ_Private(mat); CHKERRQ(ierr);
          }
          col = (baij->colmap[col/bs]) - 1 + col%bs;
          if (col < 0  && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
            ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr); 
            col = (int) PetscReal(values[3*i+1]);
          }
        }
        ierr = MatSetValues(baij->B,1,&row,1,&col,&val,addv); CHKERRQ(ierr)
      }
    }
    count--;
  }
  PetscFree(baij->recv_waits); PetscFree(baij->rvalues);
 
  /* wait on sends */
  if (baij->nsends) {
    send_status = (MPI_Status *) PetscMalloc(baij->nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(baij->nsends,baij->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(baij->send_waits); PetscFree(baij->svalues);

  ierr = MatAssemblyBegin(baij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->A,mode); CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);
  if (mat->was_assembled && !other_disassembled) {
    ierr = DisAssemble_MPIBAIJ(mat); CHKERRQ(ierr);
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBAIJ(mat); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(baij->B,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-use_hash",&flg); CHKERRQ(ierr);
  if (flg) CreateHashTable(mat);
  if (baij->rowvalues) {PetscFree(baij->rowvalues); baij->rowvalues = 0;}
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIBAIJ_Binary"
static int MatView_MPIBAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  int          ierr;

  if (baij->size == 1) {
    ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,0,"Only uniprocessor output supported");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIBAIJ_ASCIIorDraworMatlab"
static int MatView_MPIBAIJ_ASCIIorDraworMatlab(Mat mat,Viewer viewer)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ *) mat->data;
  int          ierr, format,rank,bs = baij->bs;
  FILE         *fd;
  ViewerType   vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILES_VIEWER || vtype == ASCII_FILE_VIEWER) { 
    ierr = ViewerGetFormat(viewer,&format);
    if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      MatInfo info;
      MPI_Comm_rank(mat->comm,&rank);
      ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);
      PetscSequentialPhaseBegin(mat->comm,1);
      fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d bs %d mem %d\n",
              rank,baij->m,(int)info.nz_used*bs,(int)info.nz_allocated*bs,
              baij->bs,(int)info.memory);      
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&info);
      fprintf(fd,"[%d] on-diagonal part: nz %d \n",rank,(int)info.nz_used*bs);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&info); 
      fprintf(fd,"[%d] off-diagonal part: nz %d \n",rank,(int)info.nz_used*bs); 
      fflush(fd);
      PetscSequentialPhaseEnd(mat->comm,1);
      ierr = VecScatterView(baij->Mvctx,viewer); CHKERRQ(ierr);
      return 0; 
    }
    else if (format == VIEWER_FORMAT_ASCII_INFO) {
      PetscPrintf(mat->comm,"  block size is %d\n",bs);
      return 0;
    }
  }

  if (vtype == DRAW_VIEWER) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  }

  if (vtype == ASCII_FILE_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           baij->rank,baij->m,baij->rstart*bs,baij->rend*bs,baij->n,
            baij->cstart*bs,baij->cend*bs);
    ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(baij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  }
  else {
    int size = baij->size;
    rank = baij->rank;
    if (size == 1) {
      ierr = MatView(baij->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat         A;
      Mat_SeqBAIJ *Aloc;
      int         M = baij->M, N = baij->N,*ai,*aj,row,col,i,j,k,*rvals;
      int         mbs=baij->mbs;
      Scalar      *a;

      if (!rank) {
        ierr = MatCreateMPIBAIJ(mat->comm,baij->bs,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
        CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIBAIJ(mat->comm,baij->bs,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
        CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* copy over the A part */
      Aloc = (Mat_SeqBAIJ*) baij->A->data;
      ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = baij->rstart;
      rvals = (int *) PetscMalloc(bs*sizeof(int)); CHKPTRQ(rvals);

      for ( i=0; i<mbs; i++ ) {
        rvals[0] = bs*(baij->rstart + i);
        for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
        for ( j=ai[i]; j<ai[i+1]; j++ ) {
          col = (baij->cstart+aj[j])*bs;
          for (k=0; k<bs; k++ ) {
            ierr = MatSetValues(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
            col++; a += bs;
          }
        }
      } 
      /* copy over the B part */
      Aloc = (Mat_SeqBAIJ*) baij->B->data;
      ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = baij->rstart*bs;
      for ( i=0; i<mbs; i++ ) {
        rvals[0] = bs*(baij->rstart + i);
        for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
        for ( j=ai[i]; j<ai[i+1]; j++ ) {
          col = baij->garray[aj[j]]*bs;
          for (k=0; k<bs; k++ ) { 
            ierr = MatSetValues(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
            col++; a += bs;
          }
        }
      } 
      PetscFree(rvals);
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
        ierr = MatView(((Mat_MPIBAIJ*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}



#undef __FUNC__  
#define __FUNC__ "MatView_MPIBAIJ"
int MatView_MPIBAIJ(PetscObject obj,Viewer viewer)
{
  Mat         mat = (Mat) obj;
  int         ierr;
  ViewerType  vtype;
 
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER ||
      vtype == DRAW_VIEWER       || vtype == MATLAB_VIEWER) { 
    ierr = MatView_MPIBAIJ_ASCIIorDraworMatlab(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_MPIBAIJ_Binary(mat,viewer);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIBAIJ"
int MatDestroy_MPIBAIJ(PetscObject obj)
{
  Mat         mat = (Mat) obj;
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  int         ierr;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",baij->M,baij->N);
#endif

  ierr = StashDestroy_Private(&baij->stash); CHKERRQ(ierr);
  PetscFree(baij->rowners); 
  ierr = MatDestroy(baij->A); CHKERRQ(ierr);
  ierr = MatDestroy(baij->B); CHKERRQ(ierr);
  if (baij->colmap) PetscFree(baij->colmap);
  if (baij->garray) PetscFree(baij->garray);
  if (baij->lvec)   VecDestroy(baij->lvec);
  if (baij->Mvctx)  VecScatterDestroy(baij->Mvctx);
  if (baij->rowvalues) PetscFree(baij->rowvalues);
  if (baij->barray) PetscFree(baij->barray);
  PetscFree(baij); 
  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMult_MPIBAIJ"
int MatMult_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int         ierr, nt;

  VecGetLocalSize_Fast(xx,nt);
  if (nt != a->n) {
    SETERRQ(1,0,"Incompatible partition of A and xx");
  }
  VecGetLocalSize_Fast(yy,nt);
  if (nt != a->m) {
    SETERRQ(1,0,"Incompatible parition of A and yy");
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops.mult)(a->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops.multadd)(a->B,a->lvec,yy,yy); CHKERRQ(ierr);
  ierr = VecScatterPostRecvs(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIBAIJ"
int MatMultAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops.multadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops.multadd)(a->B,a->lvec,zz,zz); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIBAIJ"
int MatMultTrans_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;

  /* do nondiagonal part */
  ierr = (*a->B->ops.multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops.multtrans)(a->A,xx,yy); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIBAIJ"
int MatMultTransAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;

  /* do nondiagonal part */
  ierr = (*a->B->ops.multtrans)(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops.multtransadd)(a->A,xx,yy,zz); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  return 0;
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIBAIJ"
int MatGetDiagonal_MPIBAIJ(Mat A,Vec v)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  if (a->M != a->N) 
    SETERRQ(1,0,"Supports only square matrix where A->A is diag block");
  return MatGetDiagonal(a->A,v);
}

#undef __FUNC__  
#define __FUNC__ "MatScale_MPIBAIJ"
int MatScale_MPIBAIJ(Scalar *aa,Mat A)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  int        ierr;
  ierr = MatScale(aa,a->A); CHKERRQ(ierr);
  ierr = MatScale(aa,a->B); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIBAIJ"
int MatGetSize_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->M; *n = mat->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIBAIJ"
int MatGetLocalSize_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->m; *n = mat->N;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIBAIJ"
int MatGetOwnershipRange_MPIBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  *m = mat->rstart*mat->bs; *n = mat->rend*mat->bs;
  return 0;
}

extern int MatGetRow_SeqBAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqBAIJ(Mat,int,int*,int**,Scalar**);

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIBAIJ"
int MatGetRow_MPIBAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBAIJ *mat = (Mat_MPIBAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB,*v_p;
  int        bs = mat->bs, bs2 = mat->bs2, i, ierr, *cworkA, *cworkB, **pcA, **pcB;
  int        nztot, nzA, nzB, lrow, brstart = mat->rstart*bs, brend = mat->rend*bs;
  int        *cmap, *idx_p,cstart = mat->cstart;

  if (mat->getrowactive == PETSC_TRUE) SETERRQ(1,0,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqBAIJ *Aa = (Mat_SeqBAIJ *) mat->A->data,*Ba = (Mat_SeqBAIJ *) mat->B->data; 
    int     max = 1,mbs = mat->mbs,tmp;
    for ( i=0; i<mbs; i++ ) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues = (Scalar *) PetscMalloc( max*bs2*(sizeof(int)+sizeof(Scalar))); 
    CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int *) (mat->rowvalues + max*bs2);
  }
       

  if (row < brstart || row >= brend) SETERRQ(1,0,"Only local rows")
  lrow = row - brstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = (*mat->A->ops.getrow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops.getrow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for ( i=0; i<nzB; i++ ) {
          if (cmap[cworkB[i]/bs] < cstart)   v_p[i] = vworkB[i];
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
            idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs;
          }
        } else {
          for ( i=0; i<nzB; i++ ) {
            if (cmap[cworkB[i]/bs] < cstart)   
              idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
            else break;
          }
          imark = i;
        }
        for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart*bs + cworkA[i];
        for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
      } 
    } 
    else {
      if (idx) *idx = 0;
      if (v)   *v   = 0;
    }
  }
  *nz = nztot;
  ierr = (*mat->A->ops.restorerow)(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = (*mat->B->ops.restorerow)(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIBAIJ"
int MatRestoreRow_MPIBAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  if (baij->getrowactive == PETSC_FALSE) {
    SETERRQ(1,0,"MatGetRow not called");
  }
  baij->getrowactive = PETSC_FALSE;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIBAIJ"
int MatGetBlockSize_MPIBAIJ(Mat mat,int *bs)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  *bs = baij->bs;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIBAIJ"
int MatZeroEntries_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *l = (Mat_MPIBAIJ *) A->data;
  int         ierr;
  ierr = MatZeroEntries(l->A); CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIBAIJ"
int MatGetInfo_MPIBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) matin->data;
  Mat         A = a->A, B = a->B;
  int         ierr;
  double      isend[5], irecv[5];

  info->rows_global    = (double)a->M;
  info->columns_global = (double)a->N;
  info->rows_local     = (double)a->m;
  info->columns_local  = (double)a->N;
  info->block_size     = (double)a->bs;
  ierr = MatGetInfo(A,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->memory;
  ierr = MatGetInfo(B,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] += info->nz_used; isend[1] += info->nz_allocated; isend[2] += info->memory;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce(isend,irecv,5,MPI_INT,MPI_MAX,matin->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce(isend,irecv,5,MPI_INT,MPI_SUM,matin->comm);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIBAIJ"
int MatSetOption_MPIBAIJ(Mat A,MatOption op)
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *) A->data;
  
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_UNSORTED ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERROR ||
      op == MAT_NEW_NONZERO_LOCATION_ERROR) {
        MatSetOption(a->A,op);
        MatSetOption(a->B,op);
  } else if (op == MAT_ROW_ORIENTED) {
        a->roworiented = 1;
        MatSetOption(a->A,op);
        MatSetOption(a->B,op);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_MPIBAIJ:Option ignored\n");
  else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = 0;
    MatSetOption(a->A,op);
    MatSetOption(a->B,op);
  } else if (op == MAT_IGNORE_OFF_PROC_ENTRIES) {
    a->donotstash = 1;
  } else if (op == MAT_NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,0,"unknown option");}
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_MPIBAIJ("
int MatTranspose_MPIBAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) A->data;
  Mat_SeqBAIJ *Aloc;
  Mat        B;
  int        ierr,M=baij->M,N=baij->N,*ai,*aj,row,i,*rvals,j,k,col;
  int        bs=baij->bs,mbs=baij->mbs;
  Scalar     *a;
  
  if (matout == PETSC_NULL && M != N) 
    SETERRQ(1,0,"Square matrix only for in-place");
  ierr = MatCreateMPIBAIJ(A->comm,baij->bs,PETSC_DECIDE,PETSC_DECIDE,N,M,0,PETSC_NULL,0,PETSC_NULL,&B); 
  CHKERRQ(ierr);
  
  /* copy over the A part */
  Aloc = (Mat_SeqBAIJ*) baij->A->data;
  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
  row = baij->rstart;
  rvals = (int *) PetscMalloc(bs*sizeof(int)); CHKPTRQ(rvals);
  
  for ( i=0; i<mbs; i++ ) {
    rvals[0] = bs*(baij->rstart + i);
    for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
    for ( j=ai[i]; j<ai[i+1]; j++ ) {
      col = (baij->cstart+aj[j])*bs;
      for (k=0; k<bs; k++ ) {
        ierr = MatSetValues(B,1,&col,bs,rvals,a,INSERT_VALUES);CHKERRQ(ierr);
        col++; a += bs;
      }
    }
  } 
  /* copy over the B part */
  Aloc = (Mat_SeqBAIJ*) baij->B->data;
  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
  row = baij->rstart*bs;
  for ( i=0; i<mbs; i++ ) {
    rvals[0] = bs*(baij->rstart + i);
    for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
    for ( j=ai[i]; j<ai[i+1]; j++ ) {
      col = baij->garray[aj[j]]*bs;
      for (k=0; k<bs; k++ ) { 
        ierr = MatSetValues(B,1,&col,bs,rvals,a,INSERT_VALUES);CHKERRQ(ierr);
        col++; a += bs;
      }
    }
  } 
  PetscFree(rvals);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    /* This isn't really an in-place transpose .... but free data structures from baij */
    PetscFree(baij->rowners); 
    ierr = MatDestroy(baij->A); CHKERRQ(ierr);
    ierr = MatDestroy(baij->B); CHKERRQ(ierr);
    if (baij->colmap) PetscFree(baij->colmap);
    if (baij->garray) PetscFree(baij->garray);
    if (baij->lvec) VecDestroy(baij->lvec);
    if (baij->Mvctx) VecScatterDestroy(baij->Mvctx);
    PetscFree(baij); 
    PetscMemcpy(A,B,sizeof(struct _p_Mat)); 
    PetscHeaderDestroy(B);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_MPIBAIJ"
int MatDiagonalScale_MPIBAIJ(Mat A,Vec ll,Vec rr)
{
  Mat a = ((Mat_MPIBAIJ *) A->data)->A;
  Mat b = ((Mat_MPIBAIJ *) A->data)->B;
  int ierr,s1,s2,s3;

  if (ll)  {
    ierr = VecGetLocalSize(ll,&s1); CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&s2,&s3); CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(1,0,"non-conforming local sizes");
    ierr = MatDiagonalScale(a,ll,0); CHKERRQ(ierr);
    ierr = MatDiagonalScale(b,ll,0); CHKERRQ(ierr);
  }
  if (rr) SETERRQ(1,0,"not supported for right vector");
  return 0;
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   baij->A and baij->B directly and not through the MatZeroRows() 
   routine. 
*/
#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIBAIJ"
int MatZeroRows_MPIBAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_MPIBAIJ    *l = (Mat_MPIBAIJ *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values,bs=l->bs;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j]*bs && idx < owners[j+1]*bs) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,0,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
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
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank]*bs;

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  PetscFree(recv_waits); 
  
  /* move the data into the send scatter */
  lrows = (int *) PetscMalloc( (slen+1)*sizeof(int) ); CHKPTRQ(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  PetscFree(rvalues); PetscFree(lens);
  PetscFree(owner); PetscFree(nprocs);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);
  PetscFree(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = MatZeroRows(l->B,istmp,0); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  return 0;
}
extern int MatPrintHelp_SeqBAIJ(Mat);
#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_MPIBAIJ"
int MatPrintHelp_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *a   = (Mat_MPIBAIJ*) A->data;

  if (!a->rank) return MatPrintHelp_SeqBAIJ(a->A);
  else return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatSetUnfactored_MPIBAIJ"
int MatSetUnfactored_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *a   = (Mat_MPIBAIJ*) A->data;
  int         ierr;
  ierr = MatSetUnfactored(a->A); CHKERRQ(ierr);
  return 0;
}

static int MatConvertSameType_MPIBAIJ(Mat,Mat *,int);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {
  MatSetValues_MPIBAIJ,MatGetRow_MPIBAIJ,MatRestoreRow_MPIBAIJ,MatMult_MPIBAIJ,
  MatMultAdd_MPIBAIJ,MatMultTrans_MPIBAIJ,MatMultTransAdd_MPIBAIJ,0,
  0,0,0,0,
  0,0,MatTranspose_MPIBAIJ,MatGetInfo_MPIBAIJ,
  0,MatGetDiagonal_MPIBAIJ,MatDiagonalScale_MPIBAIJ,MatNorm_MPIBAIJ,
  MatAssemblyBegin_MPIBAIJ,MatAssemblyEnd_MPIBAIJ,0,MatSetOption_MPIBAIJ,
  MatZeroEntries_MPIBAIJ,MatZeroRows_MPIBAIJ,0,
  0,0,0,MatGetSize_MPIBAIJ,
  MatGetLocalSize_MPIBAIJ,MatGetOwnershipRange_MPIBAIJ,0,0,
  0,0,MatConvertSameType_MPIBAIJ,0,0,
  0,0,0,MatGetSubMatrices_MPIBAIJ,
  MatIncreaseOverlap_MPIBAIJ,MatGetValues_MPIBAIJ,0,MatPrintHelp_MPIBAIJ,
  MatScale_MPIBAIJ,0,0,0,MatGetBlockSize_MPIBAIJ,
  0,0,0,0,0,0,MatSetUnfactored_MPIBAIJ,0,MatSetValuesBlocked_MPIBAIJ};
                                

#undef __FUNC__  
#define __FUNC__ "MatCreateMPIBAIJ"
/*@C
   MatCreateMPIBAIJ - Creates a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator
.  bs   - size of blockk
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the 
           y vector for the matrix-vector product y = Ax.
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the 
           x vector for the matrix-vector product y = Ax.
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
.  d_nz  - number of block nonzeros per block row in diagonal portion of local 
           submatrix  (same for all local rows)
.  d_nzz - array containing the number of block nonzeros in the various block rows 
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  You must leave room for the diagonal entry even if
           it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
.  o_nzz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

   Output Parameter:
.  A - the matrix 

   Notes:
   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set 
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

$          0 1 2 3 4 5 6 7 8 9 10 11
$         -------------------
$  row 3  |  o o o d d d o o o o o o
$  row 4  |  o o o d d d o o o o o o
$  row 5  |  o o o d d d o o o o o o
$         -------------------
$ 

   Thus, any entries in the d locations are stored in the d (diagonal) 
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues()
@*/
int MatCreateMPIBAIJ(MPI_Comm comm,int bs,int m,int n,int M,int N,
                    int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIBAIJ  *b;
  int          ierr, i,sum[2],work[2],mbs,nbs,Mbs=PETSC_DECIDE,Nbs=PETSC_DECIDE,size;

  if (bs < 1) SETERRQ(1,0,"Invalid block size specified, must be positive");

  MPI_Comm_size(comm,&size);
  if (size == 1) {
    if (M == PETSC_DECIDE) M = m;
    if (N == PETSC_DECIDE) N = n;
    ierr = MatCreateSeqBAIJ(comm,bs,M,N,d_nz,d_nnz,A); CHKERRQ(ierr);
    return 0;
  }

  *A = 0;
  PetscHeaderCreate(B,_p_Mat,MAT_COOKIE,MATMPIBAIJ,comm);
  PLogObjectCreate(B);
  B->data       = (void *) (b = PetscNew(Mat_MPIBAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIBAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));

  B->destroy    = MatDestroy_MPIBAIJ;
  B->view       = MatView_MPIBAIJ;
  B->mapping    = 0;
  B->factor     = 0;
  B->assembled  = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&b->rank);
  MPI_Comm_size(comm,&b->size);

  if ( m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) 
    SETERRQ(1,0,"Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");
  if ( M == PETSC_DECIDE && m == PETSC_DECIDE) SETERRQ(1,0,"either M or m should be specified");
  if ( M == PETSC_DECIDE && n == PETSC_DECIDE)SETERRQ(1,0,"either N or n should be specified"); 
  if ( M != PETSC_DECIDE && m != PETSC_DECIDE) M = PETSC_DECIDE;
  if ( N != PETSC_DECIDE && n != PETSC_DECIDE) N = PETSC_DECIDE;

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    mbs = m/bs; nbs = n/bs;
    MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) {M = sum[0]; Mbs = M/bs;}
    if (N == PETSC_DECIDE) {N = sum[1]; Nbs = N/bs;}
  }
  if (m == PETSC_DECIDE) {
    Mbs = M/bs;
    if (Mbs*bs != M) SETERRQ(1,0,"No of global rows must be divisible by blocksize");
    mbs = Mbs/b->size + ((Mbs % b->size) > b->rank);
    m   = mbs*bs;
  }
  if (n == PETSC_DECIDE) {
    Nbs = N/bs;
    if (Nbs*bs != N) SETERRQ(1,0,"No of global cols must be divisible by blocksize");
    nbs = Nbs/b->size + ((Nbs % b->size) > b->rank);
    n   = nbs*bs;
  }
  if (mbs*bs != m || nbs*bs != n) SETERRQ(1,0,"No of local rows, cols must be divisible by blocksize");

  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;
  b->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = nbs;
  b->Mbs = Mbs;
  b->Nbs = Nbs;

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIBAIJ));
  b->cowners = b->rowners + b->size + 2;
  MPI_Allgather(&mbs,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart    = b->rowners[b->rank]; 
  b->rend      = b->rowners[b->rank+1]; 
  b->rstart_bs = b->rstart * bs;
  b->rend_bs   = b->rend * bs;

  MPI_Allgather(&nbs,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart    = b->cowners[b->rank]; 
  b->cend      = b->cowners[b->rank+1]; 
  b->cstart_bs = b->cstart * bs;
  b->cend_bs   = b->cend * bs;
  
  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,m,n,d_nz,d_nnz,&b->A); CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,m,N,o_nz,o_nnz,&b->B); CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);
  b->donotstash  = 0;
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used in block assembly */
  b->barray      = 0;

  /* stuff used for matrix vector multiply */
  b->lvec        = 0;
  b->Mvctx       = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  *A = B;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatConvertSameType_MPIBAIJ"
static int MatConvertSameType_MPIBAIJ(Mat matin,Mat *newmat,int cpvalues)
{
  Mat         mat;
  Mat_MPIBAIJ *a,*oldmat = (Mat_MPIBAIJ *) matin->data;
  int         ierr, len=0, flg;

  *newmat       = 0;
  PetscHeaderCreate(mat,_p_Mat,MAT_COOKIE,MATMPIBAIJ,matin->comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PetscNew(Mat_MPIBAIJ)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIBAIJ;
  mat->view       = MatView_MPIBAIJ;
  mat->factor     = matin->factor;
  mat->assembled  = PETSC_TRUE;

  a->m = mat->m   = oldmat->m;
  a->n = mat->n   = oldmat->n;
  a->M = mat->M   = oldmat->M;
  a->N = mat->N   = oldmat->N;

  a->bs  = oldmat->bs;
  a->bs2 = oldmat->bs2;
  a->mbs = oldmat->mbs;
  a->nbs = oldmat->nbs;
  a->Mbs = oldmat->Mbs;
  a->Nbs = oldmat->Nbs;
  
  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->cstart       = oldmat->cstart;
  a->cend         = oldmat->cend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  mat->insertmode = NOT_SET_VALUES;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;
  a->barray       = 0;

  a->rowners = (int *) PetscMalloc(2*(a->size+2)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,2*(a->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIBAIJ));
  a->cowners = a->rowners + a->size + 2;
  PetscMemcpy(a->rowners,oldmat->rowners,2*(a->size+2)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  if (oldmat->colmap) {
    a->colmap = (int *) PetscMalloc((a->Nbs)*sizeof(int));CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->Nbs)*sizeof(int));
    PetscMemcpy(a->colmap,oldmat->colmap,(a->Nbs)*sizeof(int));
  } else a->colmap = 0;
  if (oldmat->garray && (len = ((Mat_SeqBAIJ *) (oldmat->B->data))->nbs)) {
    a->garray = (int *) PetscMalloc(len*sizeof(int)); CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&a->B); CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat); CHKERRQ(ierr);
  }
  *newmat = mat;
  return 0;
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIBAIJ"
int MatLoad_MPIBAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*buf;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,*browners,maxnz,*cols;
  int          *locrowlens,*sndcounts = 0,*procsnz = 0, jj,*mycols,*ibuf;
  int          flg,tag = ((PetscObject)viewer)->tag,bs=1,bs2,Mbs,mbs,extra_rows;
  int          *dlens,*odlens,*mask,*masked1,*masked2,rowcount,odcount;
  int          dcount,kmax,k,nzcount,tmp;

 
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  bs2  = bs*bs;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,BINARY_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,0,"not matrix object");
  }
    
  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];

  if (M != N) SETERRQ(1,0,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows &&!rank) {
    PLogInfo(0,"MatLoad_MPIBAIJ:Padding loaded matrix to match blocksize\n");
  }

  /* determine ownership of all rows */
  mbs = Mbs/size + ((Mbs % size) > rank);
  m   = mbs * bs;
  rowners = (int *) PetscMalloc(2*(size+2)*sizeof(int)); CHKPTRQ(rowners);
  browners = rowners + size + 1;
  MPI_Allgather(&mbs,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) rowners[i] += rowners[i-1];
  for ( i=0; i<=size;  i++ ) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  locrowlens = (int*) PetscMalloc( (rend-rstart)*bs*sizeof(int) ); CHKPTRQ(locrowlens);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( (M+extra_rows)*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
    for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = browners[i+1] - browners[i];
    MPI_Scatterv(rowlengths,sndcounts,browners,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT,0,comm);
    PetscFree(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT, 0,comm);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(procsnz);
    PetscMemzero(procsnz,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]*bs; j< rowners[i+1]*bs; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PetscFree(rowlengths);
    
    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int *) PetscMalloc( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    ibuf = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ibuf);
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,BINARY_INT); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every ones (except the last) and ship off */
    for ( i=1; i<size-1; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    /* read in the stuff for the last proc */
    if ( size != 1 ) {
      nz = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) cols[nz+i] = M+i;
      MPI_Send(cols,nz+extra_rows,MPI_INT,size-1,tag,comm);
    }
    PetscFree(cols);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += locrowlens[i];
    }
    ibuf = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ibuf);
    mycols = ibuf;
    /* receive message of column indices*/
    MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,0,"something is wrong with file");
  }
  
  /* loop over local rows, determining number of off diagonal entries */
  dlens  = (int *) PetscMalloc( 2*(rend-rstart+1)*sizeof(int) ); CHKPTRQ(dlens);
  odlens = dlens + (rend-rstart);
  mask   = (int *) PetscMalloc( 3*Mbs*sizeof(int) ); CHKPTRQ(mask);
  PetscMemzero(mask,3*Mbs*sizeof(int));
  masked1 = mask    + Mbs;
  masked2 = masked1 + Mbs;
  rowcount = 0; nzcount = 0;
  for ( i=0; i<mbs; i++ ) {
    dcount  = 0;
    odcount = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = locrowlens[rowcount];
      for ( k=0; k<kmax; k++ ) {
        tmp = mycols[nzcount++]/bs;
        if (!mask[tmp]) {
          mask[tmp] = 1;
          if (tmp < rstart || tmp >= rend ) masked2[odcount++] = tmp;
          else masked1[dcount++] = tmp;
        }
      }
      rowcount++;
    }
  
    dlens[i]  = dcount;
    odlens[i] = odcount;

    /* zero out the mask elements we set */
    for ( j=0; j<dcount; j++ ) mask[masked1[j]] = 0;
    for ( j=0; j<odcount; j++ ) mask[masked2[j]] = 0; 
  }

  /* create our matrix */
  ierr = MatCreateMPIBAIJ(comm,bs,m,PETSC_DECIDE,M+extra_rows,N+extra_rows,0,dlens,0,odlens,newmat);
         CHKERRQ(ierr);
  A = *newmat;
  MatSetOption(A,MAT_COLUMNS_SORTED); 
  
  if (!rank) {
    buf = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(buf);
    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    vals = buf;
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { vals[nz+i] = 1.0; }

    /* insert into matrix */
    jj      = rstart*bs;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
    /* read in other processors (except the last one) and ship out */
    for ( i=1; i<size-1; i++ ) {
      nz = procsnz[i];
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    /* the last proc */
    if ( size != 1 ){
      nz = procsnz[i] - extra_rows;
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      for ( i=0; i<extra_rows; i++ ) vals[nz+i] = 1.0;
      MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,A->tag,comm);
    }
    PetscFree(procsnz);
  }
  else {
    /* receive numeric values */
    buf = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(buf);

    /* receive message of values*/
    vals = buf;
    mycols = ibuf;
    MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,0,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart*bs;
    for ( i=0; i<m; i++ ) {
      ierr    = MatSetValues(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
  }
  PetscFree(locrowlens); 
  PetscFree(buf); 
  PetscFree(ibuf); 
  PetscFree(rowners);
  PetscFree(dlens);
  PetscFree(mask);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


