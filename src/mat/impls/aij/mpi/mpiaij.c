
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/inline/spops.h"

/* 
  Local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  When PETSC_USE_CTABLE is used this is scalable at 
a slightly higher hash table cost; without it it is not scalable (each processor
has an order N integer array but is fast to acess.
*/
#undef __FUNCT__  
#define __FUNCT__ "CreateColmap_MPIAIJ_Private"
int CreateColmap_MPIAIJ_Private(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;
  int        n = aij->B->n,i,ierr;

  PetscFunctionBegin;
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableCreate(n,&aij->colmap);CHKERRQ(ierr); 
  for (i=0; i<n; i++){
    ierr = PetscTableAdd(aij->colmap,aij->garray[i]+1,i+1);CHKERRQ(ierr);
  }
#else
  ierr = PetscMalloc((mat->N+1)*sizeof(int),&aij->colmap);CHKERRQ(ierr);
  PetscLogObjectMemory(mat,mat->N*sizeof(int));
  ierr = PetscMemzero(aij->colmap,mat->N*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) aij->colmap[aij->garray[i]] = i+1;
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
      for (_i=low; _i<high; _i++) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto a_noinsert; \
        } \
      }  \
      if (nonew == 1) goto a_noinsert; \
      else if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%d, %d) into matrix", row, col); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = ai[am] + CHUNKSIZE,len,*new_i,*new_j; \
        PetscScalar *new_a; \
 \
        if (nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%d, %d) in the matrix", row, col); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(PetscScalar))+(am+1)*sizeof(int); \
        ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr); \
        new_j   = (int*)(new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<row+1; ii++) {new_i[ii] = ai[ii];} \
        for (ii=row+1; ii<am+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - ai[row] - nrow - shift); \
        ierr = PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow, \
                                                           len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(PetscScalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow, \
                                                           len*sizeof(PetscScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
 \
        ierr = PetscFree(a->a);CHKERRQ(ierr);  \
        if (!a->singlemalloc) { \
           ierr = PetscFree(a->i);CHKERRQ(ierr); \
           ierr = PetscFree(a->j);CHKERRQ(ierr); \
        } \
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = PETSC_TRUE; \
 \
        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift; \
        rmax = aimax[row] = aimax[row] + CHUNKSIZE; \
        PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(PetscScalar))); \
        a->maxnz += CHUNKSIZE; \
        a->reallocs++; \
      } \
      N = nrow++ - 1; a->nz++; \
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
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
       for (_i=low; _i<high; _i++) { \
        if (rp[_i] > col1) break; \
        if (rp[_i] == col1) { \
          if (addv == ADD_VALUES) ap[_i] += value;   \
          else                  ap[_i] = value; \
          goto b_noinsert; \
        } \
      }  \
      if (nonew == 1) goto b_noinsert; \
      else if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%d, %d) into matrix", row, col); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int    new_nz = bi[bm] + CHUNKSIZE,len,*new_i,*new_j; \
        PetscScalar *new_a; \
 \
        if (nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%d, %d) in the matrix", row, col); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+sizeof(PetscScalar))+(bm+1)*sizeof(int); \
        ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr); \
        new_j   = (int*)(new_a + new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<row+1; ii++) {new_i[ii] = bi[ii];} \
        for (ii=row+1; ii<bm+1; ii++) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,bj,(bi[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - bi[row] - nrow - shift); \
        ierr = PetscMemcpy(new_j+bi[row]+shift+nrow+CHUNKSIZE,bj+bi[row]+shift+nrow, \
                                                           len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,ba,(bi[row]+nrow+shift)*sizeof(PetscScalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bi[row]+shift+nrow+CHUNKSIZE,ba+bi[row]+shift+nrow, \
                                                           len*sizeof(PetscScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
 \
        ierr = PetscFree(b->a);CHKERRQ(ierr);  \
        if (!b->singlemalloc) { \
          ierr = PetscFree(b->i);CHKERRQ(ierr); \
          ierr = PetscFree(b->j);CHKERRQ(ierr); \
        } \
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = PETSC_TRUE; \
 \
        rp   = bj + bi[row] + shift; ap = ba + bi[row] + shift; \
        rmax = bimax[row] = bimax[row] + CHUNKSIZE; \
        PetscLogObjectMemory(B,CHUNKSIZE*(sizeof(int) + sizeof(PetscScalar))); \
        b->maxnz += CHUNKSIZE; \
        b->reallocs++; \
      } \
      N = nrow++ - 1; b->nz++; \
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
        rp[ii+1] = rp[ii]; \
        ap[ii+1] = ap[ii]; \
      } \
      rp[_i] = col1;  \
      ap[_i] = value;  \
      b_noinsert: ; \
      bilen[row] = nrow; \
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIAIJ"
int MatSetValues_MPIAIJ(Mat mat,int m,const int im[],int n,const int in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIAIJ   *aij = (Mat_MPIAIJ*)mat->data;
  PetscScalar  value;
  int          ierr,i,j,rstart = aij->rstart,rend = aij->rend;
  int          cstart = aij->cstart,cend = aij->cend,row,col;
  PetscTruth   roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat          A = aij->A;
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data; 
  int          *aimax = a->imax,*ai = a->i,*ailen = a->ilen,*aj = a->j;
  PetscScalar  *aa = a->a;
  PetscTruth   ignorezeroentries = (((a->ignorezeroentries)&&(addv==ADD_VALUES))?PETSC_TRUE:PETSC_FALSE); 
  Mat          B = aij->B;
  Mat_SeqAIJ   *b = (Mat_SeqAIJ*)B->data; 
  int          *bimax = b->imax,*bi = b->i,*bilen = b->ilen,*bj = b->j,bm = aij->B->m,am = aij->A->m;
  PetscScalar  *ba = b->a;

  int          *rp,ii,nrow,_i,rmax,N,col1,low,high,t; 
  int          nonew = a->nonew,shift=0; 
  PetscScalar  *ap;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
    if (im[i] >= mat->M) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %d max %d",im[i],mat->M-1);
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          if (ignorezeroentries && value != 0.0) continue;
          MatSetValues_SeqAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
        else if (in[j] >= mat->N) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %d max %d",in[j],mat->N-1);}
#endif
        else {
          if (mat->was_assembled) {
            if (!aij->colmap) {
              ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined (PETSC_USE_CTABLE)
            ierr = PetscTableFind(aij->colmap,in[j]+1,&col);CHKERRQ(ierr);
	    col--;
#else
            col = aij->colmap[in[j]] - 1;
#endif
            if (col < 0 && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
              ierr = DisAssemble_MPIAIJ(mat);CHKERRQ(ierr);
              col =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private() */
              B = aij->B;
              b = (Mat_SeqAIJ*)B->data; 
              bimax = b->imax; bi = b->i; bilen = b->ilen; bj = b->j;
              ba = b->a;
            }
          } else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          if (ignorezeroentries && value != 0.0) continue;
          MatSetValues_SeqAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqAIJ(aij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {
      if (!aij->donotstash) {
        if (roworiented) {
          if (ignorezeroentries && v[i*n] != 0.0) continue;
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n);CHKERRQ(ierr);
        } else {
          if (ignorezeroentries && v[i] != 0.0) continue;
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPIAIJ"
int MatGetValues_MPIAIJ(Mat mat,int m,const int idxm[],int n,const int idxn[],PetscScalar v[])
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;
  int        ierr,i,j,rstart = aij->rstart,rend = aij->rend;
  int        cstart = aij->cstart,cend = aij->cend,row,col;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %d",idxm[i]);
    if (idxm[i] >= mat->M) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %d max %d",idxm[i],mat->M-1);
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %d",idxn[j]);
        if (idxn[j] >= mat->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %d max %d",idxn[j],mat->N-1);
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatGetValues(aij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!aij->colmap) {
            ierr = CreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
          }
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(aij->colmap,idxn[j]+1,&col);CHKERRQ(ierr);
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
      SETERRQ(PETSC_ERR_SUP,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIAIJ"
int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)mat->data;
  int         ierr,nstash,reallocs;
  InsertMode  addv;

  PetscFunctionBegin;
  if (aij->donotstash) {
    PetscFunctionReturn(0);
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,mat->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,aij->rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PetscLogInfo(aij->A,"MatAssemblyBegin_MPIAIJ:Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIAIJ"
int MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ  *a=(Mat_SeqAIJ *)aij->A->data,*b= (Mat_SeqAIJ *)aij->B->data;
  int         i,j,rstart,ncols,n,ierr,flg;
  int         *row,*col,other_disassembled;
  PetscScalar *val;
  InsertMode  addv = mat->insertmode;

  PetscFunctionBegin;
  if (!aij->donotstash) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
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
  if (!((Mat_SeqAIJ*)aij->B->data)->nonew)  {
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

  /* used by MatAXPY() */
  a->xtoy = 0; b->xtoy = 0;  
  a->XtoY = 0; b->XtoY = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIAIJ"
int MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIAIJ"
int MatZeroRows_MPIAIJ(Mat A,IS is,const PetscScalar *diag)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  int            i,ierr,N,*rows,*owners = l->rowners,size = l->size;
  int            *nprocs,j,idx,nsends,row;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values,rstart=l->rstart;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;
  PetscTruth     found;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc(2*size*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMalloc((N+1)*sizeof(int),&owner);CHKERRQ(ierr); /* see note*/
  for (i=0; i<N; i++) {
    idx = rows[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; nprocs[2*j+1] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((N+1)*sizeof(int),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0] = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr   = PetscMalloc(2*(nrecvs+1)*sizeof(int),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr = PetscMalloc((slen+1)*sizeof(int),&lrows);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PetscLogObjectParent(A,istmp);

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
      SETERRQ(PETSC_ERR_SUP,"MatZeroRows() on rectangular matrices cannot be used with the Mat options\n\
MAT_NO_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    }
    for (i = 0; i < slen; i++) {
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
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIAIJ"
int MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr,nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->n) {
    SETERRQ2(PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%d) and xx (%d)",A->n,nt);
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIAIJ"
int MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIAIJ"
int MatMultTranspose_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatIsTranspose_MPIAIJ"
int MatIsTranspose_MPIAIJ(Mat Amat,Mat Bmat,PetscTruth tol,PetscTruth *f)
{
  MPI_Comm comm;
  Mat_MPIAIJ *Aij = (Mat_MPIAIJ *) Amat->data, *Bij;
  Mat        Adia = Aij->A, Bdia, Aoff,Boff,*Aoffs,*Boffs;
  IS         Me,Notme;
  int        M,N,first,last,*notme,ntids,i, ierr;

  PetscFunctionBegin;

  /* Easy test: symmetric diagonal block */
  Bij = (Mat_MPIAIJ *) Bmat->data; Bdia = Bij->A;
  ierr = MatIsTranspose(Adia,Bdia,tol,f);CHKERRQ(ierr);
  if (!*f) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&ntids);CHKERRQ(ierr);
  if (ntids==1) PetscFunctionReturn(0);

  /* Hard test: off-diagonal block. This takes a MatGetSubMatrix. */
  ierr = MatGetSize(Amat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&first,&last);CHKERRQ(ierr);
  ierr = PetscMalloc((N-last+first)*sizeof(int),&notme);CHKERRQ(ierr);
  for (i=0; i<first; i++) notme[i] = i;
  for (i=last; i<M; i++) notme[i-last+first] = i;
  ierr = ISCreateGeneral(MPI_COMM_SELF,N-last+first,notme,&Notme);CHKERRQ(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,last-first,first,1,&Me);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(Amat,1,&Me,&Notme,MAT_INITIAL_MATRIX,&Aoffs);CHKERRQ(ierr);
  Aoff = Aoffs[0];
  ierr = MatGetSubMatrices(Bmat,1,&Notme,&Me,MAT_INITIAL_MATRIX,&Boffs);CHKERRQ(ierr);
  Boff = Boffs[0];
  ierr = MatIsTranspose(Aoff,Boff,tol,f);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&Aoffs);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&Boffs);CHKERRQ(ierr);
  ierr = ISDestroy(Me);CHKERRQ(ierr);
  ierr = ISDestroy(Notme);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPIAIJ"
int MatMultTransposeAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransposeadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIAIJ"
int MatGetDiagonal_MPIAIJ(Mat A,Vec v)
{
  int        ierr;
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  if (A->M != A->N) SETERRQ(PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  if (a->rstart != a->cstart || a->rend != a->cend) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"row partition must equal col partition");  
  }
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIAIJ"
int MatScale_MPIAIJ(const PetscScalar aa[],Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatScale(aa,a->A);CHKERRQ(ierr);
  ierr = MatScale(aa,a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ"
int MatDestroy_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;
  int        ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",mat->M,mat->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = PetscFree(aij->rowners);CHKERRQ(ierr);
  ierr = MatDestroy(aij->A);CHKERRQ(ierr);
  ierr = MatDestroy(aij->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
  if (aij->colmap) {ierr = PetscTableDelete(aij->colmap);CHKERRQ(ierr);}
#else
  if (aij->colmap) {ierr = PetscFree(aij->colmap);CHKERRQ(ierr);}
#endif
  if (aij->garray) {ierr = PetscFree(aij->garray);CHKERRQ(ierr);}
  if (aij->lvec)   {ierr = VecDestroy(aij->lvec);CHKERRQ(ierr);}
  if (aij->Mvctx)  {ierr = VecScatterDestroy(aij->Mvctx);CHKERRQ(ierr);}
  if (aij->rowvalues) {ierr = PetscFree(aij->rowvalues);CHKERRQ(ierr);}
  ierr = PetscFree(aij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAIJ_Binary"
int MatView_MPIAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ*       A = (Mat_SeqAIJ*)aij->A->data;
  Mat_SeqAIJ*       B = (Mat_SeqAIJ*)aij->B->data;
  int               nz,fd,ierr,header[4],rank,size,*row_lengths,*range,rlen,i,tag = ((PetscObject)viewer)->tag;
  int               nzmax,*column_indices,j,k,col,*garray = aij->garray,cnt,cstart = aij->cstart,rnz;
  PetscScalar       *column_values;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);
  nz   = A->nz + B->nz;
  if (!rank) {
    header[0] = MAT_FILE_COOKIE;
    header[1] = mat->M;
    header[2] = mat->N;
    ierr = MPI_Reduce(&nz,&header[3],1,MPI_INT,MPI_SUM,0,mat->comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,header,4,PETSC_INT,1);CHKERRQ(ierr);
    /* get largest number of rows any processor has */
    rlen = mat->m;
    ierr = PetscMapGetGlobalRange(mat->rmap,&range);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      rlen = PetscMax(rlen,range[i+1] - range[i]);
    }
  } else {
    ierr = MPI_Reduce(&nz,0,1,MPI_INT,MPI_SUM,0,mat->comm);CHKERRQ(ierr);
    rlen = mat->m;
  }

  /* load up the local row counts */
  ierr = PetscMalloc((rlen+1)*sizeof(int),&row_lengths);CHKERRQ(ierr);
  for (i=0; i<mat->m; i++) {
    row_lengths[i] = A->i[i+1] - A->i[i] + B->i[i+1] - B->i[i];
  }

  /* store the row lengths to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,row_lengths,mat->m,PETSC_INT,1);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      rlen = range[i+1] - range[i];
      ierr = MPI_Recv(row_lengths,rlen,MPI_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,row_lengths,rlen,PETSC_INT,1);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(row_lengths,mat->m,MPI_INT,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(row_lengths);CHKERRQ(ierr);

  /* load up the local column indices */
  nzmax = nz; /* )th processor needs space a largest processor needs */
  ierr = MPI_Reduce(&nz,&nzmax,1,MPI_INT,MPI_MAX,0,mat->comm);CHKERRQ(ierr);
  ierr = PetscMalloc((nzmax+1)*sizeof(int),&column_indices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<mat->m; i++) {
    for (j=B->i[i]; j<B->i[i+1]; j++) {
      if ( (col = garray[B->j[j]]) > cstart) break;
      column_indices[cnt++] = col;
    }
    for (k=A->i[i]; k<A->i[i+1]; k++) {
      column_indices[cnt++] = A->j[k] + cstart;
    }
    for (; j<B->i[i+1]; j++) {
      column_indices[cnt++] = garray[B->j[j]];
    }
  }
  if (cnt != A->nz + B->nz) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: cnt = %d nz = %d",cnt,A->nz+B->nz);

  /* store the column indices to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_indices,nz,PETSC_INT,1);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&rnz,1,MPI_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      if (rnz > nzmax) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: nz = %d nzmax = %d",nz,nzmax);
      ierr = MPI_Recv(column_indices,rnz,MPI_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_indices,rnz,PETSC_INT,1);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(&nz,1,MPI_INT,0,tag,mat->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_indices,nz,MPI_INT,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_indices);CHKERRQ(ierr);

  /* load up the local column values */
  ierr = PetscMalloc((nzmax+1)*sizeof(PetscScalar),&column_values);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<mat->m; i++) {
    for (j=B->i[i]; j<B->i[i+1]; j++) {
      if ( garray[B->j[j]] > cstart) break;
      column_values[cnt++] = B->a[j];
    }
    for (k=A->i[i]; k<A->i[i+1]; k++) {
      column_values[cnt++] = A->a[k];
    }
    for (; j<B->i[i+1]; j++) {
      column_values[cnt++] = B->a[j];
    }
  }
  if (cnt != A->nz + B->nz) SETERRQ2(1,"Internal PETSc error: cnt = %d nz = %d",cnt,A->nz+B->nz);

  /* store the column values to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_values,nz,PETSC_SCALAR,1);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&rnz,1,MPI_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      if (rnz > nzmax) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: nz = %d nzmax = %d",nz,nzmax);
      ierr = MPI_Recv(column_values,rnz,MPIU_SCALAR,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_values,rnz,PETSC_SCALAR,1);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(&nz,1,MPI_INT,0,tag,mat->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_values,nz,MPIU_SCALAR,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAIJ_ASCIIorDraworSocket"
int MatView_MPIAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)mat->data;
  int               ierr,rank = aij->rank,size = aij->size;
  PetscTruth        isdraw,iascii,flg,isbinary;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) { 
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %d nz %d nz alloced %d mem %d, not using I-node routines\n",
					      rank,mat->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %d nz %d nz alloced %d mem %d, using I-node routines\n",
		    rank,mat->m,(int)info.nz_used,(int)info.nz_allocated,(int)info.memory);CHKERRQ(ierr);
      }
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %d \n",rank,(int)info.nz_used);CHKERRQ(ierr);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %d \n",rank,(int)info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecScatterView(aij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (isbinary) {
    if (size == 1) {
      ierr = PetscObjectSetName((PetscObject)aij->A,mat->name);CHKERRQ(ierr);
      ierr = MatView(aij->A,viewer);CHKERRQ(ierr);
    } else {
      ierr = MatView_MPIAIJ_Binary(mat,viewer);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  } else if (isdraw) {
    PetscDraw  draw;
    PetscTruth isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = PetscObjectSetName((PetscObject)aij->A,mat->name);CHKERRQ(ierr);
    ierr = MatView(aij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqAIJ *Aloc;
    int         M = mat->M,N = mat->N,m,*ai,*aj,row,*cols,i,*ct;
    PetscScalar *a;

    if (!rank) {
      ierr = MatCreate(mat->comm,M,N,M,N,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreate(mat->comm,0,0,M,N,&A);CHKERRQ(ierr);
    }
    /* This is just a temporary matrix, so explicitly using MATMPIAIJ is probably best */
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    PetscLogObjectParent(mat,A);

    /* copy over the A part */
    Aloc = (Mat_SeqAIJ*)aij->A->data;
    m = aij->A->m; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row = aij->rstart;
    for (i=0; i<ai[m]; i++) {aj[i] += aij->cstart ;}
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERT_VALUES);CHKERRQ(ierr);
      row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
    } 
    aj = Aloc->j;
    for (i=0; i<ai[m]; i++) {aj[i] -= aij->cstart;}

    /* copy over the B part */
    Aloc = (Mat_SeqAIJ*)aij->B->data;
    m    = aij->B->m;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row  = aij->rstart;
    ierr = PetscMalloc((ai[m]+1)*sizeof(int),&cols);CHKERRQ(ierr);
    ct   = cols;
    for (i=0; i<ai[m]; i++) {cols[i] = aij->garray[aj[i]];}
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERT_VALUES);CHKERRQ(ierr);
      row++; a += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
    } 
    ierr = PetscFree(ct);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* 
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the PetscDraw object
    */
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscObjectSetName((PetscObject)((Mat_MPIAIJ*)(A->data))->A,mat->name);CHKERRQ(ierr);
      ierr = MatView(((Mat_MPIAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAIJ"
int MatView_MPIAIJ(Mat mat,PetscViewer viewer)
{
  int        ierr;
  PetscTruth iascii,isdraw,issocket,isbinary;
 
  PetscFunctionBegin;
  ierr  = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr  = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  if (iascii || isdraw || isbinary || issocket) { 
    ierr = MatView_MPIAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by MPIAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatRelax_MPIAIJ"
int MatRelax_MPIAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,int its,int lits,Vec xx)
{
  Mat_MPIAIJ   *mat = (Mat_MPIAIJ*)matin->data;
  int          ierr; 
  Vec          bb1;
  PetscScalar  mone=-1.0;

  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %d and local its %d both positive",its,lits);

  ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
      its--; 
    }
    
    while (its--) { 
      ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */ 
      ierr = VecScale(&mone,mat->lvec);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->relax)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx);
      CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,lits,PETSC_NULL,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */ 
      ierr = VecScale(&mone,mat->lvec);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->relax)(mat->A,bb1,omega,SOR_FORWARD_SWEEP,fshift,lits,PETSC_NULL,xx);
      CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,lits,PETSC_NULL,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */ 
      ierr = VecScale(&mone,mat->lvec);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->relax)(mat->A,bb1,omega,SOR_BACKWARD_SWEEP,fshift,lits,PETSC_NULL,xx);
      CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Parallel SOR not supported");
  }

  ierr = VecDestroy(bb1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIAIJ"
int MatGetInfo_MPIAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ*)matin->data;
  Mat        A = mat->A,B = mat->B;
  int        ierr;
  PetscReal  isend[5],irecv[5];

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
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  info->rows_global       = (double)matin->M;
  info->columns_global    = (double)matin->N;
  info->rows_local        = (double)matin->m;
  info->columns_local     = (double)matin->N;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAIJ"
int MatSetOption_MPIAIJ(Mat A,MatOption op)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_NEW_NONZERO_LOCATIONS:
  case MAT_YES_NEW_NONZERO_LOCATIONS:
  case MAT_COLUMNS_UNSORTED:
  case MAT_COLUMNS_SORTED:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_KEEP_ZEROED_ROWS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_USE_INODES:
  case MAT_DO_NOT_USE_INODES:
  case MAT_IGNORE_ZERO_ENTRIES:
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = PETSC_TRUE; 
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_ROWS_SORTED:
  case MAT_ROWS_UNSORTED:
  case MAT_YES_NEW_DIAGONALS:
    PetscLogInfo(A,"MatSetOption_MPIAIJ:Option ignored\n");
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented = PETSC_FALSE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = PETSC_TRUE;
    break;
  case MAT_NO_NEW_DIAGONALS:
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    break;
  case MAT_NOT_SYMMETRIC:
  case MAT_NOT_STRUCTURALLY_SYMMETRIC:
  case MAT_NOT_HERMITIAN:
  case MAT_NOT_SYMMETRY_ETERNAL:
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIAIJ"
int MatGetRow_MPIAIJ(Mat matin,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_MPIAIJ   *mat = (Mat_MPIAIJ*)matin->data;
  PetscScalar  *vworkA,*vworkB,**pvA,**pvB,*v_p;
  int          i,ierr,*cworkA,*cworkB,**pcA,**pcB,cstart = mat->cstart;
  int          nztot,nzA,nzB,lrow,rstart = mat->rstart,rend = mat->rend;
  int          *cmap,*idx_p;

  PetscFunctionBegin;
  if (mat->getrowactive == PETSC_TRUE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ*)mat->A->data,*Ba = (Mat_SeqAIJ*)mat->B->data; 
    int     max = 1,tmp;
    for (i=0; i<matin->m; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    ierr = PetscMalloc(max*(sizeof(int)+sizeof(PetscScalar)),&mat->rowvalues);CHKERRQ(ierr);
    mat->rowindices = (int*)(mat->rowvalues + max);
  }

  if (row < rstart || row >= rend) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Only local rows")
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
        for (i=0; i<nzB; i++) {
          if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for (i=0; i<nzA; i++)     v_p[imark+i] = vworkA[i];
        for (i=imark; i<nzB; i++) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for (i=0; i<imark; i++) {
            idx_p[i] = cmap[cworkB[i]];
          }
        } else {
          for (i=0; i<nzB; i++) {
            if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
            else break;
          }
          imark = i;
        }
        for (i=0; i<nzA; i++)     idx_p[imark+i] = cstart + cworkA[i];
        for (i=imark; i<nzB; i++) idx_p[nzA+i]   = cmap[cworkB[i]];
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

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPIAIJ"
int MatRestoreRow_MPIAIJ(Mat mat,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;

  PetscFunctionBegin;
  if (aij->getrowactive == PETSC_FALSE) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"MatGetRow not called");
  }
  aij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIAIJ"
int MatNorm_MPIAIJ(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPIAIJ   *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ   *amat = (Mat_SeqAIJ*)aij->A->data,*bmat = (Mat_SeqAIJ*)aij->B->data;
  int          ierr,i,j,cstart = aij->cstart;
  PetscReal    sum = 0.0;
  PetscScalar  *v;

  PetscFunctionBegin;
  if (aij->size == 1) {
    ierr =  MatNorm(aij->A,type,norm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,norm,1,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = sqrt(*norm);
    } else if (type == NORM_1) { /* max column norm */
      PetscReal *tmp,*tmp2;
      int    *jj,*garray = aij->garray;
      ierr = PetscMalloc((mat->N+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
      ierr = PetscMalloc((mat->N+1)*sizeof(PetscReal),&tmp2);CHKERRQ(ierr);
      ierr = PetscMemzero(tmp,mat->N*sizeof(PetscReal));CHKERRQ(ierr);
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for (j=0; j<amat->nz; j++) {
        tmp[cstart + *jj++ ] += PetscAbsScalar(*v);  v++;
      }
      v = bmat->a; jj = bmat->j;
      for (j=0; j<bmat->nz; j++) {
        tmp[garray[*jj++]] += PetscAbsScalar(*v); v++;
      }
      ierr = MPI_Allreduce(tmp,tmp2,mat->N,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      for (j=0; j<mat->N; j++) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscFree(tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp = 0.0;
      for (j=0; j<aij->A->m; j++) {
        v = amat->a + amat->i[j];
        sum = 0.0;
        for (i=0; i<amat->i[j+1]-amat->i[j]; i++) {
          sum += PetscAbsScalar(*v); v++;
        }
        v = bmat->a + bmat->i[j];
        for (i=0; i<bmat->i[j+1]-bmat->i[j]; i++) {
          sum += PetscAbsScalar(*v); v++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      ierr = MPI_Allreduce(&ntemp,norm,1,MPIU_REAL,MPI_MAX,mat->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPIAIJ"
int MatTranspose_MPIAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIAIJ   *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ   *Aloc = (Mat_SeqAIJ*)a->A->data;
  int          ierr;
  int          M = A->M,N = A->N,m,*ai,*aj,row,*cols,i,*ct;
  Mat          B;
  PetscScalar  *array;

  PetscFunctionBegin;
  if (!matout && M != N) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  }

  ierr = MatCreate(A->comm,A->n,A->m,N,M,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_SeqAIJ*)a->A->data;
  m = a->A->m; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  for (i=0; i<ai[m]; i++) {aj[i] += a->cstart ;}
  for (i=0; i<m; i++) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for (i=0; i<ai[m]; i++) {aj[i] -= a->cstart ;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*)a->B->data;
  m = a->B->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row  = a->rstart;
  ierr = PetscMalloc((1+ai[m])*sizeof(int),&cols);CHKERRQ(ierr);
  ct   = cols;
  for (i=0; i<ai[m]; i++) {cols[i] = a->garray[aj[i]];}
  for (i=0; i<m; i++) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  ierr = PetscFree(ct);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (matout) {
    *matout = B;
  } else {
    ierr = MatHeaderCopy(A,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_MPIAIJ"
int MatDiagonalScale_MPIAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;
  Mat        a = aij->A,b = aij->B;
  int        ierr,s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_ERR_ARG_SIZ,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(rr,aij->lvec,INSERT_VALUES,SCATTER_FORWARD,aij->Mvctx);CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_ERR_ARG_SIZ,"left vector non-conforming local size");
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


#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_MPIAIJ"
int MatPrintHelp_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqAIJ(a->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetBlockSize_MPIAIJ"
int MatGetBlockSize_MPIAIJ(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatSetUnfactored_MPIAIJ"
int MatSetUnfactored_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIAIJ"
int MatEqual_MPIAIJ(Mat A,Mat B,PetscTruth *flag)
{
  Mat_MPIAIJ *matB = (Mat_MPIAIJ*)B->data,*matA = (Mat_MPIAIJ*)A->data;
  Mat        a,b,c,d;
  PetscTruth flg;
  int        ierr;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg == PETSC_TRUE) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_MPIAIJ"
int MatCopy_MPIAIJ(Mat A,Mat B,MatStructure str)
{
  int        ierr;
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_MPIAIJ"
int MatSetUpPreallocation_MPIAIJ(Mat A)
{
  int        ierr;

  PetscFunctionBegin;
  ierr =  MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_MPIAIJ"
int MatAXPY_MPIAIJ(const PetscScalar a[],Mat X,Mat Y,MatStructure str)
{
  int        ierr,one=1,i;
  Mat_MPIAIJ *xx = (Mat_MPIAIJ *)X->data,*yy = (Mat_MPIAIJ *)Y->data;
  Mat_SeqAIJ *x,*y;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {  
    x = (Mat_SeqAIJ *)xx->A->data;
    y = (Mat_SeqAIJ *)yy->A->data;
    BLaxpy_(&x->nz,(PetscScalar*)a,x->a,&one,y->a,&one);    
    x = (Mat_SeqAIJ *)xx->B->data;
    y = (Mat_SeqAIJ *)yy->B->data;
    BLaxpy_(&x->nz,(PetscScalar*)a,x->a,&one,y->a,&one);
  } else if (str == SUBSET_NONZERO_PATTERN) {  
    ierr = MatAXPY_SeqAIJ(a,xx->A,yy->A,str);CHKERRQ(ierr);

    x = (Mat_SeqAIJ *)xx->B->data;
    y = (Mat_SeqAIJ *)yy->B->data;
    if (y->xtoy && y->XtoY != xx->B) {
      ierr = PetscFree(y->xtoy);CHKERRQ(ierr);
      ierr = MatDestroy(y->XtoY);CHKERRQ(ierr);
    }
    if (!y->xtoy) { /* get xtoy */
      ierr = MatAXPYGetxtoy_Private(xx->B->m,x->i,x->j,xx->garray,y->i,y->j,yy->garray,&y->xtoy);CHKERRQ(ierr);
      y->XtoY = xx->B;
    } 
    for (i=0; i<x->nz; i++) y->a[y->xtoy[i]] += (*a)*(x->a[i]);   
  } else {
    ierr = MatAXPY_Basic(a,X,Y,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,
       MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,
/* 4*/ MatMultAdd_MPIAIJ,
       MatMultTranspose_MPIAIJ,
       MatMultTransposeAdd_MPIAIJ,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       MatRelax_MPIAIJ,
       MatTranspose_MPIAIJ,
/*15*/ MatGetInfo_MPIAIJ,
       MatEqual_MPIAIJ,
       MatGetDiagonal_MPIAIJ,
       MatDiagonalScale_MPIAIJ,
       MatNorm_MPIAIJ,
/*20*/ MatAssemblyBegin_MPIAIJ,
       MatAssemblyEnd_MPIAIJ,
       0,
       MatSetOption_MPIAIJ,
       MatZeroEntries_MPIAIJ,
/*25*/ MatZeroRows_MPIAIJ,
#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
       MatLUFactorSymbolic_MPIAIJ_TFS,
#else
       0,
#endif
       0,
       0,
       0,
/*30*/ MatSetUpPreallocation_MPIAIJ,
       0,
       0,
       0,
       0,
/*35*/ MatDuplicate_MPIAIJ,
       0,
       0,
       0,
       0,
/*40*/ MatAXPY_MPIAIJ,
       MatGetSubMatrices_MPIAIJ,
       MatIncreaseOverlap_MPIAIJ,
       MatGetValues_MPIAIJ,
       MatCopy_MPIAIJ,
/*45*/ MatPrintHelp_MPIAIJ,
       MatScale_MPIAIJ,
       0,
       0,
       0,
/*50*/ MatGetBlockSize_MPIAIJ,
       0,
       0,
       0,
       0,
/*55*/ MatFDColoringCreate_MPIAIJ,
       0,
       MatSetUnfactored_MPIAIJ,
       0,
       0,
/*60*/ MatGetSubMatrix_MPIAIJ,
       MatDestroy_MPIAIJ,
       MatView_MPIAIJ,
       MatGetPetscMaps_Petsc,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       MatSetColoring_MPIAIJ,
       MatSetValuesAdic_MPIAIJ,
       MatSetValuesAdifor_MPIAIJ,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
/*85*/ MatLoad_MPIAIJ,
       0,
       0,
       0,
       0,
/*90*/ 0,
       MatMatMult_MPIAIJ_MPIAIJ, 
};

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_MPIAIJ"
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
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_MPIAIJ"
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

#include "petscpc.h"
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation_MPIAIJ"
int MatMPIAIJSetPreallocation_MPIAIJ(Mat B,int d_nz,const int d_nnz[],int o_nz,const int o_nnz[])
{
  Mat_MPIAIJ   *b;
  int          ierr,i;

  PetscFunctionBegin;
  B->preallocated = PETSC_TRUE;
  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
  if (d_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %d",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %d",o_nz);
  if (d_nnz) {
    for (i=0; i<B->m; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %d value %d",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->m; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %d value %d",i,o_nnz[i]);
    }
  }
  b = (Mat_MPIAIJ*)B->data;
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATMPIAIJ - MATMPIAIJ = "mpiaij" - A matrix type to be used for parallel sparse matrices.

   Options Database Keys:
. -mat_type mpiaij - sets the matrix type to "mpiaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAIJ"
int MatCreate_MPIAIJ(Mat B)
{
  Mat_MPIAIJ *b;
  int        ierr,i,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);

  ierr            = PetscNew(Mat_MPIAIJ,&b);CHKERRQ(ierr);
  B->data         = (void*)b;
  ierr            = PetscMemzero(b,sizeof(Mat_MPIAIJ));CHKERRQ(ierr);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor       = 0;
  B->assembled    = PETSC_FALSE;
  B->mapping      = 0;

  B->insertmode      = NOT_SET_VALUES;
  b->size            = size;
  ierr = MPI_Comm_rank(B->comm,&b->rank);CHKERRQ(ierr);

  ierr = PetscSplitOwnership(B->comm,&B->m,&B->M);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(B->comm,&B->n,&B->N);CHKERRQ(ierr);

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = PetscMapCreateMPI(B->comm,B->m,B->M,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->N,&B->cmap);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  ierr = PetscMalloc(2*(b->size+2)*sizeof(int),&b->rowners);CHKERRQ(ierr);
  PetscLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIAIJ));
  b->cowners = b->rowners + b->size + 2;
  ierr = MPI_Allgather(&B->m,1,MPI_INT,b->rowners+1,1,MPI_INT,B->comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for (i=2; i<=b->size; i++) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  ierr = MPI_Allgather(&B->n,1,MPI_INT,b->cowners+1,1,MPI_INT,B->comm);CHKERRQ(ierr);
  b->cowners[0] = 0;
  for (i=2; i<=b->size; i++) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash  = PETSC_FALSE;
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = PETSC_TRUE;

  /* stuff used for matrix vector multiply */
  b->lvec      = PETSC_NULL;
  b->Mvctx     = PETSC_NULL;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  /* Explicitly create 2 MATSEQAIJ matrices. */
  ierr = MatCreate(PETSC_COMM_SELF,B->m,B->n,B->m,B->n,&b->A);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQAIJ);CHKERRQ(ierr);
  PetscLogObjectParent(B,b->A);
  ierr = MatCreate(PETSC_COMM_SELF,B->m,B->N,B->m,B->N,&b->B);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJ);CHKERRQ(ierr);
  PetscLogObjectParent(B,b->B);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_MPIAIJ",
                                     MatStoreValues_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_MPIAIJ",
                                     MatRetrieveValues_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetDiagonalBlock_C",
				     "MatGetDiagonalBlock_MPIAIJ",
                                     MatGetDiagonalBlock_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatIsTranspose_C",
				     "MatIsTranspose_MPIAIJ",
				     MatIsTranspose_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAIJSetPreallocation_C",
				     "MatMPIAIJSetPreallocation_MPIAIJ",
				     MatMPIAIJSetPreallocation_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatDiagonalScaleLocal_C",
				     "MatDiagonalScaleLocal_MPIAIJ",
				     MatDiagonalScaleLocal_MPIAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJ when constructed with a single process communicator,
   and MATMPIAIJ otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJ"
int MatCreate_AIJ(Mat A) {
  int ierr,size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATAIJ);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_MPIAIJ"
int MatDuplicate_MPIAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat        mat;
  Mat_MPIAIJ *a,*oldmat = (Mat_MPIAIJ*)matin->data;
  int        ierr;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(matin->comm,matin->m,matin->n,matin->M,matin->N,&mat);CHKERRQ(ierr);
  ierr = MatSetType(mat,matin->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  a    = (Mat_MPIAIJ*)mat->data;
  
  mat->factor       = matin->factor;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;
  mat->preallocated = PETSC_TRUE;

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

  ierr       = PetscMemcpy(a->rowners,oldmat->rowners,2*(a->size+2)*sizeof(int));CHKERRQ(ierr);
  ierr       = MatStashCreate_Private(matin->comm,1,&mat->stash);CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    ierr = PetscMalloc((mat->N)*sizeof(int),&a->colmap);CHKERRQ(ierr);
    PetscLogObjectMemory(mat,(mat->N)*sizeof(int));
    ierr      = PetscMemcpy(a->colmap,oldmat->colmap,(mat->N)*sizeof(int));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;
  if (oldmat->garray) {
    int len;
    len  = oldmat->B->n;
    ierr = PetscMalloc((len+1)*sizeof(int),&a->garray);CHKERRQ(ierr);
    PetscLogObjectMemory(mat,len*sizeof(int));
    if (len) { ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));CHKERRQ(ierr); }
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  PetscLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  PetscLogObjectParent(mat,a->Mvctx);
  ierr =  MatDestroy(a->A);CHKERRQ(ierr);
  ierr =  MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  PetscLogObjectParent(mat,a->A);
  ierr =  MatDestroy(a->B);CHKERRQ(ierr);
  ierr =  MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  PetscLogObjectParent(mat,a->B);
  ierr = PetscFListDuplicate(matin->qlist,&mat->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIAIJ"
int MatLoad_MPIAIJ(PetscViewer viewer,const MatType type,Mat *newmat)
{
  Mat          A;
  PetscScalar  *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          i,nz,ierr,j,rstart,rend,fd;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0,*offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag,cend,cstart,n;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix in special format on disk, cannot load as MPIAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  ierr = PetscMalloc((size+2)*sizeof(int),&rowners);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr    = PetscMalloc(2*(rend-rstart+1)*sizeof(int),&ourlens);CHKERRQ(ierr);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    ierr = PetscMalloc(M*sizeof(int),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(int),&sndcounts);CHKERRQ(ierr);
    for (i=0; i<size; i++) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    ierr = PetscMalloc(size*sizeof(int),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=rowners[i]; j< rowners[i+1]; j++) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(int),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz   = procsnz[0];
    ierr = PetscMalloc(nz*sizeof(int),&mycols);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    ierr = PetscMalloc((nz+1)*sizeof(int),&mycols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
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
  for (i=0; i<m; i++) {
    for (j=0; j<ourlens[i]; j++) {
      if (mycols[jj] < cstart || mycols[jj] >= cend) offlens[i]++;
      jj++;
    }
  }

  /* create our matrix */
  for (i=0; i<m; i++) {
    ourlens[i] -= offlens[i];
  }
  ierr = MatCreate(comm,m,n,M,N,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,type);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,0,ourlens,0,offlens);CHKERRQ(ierr);

  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    nz   = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
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
  *newmat = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_MPIAIJ"
/*
    Not great since it makes two copies of the submatrix, first an SeqAIJ 
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatGetSubMatrices_MPIAIJ()
*/
int MatGetSubMatrix_MPIAIJ(Mat mat,IS isrow,IS iscol,int csize,MatReuse call,Mat *newmat)
{
  int          ierr,i,m,n,rstart,row,rend,nz,*cwork,size,rank,j;
  int          *ii,*jj,nlocal,*dlens,*olens,dlen,olen,jend,mglobal;
  Mat          *local,M,Mreuse;
  PetscScalar  *vwork,*aa;
  MPI_Comm     comm = mat->comm;
  Mat_SeqAIJ   *aij;


  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject *)&Mreuse);CHKERRQ(ierr);
    if (!Mreuse) SETERRQ(1,"Submatrix passed in was not used before, cannot reuse");
    local = &Mreuse;
    ierr  = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&local);CHKERRQ(ierr);
  } else {
    ierr   = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    Mreuse = *local;
    ierr   = PetscFree(local);CHKERRQ(ierr);
  }

  /* 
      m - number of local rows
      n - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  ierr = MatGetSize(Mreuse,&m,&n);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    aij = (Mat_SeqAIJ*)(Mreuse)->data;
    ii  = aij->i;
    jj  = aij->j;

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal 
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    if (csize == PETSC_DECIDE) {
      ierr = ISGetSize(isrow,&mglobal);CHKERRQ(ierr);
      if (mglobal == n) { /* square matrix */
	nlocal = m;
      } else {
        nlocal = n/size + ((n % size) > rank);
      }
    } else {
      nlocal = csize;
    }
    ierr   = MPI_Scan(&nlocal,&rend,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart = rend - nlocal;
    if (rank == size - 1 && rend != n) {
      SETERRQ2(1,"Local column sizes %d do not add up to total number of columns %d",rend,n);
    }

    /* next, compute all the lengths */
    ierr  = PetscMalloc((2*m+1)*sizeof(int),&dlens);CHKERRQ(ierr);
    olens = dlens + m;
    for (i=0; i<m; i++) {
      jend = ii[i+1] - ii[i];
      olen = 0;
      dlen = 0;
      for (j=0; j<jend; j++) {
        if (*jj < rstart || *jj >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }
    ierr = MatCreate(comm,m,nlocal,PETSC_DECIDE,n,&M);CHKERRQ(ierr);
    ierr = MatSetType(M,mat->type_name);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(M,0,dlens,0,olens);CHKERRQ(ierr);
    ierr = PetscFree(dlens);CHKERRQ(ierr);
  } else {
    int ml,nl;

    M = *newmat;
    ierr = MatGetLocalSize(M,&ml,&nl);CHKERRQ(ierr);
    if (ml != m) SETERRQ(PETSC_ERR_ARG_SIZ,"Previous matrix must be same size/layout as request");
    ierr = MatZeroEntries(M);CHKERRQ(ierr);
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE; 
    M->assembled     = PETSC_FALSE;
  }
  ierr = MatGetOwnershipRange(M,&rstart,&rend);CHKERRQ(ierr);
  aij = (Mat_SeqAIJ*)(Mreuse)->data;
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

#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation"
/*@C
   MatMPIAIJSetPreallocation - Creates a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix 
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
int MatMPIAIJSetPreallocation(Mat B,int d_nz,const int d_nnz[],int o_nz,const int o_nnz[])
{
  int ierr,(*f)(Mat,int,const int[],int,const int[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPIAIJSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIAIJ"
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

   When calling this routine with a single process communicator, a matrix of
   type SEQAIJ is returned.  If a matrix of type MPIAIJ is desired for this
   type of communicator, use the construction mechanism:
     MatCreate(...,&A); MatSetType(A,MPIAIJ); MatMPIAIJSetPreallocation(A,...);

   By default, this format uses inodes (identical nodes) when possible.
   We search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_aij_no_inode  - Do not use inodes
.  -mat_aij_inode_limit <limit> - Sets inode limit (max limit=5)
-  -mat_aij_oneindex - Internally use indexing starting at 1
        rather than 0.  Note that when calling MatSetValues(),
        the user still MUST index entries starting at 0!


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
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,int d_nz,const int d_nnz[],int o_nz,const int o_nnz[],Mat *A)
{
  int ierr,size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,M,N,A);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJGetSeqAIJ"
int MatMPIAIJGetSeqAIJ(Mat A,Mat *Ad,Mat *Ao,int *colmap[])
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  PetscFunctionBegin;
  *Ad     = a->A;
  *Ao     = a->B;
  *colmap = a->garray;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "MatSetColoring_MPIAIJ"
int MatSetColoring_MPIAIJ(Mat A,ISColoring coloring)
{
  int        ierr,i;
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;  

  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    ISColoringValue *allcolors,*colors;
    ISColoring      ocoloring;

    /* set coloring for diagonal portion */
    ierr = MatSetColoring_SeqAIJ(a->A,coloring);CHKERRQ(ierr);

    /* set coloring for off-diagonal portion */
    ierr = ISAllGatherColors(A->comm,coloring->n,coloring->colors,PETSC_NULL,&allcolors);CHKERRQ(ierr);
    ierr = PetscMalloc((a->B->n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->B->n; i++) {
      colors[i] = allcolors[a->garray[i]];
    }
    ierr = PetscFree(allcolors);CHKERRQ(ierr);
    ierr = ISColoringCreate(MPI_COMM_SELF,a->B->n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->B,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);
  } else if (coloring->ctype == IS_COLORING_GHOSTED) {
    ISColoringValue *colors;
    int             *larray;
    ISColoring      ocoloring;

    /* set coloring for diagonal portion */
    ierr = PetscMalloc((a->A->n+1)*sizeof(int),&larray);CHKERRQ(ierr);
    for (i=0; i<a->A->n; i++) {
      larray[i] = i + a->cstart;
    }
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,a->A->n,larray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((a->A->n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->A->n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(PETSC_COMM_SELF,a->A->n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->A,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);

    /* set coloring for off-diagonal portion */
    ierr = PetscMalloc((a->B->n+1)*sizeof(int),&larray);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,a->B->n,a->garray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((a->B->n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->B->n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(MPI_COMM_SELF,a->B->n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->B,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"No support ISColoringType %d",coloring->ctype);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic_MPIAIJ"
int MatSetValuesAdic_MPIAIJ(Mat A,void *advalues)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;  
  int        ierr;

  PetscFunctionBegin;
  ierr = MatSetValuesAdic_SeqAIJ(a->A,advalues);CHKERRQ(ierr);
  ierr = MatSetValuesAdic_SeqAIJ(a->B,advalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdifor_MPIAIJ"
int MatSetValuesAdifor_MPIAIJ(Mat A,int nl,void *advalues)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;  
  int        ierr;

  PetscFunctionBegin;
  ierr = MatSetValuesAdifor_SeqAIJ(a->A,nl,advalues);CHKERRQ(ierr);
  ierr = MatSetValuesAdifor_SeqAIJ(a->B,nl,advalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMerge"
/*@C
      MatMerge - Creates a single large PETSc matrix by concatinating sequential
                 matrices from each processor

    Collective on MPI_Comm

   Input Parameters:
+    comm - the communicators the parallel matrix will live on
-    inmat - the input sequential matrices

   Output Parameter:
.    outmat - the parallel matrix generated

    Level: advanced

   Notes: The number of columns of the matrix in EACH of the seperate files
      MUST be the same.

@*/
int MatMerge(MPI_Comm comm,Mat inmat, Mat *outmat)
{
  int               ierr,m,n,i,rstart,nnz,I,*dnz,*onz;
  const int         *indx;
  const PetscScalar *values;
  PetscMap          columnmap,rowmap;

  PetscFunctionBegin;
  
  ierr = MatGetSize(inmat,&m,&n);CHKERRQ(ierr);

  /* count nonzeros in each row, for diagonal and off diagonal portion of matrix */
  ierr = PetscMapCreate(comm,&columnmap);CHKERRQ(ierr);
  ierr = PetscMapSetSize(columnmap,n);CHKERRQ(ierr);
  ierr = PetscMapSetType(columnmap,MAP_MPI);CHKERRQ(ierr);
  ierr = PetscMapGetLocalSize(columnmap,&n);CHKERRQ(ierr);
  ierr = PetscMapDestroy(columnmap);CHKERRQ(ierr);

  ierr = PetscMapCreate(comm,&rowmap);CHKERRQ(ierr);
  ierr = PetscMapSetLocalSize(rowmap,m);CHKERRQ(ierr);
  ierr = PetscMapSetType(rowmap,MAP_MPI);CHKERRQ(ierr);
  ierr = PetscMapGetLocalRange(rowmap,&rstart,0);CHKERRQ(ierr);
  ierr = PetscMapDestroy(rowmap);CHKERRQ(ierr);

  ierr = MatPreallocateInitialize(comm,m,n,dnz,onz);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = MatGetRow(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+rstart,nnz,indx,dnz,onz);CHKERRQ(ierr);
    ierr = MatRestoreRow(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  /* This routine will ONLY return MPIAIJ type matrix */
  ierr = MatCreate(comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,outmat);CHKERRQ(ierr);
  ierr = MatSetType(*outmat,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*outmat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  for (i=0;i<m;i++) {
    ierr = MatGetRow(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
    I    = i + rstart;
    ierr = MatSetValues(*outmat,1,&I,nnz,indx,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  ierr = MatDestroy(inmat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFileSplit"
int MatFileSplit(Mat A,char *outfile)
{
  int               ierr,rank,len,m,N,i,rstart,nnz;
  const int         *indx;
  PetscViewer       out;
  char              *name;
  Mat               B;
  const PetscScalar *values;

  PetscFunctionBegin;
  
  ierr = MatGetLocalSize(A,&m,0);CHKERRQ(ierr);
  ierr = MatGetSize(A,0,&N);CHKERRQ(ierr);
  /* Should this be the type of the diagonal block of A? */ 
  ierr = MatCreate(PETSC_COMM_SELF,m,N,m,N,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,0);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = MatGetRow(A,i+rstart,&nnz,&indx,&values);CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&i,nnz,indx,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i+rstart,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);
  ierr = PetscStrlen(outfile,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+5)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"%s.%d",outfile,rank);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,PETSC_FILE_CREATE,&out);CHKERRQ(ierr);
  ierr = PetscFree(name);
  ierr = MatView(B,out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(out);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
