#define PETSCMAT_DLL

#include "src/mat/impls/aij/mpi/mpiaij.h"   /*I "petscmat.h" I*/
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
PetscErrorCode CreateColmap_MPIAIJ_Private(Mat mat)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       n = aij->B->cmap.n,i;

  PetscFunctionBegin;
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableCreate(n,&aij->colmap);CHKERRQ(ierr); 
  for (i=0; i<n; i++){
    ierr = PetscTableAdd(aij->colmap,aij->garray[i]+1,i+1);CHKERRQ(ierr);
  }
#else
  ierr = PetscMalloc((mat->cmap.N+1)*sizeof(PetscInt),&aij->colmap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(mat,mat->cmap.N*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(aij->colmap,mat->cmap.N*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<n; i++) aij->colmap[aij->garray[i]] = i+1;
#endif
  PetscFunctionReturn(0);
}


#define CHUNKSIZE   15
#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv) \
{ \
    if (col <= lastcol1) low1 = 0; else high1 = nrow1; \
    lastcol1 = col;\
    while (high1-low1 > 5) { \
      t = (low1+high1)/2; \
      if (rp1[t] > col) high1 = t; \
      else             low1  = t; \
    } \
      for (_i=low1; _i<high1; _i++) { \
        if (rp1[_i] > col) break; \
        if (rp1[_i] == col) { \
          if (addv == ADD_VALUES) ap1[_i] += value;   \
          else                    ap1[_i] = value; \
          goto a_noinsert; \
        } \
      }  \
      if (value == 0.0 && ignorezeroentries) goto a_noinsert; \
      if (nonew == 1) goto a_noinsert; \
      if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      MatSeqXAIJReallocateAIJ(A,am,1,nrow1,row,col,rmax1,aa,ai,aj,rp1,ap1,aimax,nonew,MatScalar); \
      N = nrow1++ - 1; a->nz++; high1++; \
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
        rp1[ii+1] = rp1[ii]; \
        ap1[ii+1] = ap1[ii]; \
      } \
      rp1[_i] = col;  \
      ap1[_i] = value;  \
      a_noinsert: ; \
      ailen[row] = nrow1; \
} 


#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv) \
{ \
    if (col <= lastcol2) low2 = 0; else high2 = nrow2; \
    lastcol2 = col;\
    while (high2-low2 > 5) { \
      t = (low2+high2)/2; \
      if (rp2[t] > col) high2 = t; \
      else             low2  = t; \
    } \
       for (_i=low2; _i<high2; _i++) { \
        if (rp2[_i] > col) break; \
        if (rp2[_i] == col) { \
          if (addv == ADD_VALUES) ap2[_i] += value;   \
          else                    ap2[_i] = value; \
          goto b_noinsert; \
        } \
      }  \
      if (value == 0.0 && ignorezeroentries) goto b_noinsert; \
      if (nonew == 1) goto b_noinsert; \
      if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar); \
      N = nrow2++ - 1; b->nz++; high2++;\
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
        rp2[ii+1] = rp2[ii]; \
        ap2[ii+1] = ap2[ii]; \
      } \
      rp2[_i] = col;  \
      ap2[_i] = value;  \
      b_noinsert: ; \
      bilen[row] = nrow2; \
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesRow_MPIAIJ"
PetscErrorCode MatSetValuesRow_MPIAIJ(Mat A,PetscInt row,const PetscScalar v[])
{
  Mat_MPIAIJ     *mat = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)mat->A->data,*b = (Mat_SeqAIJ*)mat->B->data;
  PetscErrorCode ierr;
  PetscInt       l,*garray = mat->garray,diag;

  PetscFunctionBegin;  
  /* code only works for square matrices A */

  /* find size of row to the left of the diagonal part */
  ierr = MatGetOwnershipRange(A,&diag,0);CHKERRQ(ierr);
  row  = row - diag;
  for (l=0; l<b->i[row+1]-b->i[row]; l++) {
    if (garray[b->j[b->i[row]+l]] > diag) break;
  }
  ierr = PetscMemcpy(b->a+b->i[row],v,l*sizeof(PetscScalar));CHKERRQ(ierr);

  /* diagonal part */  
  ierr = PetscMemcpy(a->a+a->i[row],v+l,(a->i[row+1]-a->i[row])*sizeof(PetscScalar));CHKERRQ(ierr);

  /* right of diagonal part */
  ierr = PetscMemcpy(b->a+b->i[row]+l,v+l+a->i[row+1]-a->i[row],(b->i[row+1]-b->i[row]-l)*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIAIJ"
PetscErrorCode MatSetValues_MPIAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscScalar    value;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap.rstart,rend = mat->rmap.rend;
  PetscInt       cstart = mat->cmap.rstart,cend = mat->cmap.rend,row,col;
  PetscTruth     roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat            A = aij->A;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  PetscInt       *aimax = a->imax,*ai = a->i,*ailen = a->ilen,*aj = a->j;
  PetscScalar    *aa = a->a;
  PetscTruth     ignorezeroentries = (((a->ignorezeroentries)&&(addv==ADD_VALUES))?PETSC_TRUE:PETSC_FALSE); 
  Mat            B = aij->B;
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data; 
  PetscInt       *bimax = b->imax,*bi = b->i,*bilen = b->ilen,*bj = b->j,bm = aij->B->rmap.n,am = aij->A->rmap.n;
  PetscScalar    *ba = b->a;

  PetscInt       *rp1,*rp2,ii,nrow1,nrow2,_i,rmax1,rmax2,N,low1,high1,low2,high2,t,lastcol1,lastcol2; 
  PetscInt       nonew = a->nonew; 
  PetscScalar    *ap1,*ap2;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap.N-1);
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row      = im[i] - rstart;
      lastcol1 = -1;
      rp1      = aj + ai[row]; 
      ap1      = aa + ai[row];
      rmax1    = aimax[row]; 
      nrow1    = ailen[row];  
      low1     = 0; 
      high1    = nrow1;
      lastcol2 = -1;
      rp2      = bj + bi[row]; 
      ap2      = ba + bi[row]; 
      rmax2    = bimax[row]; 
      nrow2    = bilen[row];  
      low2     = 0; 
      high2    = nrow2;

      for (j=0; j<n; j++) {
        if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES)) continue;
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          MatSetValues_SeqAIJ_A_Private(row,col,value,addv);
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->cmap.N) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap.N-1);}
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
              rp2      = bj + bi[row]; 
              ap2      = ba + bi[row]; 
              rmax2    = bimax[row]; 
              nrow2    = bilen[row];  
              low2     = 0; 
              high2    = nrow2;
              bm       = aij->B->rmap.n;
              ba = b->a;
            }
          } else col = in[j];
          MatSetValues_SeqAIJ_B_Private(row,col,value,addv);
        }
      }
    } else {
      if (!aij->donotstash) {
        if (roworiented) {
          if (ignorezeroentries && v[i*n] == 0.0) continue;
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n);CHKERRQ(ierr);
        } else {
          if (ignorezeroentries && v[i] == 0.0) continue;
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPIAIJ"
PetscErrorCode MatGetValues_MPIAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap.rstart,rend = mat->rmap.rend;
  PetscInt       cstart = mat->cmap.rstart,cend = mat->cmap.rend,row,col;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",idxm[i]);
    if (idxm[i] >= mat->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm[i],mat->rmap.N-1);
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",idxn[j]);
        if (idxn[j] >= mat->cmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",idxn[j],mat->cmap.N-1);
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
PetscErrorCode MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;
  InsertMode     addv;

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

  ierr = MatStashScatterBegin_Private(&mat->stash,mat->rmap.range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(aij->A,"Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIAIJ"
PetscErrorCode MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ *)aij->A->data;
  PetscErrorCode ierr;
  PetscMPIInt    n;
  PetscInt       i,j,rstart,ncols,flg;
  PetscInt       *row,*col,other_disassembled;
  PetscScalar    *val;
  InsertMode     addv = mat->insertmode;

  /* do not use 'b = (Mat_SeqAIJ *)aij->B->data' as B can be reset in disassembly */
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
  a->compressedrow.use     = PETSC_FALSE;
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
  ierr = MatSetOption(aij->B,MAT_DO_NOT_USE_INODES);CHKERRQ(ierr);
  ((Mat_SeqAIJ *)aij->B->data)->compressedrow.use = PETSC_TRUE; /* b->compressedrow.use */
  ierr = MatAssemblyBegin(aij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->B,mode);CHKERRQ(ierr);

  ierr = PetscFree(aij->rowvalues);CHKERRQ(ierr);
  aij->rowvalues = 0;

  /* used by MatAXPY() */
  a->xtoy = 0; ((Mat_SeqAIJ *)aij->B->data)->xtoy = 0;  /* b->xtoy = 0 */
  a->XtoY = 0; ((Mat_SeqAIJ *)aij->B->data)->XtoY = 0;  /* b->XtoY = 0 */

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIAIJ"
PetscErrorCode MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIAIJ"
PetscErrorCode MatZeroRows_MPIAIJ(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscMPIInt    size = l->size,imdex,n,rank = l->rank,tag = A->tag,lastidx = -1;
  PetscInt       i,*owners = A->rmap.range;
  PetscInt       *nprocs,j,idx,nsends,row;
  PetscInt       nmax,*svalues,*starts,*owner,nrecvs;
  PetscInt       *rvalues,count,base,slen,*source;
  PetscInt       *lens,*lrows,*values,rstart=A->rmap.rstart;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
#if defined(PETSC_DEBUG)
  PetscTruth     found = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  /*  first count number of contributors to each processor */
  ierr = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&owner);CHKERRQ(ierr); /* see note*/
  j = 0;
  for (i=0; i<N; i++) {
    if (lastidx > (idx = rows[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; 
        nprocs[2*j+1] = 1; 
        owner[i] = j; 
#if defined(PETSC_DEBUG)
        found = PETSC_TRUE; 
#endif
        break;
      }
    }
#if defined(PETSC_DEBUG)
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(PetscInt),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&starts);CHKERRQ(ierr);
  starts[0] = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr   = PetscMalloc(2*(nrecvs+1)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr = PetscMalloc((slen+1)*sizeof(PetscInt),&lrows);CHKERRQ(ierr);
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
  /*
        Zero the required rows. If the "diagonal block" of the matrix
     is square and the user wishes to set the diagonal we use separate
     code so that MatSetValues() is not called for each diagonal allocating
     new memory, thus calling lots of mallocs and slowing things down.

       Contributed by: Matthew Knepley
  */
  /* must zero l->B before l->A because the (diag) case below may put values into l->B*/
  ierr = MatZeroRows(l->B,slen,lrows,0.0);CHKERRQ(ierr); 
  if ((diag != 0.0) && (l->A->rmap.N == l->A->cmap.N)) {
    ierr      = MatZeroRows(l->A,slen,lrows,diag);CHKERRQ(ierr);
  } else if (diag != 0.0) {
    ierr = MatZeroRows(l->A,slen,lrows,0.0);CHKERRQ(ierr);
    if (((Mat_SeqAIJ*)l->A->data)->nonew) {
      SETERRQ(PETSC_ERR_SUP,"MatZeroRows() on rectangular matrices cannot be used with the Mat options\n\
MAT_NO_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    }
    for (i = 0; i < slen; i++) {
      row  = lrows[i] + rstart;
      ierr = MatSetValues(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows(l->A,slen,lrows,0.0);CHKERRQ(ierr);
  }
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
PetscErrorCode MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap.n) {
    SETERRQ2(PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap.n,nt);
  }
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIAIJ"
PetscErrorCode MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIAIJ"
PetscErrorCode MatMultTranspose_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscTruth     merged;

  PetscFunctionBegin;
  ierr = VecScatterGetMerged(a->Mvctx,&merged);CHKERRQ(ierr);
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  if (!merged) {
    /* send it on its way */
    ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
    /* do local part */
    ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
    /* receive remote parts: note this assumes the values are not actually */
    /* added in yy until the next line, */
    ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  } else {
    /* do local part */
    ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
    /* send it on its way */
    ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
    /* values actually were received in the Begin() but we need to call this nop */
    ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatIsTranspose_MPIAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatIsTranspose_MPIAIJ(Mat Amat,Mat Bmat,PetscReal tol,PetscTruth *f)
{
  MPI_Comm       comm;
  Mat_MPIAIJ     *Aij = (Mat_MPIAIJ *) Amat->data, *Bij;
  Mat            Adia = Aij->A, Bdia, Aoff,Boff,*Aoffs,*Boffs;
  IS             Me,Notme;
  PetscErrorCode ierr;
  PetscInt       M,N,first,last,*notme,i;
  PetscMPIInt    size;

  PetscFunctionBegin;

  /* Easy test: symmetric diagonal block */
  Bij = (Mat_MPIAIJ *) Bmat->data; Bdia = Bij->A;
  ierr = MatIsTranspose(Adia,Bdia,tol,f);CHKERRQ(ierr);
  if (!*f) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  /* Hard test: off-diagonal block. This takes a MatGetSubMatrix. */
  ierr = MatGetSize(Amat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&first,&last);CHKERRQ(ierr);
  ierr = PetscMalloc((N-last+first)*sizeof(PetscInt),&notme);CHKERRQ(ierr);
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
PetscErrorCode MatMultTransposeAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransposeadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* receive remote parts */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIAIJ"
PetscErrorCode MatGetDiagonal_MPIAIJ(Mat A,Vec v)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  if (A->rmap.N != A->cmap.N) SETERRQ(PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  if (A->rmap.rstart != A->cmap.rstart || A->rmap.rend != A->cmap.rend) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"row partition must equal col partition");  
  }
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIAIJ"
PetscErrorCode MatScale_MPIAIJ(Mat A,PetscScalar aa)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ"
PetscErrorCode MatDestroy_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D",mat->rmap.N,mat->cmap.N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatDestroy(aij->A);CHKERRQ(ierr);
  ierr = MatDestroy(aij->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
  if (aij->colmap) {ierr = PetscTableDelete(aij->colmap);CHKERRQ(ierr);}
#else
  ierr = PetscFree(aij->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(aij->garray);CHKERRQ(ierr);
  if (aij->lvec)   {ierr = VecDestroy(aij->lvec);CHKERRQ(ierr);}
  if (aij->Mvctx)  {ierr = VecScatterDestroy(aij->Mvctx);CHKERRQ(ierr);}
  ierr = PetscFree(aij->rowvalues);CHKERRQ(ierr);
  ierr = PetscFree(aij);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatIsTranspose_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIAIJSetPreallocationCSR_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDiagonalScaleLocal_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAIJ_Binary"
PetscErrorCode MatView_MPIAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ*       A = (Mat_SeqAIJ*)aij->A->data;
  Mat_SeqAIJ*       B = (Mat_SeqAIJ*)aij->B->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size,tag = ((PetscObject)viewer)->tag;
  int               fd;
  PetscInt          nz,header[4],*row_lengths,*range=0,rlen,i;
  PetscInt          nzmax,*column_indices,j,k,col,*garray = aij->garray,cnt,cstart = mat->cmap.rstart,rnz;
  PetscScalar       *column_values;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);
  nz   = A->nz + B->nz;
  if (!rank) {
    header[0] = MAT_FILE_COOKIE;
    header[1] = mat->rmap.N;
    header[2] = mat->cmap.N;
    ierr = MPI_Reduce(&nz,&header[3],1,MPIU_INT,MPI_SUM,0,mat->comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,header,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    /* get largest number of rows any processor has */
    rlen = mat->rmap.n;
    range = mat->rmap.range;
    for (i=1; i<size; i++) {
      rlen = PetscMax(rlen,range[i+1] - range[i]);
    }
  } else {
    ierr = MPI_Reduce(&nz,0,1,MPIU_INT,MPI_SUM,0,mat->comm);CHKERRQ(ierr);
    rlen = mat->rmap.n;
  }

  /* load up the local row counts */
  ierr = PetscMalloc((rlen+1)*sizeof(PetscInt),&row_lengths);CHKERRQ(ierr);
  for (i=0; i<mat->rmap.n; i++) {
    row_lengths[i] = A->i[i+1] - A->i[i] + B->i[i+1] - B->i[i];
  }

  /* store the row lengths to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,row_lengths,mat->rmap.n,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      rlen = range[i+1] - range[i];
      ierr = MPI_Recv(row_lengths,rlen,MPIU_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,row_lengths,rlen,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(row_lengths,mat->rmap.n,MPIU_INT,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(row_lengths);CHKERRQ(ierr);

  /* load up the local column indices */
  nzmax = nz; /* )th processor needs space a largest processor needs */
  ierr = MPI_Reduce(&nz,&nzmax,1,MPIU_INT,MPI_MAX,0,mat->comm);CHKERRQ(ierr);
  ierr = PetscMalloc((nzmax+1)*sizeof(PetscInt),&column_indices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<mat->rmap.n; i++) {
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
  if (cnt != A->nz + B->nz) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: cnt = %D nz = %D",cnt,A->nz+B->nz);

  /* store the column indices to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_indices,nz,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&rnz,1,MPIU_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      if (rnz > nzmax) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: nz = %D nzmax = %D",nz,nzmax);
      ierr = MPI_Recv(column_indices,rnz,MPIU_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_indices,rnz,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(&nz,1,MPIU_INT,0,tag,mat->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_indices,nz,MPIU_INT,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_indices);CHKERRQ(ierr);

  /* load up the local column values */
  ierr = PetscMalloc((nzmax+1)*sizeof(PetscScalar),&column_values);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<mat->rmap.n; i++) {
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
  if (cnt != A->nz + B->nz) SETERRQ2(PETSC_ERR_PLIB,"Internal PETSc error: cnt = %D nz = %D",cnt,A->nz+B->nz);

  /* store the column values to the file */
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_values,nz,PETSC_SCALAR,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&rnz,1,MPIU_INT,i,tag,mat->comm,&status);CHKERRQ(ierr);
      if (rnz > nzmax) SETERRQ2(PETSC_ERR_LIB,"Internal PETSc error: nz = %D nzmax = %D",nz,nzmax);
      ierr = MPI_Recv(column_values,rnz,MPIU_SCALAR,i,tag,mat->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_values,rnz,PETSC_SCALAR,PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Send(&nz,1,MPIU_INT,0,tag,mat->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_values,nz,MPIU_SCALAR,0,tag,mat->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIAIJ_ASCIIorDraworSocket"
PetscErrorCode MatView_MPIAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank = aij->rank,size = aij->size;
  PetscTruth        isdraw,iascii,isbinary;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) { 
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo    info;
      PetscTruth inodes;

      ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatInodeGetInodeSizes(aij->A,PETSC_NULL,(PetscInt **)&inodes,PETSC_NULL);CHKERRQ(ierr);
      if (!inodes) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D mem %D, not using I-node routines\n",
					      rank,mat->rmap.n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D mem %D, using I-node routines\n",
		    rank,mat->rmap.n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
      }
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecScatterView(aij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscInt   inodecount,inodelimit,*inodes;
      ierr = MatInodeGetInodeSizes(aij->A,&inodecount,&inodes,&inodelimit);CHKERRQ(ierr);
      if (inodes) {
        ierr = PetscViewerASCIIPrintf(viewer,"using I-node (on process 0) routines: found %D nodes, limit used is %D\n",inodecount,inodelimit);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"not using I-node (on process 0) routines\n");CHKERRQ(ierr);
      }
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
    Mat_SeqAIJ  *Aloc;
    PetscInt    M = mat->rmap.N,N = mat->cmap.N,m,*ai,*aj,row,*cols,i,*ct;
    PetscScalar *a;

    ierr = MatCreate(mat->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    /* This is just a temporary matrix, so explicitly using MATMPIAIJ is probably best */
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc = (Mat_SeqAIJ*)aij->A->data;
    m = aij->A->rmap.n; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row = mat->rmap.rstart;
    for (i=0; i<ai[m]; i++) {aj[i] += mat->cmap.rstart ;}
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERT_VALUES);CHKERRQ(ierr);
      row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
    } 
    aj = Aloc->j;
    for (i=0; i<ai[m]; i++) {aj[i] -= mat->cmap.rstart;}

    /* copy over the B part */
    Aloc = (Mat_SeqAIJ*)aij->B->data;
    m    = aij->B->rmap.n;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
    row  = mat->rmap.rstart;
    ierr = PetscMalloc((ai[m]+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
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
PetscErrorCode MatView_MPIAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii,isdraw,issocket,isbinary;
 
  PetscFunctionBegin;
  ierr  = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr  = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  if (iascii || isdraw || isbinary || issocket) { 
    ierr = MatView_MPIAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by MPIAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatRelax_MPIAIJ"
PetscErrorCode MatRelax_MPIAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPIAIJ     *mat = (Mat_MPIAIJ*)matin->data;
  PetscErrorCode ierr; 
  Vec            bb1;

  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);

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
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
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
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
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
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
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
#define __FUNCT__ "MatPermute_MPIAIJ"
PetscErrorCode MatPermute_MPIAIJ(Mat A,IS rowp,IS colp,Mat *B)
{
  MPI_Comm       comm,pcomm;
  PetscInt       first,local_size,nrows,*rows;
  int            ntids;
  IS             crowp,growp,irowp,lrowp,lcolp,icolp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm); CHKERRQ(ierr);
  /* make a collective version of 'rowp' */
  ierr = PetscObjectGetComm((PetscObject)rowp,&pcomm); CHKERRQ(ierr);
  if (pcomm==comm) {
    crowp = rowp;
  } else {
    ierr = ISGetSize(rowp,&nrows); CHKERRQ(ierr);
    ierr = ISGetIndices(rowp,&rows); CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nrows,rows,&crowp); CHKERRQ(ierr);
    ierr = ISRestoreIndices(rowp,&rows); CHKERRQ(ierr);
  }
  /* collect the global row permutation and invert it */
  ierr = ISAllGather(crowp,&growp); CHKERRQ(ierr);
  ierr = ISSetPermutation(growp); CHKERRQ(ierr);
  if (pcomm!=comm) {
    ierr = ISDestroy(crowp); CHKERRQ(ierr);
  }
  ierr = ISInvertPermutation(growp,PETSC_DECIDE,&irowp);CHKERRQ(ierr);
  /* get the local target indices */
  ierr = MatGetOwnershipRange(A,&first,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&local_size,PETSC_NULL); CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&rows); CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,local_size,rows+first,&lrowp); CHKERRQ(ierr);
  ierr = ISRestoreIndices(irowp,&rows); CHKERRQ(ierr);
  ierr = ISDestroy(irowp); CHKERRQ(ierr);
  /* the column permutation is so much easier;
     make a local version of 'colp' and invert it */
  ierr = PetscObjectGetComm((PetscObject)colp,&pcomm); CHKERRQ(ierr);
  ierr = MPI_Comm_size(pcomm,&ntids); CHKERRQ(ierr);
  if (ntids==1) {
    lcolp = colp;
  } else {
    ierr = ISGetSize(colp,&nrows); CHKERRQ(ierr);
    ierr = ISGetIndices(colp,&rows); CHKERRQ(ierr);
    ierr = ISCreateGeneral(MPI_COMM_SELF,nrows,rows,&lcolp); CHKERRQ(ierr);
  }
  ierr = ISInvertPermutation(lcolp,PETSC_DECIDE,&icolp); CHKERRQ(ierr);
  ierr = ISSetPermutation(lcolp); CHKERRQ(ierr);
  if (ntids>1) {
    ierr = ISRestoreIndices(colp,&rows); CHKERRQ(ierr);
    ierr = ISDestroy(lcolp); CHKERRQ(ierr);
  }
  /* now we just get the submatrix */
  ierr = MatGetSubMatrix(A,lrowp,icolp,local_size,MAT_INITIAL_MATRIX,B); CHKERRQ(ierr);
  /* clean up */
  ierr = ISDestroy(lrowp); CHKERRQ(ierr);
  ierr = ISDestroy(icolp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIAIJ"
PetscErrorCode MatGetInfo_MPIAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIAIJ     *mat = (Mat_MPIAIJ*)matin->data;
  Mat            A = mat->A,B = mat->B;
  PetscErrorCode ierr;
  PetscReal      isend[5],irecv[5];

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
  info->rows_global       = (double)matin->rmap.N;
  info->columns_global    = (double)matin->cmap.N;
  info->rows_local        = (double)matin->rmap.n;
  info->columns_local     = (double)matin->cmap.N;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAIJ"
PetscErrorCode MatSetOption_MPIAIJ(Mat A,MatOption op)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

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
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
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
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    break;
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
    SETERRQ1(PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIAIJ"
PetscErrorCode MatGetRow_MPIAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIAIJ     *mat = (Mat_MPIAIJ*)matin->data;
  PetscScalar    *vworkA,*vworkB,**pvA,**pvB,*v_p;
  PetscErrorCode ierr;
  PetscInt       i,*cworkA,*cworkB,**pcA,**pcB,cstart = matin->cmap.rstart;
  PetscInt       nztot,nzA,nzB,lrow,rstart = matin->rmap.rstart,rend = matin->rmap.rend;
  PetscInt       *cmap,*idx_p;

  PetscFunctionBegin;
  if (mat->getrowactive) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ*)mat->A->data,*Ba = (Mat_SeqAIJ*)mat->B->data; 
    PetscInt     max = 1,tmp;
    for (i=0; i<matin->rmap.n; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    ierr = PetscMalloc(max*(sizeof(PetscInt)+sizeof(PetscScalar)),&mat->rowvalues);CHKERRQ(ierr);
    mat->rowindices = (PetscInt*)(mat->rowvalues + max);
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
      PetscInt imark = -1;
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
PetscErrorCode MatRestoreRow_MPIAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*)mat->data;

  PetscFunctionBegin;
  if (!aij->getrowactive) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"MatGetRow() must be called first");
  }
  aij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIAIJ"
PetscErrorCode MatNorm_MPIAIJ(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *amat = (Mat_SeqAIJ*)aij->A->data,*bmat = (Mat_SeqAIJ*)aij->B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,cstart = mat->cmap.rstart;
  PetscReal      sum = 0.0;
  PetscScalar    *v;

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
      PetscInt    *jj,*garray = aij->garray;
      ierr = PetscMalloc((mat->cmap.N+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
      ierr = PetscMalloc((mat->cmap.N+1)*sizeof(PetscReal),&tmp2);CHKERRQ(ierr);
      ierr = PetscMemzero(tmp,mat->cmap.N*sizeof(PetscReal));CHKERRQ(ierr);
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for (j=0; j<amat->nz; j++) {
        tmp[cstart + *jj++ ] += PetscAbsScalar(*v);  v++;
      }
      v = bmat->a; jj = bmat->j;
      for (j=0; j<bmat->nz; j++) {
        tmp[garray[*jj++]] += PetscAbsScalar(*v); v++;
      }
      ierr = MPI_Allreduce(tmp,tmp2,mat->cmap.N,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      for (j=0; j<mat->cmap.N; j++) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscFree(tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp = 0.0;
      for (j=0; j<aij->A->rmap.n; j++) {
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
PetscErrorCode MatTranspose_MPIAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *Aloc = (Mat_SeqAIJ*)a->A->data;
  PetscErrorCode ierr;
  PetscInt       M = A->rmap.N,N = A->cmap.N,m,*ai,*aj,row,*cols,i,*ct;
  Mat            B;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (!matout && M != N) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  }

  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->cmap.n,A->rmap.n,N,M);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_SeqAIJ*)a->A->data;
  m = a->A->rmap.n; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = A->rmap.rstart;
  for (i=0; i<ai[m]; i++) {aj[i] += A->cmap.rstart ;}
  for (i=0; i<m; i++) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for (i=0; i<ai[m]; i++) {aj[i] -= A->cmap.rstart ;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*)a->B->data;
  m = a->B->rmap.n;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row  = A->rmap.rstart;
  ierr = PetscMalloc((1+ai[m])*sizeof(PetscInt),&cols);CHKERRQ(ierr);
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
PetscErrorCode MatDiagonalScale_MPIAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat            a = aij->A,b = aij->B;
  PetscErrorCode ierr;
  PetscInt       s1,s2,s3;

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
#define __FUNCT__ "MatSetBlockSize_MPIAIJ"
PetscErrorCode MatSetBlockSize_MPIAIJ(Mat A,PetscInt bs)
{
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetBlockSize(a->A,bs);CHKERRQ(ierr);
  ierr = MatSetBlockSize(a->B,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatSetUnfactored_MPIAIJ"
PetscErrorCode MatSetUnfactored_MPIAIJ(Mat A)
{
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIAIJ"
PetscErrorCode MatEqual_MPIAIJ(Mat A,Mat B,PetscTruth *flag)
{
  Mat_MPIAIJ     *matB = (Mat_MPIAIJ*)B->data,*matA = (Mat_MPIAIJ*)A->data;
  Mat            a,b,c,d;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_MPIAIJ"
PetscErrorCode MatCopy_MPIAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJ     *b = (Mat_MPIAIJ *)B->data;

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
PetscErrorCode MatSetUpPreallocation_MPIAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_MPIAIJ"
PetscErrorCode MatAXPY_MPIAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Mat_MPIAIJ     *xx = (Mat_MPIAIJ *)X->data,*yy = (Mat_MPIAIJ *)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqAIJ     *x,*y;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {  
    PetscScalar alpha = a;
    x = (Mat_SeqAIJ *)xx->A->data;
    y = (Mat_SeqAIJ *)yy->A->data;
    bnz = (PetscBLASInt)x->nz;
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);    
    x = (Mat_SeqAIJ *)xx->B->data;
    y = (Mat_SeqAIJ *)yy->B->data;
    bnz = (PetscBLASInt)x->nz;
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);
  } else if (str == SUBSET_NONZERO_PATTERN) {  
    ierr = MatAXPY_SeqAIJ(yy->A,a,xx->A,str);CHKERRQ(ierr);

    x = (Mat_SeqAIJ *)xx->B->data;
    y = (Mat_SeqAIJ *)yy->B->data;
    if (y->xtoy && y->XtoY != xx->B) {
      ierr = PetscFree(y->xtoy);CHKERRQ(ierr);
      ierr = MatDestroy(y->XtoY);CHKERRQ(ierr);
    }
    if (!y->xtoy) { /* get xtoy */
      ierr = MatAXPYGetxtoy_Private(xx->B->rmap.n,x->i,x->j,xx->garray,y->i,y->j,yy->garray,&y->xtoy);CHKERRQ(ierr);
      y->XtoY = xx->B;
    } 
    for (i=0; i<x->nz; i++) y->a[y->xtoy[i]] += a*(x->a[i]);   
  } else {
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConjugate_SeqAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatConjugate_MPIAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConjugate_MPIAIJ(Mat mat)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  ierr = MatConjugate_SeqAIJ(aij->A);CHKERRQ(ierr);
  ierr = MatConjugate_SeqAIJ(aij->B);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_MPIAIJ"
PetscErrorCode MatRealPart_MPIAIJ(Mat A)
{
  Mat_MPIAIJ   *a = (Mat_MPIAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_MPIAIJ"
PetscErrorCode MatImaginaryPart_MPIAIJ(Mat A)
{
  Mat_MPIAIJ   *a = (Mat_MPIAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_PBGL
#include <boost/parallel/mpi/bsp_process_group.hpp>
typedef boost::parallel::mpi::bsp_process_group            process_group_type;

#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/parallel/mpi/bsp_process_group.hpp>

#include <boost/graph/distributed/petsc/interface.hpp>
#include <boost/graph/distributed/ilu_0.hpp>

namespace petsc = boost::distributed::petsc;
using namespace std;
typedef double                                             value_type;
typedef boost::graph::distributed::ilu_elimination_state   elimination_state;
typedef boost::adjacency_list<boost::listS,
                       boost::distributedS<process_group_type, boost::vecS>,
                       boost::bidirectionalS,
                       // Vertex properties
                       boost::no_property,
                       // Edge properties
                       boost::property<boost::edge_weight_t, value_type,
                         boost::property<boost::edge_finished_t, elimination_state> > > graph_type;

typedef boost::graph_traits<graph_type>::vertex_descriptor        vertex_type;
typedef boost::graph_traits<graph_type>::edge_descriptor          edge_type;
typedef boost::property_map<graph_type, boost::edge_weight_t>::type      weight_map_type;

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_MPIAIJ"
/*
  This uses the parallel ILU factorization of Peter Gottschling <pgottsch@osl.iu.edu>
*/
PetscErrorCode MatILUFactorSymbolic_MPIAIJ(Mat A, IS isrow, IS iscol, MatFactorInfo *info, Mat *fact)
{
  PetscTruth           row_identity, col_identity;
  PetscObjectContainer c;
  PetscInt             m, n, M, N;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only levels = 0 supported for parallel ilu");
  ierr = ISIdentity(isrow, &row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol, &col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Row and column permutations must be identity for parallel ILU");
  }

  process_group_type pg;
  graph_type*        graph_p = new graph_type(petsc::num_global_vertices(A), pg, petsc::matrix_distribution(A, pg));
  graph_type&        graph   = *graph_p;
  petsc::read_matrix(A, graph, get(boost::edge_weight, graph));

  //write_graphviz("petsc_matrix_as_graph.dot", graph, default_writer(), matrix_graph_writer<graph_type>(graph));
  boost::property_map<graph_type, boost::edge_finished_t>::type finished = get(boost::edge_finished, graph);
  BGL_FORALL_EDGES(e, graph, graph_type)
    put(finished, e, boost::graph::distributed::unseen);

  ilu_0(graph, get(boost::edge_weight, graph), get(boost::edge_finished, graph));

  /* put together the new matrix */
  ierr = MatCreate(A->comm, fact);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
  ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact, m, n, M, N);CHKERRQ(ierr);
  ierr = MatSetType(*fact, A->type_name);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*fact, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*fact, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  (*fact)->factor = FACTOR_LU;

  ierr = PetscObjectContainerCreate(A->comm, &c);
  ierr = PetscObjectContainerSetPointer(c, graph_p);
  ierr = PetscObjectCompose((PetscObject) (*fact), "graph", (PetscObject) c);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_MPIAIJ"
PetscErrorCode MatLUFactorNumeric_MPIAIJ(Mat A, MatFactorInfo *info, Mat *B)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ"
/*
  This uses the parallel ILU factorization of Peter Gottschling <pgottsch@osl.iu.edu>
*/
PetscErrorCode MatSolve_MPIAIJ(Mat A, Vec b, Vec x)
{
  graph_type*          graph_p;
  PetscObjectContainer c;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "graph", (PetscObject *) &c);CHKERRQ(ierr);
  ierr = PetscObjectContainerGetPointer(c, (void **) &graph_p);CHKERRQ(ierr);
  ierr = VecCopy(b, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,
       MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,
/* 4*/ MatMultAdd_MPIAIJ,
       MatMultTranspose_MPIAIJ,
       MatMultTransposeAdd_MPIAIJ,
#ifdef PETSC_HAVE_PBGL
       MatSolve_MPIAIJ,
#else
       0,
#endif
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
       0,
#ifdef PETSC_HAVE_PBGL
       MatLUFactorNumeric_MPIAIJ,
#else
       0,
#endif
       0,
       0,
/*30*/ MatSetUpPreallocation_MPIAIJ,
#ifdef PETSC_HAVE_PBGL
       MatILUFactorSymbolic_MPIAIJ,
#else
       0,
#endif
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
/*45*/ 0,
       MatScale_MPIAIJ,
       0,
       0,
       0,
/*50*/ MatSetBlockSize_MPIAIJ,
       0,
       0,
       0,
       0,
/*55*/ MatFDColoringCreate_MPIAIJ,
       0,
       MatSetUnfactored_MPIAIJ,
       MatPermute_MPIAIJ,
       0,
/*60*/ MatGetSubMatrix_MPIAIJ,
       MatDestroy_MPIAIJ,
       MatView_MPIAIJ,
       0,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       MatSetColoring_MPIAIJ,
#if defined(PETSC_HAVE_ADIC)
       MatSetValuesAdic_MPIAIJ,
#else
       0,
#endif
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
/*84*/ MatLoad_MPIAIJ,
       0,
       0,
       0,
       0,
       0,
/*90*/ MatMatMult_MPIAIJ_MPIAIJ, 
       MatMatMultSymbolic_MPIAIJ_MPIAIJ,  
       MatMatMultNumeric_MPIAIJ_MPIAIJ, 
       MatPtAP_Basic,
       MatPtAPSymbolic_MPIAIJ,
/*95*/ MatPtAPNumeric_MPIAIJ,                                
       0,
       0,
       0,
       0,
/*100*/0,
       MatPtAPSymbolic_MPIAIJ_MPIAIJ,
       MatPtAPNumeric_MPIAIJ_MPIAIJ,
       MatConjugate_MPIAIJ,
       0,
/*105*/MatSetValuesRow_MPIAIJ,
       MatRealPart_MPIAIJ,
       MatImaginaryPart_MPIAIJ};

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_MPIAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatStoreValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_MPIAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatRetrieveValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ *)mat->data;
  PetscErrorCode ierr;

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
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocation_MPIAIJ(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ     *b;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  B->preallocated = PETSC_TRUE;
  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
  if (d_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);

  B->rmap.bs = B->cmap.bs = 1;
  ierr = PetscMapInitialize(B->comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(B->comm,&B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap.n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap.n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  b = (Mat_MPIAIJ*)B->data;

  /* Explicitly create 2 MATSEQAIJ matrices. */
  ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
  ierr = MatSetSizes(b->A,B->rmap.n,B->cmap.n,B->rmap.n,B->cmap.n);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
  ierr = MatSetSizes(b->B,B->rmap.n,B->cmap.N,B->rmap.n,B->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);

  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_MPIAIJ"
PetscErrorCode MatDuplicate_MPIAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIAIJ     *a,*oldmat = (Mat_MPIAIJ*)matin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(matin->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,matin->rmap.n,matin->cmap.n,matin->rmap.N,matin->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(mat,matin->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  a    = (Mat_MPIAIJ*)mat->data;
  
  mat->factor       = matin->factor;
  mat->rmap.bs      = matin->rmap.bs;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;
  mat->preallocated = PETSC_TRUE;

  a->size           = oldmat->size;
  a->rank           = oldmat->rank;
  a->donotstash     = oldmat->donotstash;
  a->roworiented    = oldmat->roworiented;
  a->rowindices     = 0;
  a->rowvalues      = 0;
  a->getrowactive   = PETSC_FALSE;

  ierr = PetscMapCopy(mat->comm,&matin->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscMapCopy(mat->comm,&matin->cmap,&mat->cmap);CHKERRQ(ierr);

  ierr = MatStashCreate_Private(matin->comm,1,&mat->stash);CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    ierr = PetscMalloc((mat->cmap.N)*sizeof(PetscInt),&a->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(mat,(mat->cmap.N)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->colmap,oldmat->colmap,(mat->cmap.N)*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;
  if (oldmat->garray) {
    PetscInt len;
    len  = oldmat->B->cmap.n;
    ierr = PetscMalloc((len+1)*sizeof(PetscInt),&a->garray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(mat,len*sizeof(PetscInt));CHKERRQ(ierr);
    if (len) { ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(PetscInt));CHKERRQ(ierr); }
  } else a->garray = 0;
  
  ierr = VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->Mvctx);CHKERRQ(ierr);
  ierr = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  ierr = MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->B);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(matin->qlist,&mat->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIAIJ"
PetscErrorCode MatLoad_MPIAIJ(PetscViewer viewer, MatType type,Mat *newmat)
{
  Mat            A;
  PetscScalar    *vals,*svals;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag,maxnz;
  PetscInt       i,nz,j,rstart,rend,mmax;
  PetscInt       header[4],*rowlengths = 0,M,N,m,*cols;
  PetscInt       *ourlens = PETSC_NULL,*procsnz = PETSC_NULL,*offlens = PETSC_NULL,jj,*mycols,*smycols;
  PetscInt       cend,cstart,n,*rowners;
  int            fd;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
  }

  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m    = M/size + ((M % size) > rank);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);

  /* First process needs enough room for process with most rows */
  if (!rank) {
    mmax       = rowners[1];
    for (i=2; i<size; i++) {
      mmax = PetscMax(mmax,rowners[i]);
    }
  } else mmax = m;

  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    mmax       = PetscMax(mmax,rowners[i]);
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr    = PetscMalloc2(mmax,PetscInt,&ourlens,mmax,PetscInt,&offlens);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryRead(fd,ourlens,m,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc(m*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(PetscInt),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(PetscInt));CHKERRQ(ierr);
    for (j=0; j<m; j++) {
      procsnz[0] += ourlens[j];
    }
    for (i=1; i<size; i++) {
      ierr = PetscBinaryRead(fd,rowlengths,rowners[i+1]-rowners[i],PETSC_INT);CHKERRQ(ierr);
      /* calculate the number of nonzeros on each processor */
      for (j=0; j<rowners[i+1]-rowners[i]; j++) {
        procsnz[i] += rowlengths[j];
      }
      ierr = MPI_Send(rowlengths,rowners[i+1]-rowners[i],MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  } else {
    ierr = MPI_Recv(ourlens,m,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
  }

  if (!rank) {
    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(PetscInt),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz   = procsnz[0];
    ierr = PetscMalloc(nz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    ierr = PetscMalloc(nz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

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

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,type);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,0,ourlens,0,offlens);CHKERRQ(ierr);

  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    ierr = PetscMalloc((maxnz+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    nz   = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues_MPIAIJ(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr     = MatSetValues_MPIAIJ(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  ierr = PetscFree2(ourlens,offlens);CHKERRQ(ierr);
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
PetscErrorCode MatGetSubMatrix_MPIAIJ(Mat mat,IS isrow,IS iscol,PetscInt csize,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,m,n,rstart,row,rend,nz,*cwork,j;
  PetscInt       *ii,*jj,nlocal,*dlens,*olens,dlen,olen,jend,mglobal;
  Mat            *local,M,Mreuse;
  PetscScalar    *vwork,*aa;
  MPI_Comm       comm = mat->comm;
  Mat_SeqAIJ     *aij;


  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject *)&Mreuse);CHKERRQ(ierr);
    if (!Mreuse) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
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
    ierr   = MPI_Scan(&nlocal,&rend,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart = rend - nlocal;
    if (rank == size - 1 && rend != n) {
      SETERRQ2(PETSC_ERR_ARG_SIZ,"Local column sizes %D do not add up to total number of columns %D",rend,n);
    }

    /* next, compute all the lengths */
    ierr  = PetscMalloc((2*m+1)*sizeof(PetscInt),&dlens);CHKERRQ(ierr);
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
    ierr = MatCreate(comm,&M);CHKERRQ(ierr);
    ierr = MatSetSizes(M,m,nlocal,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = MatSetType(M,mat->type_name);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(M,0,dlens,0,olens);CHKERRQ(ierr);
    ierr = PetscFree(dlens);CHKERRQ(ierr);
  } else {
    PetscInt ml,nl;

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocationCSR_MPIAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocationCSR_MPIAIJ(Mat B,const PetscInt Ii[],const PetscInt J[],const PetscScalar v[])
{
  PetscInt       m,cstart, cend,j,nnz,i,d; 
  PetscInt       *d_nnz,*o_nnz,nnz_max = 0,rstart,ii;
  const PetscInt *JJ;
  PetscScalar    *values;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Ii[0]) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Ii[0] must be 0 it is %D",Ii[0]);

  B->rmap.bs = B->cmap.bs = 1;
  ierr = PetscMapInitialize(B->comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(B->comm,&B->cmap);CHKERRQ(ierr);
  m      = B->rmap.n;
  cstart = B->cmap.rstart;
  cend   = B->cmap.rend;
  rstart = B->rmap.rstart;

  ierr  = PetscMalloc((2*m+1)*sizeof(PetscInt),&d_nnz);CHKERRQ(ierr);
  o_nnz = d_nnz + m;

  for (i=0; i<m; i++) {
    nnz     = Ii[i+1]- Ii[i];
    JJ      = J + Ii[i];
    nnz_max = PetscMax(nnz_max,nnz);
    if (nnz < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local row %D has a negative %D number of columns",i,nnz);
    for (j=0; j<nnz; j++) {
      if (*JJ >= cstart) break;
      JJ++;
    }
    d = 0;
    for (; j<nnz; j++) {
      if (*JJ++ >= cend) break;
      d++;
    }
    d_nnz[i] = d; 
    o_nnz[i] = nnz - d;
  }
  ierr = MatMPIAIJSetPreallocation(B,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree(d_nnz);CHKERRQ(ierr);

  if (v) values = (PetscScalar*)v;
  else {
    ierr = PetscMalloc((nnz_max+1)*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,nnz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ii   = i + rstart;
    nnz  = Ii[i+1]- Ii[i];
    ierr = MatSetValues_MPIAIJ(B,1,&ii,nnz,J+Ii[i],values+(v ? Ii[i] : 0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_UNSORTED);CHKERRQ(ierr);

  if (!v) {
    ierr = PetscFree(values);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocationCSR"
/*@
   MatMPIAIJSetPreallocationCSR - Allocates memory for a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).  

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix 
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: developer

   Notes: this actually copies the values from i[], j[], and a[] to put them into PETSc's internal
     storage format. Thus changing the values in a[] after this call will not effect the matrix values.

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatCreateMPIAIJ(), MPIAIJ,
          MatCreateSeqAIJWithArrays(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocationCSR(Mat B,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr,(*f)(Mat,const PetscInt[],const PetscInt[],const PetscScalar[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPIAIJSetPreallocationCSR_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,i,j,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation"
/*@C
   MatMPIAIJSetPreallocation - Preallocates memory for a sparse parallel matrix in AIJ format
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

   If the *_nnz parameter is given then the *_nz parameter is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage (CSR)), is fully compatible with standard Fortran 77
   storage.  The stored row and column indices begin with zero.  See the users manual for details.

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

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIAIJ(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPIAIJSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIAIJWithArrays"
/*@C
     MatCreateMPIAIJWithArrays - creates a MPI AIJ matrix using arrays that contain in standard
         CSR format the local rows.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (Cannot be PETSC_DECIDE)
.  n - This value should be the same as the local size used in creating the 
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.   i - row indices
.   j - column indices
-   a - matrix values

   Output Parameter:
.   mat - the matrix

   Level: intermediate

   Notes:
       The i, j, and a arrays ARE copied by this routine into the internal format used by PETSc;
     thus you CANNOT change the matrix entries by changing the values of a[] after you have 
     called this routine. Use MatCreateMPIAIJWithSplitArrays() to avoid needing to copy the arrays.

       The i and j indices are 0 based

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateMPIAIJ(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIIJWithArrays(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;

 PetscFunctionBegin;
  if (i[0]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  }
  if (m < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(*mat,i,j,a);CHKERRQ(ierr);
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
   If the *_nnz parameter is given then the *_nz parameter is ignored

   m,n,M,N parameters specify the size of the matrix, and its partitioning across
   processors, while d_nz,d_nnz,o_nz,o_nnz parameters specify the approximate
   storage requirements for this matrix.

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one 
   processor than it must be used on all processors that share the object for 
   that argument.

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
+  -mat_no_inode  - Do not use inodes
.  -mat_inode_limit <limit> - Sets inode limit (max limit=5)
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

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateMPIAIJWithArrays()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAIJ(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIAIJGetSeqAIJ(Mat A,Mat *Ad,Mat *Ao,PetscInt *colmap[])
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
PetscErrorCode MatSetColoring_MPIAIJ(Mat A,ISColoring coloring)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;  

  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    ISColoringValue *allcolors,*colors;
    ISColoring      ocoloring;

    /* set coloring for diagonal portion */
    ierr = MatSetColoring_SeqAIJ(a->A,coloring);CHKERRQ(ierr);

    /* set coloring for off-diagonal portion */
    ierr = ISAllGatherColors(A->comm,coloring->n,coloring->colors,PETSC_NULL,&allcolors);CHKERRQ(ierr);
    ierr = PetscMalloc((a->B->cmap.n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->B->cmap.n; i++) {
      colors[i] = allcolors[a->garray[i]];
    }
    ierr = PetscFree(allcolors);CHKERRQ(ierr);
    ierr = ISColoringCreate(MPI_COMM_SELF,coloring->n,a->B->cmap.n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->B,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);
  } else if (coloring->ctype == IS_COLORING_GHOSTED) {
    ISColoringValue *colors;
    PetscInt        *larray;
    ISColoring      ocoloring;

    /* set coloring for diagonal portion */
    ierr = PetscMalloc((a->A->cmap.n+1)*sizeof(PetscInt),&larray);CHKERRQ(ierr);
    for (i=0; i<a->A->cmap.n; i++) {
      larray[i] = i + A->cmap.rstart;
    }
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,a->A->cmap.n,larray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((a->A->cmap.n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->A->cmap.n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(PETSC_COMM_SELF,coloring->n,a->A->cmap.n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->A,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);

    /* set coloring for off-diagonal portion */
    ierr = PetscMalloc((a->B->cmap.n+1)*sizeof(PetscInt),&larray);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,a->B->cmap.n,a->garray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((a->B->cmap.n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<a->B->cmap.n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(MPI_COMM_SELF,coloring->n,a->B->cmap.n,colors,&ocoloring);CHKERRQ(ierr);
    ierr = MatSetColoring_SeqAIJ(a->B,ocoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(ocoloring);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"No support ISColoringType %d",(int)coloring->ctype);
  }

  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ADIC)
#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic_MPIAIJ"
PetscErrorCode MatSetValuesAdic_MPIAIJ(Mat A,void *advalues)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetValuesAdic_SeqAIJ(a->A,advalues);CHKERRQ(ierr);
  ierr = MatSetValuesAdic_SeqAIJ(a->B,advalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdifor_MPIAIJ"
PetscErrorCode MatSetValuesAdifor_MPIAIJ(Mat A,PetscInt nl,void *advalues)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;  
  PetscErrorCode ierr;

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
.    inmat - the input sequential matrices
.    n - number of local columns (or PETSC_DECIDE)
-    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.    outmat - the parallel matrix generated

    Level: advanced

   Notes: The number of columns of the matrix in EACH processor MUST be the same.

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMerge(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscInt       m,N,i,rstart,nnz,Ii,*dnz,*onz;
  PetscInt       *indx;
  PetscScalar    *values;

  PetscFunctionBegin;
  ierr = MatGetSize(inmat,&m,&N);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX){ 
    /* count nonzeros in each row, for diagonal and off diagonal portion of matrix */
    if (n == PETSC_DECIDE){ 
      ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
    } 
    ierr = MPI_Scan(&m, &rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart -= m;

    ierr = MatPreallocateInitialize(comm,m,n,dnz,onz);CHKERRQ(ierr);
    for (i=0;i<m;i++) {
      ierr = MatGetRow_SeqAIJ(inmat,i,&nnz,&indx,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatPreallocateSet(i+rstart,nnz,indx,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow_SeqAIJ(inmat,i,&nnz,&indx,PETSC_NULL);CHKERRQ(ierr);
    }
    /* This routine will ONLY return MPIAIJ type matrix */
    ierr = MatCreate(comm,outmat);CHKERRQ(ierr);
    ierr = MatSetSizes(*outmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(*outmat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*outmat,0,dnz,0,onz);CHKERRQ(ierr);
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  
  } else if (scall == MAT_REUSE_MATRIX){
    ierr = MatGetOwnershipRange(*outmat,&rstart,PETSC_NULL);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",(int)scall);
  }

  for (i=0;i<m;i++) {
    ierr = MatGetRow_SeqAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
    Ii    = i + rstart;
    ierr = MatSetValues(*outmat,1,&Ii,nnz,indx,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  ierr = MatDestroy(inmat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFileSplit"
PetscErrorCode MatFileSplit(Mat A,char *outfile)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  PetscInt          m,N,i,rstart,nnz;
  size_t            len;
  const PetscInt    *indx;
  PetscViewer       out;
  char              *name;
  Mat               B;
  const PetscScalar *values;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,0);CHKERRQ(ierr);
  ierr = MatGetSize(A,0,&N);CHKERRQ(ierr);
  /* Should this be the type of the diagonal block of A? */ 
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,N,m,N);CHKERRQ(ierr);
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
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_APPEND,&out);CHKERRQ(ierr);
  ierr = PetscFree(name);
  ierr = MatView(B,out);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(out);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatDestroy_MPIAIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_SeqsToMPI"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_MPIAIJ_SeqsToMPI(Mat A)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  PetscObjectContainer container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"MatMergeSeqsToMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectContainerGetPointer(container,(void **)&merge);CHKERRQ(ierr); 
    ierr = PetscFree(merge->id_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_s);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->bi);CHKERRQ(ierr);
    ierr = PetscFree(merge->bj);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_ri);CHKERRQ(ierr); 
    ierr = PetscFree(merge->buf_rj);CHKERRQ(ierr);
    ierr = PetscFree(merge->coi);CHKERRQ(ierr);
    ierr = PetscFree(merge->coj);CHKERRQ(ierr);
    ierr = PetscFree(merge->owners_co);CHKERRQ(ierr);
    ierr = PetscFree(merge->rowmap.range);CHKERRQ(ierr);
    
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"MatMergeSeqsToMPI",0);CHKERRQ(ierr);
  }
  ierr = PetscFree(merge);CHKERRQ(ierr);

  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "src/mat/utils/freespace.h"
#include "petscbt.h"
static PetscEvent logkey_seqstompinum = 0;
#undef __FUNCT__  
#define __FUNCT__ "MatMerge_SeqsToMPINumeric"
/*@C
      MatMerge_SeqsToMPI - Creates a MPIAIJ matrix by adding sequential
                 matrices from each processor

    Collective on MPI_Comm

   Input Parameters:
+    comm - the communicators the parallel matrix will live on
.    seqmat - the input sequential matrices
.    m - number of local rows (or PETSC_DECIDE)
.    n - number of local columns (or PETSC_DECIDE)
-    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.    mpimat - the parallel matrix generated

    Level: advanced

   Notes: 
     The dimensions of the sequential matrix in each processor MUST be the same.
     The input seqmat is included into the container "Mat_Merge_SeqsToMPI", and will be
     destroyed when mpimat is destroyed. Call PetscObjectQuery() to access seqmat.
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPINumeric(Mat seqmat,Mat mpimat) 
{
  PetscErrorCode       ierr; 
  MPI_Comm             comm=mpimat->comm;
  Mat_SeqAIJ           *a=(Mat_SeqAIJ*)seqmat->data;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             N=mpimat->cmap.N,i,j,*owners,*ai=a->i,*aj=a->j; 
  PetscInt             proc,m;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             k,anzi,*bj_i,*bi,*bj,arow,bnzi,nextaj; 
  PetscInt             nrows,**buf_ri_k,**nextrow,**nextai;
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            *aa=a->a,**abuf_r,*ba_i;
  Mat_Merge_SeqsToMPI  *merge;
  PetscObjectContainer container;
  
  PetscFunctionBegin;
  if (!logkey_seqstompinum) {
    ierr = PetscLogEventRegister(&logkey_seqstompinum,"MatMerge_SeqsToMPINumeric",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_seqstompinum,seqmat,0,0,0);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)mpimat,"MatMergeSeqsToMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void **)&merge);CHKERRQ(ierr); 
  }
  bi     = merge->bi;
  bj     = merge->bj;
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;

  ierr   = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);
  owners = merge->rowmap.range;
  len_s  = merge->len_s;

  /* send and recv matrix values */
  /*-----------------------------*/
  ierr = PetscObjectGetNewTag((PetscObject)mpimat,&taga);CHKERRQ(ierr);
  ierr = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners[proc];
    ierr = MPI_Isend(aa+ai[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  } 

  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);}
  ierr = PetscFree(status);CHKERRQ(ierr);

  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);

  /* insert mat values of mpimat */
  /*----------------------------*/
  ierr = PetscMalloc(N*sizeof(MatScalar),&ba_i);CHKERRQ(ierr);
  ierr = PetscMalloc((3*merge->nrecv+1)*sizeof(PetscInt**),&buf_ri_k);CHKERRQ(ierr);
  nextrow = buf_ri_k + merge->nrecv;
  nextai  = nextrow + merge->nrecv;

  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextai[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  /* set values of ba */
  m = merge->rowmap.n;
  for (i=0; i<m; i++) {
    arow = owners[rank] + i; 
    bj_i = bj+bi[i];  /* col indices of the i-th row of mpimat */
    bnzi = bi[i+1] - bi[i];
    ierr = PetscMemzero(ba_i,bnzi*sizeof(MatScalar));CHKERRQ(ierr);

    /* add local non-zero vals of this proc's seqmat into ba */
    anzi = ai[arow+1] - ai[arow];
    aj   = a->j + ai[arow]; 
    aa   = a->a + ai[arow]; 
    nextaj = 0;
    for (j=0; nextaj<anzi; j++){
      if (*(bj_i + j) == aj[nextaj]){ /* bcol == acol */
        ba_i[j] += aa[nextaj++];
      }
    }

    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        anzi = *(nextai[k]+1) - *nextai[k]; 
        aj   = buf_rj[k] + *(nextai[k]);
        aa   = abuf_r[k] + *(nextai[k]);
        nextaj = 0;
        for (j=0; nextaj<anzi; j++){ 
          if (*(bj_i + j) == aj[nextaj]){ /* bcol == acol */
            ba_i[j] += aa[nextaj++];
          }
        }
        nextrow[k]++; nextai[k]++;
      } 
    }
    ierr = MatSetValues(mpimat,1,&arow,bnzi,bj_i,ba_i,INSERT_VALUES);CHKERRQ(ierr);
  } 
  ierr = MatAssemblyBegin(mpimat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mpimat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 

  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree(ba_i);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri_k);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_seqstompinum,seqmat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscEvent logkey_seqstompisym = 0;
#undef __FUNCT__  
#define __FUNCT__ "MatMerge_SeqsToMPISymbolic"
PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPISymbolic(MPI_Comm comm,Mat seqmat,PetscInt m,PetscInt n,Mat *mpimat) 
{
  PetscErrorCode       ierr; 
  Mat                  B_mpi;
  Mat_SeqAIJ           *a=(Mat_SeqAIJ*)seqmat->data;
  PetscMPIInt          size,rank,tagi,tagj,*len_s,*len_si,*len_ri;
  PetscInt             **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt             M=seqmat->rmap.n,N=seqmat->cmap.n,i,*owners,*ai=a->i,*aj=a->j;
  PetscInt             len,proc,*dnz,*onz;
  PetscInt             k,anzi,*bi,*bj,*lnk,nlnk,arow,bnzi,nspacedouble=0; 
  PetscInt             nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextai;
  MPI_Request          *si_waits,*sj_waits,*ri_waits,*rj_waits; 
  MPI_Status           *status;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT              lnkbt;
  Mat_Merge_SeqsToMPI  *merge;
  PetscObjectContainer container;

  PetscFunctionBegin;
  if (!logkey_seqstompisym) {
    ierr = PetscLogEventRegister(&logkey_seqstompisym,"MatMerge_SeqsToMPISymbolic",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_seqstompisym,seqmat,0,0,0);CHKERRQ(ierr);

  /* make sure it is a PETSc comm */
  ierr = PetscCommDuplicate(comm,&comm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  ierr = PetscNew(Mat_Merge_SeqsToMPI,&merge);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);

  /* determine row ownership */
  /*---------------------------------------------------------*/
  merge->rowmap.n = m;
  merge->rowmap.N = M;
  merge->rowmap.bs = 1;
  ierr = PetscMapInitialize(comm,&merge->rowmap);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&len_si);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&merge->len_s);CHKERRQ(ierr);
  
  m      = merge->rowmap.n;
  M      = merge->rowmap.N;
  owners = merge->rowmap.range;

  /* determine the number of messages to send, their lengths */
  /*---------------------------------------------------------*/
  len_s  = merge->len_s;

  len = 0;  /* length of buf_si[] */
  merge->nsend = 0;
  for (proc=0; proc<size; proc++){
    len_si[proc] = 0;
    if (proc == rank){
      len_s[proc] = 0;  
    } else {
      len_si[proc] = owners[proc+1] - owners[proc] + 1;
      len_s[proc] = ai[owners[proc+1]] - ai[owners[proc]]; /* num of rows to be sent to [proc] */
    }
    if (len_s[proc]) {
      merge->nsend++;
      nrows = 0;
      for (i=owners[proc]; i<owners[proc+1]; i++){
        if (ai[i+1] > ai[i]) nrows++;
      }
      len_si[proc] = 2*(nrows+1);
      len += len_si[proc];
    } 
  }

  /* determine the number and length of messages to receive for ij-structure */
  /*-------------------------------------------------------------------------*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);      

  /* post the Irecv of j-structure */
  /*-------------------------------*/
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rj_waits);CHKERRQ(ierr);

  /* post the Isend of j-structure */
  /*--------------------------------*/
  ierr = PetscMalloc((2*merge->nsend+1)*sizeof(MPI_Request),&si_waits);CHKERRQ(ierr);
  sj_waits = si_waits + merge->nsend;

  for (proc=0, k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners[proc];
    ierr = MPI_Isend(aj+ai[i],len_s[proc],MPIU_INT,proc,tagj,comm,sj_waits+k);CHKERRQ(ierr);
    k++;
  } 

  /* receives and sends of j-structure are complete */
  /*------------------------------------------------*/
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,rj_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,sj_waits,status);CHKERRQ(ierr);}
  
  /* send and recv i-structure */
  /*---------------------------*/  
  ierr = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&ri_waits);CHKERRQ(ierr);
    
  ierr = PetscMalloc((len+1)*sizeof(PetscInt),&buf_s);CHKERRQ(ierr); 
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure: 
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/      
    nrows = len_si[proc]/2 - 1; 
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows = 0;
    for (i=owners[proc]; i<owners[proc+1]; i++){
      anzi = ai[i+1] - ai[i];
      if (anzi) {
        buf_si_i[nrows+1] = buf_si_i[nrows] + anzi; /* i-structure */
        buf_si[nrows+1] = i-owners[proc]; /* local row index */
        nrows++;
      }
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,si_waits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  } 

  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,ri_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,si_waits,status);CHKERRQ(ierr);}

  ierr = PetscInfo2(seqmat,"nsend: %D, nrecv: %D\n",merge->nsend,merge->nrecv);CHKERRQ(ierr);
  for (i=0; i<merge->nrecv; i++){
    ierr = PetscInfo3(seqmat,"recv len_ri=%D, len_rj=%D from [%D]\n",len_ri[i],merge->len_r[i],merge->id_r[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(rj_waits);CHKERRQ(ierr);
  ierr = PetscFree(si_waits);CHKERRQ(ierr);
  ierr = PetscFree(ri_waits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);
  ierr = PetscFree(status);CHKERRQ(ierr);

  /* compute a local seq matrix in each processor */
  /*----------------------------------------------*/
  /* allocate bi array and free space for accumulating nonzero column info */
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* create and initialize a linked list */
  nlnk = N+1;
  ierr = PetscLLCreate(N,N,nlnk,lnk,lnkbt);CHKERRQ(ierr);
  
  /* initial FreeSpace size is 2*(num of local nnz(seqmat)) */
  len = 0;
  len  = ai[owners[rank+1]] - ai[owners[rank]];
  ierr = PetscFreeSpaceGet((PetscInt)(2*len+1),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* determine symbolic info for each local row */
  ierr = PetscMalloc((3*merge->nrecv+1)*sizeof(PetscInt**),&buf_ri_k);CHKERRQ(ierr);
  nextrow = buf_ri_k + merge->nrecv;
  nextai  = nextrow + merge->nrecv;
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextai[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,m,n,dnz,onz);CHKERRQ(ierr);
  len = 0;  
  for (i=0;i<m;i++) {
    bnzi   = 0;
    /* add local non-zero cols of this proc's seqmat into lnk */
    arow   = owners[rank] + i;
    anzi   = ai[arow+1] - ai[arow];
    aj     = a->j + ai[arow]; 
    ierr = PetscLLAdd(anzi,aj,N,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    bnzi += nlnk;
    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        anzi = *(nextai[k]+1) - *nextai[k]; 
        aj   = buf_rj[k] + *nextai[k];
        ierr = PetscLLAdd(anzi,aj,N,nlnk,lnk,lnkbt);CHKERRQ(ierr);
        bnzi += nlnk;
        nextrow[k]++; nextai[k]++;
      }
    }
    if (len < bnzi) len = bnzi;  /* =max(bnzi) */

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<bnzi) {
      ierr = PetscFreeSpaceGet(current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(N,N,bnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],bnzi,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += bnzi;
    current_space->local_used      += bnzi;
    current_space->local_remaining -= bnzi;
   
    bi[i+1] = bi[i] + bnzi;
  }
  
  ierr = PetscFree(buf_ri_k);CHKERRQ(ierr);

  ierr = PetscMalloc((bi[m]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* create symbolic parallel matrix B_mpi */
  /*---------------------------------------*/
  ierr = MatCreate(comm,&B_mpi);CHKERRQ(ierr);
  if (n==PETSC_DECIDE) {
    ierr = MatSetSizes(B_mpi,m,n,PETSC_DETERMINE,N);CHKERRQ(ierr);
  } else {
    ierr = MatSetSizes(B_mpi,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  }
  ierr = MatSetType(B_mpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B_mpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* B_mpi is not ready for use - assembly will be done by MatMerge_SeqsToMPINumeric() */
  B_mpi->assembled     = PETSC_FALSE; 
  B_mpi->ops->destroy  = MatDestroy_MPIAIJ_SeqsToMPI; 
  merge->bi            = bi;
  merge->bj            = bj;
  merge->buf_ri        = buf_ri;
  merge->buf_rj        = buf_rj;
  merge->coi           = PETSC_NULL;
  merge->coj           = PETSC_NULL;
  merge->owners_co     = PETSC_NULL;

  /* attach the supporting struct to B_mpi for reuse */
  ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscObjectContainerSetPointer(container,merge);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B_mpi,"MatMergeSeqsToMPI",(PetscObject)container);CHKERRQ(ierr);
  *mpimat = B_mpi;

  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_seqstompisym,seqmat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscEvent logkey_seqstompi = 0;
#undef __FUNCT__
#define __FUNCT__ "MatMerge_SeqsToMPI"
PetscErrorCode PETSCMAT_DLLEXPORT MatMerge_SeqsToMPI(MPI_Comm comm,Mat seqmat,PetscInt m,PetscInt n,MatReuse scall,Mat *mpimat) 
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!logkey_seqstompi) {
    ierr = PetscLogEventRegister(&logkey_seqstompi,"MatMerge_SeqsToMPI",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_seqstompi,seqmat,0,0,0);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX){ 
    ierr = MatMerge_SeqsToMPISymbolic(comm,seqmat,m,n,mpimat);CHKERRQ(ierr);
  } 
  ierr = MatMerge_SeqsToMPINumeric(seqmat,*mpimat);CHKERRQ(ierr); 
  ierr = PetscLogEventEnd(logkey_seqstompi,seqmat,0,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
static PetscEvent logkey_getlocalmat = 0;
#undef __FUNCT__
#define __FUNCT__ "MatGetLocalMat"
/*@C
     MatGetLocalMat - Creates a SeqAIJ matrix by taking all its local rows

    Not Collective

   Input Parameters:
+    A - the matrix 
.    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX 

   Output Parameter:
.    A_loc - the local sequential matrix generated

    Level: developer

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalMat(Mat A,MatReuse scall,Mat *A_loc) 
{
  PetscErrorCode  ierr;
  Mat_MPIAIJ      *mpimat=(Mat_MPIAIJ*)A->data; 
  Mat_SeqAIJ      *mat,*a=(Mat_SeqAIJ*)(mpimat->A)->data,*b=(Mat_SeqAIJ*)(mpimat->B)->data;
  PetscInt        *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*cmap=mpimat->garray;
  PetscScalar     *aa=a->a,*ba=b->a,*ca;
  PetscInt        am=A->rmap.n,i,j,k,cstart=A->cmap.rstart;
  PetscInt        *ci,*cj,col,ncols_d,ncols_o,jo;

  PetscFunctionBegin;
  if (!logkey_getlocalmat) {
    ierr = PetscLogEventRegister(&logkey_getlocalmat,"MatGetLocalMat",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_getlocalmat,A,0,0,0);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscMalloc((1+am)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
    ci[0] = 0;
    for (i=0; i<am; i++){
      ci[i+1] = ci[i] + (ai[i+1] - ai[i]) + (bi[i+1] - bi[i]);
    }
    ierr = PetscMalloc((1+ci[am])*sizeof(PetscInt),&cj);CHKERRQ(ierr);
    ierr = PetscMalloc((1+ci[am])*sizeof(PetscScalar),&ca);CHKERRQ(ierr);
    k = 0;
    for (i=0; i<am; i++) {
      ncols_o = bi[i+1] - bi[i];
      ncols_d = ai[i+1] - ai[i];
      /* off-diagonal portion of A */
      for (jo=0; jo<ncols_o; jo++) {
        col = cmap[*bj];
        if (col >= cstart) break;
        cj[k]   = col; bj++;
        ca[k++] = *ba++; 
      }
      /* diagonal portion of A */
      for (j=0; j<ncols_d; j++) {
        cj[k]   = cstart + *aj++; 
        ca[k++] = *aa++; 
      }
      /* off-diagonal portion of A */
      for (j=jo; j<ncols_o; j++) {
        cj[k]   = cmap[*bj++]; 
        ca[k++] = *ba++; 
      }
    }
    /* put together the new matrix */
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,A->cmap.N,ci,cj,ca,A_loc);CHKERRQ(ierr);
    /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
    /* Since these are PETSc arrays, change flags to free them as necessary. */
    mat          = (Mat_SeqAIJ*)(*A_loc)->data;
    mat->free_a  = PETSC_TRUE;
    mat->free_ij = PETSC_TRUE;
    mat->nonew   = 0;
  } else if (scall == MAT_REUSE_MATRIX){
    mat=(Mat_SeqAIJ*)(*A_loc)->data; 
    ci = mat->i; cj = mat->j; ca = mat->a;
    for (i=0; i<am; i++) {
      /* off-diagonal portion of A */
      ncols_o = bi[i+1] - bi[i];
      for (jo=0; jo<ncols_o; jo++) {
        col = cmap[*bj];
        if (col >= cstart) break;
        *ca++ = *ba++; bj++;
      }
      /* diagonal portion of A */
      ncols_d = ai[i+1] - ai[i];
      for (j=0; j<ncols_d; j++) *ca++ = *aa++; 
      /* off-diagonal portion of A */
      for (j=jo; j<ncols_o; j++) {
        *ca++ = *ba++; bj++;
      }
    }
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",(int)scall);
  }

  ierr = PetscLogEventEnd(logkey_getlocalmat,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscEvent logkey_getlocalmatcondensed = 0;
#undef __FUNCT__
#define __FUNCT__ "MatGetLocalMatCondensed"
/*@C
     MatGetLocalMatCondensed - Creates a SeqAIJ matrix by taking all its local rows and NON-ZERO columns

    Not Collective

   Input Parameters:
+    A - the matrix 
.    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-    row, col - index sets of rows and columns to extract (or PETSC_NULL)  

   Output Parameter:
.    A_loc - the local sequential matrix generated

    Level: developer

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalMatCondensed(Mat A,MatReuse scall,IS *row,IS *col,Mat *A_loc) 
{
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,start,end,ncols,nzA,nzB,*cmap,imark,*idx;
  IS                isrowa,iscola;
  Mat               *aloc;

  PetscFunctionBegin;
  if (!logkey_getlocalmatcondensed) {
    ierr = PetscLogEventRegister(&logkey_getlocalmatcondensed,"MatGetLocalMatCondensed",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_getlocalmatcondensed,A,0,0,0);CHKERRQ(ierr);
  if (!row){
    start = A->rmap.rstart; end = A->rmap.rend;
    ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&isrowa);CHKERRQ(ierr); 
  } else {
    isrowa = *row;
  }
  if (!col){
    start = A->cmap.rstart;
    cmap  = a->garray;
    nzA   = a->A->cmap.n; 
    nzB   = a->B->cmap.n;
    ierr  = PetscMalloc((nzA+nzB)*sizeof(PetscInt), &idx);CHKERRQ(ierr);
    ncols = 0;
    for (i=0; i<nzB; i++) {
      if (cmap[i] < start) idx[ncols++] = cmap[i];
      else break;
    }
    imark = i;
    for (i=0; i<nzA; i++) idx[ncols++] = start + i;
    for (i=imark; i<nzB; i++) idx[ncols++] = cmap[i];
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,&iscola);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr); 
  } else {
    iscola = *col;
  }
  if (scall != MAT_INITIAL_MATRIX){
    ierr = PetscMalloc(sizeof(Mat),&aloc);CHKERRQ(ierr); 
    aloc[0] = *A_loc;
  }
  ierr = MatGetSubMatrices(A,1,&isrowa,&iscola,scall,&aloc);CHKERRQ(ierr); 
  *A_loc = aloc[0];
  ierr = PetscFree(aloc);CHKERRQ(ierr);
  if (!row){ 
    ierr = ISDestroy(isrowa);CHKERRQ(ierr);
  } 
  if (!col){ 
    ierr = ISDestroy(iscola);CHKERRQ(ierr);
  } 
  ierr = PetscLogEventEnd(logkey_getlocalmatcondensed,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscEvent logkey_GetBrowsOfAcols = 0;
#undef __FUNCT__
#define __FUNCT__ "MatGetBrowsOfAcols"
/*@C
    MatGetBrowsOfAcols - Creates a SeqAIJ matrix by taking rows of B that equal to nonzero columns of local A 

    Collective on Mat

   Input Parameters:
+    A,B - the matrices in mpiaij format
.    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-    rowb, colb - index sets of rows and columns of B to extract (or PETSC_NULL)   

   Output Parameter:
+    rowb, colb - index sets of rows and columns of B to extract 
.    brstart - row index of B_seq from which next B->rmap.n rows are taken from B's local rows
-    B_seq - the sequential matrix generated

    Level: developer

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetBrowsOfAcols(Mat A,Mat B,MatReuse scall,IS *rowb,IS *colb,PetscInt *brstart,Mat *B_seq) 
{
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          *idx,i,start,ncols,nzA,nzB,*cmap,imark;
  IS                isrowb,iscolb;
  Mat               *bseq;
 
  PetscFunctionBegin;
  if (A->cmap.rstart != B->rmap.rstart || A->cmap.rend != B->rmap.rend){
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap.rstart,A->cmap.rend,B->rmap.rstart,B->rmap.rend);
  }
  if (!logkey_GetBrowsOfAcols) {
    ierr = PetscLogEventRegister(&logkey_GetBrowsOfAcols,"MatGetBrowsOfAcols",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_GetBrowsOfAcols,A,B,0,0);CHKERRQ(ierr);
  
  if (scall == MAT_INITIAL_MATRIX){
    start = A->cmap.rstart;
    cmap  = a->garray;
    nzA   = a->A->cmap.n; 
    nzB   = a->B->cmap.n;
    ierr  = PetscMalloc((nzA+nzB)*sizeof(PetscInt), &idx);CHKERRQ(ierr);
    ncols = 0;
    for (i=0; i<nzB; i++) {  /* row < local row index */
      if (cmap[i] < start) idx[ncols++] = cmap[i];
      else break;
    }
    imark = i;
    for (i=0; i<nzA; i++) idx[ncols++] = start + i;  /* local rows */
    for (i=imark; i<nzB; i++) idx[ncols++] = cmap[i]; /* row > local row index */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,&isrowb);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr); 
    *brstart = imark;
    ierr = ISCreateStride(PETSC_COMM_SELF,B->cmap.N,0,1,&iscolb);CHKERRQ(ierr);
  } else {
    if (!rowb || !colb) SETERRQ(PETSC_ERR_SUP,"IS rowb and colb must be provided for MAT_REUSE_MATRIX");
    isrowb = *rowb; iscolb = *colb;
    ierr = PetscMalloc(sizeof(Mat),&bseq);CHKERRQ(ierr);
    bseq[0] = *B_seq;
  }
  ierr = MatGetSubMatrices(B,1,&isrowb,&iscolb,scall,&bseq);CHKERRQ(ierr);
  *B_seq = bseq[0];
  ierr = PetscFree(bseq);CHKERRQ(ierr);
  if (!rowb){ 
    ierr = ISDestroy(isrowb);CHKERRQ(ierr);
  } else {
    *rowb = isrowb;
  }
  if (!colb){ 
    ierr = ISDestroy(iscolb);CHKERRQ(ierr);
  } else {
    *colb = iscolb;
  }
  ierr = PetscLogEventEnd(logkey_GetBrowsOfAcols,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscEvent logkey_GetBrowsOfAocols = 0;
#undef __FUNCT__
#define __FUNCT__ "MatGetBrowsOfAoCols"
/*@C
    MatGetBrowsOfAoCols - Creates a SeqAIJ matrix by taking rows of B that equal to nonzero columns
    of the OFF-DIAGONAL portion of local A 

    Collective on Mat

   Input Parameters:
+    A,B - the matrices in mpiaij format
.    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
.    startsj - starting point in B's sending and receiving j-arrays, saved for MAT_REUSE (or PETSC_NULL) 
-    bufa_ptr - array for sending matrix values, saved for MAT_REUSE (or PETSC_NULL) 

   Output Parameter:
+    B_oth - the sequential matrix generated

    Level: developer

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetBrowsOfAoCols(Mat A,Mat B,MatReuse scall,PetscInt **startsj,PetscScalar **bufa_ptr,Mat *B_oth) 
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscErrorCode         ierr;
  Mat_MPIAIJ             *a=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ             *b_oth;
  VecScatter             ctx=a->Mvctx;
  MPI_Comm               comm=ctx->comm;
  PetscMPIInt            *rprocs,*sprocs,tag=ctx->tag,rank; 
  PetscInt               *rowlen,*bufj,*bufJ,ncols,aBn=a->B->cmap.n,row,*b_othi,*b_othj;
  PetscScalar            *rvalues,*svalues,*b_otha,*bufa,*bufA;
  PetscInt               i,k,l,nrecvs,nsends,nrows,*srow,*rstarts,*rstartsj = 0,*sstarts,*sstartsj,len;
  MPI_Request            *rwaits = PETSC_NULL,*swaits = PETSC_NULL;
  MPI_Status             *sstatus,rstatus;
  PetscInt               *cols;
  PetscScalar            *vals;
  PetscMPIInt            j;
 
  PetscFunctionBegin;
  if (A->cmap.rstart != B->rmap.rstart || A->cmap.rend != B->rmap.rend){
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%d, %d) != (%d,%d)",A->cmap.rstart,A->cmap.rend,B->rmap.rstart,B->rmap.rend);
  }
  if (!logkey_GetBrowsOfAocols) {
    ierr = PetscLogEventRegister(&logkey_GetBrowsOfAocols,"MatGetBrAoCol",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_GetBrowsOfAocols,A,B,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  gen_to   = (VecScatter_MPI_General*)ctx->todata;
  gen_from = (VecScatter_MPI_General*)ctx->fromdata;
  rvalues  = gen_from->values; /* holds the length of sending row */
  svalues  = gen_to->values;   /* holds the length of receiving row */
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;

  ierr = PetscMalloc2(nrecvs,MPI_Request,&rwaits,nsends,MPI_Request,&swaits);CHKERRQ(ierr);
  srow     = gen_to->indices;   /* local row index to be sent */
  rstarts  = gen_from->starts;
  sstarts  = gen_to->starts; 
  rprocs   = gen_from->procs;
  sprocs   = gen_to->procs;
  sstatus  = gen_to->sstatus;

  if (!startsj || !bufa_ptr) scall = MAT_INITIAL_MATRIX;
  if (scall == MAT_INITIAL_MATRIX){
    /* i-array */
    /*---------*/
    /*  post receives */
    for (i=0; i<nrecvs; i++){
      rowlen = (PetscInt*)rvalues + rstarts[i];
      nrows = rstarts[i+1]-rstarts[i];
      ierr = MPI_Irecv(rowlen,nrows,MPIU_INT,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }

    /* pack the outgoing message */
    ierr = PetscMalloc((nsends+nrecvs+3)*sizeof(PetscInt),&sstartsj);CHKERRQ(ierr); 
    rstartsj = sstartsj + nsends +1;
    sstartsj[0] = 0;  rstartsj[0] = 0;
    len = 0; /* total length of j or a array to be sent */
    k = 0; 
    for (i=0; i<nsends; i++){
      rowlen = (PetscInt*)svalues + sstarts[i];
      nrows = sstarts[i+1]-sstarts[i]; /* num of rows */
      for (j=0; j<nrows; j++) {
        row = srow[k] + B->rmap.range[rank]; /* global row idx */
        ierr = MatGetRow_MPIAIJ(B,row,&rowlen[j],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); /* rowlength */
        len += rowlen[j];  
        ierr = MatRestoreRow_MPIAIJ(B,row,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        k++;
      } 
      ierr = MPI_Isend(rowlen,nrows,MPIU_INT,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
       sstartsj[i+1] = len;  /* starting point of (i+1)-th outgoing msg in bufj and bufa */
    }
    /* recvs and sends of i-array are completed */
    i = nrecvs;
    while (i--) {
      ierr = MPI_Waitany(nrecvs,rwaits,&j,&rstatus);CHKERRQ(ierr);
    }
    if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}  
    /* allocate buffers for sending j and a arrays */
    ierr = PetscMalloc((len+1)*sizeof(PetscInt),&bufj);CHKERRQ(ierr);
    ierr = PetscMalloc((len+1)*sizeof(PetscScalar),&bufa);CHKERRQ(ierr);

    /* create i-array of B_oth */
    ierr = PetscMalloc((aBn+2)*sizeof(PetscInt),&b_othi);CHKERRQ(ierr);
    b_othi[0] = 0;
    len = 0; /* total length of j or a array to be received */
    k = 0;
    for (i=0; i<nrecvs; i++){
      rowlen = (PetscInt*)rvalues + rstarts[i];
      nrows = rstarts[i+1]-rstarts[i];
      for (j=0; j<nrows; j++) {
        b_othi[k+1] = b_othi[k] + rowlen[j]; 
        len += rowlen[j]; k++;
      }
      rstartsj[i+1] = len; /* starting point of (i+1)-th incoming msg in bufj and bufa */
    }

    /* allocate space for j and a arrrays of B_oth */
    ierr = PetscMalloc((b_othi[aBn]+1)*sizeof(PetscInt),&b_othj);CHKERRQ(ierr);
    ierr = PetscMalloc((b_othi[aBn]+1)*sizeof(PetscScalar),&b_otha);CHKERRQ(ierr);

    /* j-array */
    /*---------*/
    /*  post receives of j-array */
    for (i=0; i<nrecvs; i++){
      nrows = rstartsj[i+1]-rstartsj[i]; /* length of the msg received */
      ierr = MPI_Irecv(b_othj+rstartsj[i],nrows,MPIU_INT,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }
    k = 0; 
    for (i=0; i<nsends; i++){
      nrows = sstarts[i+1]-sstarts[i]; /* num of rows */
      bufJ = bufj+sstartsj[i];
      for (j=0; j<nrows; j++) {
        row  = srow[k++] + B->rmap.range[rank]; /* global row idx */
        ierr = MatGetRow_MPIAIJ(B,row,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
        for (l=0; l<ncols; l++){
          *bufJ++ = cols[l];
        }
        ierr = MatRestoreRow_MPIAIJ(B,row,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);  
      }
      ierr = MPI_Isend(bufj+sstartsj[i],sstartsj[i+1]-sstartsj[i],MPIU_INT,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr); 
    }

    /* recvs and sends of j-array are completed */  
    i = nrecvs;
    while (i--) {
      ierr = MPI_Waitany(nrecvs,rwaits,&j,&rstatus);CHKERRQ(ierr);
    }
    if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  } else if (scall == MAT_REUSE_MATRIX){
    sstartsj = *startsj;
    rstartsj = sstartsj + nsends +1;
    bufa     = *bufa_ptr;
    b_oth    = (Mat_SeqAIJ*)(*B_oth)->data;
    b_otha   = b_oth->a;  
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
  }

  /* a-array */
  /*---------*/
  /*  post receives of a-array */
  for (i=0; i<nrecvs; i++){
    nrows = rstartsj[i+1]-rstartsj[i]; /* length of the msg received */
    ierr = MPI_Irecv(b_otha+rstartsj[i],nrows,MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
  }
  k = 0; 
  for (i=0; i<nsends; i++){
    nrows = sstarts[i+1]-sstarts[i];
    bufA = bufa+sstartsj[i];
    for (j=0; j<nrows; j++) {
      row  = srow[k++] + B->rmap.range[rank]; /* global row idx */
      ierr = MatGetRow_MPIAIJ(B,row,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);
      for (l=0; l<ncols; l++){
        *bufA++ = vals[l]; 
      }
      ierr = MatRestoreRow_MPIAIJ(B,row,&ncols,PETSC_NULL,&vals);CHKERRQ(ierr);  

    }
    ierr = MPI_Isend(bufa+sstartsj[i],sstartsj[i+1]-sstartsj[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr); 
  }
  /* recvs and sends of a-array are completed */
  i = nrecvs;
  while (i--) {
    ierr = MPI_Waitany(nrecvs,rwaits,&j,&rstatus);CHKERRQ(ierr);
  }
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}  
  ierr = PetscFree2(rwaits,swaits);CHKERRQ(ierr); 

  if (scall == MAT_INITIAL_MATRIX){
    /* put together the new matrix */
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,aBn,B->cmap.N,b_othi,b_othj,b_otha,B_oth);CHKERRQ(ierr);

    /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
    /* Since these are PETSc arrays, change flags to free them as necessary. */
    b_oth          = (Mat_SeqAIJ *)(*B_oth)->data;
    b_oth->free_a  = PETSC_TRUE;
    b_oth->free_ij = PETSC_TRUE;
    b_oth->nonew   = 0;

    ierr = PetscFree(bufj);CHKERRQ(ierr);
    if (!startsj || !bufa_ptr){
      ierr = PetscFree(sstartsj);CHKERRQ(ierr);
      ierr = PetscFree(bufa_ptr);CHKERRQ(ierr);
    } else {
      *startsj  = sstartsj;
      *bufa_ptr = bufa;
    }
  }
  ierr = PetscLogEventEnd(logkey_GetBrowsOfAocols,A,B,0,0);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetCommunicationStructs"
/*@C
  MatGetCommunicationStructs - Provides access to the communication structures used in matrix-vector multiplication.

  Not Collective

  Input Parameters:
. A - The matrix in mpiaij format

  Output Parameter:
+ lvec - The local vector holding off-process values from the argument to a matrix-vector product
. colmap - A map from global column index to local index into lvec
- multScatter - A scatter from the argument of a matrix-vector product to lvec

  Level: developer

@*/
#if defined (PETSC_USE_CTABLE)
PetscErrorCode PETSCMAT_DLLEXPORT MatGetCommunicationStructs(Mat A, Vec *lvec, PetscTable *colmap, VecScatter *multScatter)
#else
PetscErrorCode PETSCMAT_DLLEXPORT MatGetCommunicationStructs(Mat A, Vec *lvec, PetscInt *colmap[], VecScatter *multScatter)
#endif
{
  Mat_MPIAIJ *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_COOKIE, 1);
  PetscValidPointer(lvec, 2)
  PetscValidPointer(colmap, 3)
  PetscValidPointer(multScatter, 4)
  a = (Mat_MPIAIJ *) A->data;
  if (lvec) *lvec = a->lvec;
  if (colmap) *colmap = a->colmap;
  if (multScatter) *multScatter = a->Mvctx;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_MPICRL(Mat,MatType,MatReuse,Mat*);
extern PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_MPICSRPERM(Mat,MatType,MatReuse,Mat*);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIAIJ(Mat B)
{
  Mat_MPIAIJ     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);

  ierr            = PetscNew(Mat_MPIAIJ,&b);CHKERRQ(ierr);
  B->data         = (void*)b;
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor       = 0;
  B->rmap.bs      = 1;
  B->assembled    = PETSC_FALSE;
  B->mapping      = 0;

  B->insertmode      = NOT_SET_VALUES;
  b->size            = size;
  ierr = MPI_Comm_rank(B->comm,&b->rank);CHKERRQ(ierr);

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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAIJSetPreallocationCSR_C",
				     "MatMPIAIJSetPreallocationCSR_MPIAIJ",
				     MatMPIAIJSetPreallocationCSR_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatDiagonalScaleLocal_C",
				     "MatDiagonalScaleLocal_MPIAIJ",
				     MatDiagonalScaleLocal_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaij_mpicsrperm_C",
                                     "MatConvert_MPIAIJ_MPICSRPERM",
                                      MatConvert_MPIAIJ_MPICSRPERM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaij_mpicrl_C",
                                     "MatConvert_MPIAIJ_MPICRL",
                                      MatConvert_MPIAIJ_MPICRL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIAIJWithSplitArrays"
/*@C
     MatCreateMPIAIJWithSplitArrays - creates a MPI AIJ matrix using arrays that contain the "diagonal"
         and "off-diagonal" part of the matrix in CSR format.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (Cannot be PETSC_DECIDE)
.  n - This value should be the same as the local size used in creating the 
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.   i - row indices for "diagonal" portion of matrix
.   j - column indices
.   a - matrix values
.   oi - row indices for "off-diagonal" portion of matrix
.   oj - column indices
-   oa - matrix values

   Output Parameter:
.   mat - the matrix

   Level: advanced

   Notes:
       The i, j, and a arrays ARE NOT copied by this routine into the internal format used by PETSc.

       The i and j indices are 0 based
 
       See MatCreateMPIAIJ() for the definition of "diagonal" and "off-diagonal" portion of the matrix


.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateMPIAIJ(), MatCreateMPIAIJWithArrays()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIAIJWithSplitArrays(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt i[],PetscInt j[],PetscScalar a[],
								PetscInt oi[], PetscInt oj[],PetscScalar oa[],Mat *mat)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *maij;

 PetscFunctionBegin;
  if (m < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  if (i[0]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  }
  if (oi[0]) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"oi (row indices) must start with 0");
  }
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
  maij = (Mat_MPIAIJ*) (*mat)->data;
  maij->donotstash     = PETSC_TRUE;
  (*mat)->preallocated = PETSC_TRUE;

  (*mat)->rmap.bs = (*mat)->cmap.bs = 1;
  ierr = PetscMapInitialize((*mat)->comm,&(*mat)->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize((*mat)->comm,&(*mat)->cmap);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,m,n,i,j,a,&maij->A);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,m,(*mat)->cmap.N,oi,oj,oa,&maij->B);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(maij->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(maij->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(maij->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(maij->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Special version for direct calls from Fortran 
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvaluesmpiaij_ MATSETVALUESMPIAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvaluesmpiaij_ matsetvaluesmpiaij
#endif

/* Change these macros so can be used in void function */
#undef CHKERRQ
#define CHKERRQ(ierr) CHKERRABORT(mat->comm,ierr) 
#undef SETERRQ2
#define SETERRQ2(ierr,b,c,d) CHKERRABORT(mat->comm,ierr) 
#undef SETERRQ
#define SETERRQ(ierr,b) CHKERRABORT(mat->comm,ierr) 

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "matsetvaluesmpiaij_"
void PETSC_STDCALL matsetvaluesmpiaij_(Mat *mmat,PetscInt *mm,const PetscInt im[],PetscInt *mn,const PetscInt in[],const PetscScalar v[],InsertMode *maddv,PetscErrorCode *_ierr)
{
  Mat            mat = *mmat;
  PetscInt       m = *mm, n = *mn;
  InsertMode     addv = *maddv;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  PetscScalar    value;
  PetscErrorCode ierr;

  MatPreallocated(mat);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG)
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
#endif
  { 
  PetscInt       i,j,rstart = mat->rmap.rstart,rend = mat->rmap.rend;
  PetscInt       cstart = mat->cmap.rstart,cend = mat->cmap.rend,row,col;
  PetscTruth     roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat            A = aij->A;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  PetscInt       *aimax = a->imax,*ai = a->i,*ailen = a->ilen,*aj = a->j;
  PetscScalar    *aa = a->a;
  PetscTruth     ignorezeroentries = (((a->ignorezeroentries)&&(addv==ADD_VALUES))?PETSC_TRUE:PETSC_FALSE); 
  Mat            B = aij->B;
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data; 
  PetscInt       *bimax = b->imax,*bi = b->i,*bilen = b->ilen,*bj = b->j,bm = aij->B->rmap.n,am = aij->A->rmap.n;
  PetscScalar    *ba = b->a;

  PetscInt       *rp1,*rp2,ii,nrow1,nrow2,_i,rmax1,rmax2,N,low1,high1,low2,high2,t,lastcol1,lastcol2; 
  PetscInt       nonew = a->nonew; 
  PetscScalar    *ap1,*ap2;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap.N-1);
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row      = im[i] - rstart;
      lastcol1 = -1;
      rp1      = aj + ai[row]; 
      ap1      = aa + ai[row];
      rmax1    = aimax[row]; 
      nrow1    = ailen[row];  
      low1     = 0; 
      high1    = nrow1;
      lastcol2 = -1;
      rp2      = bj + bi[row]; 
      ap2      = ba + bi[row]; 
      rmax2    = bimax[row]; 
      nrow2    = bilen[row];  
      low2     = 0; 
      high2    = nrow2;

      for (j=0; j<n; j++) {
        if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES)) continue;
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          MatSetValues_SeqAIJ_A_Private(row,col,value,addv);
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->cmap.N) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap.N-1);}
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
              rp2      = bj + bi[row]; 
              ap2      = ba + bi[row]; 
              rmax2    = bimax[row]; 
              nrow2    = bilen[row];  
              low2     = 0; 
              high2    = nrow2;
              bm       = aij->B->rmap.n;
              ba = b->a;
            }
          } else col = in[j];
          MatSetValues_SeqAIJ_B_Private(row,col,value,addv);
        }
      }
    } else {
      if (!aij->donotstash) {
        if (roworiented) {
          if (ignorezeroentries && v[i*n] == 0.0) continue;
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n);CHKERRQ(ierr);
        } else {
          if (ignorezeroentries && v[i] == 0.0) continue;
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }}
  PetscFunctionReturnVoid();
}
EXTERN_C_END

