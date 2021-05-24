#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/sell/mpi/mpisell.h>   /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>
#include <petscblaslapack.h>
#include <petscsf.h>

/*MC
   MATSELL - MATSELL = "sell" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQSELL when constructed with a single process communicator,
   and MATMPISELL otherwise.  As a result, for single process communicators,
  MatSeqSELLSetPreallocation is supported, and similarly MatMPISELLSetPreallocation is supported
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type sell - sets the matrix type to "sell" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSELL(), MatCreateSeqSELL(), MATSEQSELL, MATMPISELL
M*/

PetscErrorCode MatDiagonalSet_MPISELL(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *sell=(Mat_MPISELL*)Y->data;

  PetscFunctionBegin;
  if (Y->assembled && Y->rmap->rstart == Y->cmap->rstart && Y->rmap->rend == Y->cmap->rend) {
    ierr = MatDiagonalSet(sell->A,D,is);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalSet_Default(Y,D,is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Local utility routine that creates a mapping from the global column
number to the local number in the off-diagonal part of the local
storage of the matrix.  When PETSC_USE_CTABLE is used this is scalable at
a slightly higher hash table cost; without it it is not scalable (each processor
has an order N integer array but is fast to acess.
*/
PetscErrorCode MatCreateColmap_MPISELL_Private(Mat mat)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       n=sell->B->cmap->n,i;

  PetscFunctionBegin;
  if (!sell->garray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPISELL Matrix was assembled but is missing garray");
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableCreate(n,mat->cmap->N+1,&sell->colmap);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscTableAdd(sell->colmap,sell->garray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
#else
  ierr = PetscCalloc1(mat->cmap->N+1,&sell->colmap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)mat,(mat->cmap->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<n; i++) sell->colmap[sell->garray[i]] = i+1;
#endif
  PetscFunctionReturn(0);
}

#define MatSetValues_SeqSELL_A_Private(row,col,value,addv,orow,ocol) \
  { \
    if (col <= lastcol1) low1 = 0; \
    else                high1 = nrow1; \
    lastcol1 = col; \
    while (high1-low1 > 5) { \
      t = (low1+high1)/2; \
      if (*(cp1+8*t) > col) high1 = t; \
      else                   low1 = t; \
    } \
    for (_i=low1; _i<high1; _i++) { \
      if (*(cp1+8*_i) > col) break; \
      if (*(cp1+8*_i) == col) { \
        if (addv == ADD_VALUES) *(vp1+8*_i) += value;   \
        else                     *(vp1+8*_i) = value; \
        goto a_noinsert; \
      } \
    }  \
    if (value == 0.0 && ignorezeroentries) {low1 = 0; high1 = nrow1;goto a_noinsert;} \
    if (nonew == 1) {low1 = 0; high1 = nrow1; goto a_noinsert;} \
    if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXSELLReallocateSELL(A,am,1,nrow1,a->sliidx,row/8,row,col,a->colidx,a->val,cp1,vp1,nonew,MatScalar); \
    /* shift up all the later entries in this row */ \
    for (ii=nrow1-1; ii>=_i; ii--) { \
      *(cp1+8*(ii+1)) = *(cp1+8*ii); \
      *(vp1+8*(ii+1)) = *(vp1+8*ii); \
    } \
    *(cp1+8*_i) = col; \
    *(vp1+8*_i) = value; \
    a->nz++; nrow1++; A->nonzerostate++; \
    a_noinsert: ; \
    a->rlen[row] = nrow1; \
  }

#define MatSetValues_SeqSELL_B_Private(row,col,value,addv,orow,ocol) \
  { \
    if (col <= lastcol2) low2 = 0; \
    else                high2 = nrow2; \
    lastcol2 = col; \
    while (high2-low2 > 5) { \
      t = (low2+high2)/2; \
      if (*(cp2+8*t) > col) high2 = t; \
      else low2  = t; \
    } \
    for (_i=low2; _i<high2; _i++) { \
      if (*(cp2+8*_i) > col) break; \
      if (*(cp2+8*_i) == col) { \
        if (addv == ADD_VALUES) *(vp2+8*_i) += value; \
        else                     *(vp2+8*_i) = value; \
        goto b_noinsert; \
      } \
    } \
    if (value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2; goto b_noinsert;} \
    if (nonew == 1) {low2 = 0; high2 = nrow2; goto b_noinsert;} \
    if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXSELLReallocateSELL(B,bm,1,nrow2,b->sliidx,row/8,row,col,b->colidx,b->val,cp2,vp2,nonew,MatScalar); \
    /* shift up all the later entries in this row */ \
    for (ii=nrow2-1; ii>=_i; ii--) { \
      *(cp2+8*(ii+1)) = *(cp2+8*ii); \
      *(vp2+8*(ii+1)) = *(vp2+8*ii); \
    } \
    *(cp2+8*_i) = col; \
    *(vp2+8*_i) = value; \
    b->nz++; nrow2++; B->nonzerostate++; \
    b_noinsert: ; \
    b->rlen[row] = nrow2; \
  }

PetscErrorCode MatSetValues_MPISELL(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscScalar    value;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart=mat->rmap->rstart,rend=mat->rmap->rend,shift1,shift2;
  PetscInt       cstart=mat->cmap->rstart,cend=mat->cmap->rend,row,col;
  PetscBool      roworiented=sell->roworiented;

  /* Some Variables required in the macro */
  Mat            A=sell->A;
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscBool      ignorezeroentries=a->ignorezeroentries,found;
  Mat            B=sell->B;
  Mat_SeqSELL    *b=(Mat_SeqSELL*)B->data;
  PetscInt       *cp1,*cp2,ii,_i,nrow1,nrow2,low1,high1,low2,high2,t,lastcol1,lastcol2;
  MatScalar      *vp1,*vp2;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    if (PetscUnlikelyDebug(im[i] >= mat->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap->N-1);
    if (im[i] >= rstart && im[i] < rend) {
      row      = im[i] - rstart;
      lastcol1 = -1;
      shift1   = a->sliidx[row>>3]+(row&0x07); /* starting index of the row */
      cp1      = a->colidx+shift1;
      vp1      = a->val+shift1;
      nrow1    = a->rlen[row];
      low1     = 0;
      high1    = nrow1;
      lastcol2 = -1;
      shift2   = b->sliidx[row>>3]+(row&0x07); /* starting index of the row */
      cp2      = b->colidx+shift2;
      vp2      = b->val+shift2;
      nrow2    = b->rlen[row];
      low2     = 0;
      high2    = nrow2;

      for (j=0; j<n; j++) {
        if (roworiented) value = v[i*n+j];
        else             value = v[i+j*m];
        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES)) continue;
        if (in[j] >= cstart && in[j] < cend) {
          col   = in[j] - cstart;
          MatSetValue_SeqSELL_Private(A,row,col,value,addv,im[i],in[j],cp1,vp1,lastcol1,low1,high1); /* set one value */
        } else if (in[j] < 0) continue;
        else if (PetscUnlikelyDebug(in[j] >= mat->cmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap->N-1);
        else {
          if (mat->was_assembled) {
            if (!sell->colmap) {
              ierr = MatCreateColmap_MPISELL_Private(mat);CHKERRQ(ierr);
            }
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(sell->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col--;
#else
            col = sell->colmap[in[j]] - 1;
#endif
            if (col < 0 && !((Mat_SeqSELL*)(sell->B->data))->nonew) {
              ierr   = MatDisAssemble_MPISELL(mat);CHKERRQ(ierr);
              col    = in[j];
              /* Reinitialize the variables required by MatSetValues_SeqSELL_B_Private() */
              B      = sell->B;
              b      = (Mat_SeqSELL*)B->data;
              shift2 = b->sliidx[row>>3]+(row&0x07); /* starting index of the row */
              cp2    = b->colidx+shift2;
              vp2    = b->val+shift2;
              nrow2  = b->rlen[row];
              low2   = 0;
              high2  = nrow2;
            } else if (col < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", im[i], in[j]);
          } else col = in[j];
          MatSetValue_SeqSELL_Private(B,row,col,value,addv,im[i],in[j],cp2,vp2,lastcol2,low2,high2); /* set one value */
        }
      }
    } else {
      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!sell->donotstash) {
        mat->assembled = PETSC_FALSE;
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_MPISELL(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart=mat->rmap->rstart,rend=mat->rmap->rend;
  PetscInt       cstart=mat->cmap->rstart,cend=mat->cmap->rend,row,col;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",idxm[i]);*/
    if (idxm[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm[i],mat->rmap->N-1);
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",idxn[j]); */
        if (idxn[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",idxn[j],mat->cmap->N-1);
        if (idxn[j] >= cstart && idxn[j] < cend) {
          col  = idxn[j] - cstart;
          ierr = MatGetValues(sell->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!sell->colmap) {
            ierr = MatCreateColmap_MPISELL_Private(mat);CHKERRQ(ierr);
          }
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(sell->colmap,idxn[j]+1,&col);CHKERRQ(ierr);
          col--;
#else
          col = sell->colmap[idxn[j]] - 1;
#endif
          if ((col < 0) || (sell->garray[col] != idxn[j])) *(v+i*n+j) = 0.0;
          else {
            ierr = MatGetValues(sell->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          }
        }
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatMultDiagonalBlock_MPISELL(Mat,Vec,Vec);

PetscErrorCode MatAssemblyBegin_MPISELL(Mat mat,MatAssemblyType mode)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (sell->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(sell->A,"Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPISELL(Mat mat,MatAssemblyType mode)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;
  PetscMPIInt    n;
  PetscInt       i,flg;
  PetscInt       *row,*col;
  PetscScalar    *val;
  PetscBool      other_disassembled;
  /* do not use 'b = (Mat_SeqSELL*)sell->B->data' as B can be reset in disassembly */
  PetscFunctionBegin;
  if (!sell->donotstash && !mat->nooffprocentries) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n; i++) { /* assemble one by one */
        ierr = MatSetValues_MPISELL(mat,1,row+i,1,col+i,val+i,mat->insertmode);CHKERRQ(ierr);
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(sell->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sell->A,mode);CHKERRQ(ierr);

  /*
     determine if any processor has disassembled, if so we must
     also disassemble ourselfs, in order that we may reassemble.
  */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqSELL*)sell->B->data)->nonew) {
    ierr = MPIU_Allreduce(&mat->was_assembled,&other_disassembled,1,MPIU_BOOL,MPI_PROD,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
    if (mat->was_assembled && !other_disassembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatDisAssemble not implemented yet\n");
  }
  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPISELL(mat);CHKERRQ(ierr);
  }
  /*
  ierr = MatSetOption(sell->B,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);
  */
  ierr = MatAssemblyBegin(sell->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sell->B,mode);CHKERRQ(ierr);
  ierr = PetscFree2(sell->rowvalues,sell->rowindices);CHKERRQ(ierr);
  sell->rowvalues = NULL;
  ierr = VecDestroy(&sell->diag);CHKERRQ(ierr);

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqSELL*)(sell->A->data))->nonew) {
    PetscObjectState state = sell->A->nonzerostate + sell->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&mat->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPISELL(Mat A)
{
  Mat_MPISELL    *l=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISELL(Mat A,Vec xx,Vec yy)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultDiagonalBlock_MPISELL(Mat A,Vec bb,Vec xx)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultDiagonalBlock(a->A,bb,xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPISELL(Mat A,Vec xx,Vec yy)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  /* add partial results together */
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsTranspose_MPISELL(Mat Amat,Mat Bmat,PetscReal tol,PetscBool *f)
{
  MPI_Comm       comm;
  Mat_MPISELL    *Asell=(Mat_MPISELL*)Amat->data,*Bsell;
  Mat            Adia=Asell->A,Bdia,Aoff,Boff,*Aoffs,*Boffs;
  IS             Me,Notme;
  PetscErrorCode ierr;
  PetscInt       M,N,first,last,*notme,i;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Easy test: symmetric diagonal block */
  Bsell = (Mat_MPISELL*)Bmat->data; Bdia = Bsell->A;
  ierr = MatIsTranspose(Adia,Bdia,tol,f);CHKERRQ(ierr);
  if (!*f) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) PetscFunctionReturn(0);

  /* Hard test: off-diagonal block. This takes a MatCreateSubMatrix. */
  ierr = MatGetSize(Amat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&first,&last);CHKERRQ(ierr);
  ierr = PetscMalloc1(N-last+first,&notme);CHKERRQ(ierr);
  for (i=0; i<first; i++) notme[i] = i;
  for (i=last; i<M; i++) notme[i-last+first] = i;
  ierr = ISCreateGeneral(MPI_COMM_SELF,N-last+first,notme,PETSC_COPY_VALUES,&Notme);CHKERRQ(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,last-first,first,1,&Me);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(Amat,1,&Me,&Notme,MAT_INITIAL_MATRIX,&Aoffs);CHKERRQ(ierr);
  Aoff = Aoffs[0];
  ierr = MatCreateSubMatrices(Bmat,1,&Notme,&Me,MAT_INITIAL_MATRIX,&Boffs);CHKERRQ(ierr);
  Boff = Boffs[0];
  ierr = MatIsTranspose(Aoff,Boff,tol,f);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&Aoffs);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&Boffs);CHKERRQ(ierr);
  ierr = ISDestroy(&Me);CHKERRQ(ierr);
  ierr = ISDestroy(&Notme);CHKERRQ(ierr);
  ierr = PetscFree(notme);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_MPISELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransposeadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* add partial results together */
  ierr = VecScatterBegin(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
PetscErrorCode MatGetDiagonal_MPISELL(Mat A,Vec v)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  if (A->rmap->rstart != A->cmap->rstart || A->rmap->rend != A->cmap->rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"row partition must equal col partition");
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_MPISELL(Mat A,PetscScalar aa)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPISELL(Mat mat)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D",mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = VecDestroy(&sell->diag);CHKERRQ(ierr);
  ierr = MatDestroy(&sell->A);CHKERRQ(ierr);
  ierr = MatDestroy(&sell->B);CHKERRQ(ierr);
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&sell->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(sell->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(sell->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&sell->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sell->Mvctx);CHKERRQ(ierr);
  ierr = PetscFree2(sell->rowvalues,sell->rowindices);CHKERRQ(ierr);
  ierr = PetscFree(sell->ld);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatIsTranspose_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPISELLSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisell_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDiagonalScaleLocal_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode MatView_MPISELL_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPISELL       *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank=sell->rank,size=sell->size;
  PetscBool         isdraw,iascii,isbinary;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo   info;
      PetscInt *inodes;

      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank);CHKERRMPI(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatInodeGetInodeSizes(sell->A,NULL,&inodes,NULL);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      if (!inodes) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D mem %D, not using I-node routines\n",
                                                  rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D mem %D, using I-node routines\n",
                                                  rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
      }
      ierr = MatGetInfo(sell->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = MatGetInfo(sell->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Information on VecScatter used in matrix-vector product: \n");CHKERRQ(ierr);
      ierr = VecScatterView(sell->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscInt inodecount,inodelimit,*inodes;
      ierr = MatInodeGetInodeSizes(sell->A,&inodecount,&inodes,&inodelimit);CHKERRQ(ierr);
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
      ierr = PetscObjectSetName((PetscObject)sell->A,((PetscObject)mat)->name);CHKERRQ(ierr);
      ierr = MatView(sell->A,viewer);CHKERRQ(ierr);
    } else {
      /* ierr = MatView_MPISELL_Binary(mat,viewer);CHKERRQ(ierr); */
    }
    PetscFunctionReturn(0);
  } else if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);
  }

  {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqSELL *Aloc;
    PetscInt    M=mat->rmap->N,N=mat->cmap->N,*acolidx,row,col,i,j;
    MatScalar   *aval;
    PetscBool   isnonzero;

    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    /* This is just a temporary matrix, so explicitly using MATMPISELL is probably best */
    ierr = MatSetType(A,MATMPISELL);CHKERRQ(ierr);
    ierr = MatMPISELLSetPreallocation(A,0,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc = (Mat_SeqSELL*)sell->A->data;
    acolidx = Aloc->colidx; aval = Aloc->val;
    for (i=0; i<Aloc->totalslices; i++) { /* loop over slices */
      for (j=Aloc->sliidx[i]; j<Aloc->sliidx[i+1]; j++) {
        isnonzero = (PetscBool)((j-Aloc->sliidx[i])/8 < Aloc->rlen[(i<<3)+(j&0x07)]);
        if (isnonzero) { /* check the mask bit */
          row  = (i<<3)+(j&0x07) + mat->rmap->rstart; /* i<<3 is the starting row of this slice */
          col  = *acolidx + mat->rmap->rstart;
          ierr = MatSetValues(A,1,&row,1,&col,aval,INSERT_VALUES);CHKERRQ(ierr);
        }
        aval++; acolidx++;
      }
    }

    /* copy over the B part */
    Aloc = (Mat_SeqSELL*)sell->B->data;
    acolidx = Aloc->colidx; aval = Aloc->val;
    for (i=0; i<Aloc->totalslices; i++) {
      for (j=Aloc->sliidx[i]; j<Aloc->sliidx[i+1]; j++) {
        isnonzero = (PetscBool)((j-Aloc->sliidx[i])/8 < Aloc->rlen[(i<<3)+(j&0x07)]);
        if (isnonzero) {
          row  = (i<<3)+(j&0x07) + mat->rmap->rstart;
          col  = sell->garray[*acolidx];
          ierr = MatSetValues(A,1,&row,1,&col,aval,INSERT_VALUES);CHKERRQ(ierr);
        }
        aval++; acolidx++;
      }
    }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /*
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the PetscDraw object
    */
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscObjectSetName((PetscObject)((Mat_MPISELL*)(A->data))->A,((PetscObject)mat)->name);CHKERRQ(ierr);
      ierr = MatView_SeqSELL(((Mat_MPISELL*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MPISELL(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  if (iascii || isdraw || isbinary || issocket) {
    ierr = MatView_MPISELL_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetGhosts_MPISELL(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(sell->B,NULL,nghosts);CHKERRQ(ierr);
  if (ghosts) *ghosts = sell->garray;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPISELL(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPISELL    *mat=(Mat_MPISELL*)matin->data;
  Mat            A=mat->A,B=mat->B;
  PetscErrorCode ierr;
  PetscLogDouble isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size = 1.0;
  ierr             = MatGetInfo(A,MAT_LOCAL,info);CHKERRQ(ierr);

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
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)matin));CHKERRMPI(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)matin));CHKERRMPI(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_MPISELL(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_USE_INODES:
  case MAT_IGNORE_ZERO_ENTRIES:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A,1);
    a->roworiented = flg;

    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_SORTED_FULL:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_SPD:
    A->spd_set = PETSC_TRUE;
    A->spd     = flg;
    if (flg) {
      A->symmetric                  = PETSC_TRUE;
      A->structurally_symmetric     = PETSC_TRUE;
      A->symmetric_set              = PETSC_TRUE;
      A->structurally_symmetric_set = PETSC_TRUE;
    }
    break;
  case MAT_SYMMETRIC:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_HERMITIAN:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_SYMMETRY_ETERNAL:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPISELL(Mat mat,Vec ll,Vec rr)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  Mat            a=sell->A,b=sell->B;
  PetscErrorCode ierr;
  PetscInt       s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(sell->Mvctx,rr,sell->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,NULL);CHKERRQ(ierr);
  }
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(sell->Mvctx,rr,sell->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,NULL,sell->lvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_MPISELL(Mat A)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatEqual_MPISELL(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPISELL    *matB=(Mat_MPISELL*)B->data,*matA=(Mat_MPISELL*)A->data;
  Mat            a,b,c,d;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPIU_Allreduce(&flg,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_MPISELL(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  Mat_MPISELL    *b=(Mat_MPISELL*)B->data;

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

PetscErrorCode MatSetUp_MPISELL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPISELLSetPreallocation(A,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatConjugate_SeqSELL(Mat);

PetscErrorCode MatConjugate_MPISELL(Mat mat)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;

  PetscFunctionBegin;
  ierr = MatConjugate_SeqSELL(sell->A);CHKERRQ(ierr);
  ierr = MatConjugate_SeqSELL(sell->B);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPISELL(Mat A)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPISELL(Mat A)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_MPISELL(Mat A,const PetscScalar **values)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInvertBlockDiagonal(a->A,values);CHKERRQ(ierr);
  A->factorerrortype = a->A->factorerrortype;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetRandom_MPISELL(Mat x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *sell=(Mat_MPISELL*)x->data;

  PetscFunctionBegin;
  ierr = MatSetRandom(sell->A,rctx);CHKERRQ(ierr);
  ierr = MatSetRandom(sell->B,rctx);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(x,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPISELL(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPISELL options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_MPISELL(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *msell=(Mat_MPISELL*)Y->data;
  Mat_SeqSELL    *sell=(Mat_SeqSELL*)msell->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    ierr = MatMPISELLSetPreallocation(Y,1,NULL,0,NULL);CHKERRQ(ierr);
  } else if (!sell->nz) {
    PetscInt nonew = sell->nonew;
    ierr = MatSeqSELLSetPreallocation(msell->A,1,NULL);CHKERRQ(ierr);
    sell->nonew = nonew;
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMissingDiagonal_MPISELL(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only works for square matrices");
  ierr = MatMissingDiagonal(a->A,missing,d);CHKERRQ(ierr);
  if (d) {
    PetscInt rstart;
    ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
    *d += rstart;

  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonalBlock_MPISELL(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPISELL*)A->data)->A;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPISELL,
                                       NULL,
                                       NULL,
                                       MatMult_MPISELL,
                                /* 4*/ MatMultAdd_MPISELL,
                                       MatMultTranspose_MPISELL,
                                       MatMultTransposeAdd_MPISELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_MPISELL,
                                       NULL,
                                /*15*/ MatGetInfo_MPISELL,
                                       MatEqual_MPISELL,
                                       MatGetDiagonal_MPISELL,
                                       MatDiagonalScale_MPISELL,
                                       NULL,
                                /*20*/ MatAssemblyBegin_MPISELL,
                                       MatAssemblyEnd_MPISELL,
                                       MatSetOption_MPISELL,
                                       MatZeroEntries_MPISELL,
                                /*24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*29*/ MatSetUp_MPISELL,
                                       NULL,
                                       NULL,
                                       MatGetDiagonalBlock_MPISELL,
                                       NULL,
                                /*34*/ MatDuplicate_MPISELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*39*/ NULL,
                                       NULL,
                                       NULL,
                                       MatGetValues_MPISELL,
                                       MatCopy_MPISELL,
                                /*44*/ NULL,
                                       MatScale_MPISELL,
                                       MatShift_MPISELL,
                                       MatDiagonalSet_MPISELL,
                                       NULL,
                                /*49*/ MatSetRandom_MPISELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*54*/ MatFDColoringCreate_MPIXAIJ,
                                       NULL,
                                       MatSetUnfactored_MPISELL,
                                       NULL,
                                       NULL,
                                /*59*/ NULL,
                                       MatDestroy_MPISELL,
                                       MatView_MPISELL,
                                       NULL,
                                       NULL,
                                /*64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*75*/ MatFDColoringApply_AIJ, /* reuse AIJ function */
                                       MatSetFromOptions_MPISELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*80*/ NULL,
                                       NULL,
                                       NULL,
                                /*83*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_MPISELL,
                                       NULL,
                                /*104*/NULL,
                                       MatRealPart_MPISELL,
                                       MatImaginaryPart_MPISELL,
                                       NULL,
                                       NULL,
                                /*109*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_MPISELL,
                                /*114*/NULL,
                                       NULL,
                                       MatGetGhosts_MPISELL,
                                       NULL,
                                       NULL,
                                /*119*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*124*/NULL,
                                       NULL,
                                       MatInvertBlockDiagonal_MPISELL,
                                       NULL,
                                       NULL,
                                /*129*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*134*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*139*/NULL,
                                       NULL,
                                       NULL,
                                       MatFDColoringSetUp_MPIXAIJ,
                                       NULL,
                                /*144*/NULL
};

/* ----------------------------------------------------------------------------------------*/

PetscErrorCode MatStoreValues_MPISELL(Mat mat)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(sell->A);CHKERRQ(ierr);
  ierr = MatStoreValues(sell->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRetrieveValues_MPISELL(Mat mat)
{
  Mat_MPISELL    *sell=(Mat_MPISELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(sell->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(sell->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPISELLSetPreallocation_MPISELL(Mat B,PetscInt d_rlenmax,const PetscInt d_rlen[],PetscInt o_rlenmax,const PetscInt o_rlen[])
{
  Mat_MPISELL    *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  b = (Mat_MPISELL*)B->data;

  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQSELL matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(b->A,B,B);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQSELL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(b->B,B,B);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQSELL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }

  ierr = MatSeqSELLSetPreallocation(b->A,d_rlenmax,d_rlen);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(b->B,o_rlenmax,o_rlen);CHKERRQ(ierr);
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  /*
    critical for MatAssemblyEnd to work.
    MatAssemblyBegin checks it to set up was_assembled
    and MatAssemblyEnd checks was_assembled to determine whether to build garray
  */
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_MPISELL(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPISELL    *a,*oldmat=(Mat_MPISELL*)matin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *newmat = NULL;
  ierr    = MatCreate(PetscObjectComm((PetscObject)matin),&mat);CHKERRQ(ierr);
  ierr    = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr    = MatSetBlockSizesFromMats(mat,matin,matin);CHKERRQ(ierr);
  ierr    = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);
  a       = (Mat_MPISELL*)mat->data;

  mat->factortype   = matin->factortype;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;
  mat->preallocated = PETSC_TRUE;

  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = NULL;
  a->rowvalues    = NULL;
  a->getrowactive = PETSC_FALSE;

  ierr = PetscLayoutReference(matin->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(matin->cmap,&mat->cmap);CHKERRQ(ierr);

  if (oldmat->colmap) {
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    ierr = PetscMalloc1(mat->cmap->N,&a->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(mat->cmap->N)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(a->colmap,oldmat->colmap,mat->cmap->N);CHKERRQ(ierr);
#endif
  } else a->colmap = NULL;
  if (oldmat->garray) {
    PetscInt len;
    len  = oldmat->B->cmap->n;
    ierr = PetscMalloc1(len+1,&a->garray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,len*sizeof(PetscInt));CHKERRQ(ierr);
    if (len) { ierr = PetscArraycpy(a->garray,oldmat->garray,len);CHKERRQ(ierr); }
  } else a->garray = NULL;

  ierr    = VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->lvec);CHKERRQ(ierr);
  ierr    = VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->Mvctx);CHKERRQ(ierr);
  ierr    = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A);CHKERRQ(ierr);
  ierr    = MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->B);CHKERRQ(ierr);
  ierr    = PetscFunctionListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

/*@C
   MatMPISELLSetPreallocation - Preallocates memory for a sparse parallel matrix in sell format.
   For good matrix assembly performance the user should preallocate the matrix storage by
   setting the parameters d_nz (or d_nnz) and o_nz (or o_nnz).

   Collective

   Input Parameters:
+  B - the matrix
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or NULL (PETSC_NULL_INTEGER in Fortran), if d_nz is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e 'm'.
           For matrices that will be factored, you must leave room for (and set)
           the diagonal entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or NULL (PETSC_NULL_INTEGER in Fortran), if o_nz is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e 'm'.

   If the *_nnz parameter is given then the *_nz parameter is ignored

   The stored row and column indices begin with zero.

   The parallel matrix is partitioned such that the first m0 rows belong to
   process 0, the next m1 rows belong to process 1, the next m2 rows belong
   to process 2 etc.. where m0,m1,m2... are the input parameter 'm'.

   The DIAGONAL portion of the local submatrix of a processor can be defined
   as the submatrix which is obtained by extraction the part corresponding to
   the rows r1-r2 and columns c1-c2 of the global matrix, where r1 is the
   first row that belongs to the processor, r2 is the last row belonging to
   the this processor, and c1-c2 is range of indices of the local part of a
   vector suitable for applying the matrix to.  This is an mxn matrix.  In the
   common case of a square matrix, the row and column ranges are the same and
   the DIAGONAL part is also square. The remaining portion of the local
   submatrix (mxN) constitute the OFF-DIAGONAL portion.

   If o_nnz, d_nnz are specified, then o_nz, and d_nz are ignored.

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string
   malloc in them to see if additional memory allocation was needed.

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
   part as SeqSELL matrices. for eg: proc1 will store [E] as a SeqSELL
   matrix, ans [DF] as another SeqSELL matrix.

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
   Here the space allocated is according to nz (or maximum values in the nnz
   if nnz is provided) for DIAGONAL and OFF-DIAGONAL submatrices, i.e (2+2+3+2)*3+(1+4)*2=37

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqSELL(), MatSetValues(), MatCreatesell(),
          MATMPISELL, MatGetInfo(), PetscSplitOwnership()
@*/
PetscErrorCode MatMPISELLSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatMPISELLSetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSELL - Creates a sparse parallel matrix in SELL format.

   Collective

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
.  d_rlenmax - max number of nonzeros per row in DIAGONAL portion of local submatrix
               (same value is used for all local rows)
.  d_rlen - array containing the number of nonzeros in the various rows of the
            DIAGONAL portion of the local submatrix (possibly different for each row)
            or NULL, if d_rlenmax is used to specify the nonzero structure.
            The size of this array is equal to the number of local rows, i.e 'm'.
.  o_rlenmax - max number of nonzeros per row in the OFF-DIAGONAL portion of local
               submatrix (same value is used for all local rows).
-  o_rlen - array containing the number of nonzeros in the various rows of the
            OFF-DIAGONAL portion of the local submatrix (possibly different for
            each row) or NULL, if o_rlenmax is used to specify the nonzero
            structure. The size of this array is equal to the number
            of local rows, i.e 'm'.

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqSELLSetPreallocation]

   Notes:
   If the *_rlen parameter is given then the *_rlenmax parameter is ignored

   m,n,M,N parameters specify the size of the matrix, and its partitioning across
   processors, while d_rlenmax,d_rlen,o_rlenmax,o_rlen parameters specify the approximate
   storage requirements for this matrix.

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one
   processor than it must be used on all processors that share the object for
   that argument.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   The parallel matrix is partitioned across processors such that the
   first m0 rows belong to process 0, the next m1 rows belong to
   process 1, the next m2 rows belong to process 2 etc.. where
   m0,m1,m2,.. are the input parameter 'm'. i.e each processor stores
   values corresponding to [m x N] submatrix.

   The columns are logically partitioned with the n0 columns belonging
   to 0th partition, the next n1 columns belonging to the next
   partition etc.. where n0,n1,n2... are the input parameter 'n'.

   The DIAGONAL portion of the local submatrix on any given processor
   is the submatrix corresponding to the rows and columns m,n
   corresponding to the given processor. i.e diagonal matrix on
   process 0 is [m0 x n0], diagonal matrix on process 1 is [m1 x n1]
   etc. The remaining portion of the local submatrix [m x (N-n)]
   constitute the OFF-DIAGONAL portion. The example below better
   illustrates this concept.

   For a square global matrix we define each processor's diagonal portion
   to be its local rows and the corresponding columns (a square submatrix);
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix).

   If o_rlen, d_rlen are specified, then o_rlenmax, and d_rlenmax are ignored.

   When calling this routine with a single process communicator, a matrix of
   type SEQSELL is returned.  If a matrix of type MATMPISELL is desired for this
   type of communicator, use the construction mechanism:
     MatCreate(...,&A); MatSetType(A,MATMPISELL); MatSetSizes(A, m,n,M,N); MatMPISELLSetPreallocation(A,...);

   Options Database Keys:
-  -mat_sell_oneindex - Internally use indexing starting at 1
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
   part as SeqSELL matrices. for eg: proc1 will store [E] as a SeqSELL
   matrix, ans [DF] as another SeqSELL matrix.

   When d_rlenmax, o_rlenmax parameters are specified, d_rlenmax storage elements are
   allocated for every row of the local diagonal submatrix, and o_rlenmax
   storage locations are allocated for every row of the OFF-DIAGONAL submat.
   One way to choose d_rlenmax and o_rlenmax is to use the max nonzerors per local
   rows for each of the local DIAGONAL, and the OFF-DIAGONAL submatrices.
   In this case, the values of d_rlenmax,o_rlenmax are:
.vb
     proc0 : d_rlenmax = 2, o_rlenmax = 2
     proc1 : d_rlenmax = 3, o_rlenmax = 2
     proc2 : d_rlenmax = 1, o_rlenmax = 4
.ve
   We are allocating m*(d_rlenmax+o_rlenmax) storage locations for every proc. This
   translates to 3*(2+2)=12 for proc0, 3*(3+2)=15 for proc1, 2*(1+4)=10
   for proc3. i.e we are using 12+15+10=37 storage locations to store
   34 values.

   When d_rlen, o_rlen parameters are specified, the storage is specified
   for every row, coresponding to both DIAGONAL and OFF-DIAGONAL submatrices.
   In the above case the values for d_nnz,o_nnz are:
.vb
     proc0: d_nnz = [2,2,2] and o_nnz = [2,2,2]
     proc1: d_nnz = [3,3,2] and o_nnz = [2,1,1]
     proc2: d_nnz = [1,1]   and o_nnz = [4,4]
.ve
   Here the space allocated is still 37 though there are 34 nonzeros because
   the allocation is always done according to rlenmax.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqSELL(), MatSetValues(), MatMPISELLSetPreallocation(), MatMPISELLSetPreallocationSELL(),
          MATMPISELL, MatCreateMPISELLWithArrays()
@*/
PetscErrorCode MatCreateSELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_rlenmax,const PetscInt d_rlen[],PetscInt o_rlenmax,const PetscInt o_rlen[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPISELL);CHKERRQ(ierr);
    ierr = MatMPISELLSetPreallocation(*A,d_rlenmax,d_rlen,o_rlenmax,o_rlen);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQSELL);CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(*A,d_rlenmax,d_rlen);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPISELLGetSeqSELL(Mat A,Mat *Ad,Mat *Ao,const PetscInt *colmap[])
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPISELL,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"This function requires a MATMPISELL matrix as input");
  if (Ad)     *Ad     = a->A;
  if (Ao)     *Ao     = a->B;
  if (colmap) *colmap = a->garray;
  PetscFunctionReturn(0);
}

/*@C
     MatMPISELLGetLocalMatCondensed - Creates a SeqSELL matrix from an MATMPISELL matrix by taking all its local rows and NON-ZERO columns

    Not Collective

   Input Parameters:
+    A - the matrix
.    scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-    row, col - index sets of rows and columns to extract (or NULL)

   Output Parameter:
.    A_loc - the local sequential matrix generated

    Level: developer

.seealso: MatGetOwnershipRange(), MatMPISELLGetLocalMat()

@*/
PetscErrorCode MatMPISELLGetLocalMatCondensed(Mat A,MatReuse scall,IS *row,IS *col,Mat *A_loc)
{
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,start,end,ncols,nzA,nzB,*cmap,imark,*idx;
  IS             isrowa,iscola;
  Mat            *aloc;
  PetscBool      match;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPISELL,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,"Requires MATMPISELL matrix as input");
  ierr = PetscLogEventBegin(MAT_Getlocalmatcondensed,A,0,0,0);CHKERRQ(ierr);
  if (!row) {
    start = A->rmap->rstart; end = A->rmap->rend;
    ierr  = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&isrowa);CHKERRQ(ierr);
  } else {
    isrowa = *row;
  }
  if (!col) {
    start = A->cmap->rstart;
    cmap  = a->garray;
    nzA   = a->A->cmap->n;
    nzB   = a->B->cmap->n;
    ierr  = PetscMalloc1(nzA+nzB, &idx);CHKERRQ(ierr);
    ncols = 0;
    for (i=0; i<nzB; i++) {
      if (cmap[i] < start) idx[ncols++] = cmap[i];
      else break;
    }
    imark = i;
    for (i=0; i<nzA; i++) idx[ncols++] = start + i;
    for (i=imark; i<nzB; i++) idx[ncols++] = cmap[i];
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,PETSC_OWN_POINTER,&iscola);CHKERRQ(ierr);
  } else {
    iscola = *col;
  }
  if (scall != MAT_INITIAL_MATRIX) {
    ierr    = PetscMalloc1(1,&aloc);CHKERRQ(ierr);
    aloc[0] = *A_loc;
  }
  ierr   = MatCreateSubMatrices(A,1,&isrowa,&iscola,scall,&aloc);CHKERRQ(ierr);
  *A_loc = aloc[0];
  ierr   = PetscFree(aloc);CHKERRQ(ierr);
  if (!row) {
    ierr = ISDestroy(&isrowa);CHKERRQ(ierr);
  }
  if (!col) {
    ierr = ISDestroy(&iscola);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_Getlocalmatcondensed,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode MatConvert_MPISELL_MPIAIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_MPISELL    *a=(Mat_MPISELL*)A->data;
  Mat            B;
  Mat_MPIAIJ     *b;

  PetscFunctionBegin;
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Matrix must be assembled");

  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);
  }
  b    = (Mat_MPIAIJ*) B->data;

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatConvert_SeqSELL_SeqAIJ(a->A, MATSEQAIJ, MAT_REUSE_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqSELL_SeqAIJ(a->B, MATSEQAIJ, MAT_REUSE_MATRIX, &b->B);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&b->A);CHKERRQ(ierr);
    ierr = MatDestroy(&b->B);CHKERRQ(ierr);
    ierr = MatDisAssemble_MPISELL(A);CHKERRQ(ierr);
    ierr = MatConvert_SeqSELL_SeqAIJ(a->A, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqSELL_SeqAIJ(a->B, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_MPIAIJ_MPISELL(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat            B;
  Mat_MPISELL    *b;

  PetscFunctionBegin;
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Matrix must be assembled");

  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATMPISELL);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);
  }
  b    = (Mat_MPISELL*) B->data;

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatConvert_SeqAIJ_SeqSELL(a->A, MATSEQSELL, MAT_REUSE_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_SeqSELL(a->B, MATSEQSELL, MAT_REUSE_MATRIX, &b->B);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&b->A);CHKERRQ(ierr);
    ierr = MatDestroy(&b->B);CHKERRQ(ierr);
    ierr = MatDisAssemble_MPIAIJ(A);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_SeqSELL(a->A, MATSEQSELL, MAT_INITIAL_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_SeqSELL(a->B, MATSEQSELL, MAT_INITIAL_MATRIX, &b->B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_MPISELL(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISELL    *mat=(Mat_MPISELL*)matin->data;
  PetscErrorCode ierr;
  Vec            bb1=NULL;

  PetscFunctionBegin;
  if (flag == SOR_APPLY_UPPER) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (its > 1 || ~flag & SOR_ZERO_INITIAL_GUESS || flag & SOR_EISENSTAT) {
    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }

    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_FORWARD_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_BACKWARD_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)matin),PETSC_ERR_SUP,"Parallel SOR not supported");

  ierr = VecDestroy(&bb1);CHKERRQ(ierr);

  matin->factorerrortype = mat->A->factorerrortype;
  PetscFunctionReturn(0);
}

/*MC
   MATMPISELL - MATMPISELL = "MPISELL" - A matrix type to be used for parallel sparse matrices.

   Options Database Keys:
. -mat_type MPISELL - sets the matrix type to "MPISELL" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSELL()
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPISELL(Mat B)
{
  Mat_MPISELL    *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
  ierr          = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data       = (void*)b;
  ierr          = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled  = PETSC_FALSE;
  B->insertmode = NOT_SET_VALUES;
  b->size       = size;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&b->rank);CHKERRMPI(ierr);
  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash);CHKERRQ(ierr);

  b->donotstash  = PETSC_FALSE;
  b->colmap      = NULL;
  b->garray      = NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used for matrix vector multiply */
  b->lvec  = NULL;
  b->Mvctx = NULL;

  /* stuff for MatGetRow() */
  b->rowindices   = NULL;
  b->rowvalues    = NULL;
  b->getrowactive = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_MPISELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_MPISELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatIsTranspose_C",MatIsTranspose_MPISELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPISELLSetPreallocation_C",MatMPISELLSetPreallocation_MPISELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisell_mpiaij_C",MatConvert_MPISELL_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDiagonalScaleLocal_C",MatDiagonalScaleLocal_MPISELL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPISELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
