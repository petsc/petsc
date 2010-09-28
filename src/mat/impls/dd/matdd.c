#define PETSCMAT_DLL

#include "../src/mat/impls/dd/matdd.h"          /*I "petscmat.h" I*/



/*
  MatDD represents global matrix M as composed of local submatrices or blocks M_{ij}.
  The blocks assembled using scatters S_j and gather G_i:
           M = \sum_{i,j=0}^{N-1} G_i M_{ij} S_j.
  where each submatrix M_{ij} acts between the j-th block of the input Vec
  and the i-th block of the output Vec.  Therefore, M also defines 
  a splitting of the Vec space(s) it acts on (between) into nonoverlapping
  dof blocks of arbitrary sizes.  MatDD can then applied to Vecs using the submatrix 
  MatMult actions.

  Preallocation can be done by declaring which nonzero blocks are present with
  MatDDAddBlock(matdd, i,j).  This is analogous to declaring a nonzero element in an
  AIJ matrix, although such preallocation makes only sense at the block 
  granularity.  It would probably make even more sense to add all blocks at once
  with some block version of preallocation.  This, of course, would make little
  sense for MATDDDENSE and similar formats.  

 */


/* 
Use cases that must work:

1. Efficient representation of a stiffness matrix for a 2D P1 triangular mesh using just the connectivity Map and an element Mat.
2. Assembly of a stiffness matrix for a geometrically conforming mortar element in 2D on a square doublet 
  (e.g., as in P. Fischer et al).
 a) Eliminate slaved degrees of freedom on the mortar.
 b) Add Lagrange multipliers on the mortar.
3. Efficient representation of a bordered matrix.  For example, a single row/col border arising from parameter
continuation.
4. Construction of a global pressure split scatter from a local scatter for Stokes on function spaces in 1 and 2.
5. MG
*/


/* Need to provide operations for 
  1. easy construction of gathers/scatters:
   - P1 simplicial gather
   - mortar element mesh gather (with and without constraints)
   - field split-off scatter from local splits
  2. easy combinations of gathers/scatters with certain types of blocks:
   - "tensor product" of a scatter with a block
*/


#undef  __FUNCT__
#define __FUNCT__ "Mat_DDBlockSetMat"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_DDBlockSetMat(Mat A, PetscInt rowblock, PetscInt colblock, Mat_DDBlock *block, Mat B) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  PetscInt              m,n,M,N;
#if defined PETSC_USE_DEBUG
  PetscInt              actualM, actualN;
#endif
  MPI_Comm              subcomm = ((PetscObject)B)->comm;
  PetscMPIInt           subcommsize;
  
  PetscFunctionBegin;
  
  /**/
  m = dd->lrowblockoffset[rowblock+1]-dd->lrowblockoffset[rowblock];
  n = dd->lcolblockoffset[colblock+1]-dd->lcolblockoffset[colblock];
  M = dd->growblockoffset[rowblock+1]-dd->growblockoffset[rowblock];
  N = dd->gcolblockoffset[colblock+1]-dd->gcolblockoffset[colblock];

  ierr = MPI_Comm_size(subcomm, &subcommsize); CHKERRQ(ierr);
#if defined PETSC_USE_DEBUG
  /**/
  if(subcommsize == 1) {
    actualM = M; actualN = N;
  }
  else {
    ierr = MPI_Allreduce(&m, &actualM, 1, MPI_INT, MPI_SUM, subcomm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&n, &actualN, 1, MPI_INT, MPI_SUM, subcomm); CHKERRQ(ierr);
  }
  if(actualM != M) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER, "Block[%d,%d]'s actual global row size %d doesn't match declared size %d", rowblock, colblock, actualM, M);
  }
  
  if(actualN != N) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER, "Block[%d,%d]'s actual global row size %d doesn't match declared size %d", rowblock, colblock, actualN, N);
  }
#endif

  if(subcommsize == 1) {
    ierr = VecCreateSeqWithArray(subcomm,n,PETSC_NULL,&(block->invec)); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(subcomm,m,PETSC_NULL,&(block->outvec)); CHKERRQ(ierr);
  }
  else {
    ierr = VecCreateMPIWithArray(subcomm,n,N,PETSC_NULL,&(block->invec)); CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(subcomm,m,M,PETSC_NULL,&(block->outvec)); CHKERRQ(ierr);
  }
  block->mat = B;

  PetscFunctionReturn(0);
}/* Mat_DDBlockSetMat() */

#undef  __FUNCT__
#define __FUNCT__ "Mat_DDBlockInit"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_DDBlockInit(Mat A, PetscInt rowblock, PetscInt colblock, const MatType blockmattype, MatDDBlockCommType subcommtype, Mat_DDBlock *block) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  PetscInt              m,n,M,N;
  MPI_Comm              comm = ((PetscObject)A)->comm, subcomm;
  PetscMPIInt           commsize=0, commrank, subcommsize=0;
  PetscMPIInt           subcommcolor;
  
  PetscFunctionBegin;
  m = dd->lrowblockoffset[rowblock+1]-dd->lrowblockoffset[rowblock];
  n = dd->lcolblockoffset[colblock+1]-dd->lcolblockoffset[colblock];
  M = dd->growblockoffset[rowblock+1]-dd->growblockoffset[rowblock];
  N = dd->gcolblockoffset[colblock+1]-dd->gcolblockoffset[colblock];
  switch(subcommtype) {
  case MATDD_BLOCK_COMM_SELF:
    subcomm = PETSC_COMM_SELF;
    subcommsize = 1;
    break;
  case MATDD_BLOCK_COMM_DEFAULT:
    subcomm = comm;
    ierr = MPI_Comm_size(subcomm, &subcommsize); CHKERRQ(ierr);
    break;
  case MATDD_BLOCK_COMM_DETERMINE:
    /* determine and construct a subcomm that includes all of the procs with nonzero m and n */
    ierr = MPI_Comm_size(comm, &commsize); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &commrank); CHKERRQ(ierr);
    if(m || n) {
      subcommcolor = 1;
    }
    else {
      subcommcolor = MPI_UNDEFINED;
    }
    ierr = MPI_Comm_split(comm, subcommcolor, commrank, &subcomm); CHKERRQ(ierr);
    ierr = MPI_Comm_size(subcomm, &subcommsize);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER, "Unknown block comm type: %d", commsize);
    break;
  }/* switch(subcommtype) */
  if(subcomm != MPI_COMM_NULL) {
    ierr = MatCreate(subcomm, &(block->mat)); CHKERRQ(ierr);
    ierr = MatSetSizes(block->mat,m,n,M,N); CHKERRQ(ierr);
    /**/
    if(!blockmattype) {
      ierr = MatSetType(block->mat,dd->default_block_type); CHKERRQ(ierr);
    }
    else {
      ierr = MatSetType(block->mat,blockmattype); CHKERRQ(ierr);
    }
    if(subcommsize == 1) {
      ierr = VecCreateSeqWithArray(subcomm,n,PETSC_NULL,&(block->invec)); CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(subcomm,m,PETSC_NULL,&(block->outvec)); CHKERRQ(ierr);
    }
    else {
      ierr = VecCreateMPIWithArray(subcomm,n,N,PETSC_NULL,&(block->invec)); CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(subcomm,m,M,PETSC_NULL,&(block->outvec)); CHKERRQ(ierr);
    }
  }
  else {
    block->mat = PETSC_NULL;
    block->invec = PETSC_NULL;
    block->outvec = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}/* Mat_DDBlockInit() */


#undef  __FUNCT__
#define __FUNCT__ "Mat_DDBlockFinit"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_DDBlockFinit(Mat A, Mat_DDBlock *block) {
  PetscErrorCode        ierr;
  MPI_Comm              comm = ((PetscObject)A)->comm, subcomm = ((PetscObject)(block->mat))->comm;
  PetscMPIInt           flag;
  
  PetscFunctionBegin;
  
  ierr = MatDestroy(block->mat); CHKERRQ(ierr);
  ierr = VecDestroy(block->invec); CHKERRQ(ierr);
  ierr = VecDestroy(block->outvec); CHKERRQ(ierr);
  /**/
  ierr = MPI_Comm_compare(subcomm, comm, &flag); CHKERRQ(ierr);
  if(flag != MPI_IDENT) {
    ierr = MPI_Comm_compare(subcomm, PETSC_COMM_SELF, &flag); CHKERRQ(ierr);
    if(flag != MPI_IDENT) {
      ierr = MPI_Comm_free(&subcomm); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}/* Mat_DDBlockFinit() */



#undef  __FUNCT__
#define __FUNCT__ "MatDDSetScatters"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetScatters(Mat A, PetscInt blockcount, Mat scatters[]) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  PetscInt k,j0,n,N;
  PetscScalar *inarr;
  
  PetscFunctionBegin;
  /* check validity of block parameters */
  if(blockcount <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  if(dd->scatters) {
    for(k = 0; k < dd->colblockcount; ++k) {
      if(dd->scatters[k]) {
        ierr = MatDestroy(dd->scatters[k]); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->scatters); CHKERRQ(ierr);
  }
  ierr = PetscFree2(dd->lcolblockoffset, dd->gcolblockoffset); CHKERRQ(ierr);

  dd->colblockcount = blockcount;
  ierr = PetscMalloc(sizeof(Mat)*dd->colblockcount, &(dd->scatters)); CHKERRQ(ierr);
  ierr = PetscMalloc2(dd->colblockcount+1,PetscInt,&(dd->lcolblockoffset), dd->colblockcount+1,PetscInt,&(dd->gcolblockoffset)); CHKERRQ(ierr);    
  n = 0; N = 0;
  for(k = 0; k < dd->colblockcount; ++k) {
    dd->lcolblockoffset[k] = n;
    dd->gcolblockoffset[k] = N;
    dd->scatters[k] = scatters[k];
    if(!scatters[k]) continue;
    ierr = PetscObjectReference((PetscObject)scatters[k]); CHKERRQ(ierr);
    /* make sure the scatter column dimension matches that of MatDD */
    if(scatters[k]->cmap->N != A->cmap->N){
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Scatter # %d has global column dimension %d, which doesn't match MatDD's %d", k, scatters[k]->cmap->N, A->cmap->N);
    }
    if(scatters[k]->cmap->n != A->cmap->n) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Scatter # %d has local column dimension %d, which doesn't match MatDD's %d", k, scatters[k]->cmap->n, A->cmap->n);
    }
    n += scatters[k]->rmap->n;
    N += scatters[k]->rmap->N;
  }  
  dd->lcolblockoffset[dd->colblockcount] = n;
  dd->gcolblockoffset[dd->colblockcount] = N;
  /* Now create invec and invecs.  FIX: get rid of invecs as soon as scatters are merged into one. */
  ierr = VecCreateMPI(((PetscObject)A)->comm, n, N, &(dd->invec)); CHKERRQ(ierr);
  ierr = VecGetArray(dd->invec, &inarr); CHKERRQ(ierr);
  for(k = 0; k < dd->colblockcount; ++k) {
    /* FIX: get rid of invecs into a single Vec, as soon as scatters are merged */
    if(dd->invecs[k]) {
      ierr = VecResetArray(dd->invecs[k]);
      ierr = VecDestroy(dd->invecs[k]); CHKERRQ(ierr);
    }
    j0 = dd->lcolblockoffset[k];
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, dd->scatters[k]->rmap->n, dd->scatters[k]->rmap->N, inarr+j0, &(dd->invecs[k])); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(dd->invec, &inarr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
  
}/* MatDDSetScatters() */

#undef  __FUNCT__
#define __FUNCT__ "MatDDSetGathers"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetGathers(Mat A, PetscInt blockcount, Mat gathers[]) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  PetscInt              k,i0,m,M;
  PetscScalar           *outarr;
  
  PetscFunctionBegin;

  /* check validity of block parameters */
  if(blockcount <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  if(dd->gathers) {
    for(k = 0; k < dd->rowblockcount; ++k) {
      if(dd->gathers[k]) {
        ierr = MatDestroy(dd->gathers[k]); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->gathers); CHKERRQ(ierr);
  }
  ierr = PetscFree2(dd->lcolblockoffset, dd->gcolblockoffset); CHKERRQ(ierr);

  dd->rowblockcount = blockcount;
  ierr = PetscMalloc(sizeof(Mat)*dd->rowblockcount, &(dd->gathers)); CHKERRQ(ierr);
  ierr = PetscMalloc2(dd->rowblockcount+1,PetscInt,&(dd->lrowblockoffset), dd->rowblockcount+1,PetscInt,&(dd->growblockoffset)); CHKERRQ(ierr);    
  m = M = 0;
  for(k = 0; k < dd->rowblockcount; ++k) {
    dd->lrowblockoffset[k] = m;
    dd->growblockoffset[k] = M;
    dd->gathers[k] = gathers[k];
    if(!gathers[k]) continue;
    ierr = PetscObjectReference((PetscObject)gathers[k]); CHKERRQ(ierr);
    /* make sure the gather row dimension matches that of MatDD */
    if(gathers[k] && gathers[k]->rmap->N != A->rmap->N){
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Gather # %d has global row dimension %d, which doesn't match MatDD's %d", k, gathers[k]->rmap->N, A->rmap->N);
    }
    if(gathers[k] && gathers[k]->rmap->n != A->rmap->n) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Gather # %d has local row dimension %d, which doesn't match MatDD's %d", k, gathers[k]->rmap->n, A->rmap->n);
    }
    m += gathers[k]->cmap->n;
    M += gathers[k]->cmap->N;
  }  
  dd->lrowblockoffset[dd->rowblockcount] = m;
  dd->growblockoffset[dd->rowblockcount] = M;
  /* Now create outvec and outvecs.  FIX: get rid of outvecs as soon as gathers are merged into one. */
  ierr = VecCreateMPI(((PetscObject)A)->comm, m,M, &(dd->outvec)); CHKERRQ(ierr);
  ierr = VecGetArray(dd->outvec, &outarr); CHKERRQ(ierr);
  for(k = 0; k < dd->rowblockcount; ++k) {
    /* FIX: get rid of outvecs into a single Vec, as soon as scatters are merged */
    if(dd->outvecs[k]) {
      ierr = VecResetArray(dd->outvecs[k]);
      ierr = VecDestroy(dd->outvecs[k]); CHKERRQ(ierr);
    }
    i0 = dd->lrowblockoffset[k];
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, dd->gathers[k]->cmap->n, dd->gathers[k]->cmap->N, outarr+i0, &(dd->outvecs[k])); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(dd->outvec, &outarr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}/* MatDDSetGathers() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDSetDefaultBlockType"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetDefaltBlockType(Mat A, const MatType type) 
{
  Mat_DD  *dd = (Mat_DD*)A->data;
  PetscFunctionBegin;
  if (!type) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Unknown default block type");
  dd->default_block_type = type;
  PetscFunctionReturn(0);
}/* MatDDSetDefaultBlockType()*/

#undef  __FUNCT__
#define __FUNCT__ "MatDDGetDefaultBlockType"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDGetDefaltBlockType(Mat A, const MatType *type) 
{
  Mat_DD  *dd = (Mat_DD*)A->data;
  PetscFunctionBegin;
  *type = dd->default_block_type;
  PetscFunctionReturn(0);
}/* MatDDSetDefaultBlockType()*/


#undef __FUNCT__  
#define __FUNCT__ "MatDDSetPreallocation_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetPreallocation_AIJ(Mat A,PetscInt nz, PetscInt *nnz)
{
  /* Assume gather and scatter have been set */
  Mat_DD        *dd = (Mat_DD*)A->data;
  Mat_DDAIJ     *aij = (Mat_DDAIJ*)dd->data;
  PetscBool      skipallocation = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<dd->rowblockcount; ++i) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: block %d value %d",i,nnz[i]);
      if (nnz[i] > dd->colblockcount) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than the number of column blocks: block %d value %d column block count %d",i,nnz[i],dd->colblockcount);
    }
  }

  A->preallocated = PETSC_TRUE;

  if (!skipallocation) {
    if (!aij->imax) {
      ierr = PetscMalloc2(dd->rowblockcount,PetscInt,&aij->imax,dd->rowblockcount,PetscInt,&aij->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(A,2*dd->rowblockcount*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
      else if (nz <= 0)        nz = 1;
      for (i=0; i<dd->rowblockcount; ++i) aij->imax[i] = nz;
      nz = nz*dd->rowblockcount;
    } else {
      nz = 0;
      for (i=0; i<dd->rowblockcount; ++i) {aij->imax[i] = nnz[i]; nz += nnz[i];}
    }
    /* aij->ilen will count nonzeros in each row so far. */
    for (i=0; i<dd->rowblockcount; i++) { aij->ilen[i] = 0; }

    /* allocate the block space */
    ierr = MatDDXAIJFreeAIJ(A,&aij->a,&aij->j,&aij->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(nz,Mat_DDBlock,&aij->a,nz,PetscInt,&aij->j,dd->rowblockcount+1,PetscInt,&aij->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,(dd->rowblockcount+1)*sizeof(PetscInt)+nz*(sizeof(Mat_DDBlock)+sizeof(PetscInt)));CHKERRQ(ierr);
    aij->i[0] = 0;
    for (i=1; i<dd->rowblockcount+1; ++i) {
      aij->i[i] = aij->i[i-1] + aij->imax[i-1];
    }
    aij->singlemalloc = PETSC_TRUE;
    aij->free_a       = PETSC_TRUE;
    aij->free_ij      = PETSC_TRUE;
  } else {
    aij->free_a       = PETSC_FALSE;
    aij->free_ij      = PETSC_FALSE;
  }

  aij->nz                = 0;
  aij->maxnz             = nz;
  A->info.nz_unneeded  = (double)aij->maxnz;
  PetscFunctionReturn(0);
}/* MatDDSetPreallocation_AIJ() */


#undef __FUNCT__  
#define __FUNCT__ "MatDDAIJSetPreallocation"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDAIJSetPreallocation(Mat A,PetscInt nz,PetscInt *nnz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDDSetPreallocation_AIJ(A,nz,nnz); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDAIJSetPreallocation() */

#undef __FUNCT__  
#define __FUNCT__ "MatDDSetUpPreallocation_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetUpPreallocation_AIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDDSetPreallocation_AIJ(A,PETSC_DEFAULT,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDAIJSetUpPreallocation_AIJ() */

#undef __FUNCT__  
#define __FUNCT__ "MatDDSetUpPreallocation"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetUpPreallocation(Mat A) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = MatDDSetUpPreallocation_AIJ(A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDSetUpPreallocation() */

#undef  __FUNCT__
#define __FUNCT__ "MatDDLocateBlock_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDLocateBlock_AIJ(Mat M, PetscInt row, PetscInt col, PetscBool  insert, Mat_DDBlock **block_pp) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)M->data;
  Mat_DDAIJ*           a = (Mat_DDAIJ*)dd->data;
  PetscInt       *rp,low,high,t,ii,nrow,i,rmax,N;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  Mat_DDBlock   *ap,*aa = a->a;
  PetscFunctionBegin;

  *block_pp = PETSC_NULL;
  if (row < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (row >= dd->rowblockcount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block row too large: row %D max %D",row,dd->rowblockcount-1);
#endif
  rp   = aj + ai[row]; ap = aa + ai[row];
  rmax = imax[row]; nrow = ailen[row]; 
  low  = 0;
  high = nrow;
  
  if (col < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (col >= dd->colblockcount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block columnt too large: col %D max %D",col,dd->colblockcount-1);
#endif

  if (col <= lastcol) low = 0; else high = nrow;
  lastcol = col;
  while (high-low > 5) {
    t = (low+high)/2;
    if (rp[t] > col) high = t;
    else             low  = t;
  }
  for (i=low; i<high; i++) {
    if (rp[i] > col) break;
    if (rp[i] == col) {
      *block_pp = ap+i;  
      goto we_are_done;
    }
  } 
  if (!insert || nonew == 1) goto we_are_done;
  if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block at (%D,%D) in the matrix",row,col);
  MatDDXAIJReallocateAIJ(M,dd->rowblockcount,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew);
  N = nrow++ - 1; a->nz++; high++;
  /* shift up all the later entries in this row */
  for (ii=N; ii>=i; ii--) {
    rp[ii+1] = rp[ii];
    ap[ii+1] = ap[ii];
  }
  rp[i] = col; 
  *block_pp = ap+i; 
  low   = i + 1;
  ailen[row] = nrow;
  M->same_nonzero = PETSC_FALSE;

  if (M->assembled) {
    M->was_assembled = PETSC_TRUE; 
    M->assembled     = PETSC_FALSE;
  }
 we_are_done:;
  PetscFunctionReturn(0);
}/* MatDDLocateBlock_AIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDAddBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDAddBlock(Mat A, PetscInt rowblock, PetscInt colblock, const MatType blockmattype, MatDDBlockCommType blockcommtype, Mat *_BBB) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  Mat_DDBlock          *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(rowblock < 0 || rowblock > dd->rowblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, dd->rowblockcount);
  }
  if(colblock < 0 || colblock > dd->colblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, dd->colblockcount);
  }
  ierr = MatDDLocateBlock_AIJ(A,rowblock, colblock, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = Mat_DDBlockInit(A,rowblock,colblock,blockmattype, blockcommtype, _block); CHKERRQ(ierr);
  if(_BBB) {
    ierr = Mat_DDBlockGetMat(A, rowblock, colblock, _block, _BBB); CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)(*_BBB)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatDDAddBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDAddBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDSetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat B) {
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)A->data;
  Mat_DDBlock          *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(rowblock < 0 || rowblock > dd->rowblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, dd->rowblockcount);
  }
  if(colblock < 0 || colblock > dd->colblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, dd->colblockcount);
  }
  ierr = MatDDLocateBlock_AIJ(A,rowblock, colblock, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = Mat_DDBlockSetMat(A,rowblock,colblock,_block, B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(B)); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}/* MatDDSetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDGetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatDDGetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat *_BBB) {
  PetscErrorCode     ierr;
  Mat_DD*           dd = (Mat_DD*)A->data;
  Mat_DDBlock       *_block;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > dd->rowblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, dd->rowblockcount);
  }
  if(colblock < 0 || colblock > dd->colblockcount){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, dd->colblockcount);
  }
  ierr = MatDDLocateBlock_AIJ(A, rowblock, colblock, PETSC_FALSE, &_block); CHKERRQ(ierr);
  if(_block == PETSC_NULL) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "Block not found: row %d col %d", rowblock, colblock);
  }
  ierr = Mat_DDBlockGetMat(A,rowblock,colblock,_block, _BBB); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(*_BBB)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatDDGetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatMult_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_DDAIJ(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  Mat_DDAIJ *aij = (Mat_DDAIJ*)dd->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  Mat_DDBlock *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<dd->rowblockcount; ++i) {
    yoff = dd->lrowblockoffset[i];
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      xoff = dd->lcolblockoffset[j];
      ierr = VecPlaceArray(b->outvec,yarr+yoff); CHKERRQ(ierr);
      ierr = VecPlaceArray(b->invec,xarr+xoff); CHKERRQ(ierr);
      ierr = MatMultAdd(b->mat, b->invec, b->outvec, b->outvec); CHKERRQ(ierr);
      ierr = VecResetArray(b->invec); CHKERRQ(ierr);
      ierr = VecResetArray(b->outvec); CHKERRQ(ierr);
    }
  } 
  ierr = VecRestoreArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarr); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}// MatMult_DDAIJ()


#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_DDAIJ(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  Mat_DDAIJ *aij = (Mat_DDAIJ*)dd->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  Mat_DDBlock *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<dd->rowblockcount; ++i) {
    xoff = dd->lrowblockoffset[i];
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      yoff = dd->lcolblockoffset[j];
      ierr = VecPlaceArray(b->outvec,xarr+xoff); CHKERRQ(ierr);
      ierr = VecPlaceArray(b->invec,yarr+yoff); CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(b->mat, b->outvec, b->invec, b->invec); CHKERRQ(ierr);
      ierr = VecResetArray(b->invec); CHKERRQ(ierr);
      ierr = VecResetArray(b->outvec); CHKERRQ(ierr);
    }
  } 
  ierr = VecRestoreArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarr); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}// MatMultTranspose_DDAIJ()


#undef  __FUNCT__
#define __FUNCT__ "MatMult_DD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_DD(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  Vec xx, yy;
  PetscInt i,j;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Scatter */
  /* FIX: merge scatters into a single scatter */
  if(dd->scatters) {
    for(j = 0; j < dd->colblockcount; ++j) {
      ierr = MatMult(dd->scatters[j],x,dd->invecs[j]); CHKERRQ(ierr); 
    }
    xx = dd->invec;
  }
  else {
    xx = x;
  }
  if(dd->gathers) {
    yy = dd->outvec; 
  }
  else {
    yy = y;
  }
  ierr = MatMult_DDAIJ(A, xx, yy); CHKERRQ(ierr);
  /* Gather */
  if(dd->gathers) {
    for(i = 0; i < dd->rowblockcount; ++i) {
      ierr = MatMultAdd(dd->gathers[i],dd->outvecs[i],y,y); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatMult_DD()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTransporse_DD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_DD(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  PetscInt i,j;
  Vec xx,yy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Gather^T */
  if(dd->gathers) {
    for(i = 0; i < dd->rowblockcount; ++i) {
      ierr = MatMultTranspose(dd->gathers[i],x,dd->outvecs[i]); CHKERRQ(ierr); 
    }
    xx = dd->outvec;
  }
  else {
    xx = x;
  }
  if(dd->scatters) {
    yy = dd->invec;
  }
  else {
    yy = y;
  }
  ierr = MatMultTranspose_DDAIJ(A, xx, yy); CHKERRQ(ierr);
  /* Scatter^T */
  if(dd->scatters) {
    for(j = 0; j < dd->colblockcount; ++j) {
      ierr = MatMultTransposeAdd(dd->scatters[j],dd->invecs[j],y,y); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatMultTranspose_DD()

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyBegin_DDAIJ(Mat A, MatAssemblyType type) 
{
  Mat_DD         *dd = (Mat_DD*)A->data;
  Mat_DDAIJ      *aij = (Mat_DDAIJ*)dd->data;
  PetscInt       i,j,k;
  Mat            B;
  Mat_DDBlock    *ap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(i = 0; i < dd->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = Mat_DDBlockGetMat(A, i, j, ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyBegin(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatAssemblyBegin_DDAIJ()

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyEnd_DDAIJ(Mat A, MatAssemblyType type) 
{
  Mat_DD         *dd = (Mat_DD*)A->data;
  Mat_DDAIJ      *aij = (Mat_DDAIJ*)dd->data;
  PetscInt       i,j,k;
  Mat            B;
  Mat_DDBlock    *ap;
  PetscErrorCode ierr; 

  PetscFunctionBegin;
  for(i = 0; i < dd->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = Mat_DDBlockGetMat(A,i,j,ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatAssemblyEnd_DDAIJ()



#undef  __FUNCT__
#define __FUNCT__ "MatCreate_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_DDAIJ(Mat A) {
  Mat_DD     *dd = (Mat_DD *)A->data;
  Mat_DDAIJ  *aij;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,Mat_DDAIJ,&aij);CHKERRQ(ierr);
  dd->data             = (void*)aij;

  aij->reallocs         = 0;
  aij->nonew            = 0;
  
  A->ops->assemblybegin = MatAssemblyBegin_DDAIJ;
  A->ops->assemblyend   = MatAssemblyEnd_DDAIJ;
  
  PetscFunctionReturn(0);
}/* MatCreate_DDAIJ() */

#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_DDAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_DDAIJ(Mat M) {
  Mat_DD     *dd = (Mat_DD *)M->data;
  Mat_DDAIJ  *aij = (Mat_DDAIJ *)dd->data;
  PetscInt    *ai = aij->i, *ailen = aij->ilen;
  Mat_DDBlock *aa = aij->a, *bp;
  PetscInt    i,n,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy blocks */
  for(i = 0; i < dd->rowblockcount; ++i) {
    n = ailen[i];
    bp = aa + ai[i];
    for(j = 0; j < n; ++j) {
      ierr = Mat_DDBlockFinit(M, bp); CHKERRQ(ierr);
      ++bp;
    }
  }
  /* Dealloc the AIJ structures */
  ierr = MatDDXAIJFreeAIJ(M, &(aij->a), &(aij->j), &(aij->i)); CHKERRQ(ierr);
  ierr = PetscFree2(aij->imax,aij->ilen);CHKERRQ(ierr);
  ierr = PetscFree(aij);CHKERRQ(ierr);
  dd->data = 0;
  
  PetscFunctionReturn(0);
}/* MatDestroy_DDAIJ() */

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_DD"
PetscErrorCode MatDestroy_DD(Mat mat)
{
  PetscErrorCode        ierr;
  Mat_DD*              dd = (Mat_DD*)mat->data;
  PetscInt              k;

  PetscFunctionBegin;
  ierr = MatDestroy_DDAIJ(mat); CHKERRQ(ierr);

  ierr = PetscFree2(dd->lcolblockoffset, dd->gcolblockoffset);         CHKERRQ(ierr);
  ierr = PetscFree2(dd->lrowblockoffset, dd->growblockoffset);         CHKERRQ(ierr);

  
  if(dd->invecs) {
    for(k = 0; k < dd->colblockcount; ++k) {
      if(dd->invecs[k]) {
        ierr = VecResetArray(dd->invecs[k]); CHKERRQ(ierr);
        ierr = VecDestroy(dd->invecs[k]);    CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->invecs); CHKERRQ(ierr);
  }
  ierr = VecDestroy(dd->invec); CHKERRQ(ierr);
  if(dd->outvecs) {
    for(k = 0; k < dd->rowblockcount; ++k) {
      if(dd->outvecs[k]) {
        ierr = VecResetArray(dd->outvecs[k]); CHKERRQ(ierr);
        ierr = VecDestroy(dd->outvecs[k]);    CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->outvecs); CHKERRQ(ierr);
  }
  ierr = VecDestroy(dd->outvec); CHKERRQ(ierr);
  if(dd->scatters) {
    for(k = 0; k < dd->colblockcount; ++k) {
      if(dd->scatters[k]) {
        ierr = MatDestroy(dd->scatters[k]);                          CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->scatters);                                  CHKERRQ(ierr);
  }
  if(dd->gathers) {
    for(k = 0; k < dd->rowblockcount; ++k) {
      if(dd->gathers[k]) {
        ierr = MatDestroy(dd->gathers[k]);                          CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(dd->gathers);                                  CHKERRQ(ierr);
  }

  ierr = PetscFree(dd);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}/* MatDestroy_DD() */


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "MatCreate_DD"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_DD(Mat A) {
  /* Assume that this is called after MatSetSizes() */
  Mat_DD  *dd;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->ops->setuppreallocation = MatDDSetUpPreallocation;
  A->ops->mult          = MatMult_DD;
  A->ops->multtranspose = MatMultTranspose_DD;
  A->ops->destroy       = MatDestroy_DD;

  A->assembled    = PETSC_FALSE;
  A->same_nonzero = PETSC_FALSE;

  ierr = PetscNewLog(A,Mat_DD,&dd);CHKERRQ(ierr);
  A->data = (void*)dd;
  
  /* FIX: gathers and scatters need to be merged */
  dd->default_block_type = MATAIJ;
  dd->scatters = PETSC_NULL;
  dd->gathers  = PETSC_NULL;
  dd->rowblockcount = 1; 
  dd->colblockcount = 1; 
  ierr = PetscMalloc2(dd->rowblockcount+1, PetscInt, &(dd->lrowblockoffset), dd->rowblockcount+1, PetscInt, &(dd->growblockoffset)); CHKERRQ(ierr);
  ierr = PetscMalloc2(dd->colblockcount+1, PetscInt, &(dd->lcolblockoffset), dd->colblockcount+1, PetscInt, &(dd->gcolblockoffset)); CHKERRQ(ierr);
  dd->lrowblockoffset[0] = 0; dd->lrowblockoffset[1] = A->rmap->n;
  dd->growblockoffset[0] = 0; dd->growblockoffset[1] = A->rmap->N;
  dd->lcolblockoffset[0] = 0; dd->lcolblockoffset[1] = A->cmap->n;
  dd->gcolblockoffset[0] = 0; dd->gcolblockoffset[1] = A->cmap->N;

  /* FIX: get rid of invecs and outvecs as soon as gathers and scatters are merged */
  dd->invec = PETSC_NULL;
  dd->invecs = PETSC_NULL;
  dd->outvec = PETSC_NULL;
  dd->outvecs = PETSC_NULL;

  ierr = MatCreate_DDAIJ(A); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATDD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatCreate_DD() */
EXTERN_C_END
