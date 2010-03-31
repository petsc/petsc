#define PETSCMAT_DLL

#include "../src/mat/impls/fwk/matfwk.h"          /*I "petscmat.h" I*/



/*
  MatFwk represents global matrix M as composed of local submatrices or *cells* M_{ij}.
  The cells assembled using scatters S_j and gather G_i:
           M = \sum_{i,j=0}^{N-1} S_i M_{ij} S_j.
  where each submatrix M_{ij} acts between the j-th block of the input Vec
  and the i-th block of the output Vec.  Therefore, M also defines 
  a splitting of the Vec space(s) it acts on (between) into nonoverlapping
  dof blocks of arbitrary sizes.  In 

  MatFwk can then applied to Vecs using the submatrix MatMult actions.
  Furthermore, 

  The user creates a new MatFwk as any (MPI) Mat, by specifying the communicator,
  the local and global sizes:
     

  Storage scheme underlying a MatFwk can derive from some existing Mat types: 
  MAT(B)AIJ, MATDENSE, MATSCATTER.  In this case setting block values 
  with MatFwkBlockSetValues is a logical extension of MatSetValuesLocal, 
  which contextualizes the row and column indices of the inserted matrix element
  using a local-to-global map.  Likewise, MatFwk reinterpretes the indices 
  of the inserted block elements as being relative to the block.  
  The resulting MatFwk types will be MATFWK(B)AIJ, etc. They will merely apply
  in underlying MAT(B)AIJ object to a Vec, etc.

  Preallocation can be done by declaring which nonzero blocks are present with
  MatFwkAddBlock(fwk, i,j).  This is analogous to declaring a nonzero element in an
  AIJ matrix, although such preallocation makes only sense at the block 
  granularity.  It would probably make even more sense to add all blocks at once
  with some block version of preallocation.  This, of course, would make little
  sense for MATFWKDENSE and similar formats.  In absence of preallocation for
  the (i,j)-th block, MatFwkBlockSetValues on that block should be expected to
  be expensive.

  A different approach is to store blocks as Mat objects, which can be set
  with MatFwkBlockSet(fwk,i,j,block), MatFwkBlockGet(fwk,i,j,&block).
  Similarly, a previously non-Added (i,j)-th block will be expensive to Set.
  Blocks can be applied to pieces of an input Vec (and the results output to 
  the corresponding pieces of the output Vec) by wrapping the relevant array
  chunks with VecPlaceArray.  The wrappers can, naturally, be reused.
  The resulting MatFwk type is MATFWKMAT.

  A still more general approach is to allow user-defined applications of
  a matrix block to a vector block.  In other words, the user has to provide
  an implementation for MatFwk_ApplyBlock(fwk, i,j,arrin,arrout),
  which operates on input and output arrays of incoming and outgoing dofs
  extracted from the corresponding Vecs.  The idea is to have an API for
  allowing the user to set such a function: MatFwkSetBlockApplication.
  [It would be nice to be able to launch this on a GPU as a kernel, but CUDA
  does not allow for function pointers, does it?]
  This approach is a generalization of MatShell as well as of DA's FormFunctionLocal.  
  The resulting MatFwk type is MATFWKSHELL.  


  Up to now it has been described how MatFwk combines blocks additively,
  but it can also be done multiplicatively: pairs of MatFwk objects can be 
  multiplied (MatMatMult: M = A*B), if the input/output Vec decompositions 
  into blocks are compatible: each input block of A is an output block of B.
  [More generally, we could demand some kind of containment, but this isn't 
  done yet in the interest of simplicity.]
  If the underlying storage formats (MAT(B)AIJ, etc) admit a MatMatMult, they
  will be multiplied internally, and the input/output blocks of the product
  will bet set appropriately.  Otherwise the product will become a MATFWKSHELL,
  which will use the factors' MatFwk_ApplyBlock to define its own 
  MatFwk_ApplyBlock [perhaps we should also demand MatFwk_ApplyRow?].

  With the multiplicative combination capability we can, for example, 
  define disassembled stiffness matrices: K = S^T*L*S, where S is the 
  scatter from a Vec into element blocks, L is the elementwise stiffness
  matrix and S^T acts as a gather.  If MatMatMult is defined for the types
  of S and L, then K will actually be assembled.

  We start out by implementing MATFWKSHELL and MATFWKMAT, since these are 
  the most general and the simplest (offloading the complexity elsewhere:
  onto the user or by carrying around a lot of Mat objects).
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


#define CHUNKSIZE 15



#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetScatter"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetScatter(Mat A, Mat scatter, PetscInt blockcount, PetscInt *blocksizes) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt i,sum;
  
  PetscFunctionBegin;

  /* make sure the scatter column dimension matches that of MatFwk */
  if(scatter->cmap->N != A->cmap->N){
    SETERRQ2(PETSC_ERR_USER, "Scatter's global columng dimension %d doesn't match MatFwk's %d", scatter->cmap->N, A->cmap->N);
  }
  if(scatter->cmap->n != A->cmap->n) {
    SETERRQ2(PETSC_ERR_USER, "Scatter's local column dimension %d doesn't match MatFwk's %d", scatter->cmap->n, A->cmap->n);
  }
  /* check validity of block parameters */
  if(blockcount <= 0) {
    SETERRQ1(PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  }
  /* make sure block sizes are nonnegative */
  for(i = 0; i < blockcount; ++i) {
    if(blocksizes[i] < 0) {
      SETERRQ2(PETSC_ERR_USER, "Block %d has negative size %d; must be >= 0", i,blocksizes[i]);
    }
  }
  /* set validated block parameters and calculate the expanded dimension */
  fwk->colblockcount = blockcount;
  ierr = PetscFree(fwk->colblockoffset); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(blockcount+1), &fwk->colblockoffset); CHKERRQ(ierr);
  for(sum=0,i = 0; i < blockcount; ++i) {
    fwk->colblockoffset[i] = sum;
    sum += blocksizes[i];
  }
  /* make sure block sizes add up to the local scatter row count */
  if(sum != scatter->rmap->n) {
    SETERRQ2(PETSC_ERR_USER, "Local block sizes add up to %d, not the same as the number of local scatter rows %d", sum, scatter->rmap->n);
  }
  fwk->colblockoffset[blockcount] = sum;
  fwk->en = sum;
  fwk->scatter = scatter;
  ierr = PetscObjectReference((PetscObject)(fwk->scatter)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}/* MatFwkSetScatter() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetGather"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetGather(Mat A, Mat gather, PetscInt blockcount, PetscInt *blocksizes) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt i,sum;
  
  PetscFunctionBegin;
  /* make sure the gather row dimension matches that of MatFwk */
  if(gather->rmap->N != A->rmap->N){
    SETERRQ2(PETSC_ERR_USER, "Gather's global columng dimension %d doesn't match MatFwk's %d", gather->rmap->N, A->rmap->N);
  }
  if(gather->rmap->n != A->rmap->n) {
    SETERRQ2(PETSC_ERR_USER, "Gather's local column dimension %d doesn't match MatFwk's %d", gather->rmap->n, A->rmap->n);
  }

  /* check validity of block parameters */
  if(blockcount <= 0) {
    SETERRQ1(PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  }
  /* make sure block sizes are nonnegative */
  for(i = 0; i < blockcount; ++i) {
    if(blocksizes[i] < 0) {
      SETERRQ2(PETSC_ERR_USER, "Block %d has negative size %d; must be >= 0", i,blocksizes[i]);
    }
  }
  /* set validated block parameters and calculate the expanded dimension */
  fwk->rowblockcount = blockcount;
  ierr = PetscFree(fwk->rowblockoffset); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(blockcount+1), &fwk->rowblockoffset); CHKERRQ(ierr);
  for(sum=0,i = 0; i < blockcount; ++i) {
    fwk->rowblockoffset[i] = sum;
    sum += blocksizes[i];
  }
  if(sum != gather->cmap->n) {
    SETERRQ2(PETSC_ERR_USER, "Local block sizes add up to %d, not the same as the number of local gather columns %d", sum, gather->cmap->n);
  }
  fwk->rowblockoffset[blockcount] = sum;
  fwk->em = sum;
  fwk->gather = gather;
  ierr = PetscObjectReference((PetscObject)(fwk->gather)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}/* MatFwkSetGather() */




#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetDefaultBlockType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetDefaltBlockType(Mat A, const MatType type) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  PetscFunctionBegin;
  if(!type){
    SETERRQ(PETSC_ERR_USER, "Unknown default block type");
  }
  fwk->default_block_type = type;
  PetscFunctionReturn(0);
}/* MatFwkSetDefaultBlockType()*/

#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetDefaultBlockType"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetDefaltBlockType(Mat A, const MatType *type) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  PetscFunctionBegin;
  *type = fwk->default_block_type;
  PetscFunctionReturn(0);
}/* MatFwkSetDefaultBlockType()*/


#undef __FUNCT__  
#define __FUNCT__ "MatFwkSetPreallocation_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetPreallocation_AIJ(Mat A,PetscInt nz, PetscInt *nnz)
{
  /* Assume gather and scatter have been set */
  Mat_Fwk        *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ     *aij = (Mat_FwkAIJ*)fwk->data;
  PetscTruth     skipallocation = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<fwk->rowblockcount; ++i) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local block %d value %d",i,nnz[i]);
      if (nnz[i] > fwk->colblockcount) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than the number of local column blocks: local block %d value %d column block count %d",i,nnz[i],fwk->colblockcount);
    }
  }

  A->preallocated = PETSC_TRUE;

  if (!skipallocation) {
    if (!aij->imax) {
      ierr = PetscMalloc2(fwk->rowblockcount,PetscInt,&aij->imax,fwk->rowblockcount,PetscInt,&aij->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(A,2*fwk->rowblockcount*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
      else if (nz <= 0)        nz = 1;
      for (i=0; i<fwk->rowblockcount; ++i) aij->imax[i] = nz;
      nz = nz*fwk->rowblockcount;
    } else {
      nz = 0;
      for (i=0; i<fwk->rowblockcount; ++i) {aij->imax[i] = nnz[i]; nz += nnz[i];}
    }
    /* aij->ilen will count nonzeros in each row so far. */
    for (i=0; i<fwk->rowblockcount; i++) { aij->ilen[i] = 0; }

    /* allocate the block space */
    ierr = MatFwkXAIJFreeAIJ(A,&aij->a,&aij->j,&aij->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(nz,Mat_FwkBlock,&aij->a,nz,PetscInt,&aij->j,fwk->rowblockcount+1,PetscInt,&aij->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,(fwk->rowblockcount+1)*sizeof(PetscInt)+nz*(sizeof(Mat_FwkBlock)+sizeof(PetscInt)));CHKERRQ(ierr);
    aij->i[0] = 0;
    for (i=1; i<fwk->rowblockcount+1; ++i) {
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
}/* MatFwkSetPreallocation_AIJ() */


#undef __FUNCT__  
#define __FUNCT__ "MatFwkAIJSetPreallocation"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkAIJSetPreallocation(Mat A,PetscInt nz,PetscInt *nnz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatFwkSetPreallocation_AIJ(A,nz,nnz); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkAIJSetPreallocation() */

#undef __FUNCT__  
#define __FUNCT__ "MatFwkSetUpPreallocation_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetUpPreallocation_AIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatFwkSetPreallocation_AIJ(A,PETSC_DEFAULT,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkAIJSetUpPreallocation_AIJ() */

#undef __FUNCT__  
#define __FUNCT__ "MatFwkSetUpPreallocation"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetUpPreallocation(Mat A) 
{
  Mat_Fwk* fwk = (Mat_Fwk*)A->data;
  MPI_Comm comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject)A, &comm); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, fwk->en, PETSC_DECIDE, &(fwk->invec)); CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, fwk->em, PETSC_DECIDE, &(fwk->outvec)); CHKERRQ(ierr);
  ierr = MatFwkSetUpPreallocation_AIJ(A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkSetUpPreallocation() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkLocateBlock_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkLocateBlock_AIJ(Mat M, PetscInt row, PetscInt col, PetscTruth noinsert, Mat_FwkBlock **B) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)M->data;
  Mat_FwkAIJ*           a = (Mat_FwkAIJ*)fwk->data;
  PetscInt       *rp,low,high,t,ii,nrow,i,rmax,N;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  Mat_FwkBlock   *ap,*aa = a->a;
  PetscFunctionBegin;

  *B = PETSC_NULL;
  if (row < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (row >= fwk->rowblockcount) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Block row too large: row %D max %D",row,fwk->rowblockcount-1);
#endif
  rp   = aj + ai[row]; ap = aa + ai[row];
  rmax = imax[row]; nrow = ailen[row]; 
  low  = 0;
  high = nrow;
  
  if (col < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (col >= fwk->colblockcount) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Block columnt too large: col %D max %D",col,fwk->colblockcount-1);
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
      *B = ap+i;  
      goto we_are_done;
    }
  } 
  if (noinsert || nonew == 1) goto we_are_done;
  if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block at (%D,%D) in the matrix",row,col);
  MatFwkXAIJReallocateAIJ(M,fwk->rowblockcount,1,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,Mat_FwkBlock);
  N = nrow++ - 1; a->nz++; high++;
  /* shift up all the later entries in this row */
  for (ii=N; ii>=i; ii--) {
    rp[ii+1] = rp[ii];
    ap[ii+1] = ap[ii];
  }
  rp[i] = col; 
  *B = ap+i; 
  low   = i + 1;
  ailen[row] = nrow;
  M->same_nonzero = PETSC_FALSE;

  if (M->assembled) {
    M->was_assembled = PETSC_TRUE; 
    M->assembled     = PETSC_FALSE;
  }
 we_are_done:;
  PetscFunctionReturn(0);
}/* MatFwkLocateBlock_AIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatFwkAddBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkAddBlock(Mat A, PetscInt rowblock, PetscInt colblock, const MatType type, Mat *Bout) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  Mat_FwkBlock          *_block;
  Mat                   B;
  PetscInt              m,n;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  /**/
  m = fwk->rowblockoffset[rowblock+1]-fwk->rowblockoffset[rowblock];
  n = fwk->colblockoffset[colblock+1]-fwk->colblockoffset[colblock];
  ierr = MatCreate(PETSC_COMM_SELF, &B); CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n); CHKERRQ(ierr);
  /**/
  if(!type) {
    ierr = MatSetType(B,fwk->default_block_type); CHKERRQ(ierr);
  }
  else {
    ierr = MatSetType(B,type); CHKERRQ(ierr);
  }
  ierr = MatFwkLocateBlock_AIJ(A,rowblock, colblock, PETSC_FALSE, &_block); CHKERRQ(ierr);
  ierr = Mat_FwkBlock_SetMat(_block,B); CHKERRQ(ierr);
  
  ierr = PetscObjectReference((PetscObject)B); CHKERRQ(ierr);
  *Bout = B;
  PetscFunctionReturn(0);
}/* MatFwkAddBlock() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat *B) {
  PetscErrorCode     ierr;
  Mat_Fwk*           fwk = (Mat_Fwk*)A->data;
  Mat_FwkBlock       *_block;
  
  PetscFunctionBegin;
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  ierr = MatFwkLocateBlock_AIJ(A, rowblock, colblock, PETSC_TRUE, &_block); CHKERRQ(ierr);
  if(_block == PETSC_NULL) {
    SETERRQ2(PETSC_ERR_USER, "Block not found: row %d col %d", rowblock, colblock);
  }
  ierr = Mat_FwkBlock_GetMat(_block, B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(*B)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkGetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatMult_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_FwkAIJ(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ *aij = (Mat_FwkAIJ*)fwk->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  Mat_FwkBlock *aa, *b;
  Mat B;
  Vec xx,yy;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  x  = fwk->invec;
  y  = fwk->outvec;
  xx = fwk->binvec;
  yy = fwk->boutvec;
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr);
  for (i=0; i<fwk->rowblockcount; ++i) {
    /* Set up the work vector corresponding to i-th out block */
    /* 
       HACK: we directly manipulate internal Vec structures to change the Vec size 
       and place the correct array chunk there 
    */
    MatFwk_SetUpBlockVec(fwk->rowblockoffset,yy,yarr,i);
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
    /* Set up the work vector corresponding to j-th in block */
    /* 
       HACK: we directly manipulate internal Vec structures to change the Vec size 
       and place the correct array chunk there 
    */
      MatFwk_SetUpBlockVec(fwk->colblockoffset,xx,xarr,j);
      ierr = Mat_FwkBlock_GetMat(b,&B); CHKERRQ(ierr);
      ierr = MatMultAdd(B,xx,yy,yy); CHKERRQ(ierr);
    }
  } 
  PetscFunctionReturn(0);
}// MatMult_FwkAIJ()


#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_FwkAIJ(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ *aij = (Mat_FwkAIJ*)fwk->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  Mat_FwkBlock *aa, *b;
  Mat B;
  Vec xx,yy;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  x  = fwk->invec;
  y  = fwk->outvec;
  xx = fwk->binvec;
  yy = fwk->boutvec;
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr);
  for (i=0; i<fwk->rowblockcount; ++i) {
    /* Set up the work vector corresponding to i-th out block */
    /* 
       HACK: we directly manipulate internal Vec structures to change the Vec size 
       and place the correct array chunk there 
    */
    MatFwk_SetUpBlockVec(fwk->rowblockoffset,yy,yarr,i);
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
    /* Set up the work vector corresponding to j-th in block */
    /* 
       HACK: we directly manipulate internal Vec structures to change the Vec size 
       and place the correct array chunk there 
    */
      MatFwk_SetUpBlockVec(fwk->colblockoffset,xx,xarr,j);
      ierr = Mat_FwkBlock_GetMat(b,&B); CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(B,yy,xx,xx); CHKERRQ(ierr);
    }
  } 
  PetscFunctionReturn(0);
}// MatMultTranspose_FwkAIJ()




#undef  __FUNCT__
#define __FUNCT__ "MatMult_Fwk"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_Fwk(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Scatter */
  ierr = MatMult(fwk->scatter,x,fwk->invec); CHKERRQ(ierr); 
  ierr = MatMult_FwkAIJ(A, fwk->invec, fwk->outvec); CHKERRQ(ierr);
  /* Gather */
  ierr = MatMultAdd(fwk->gather,fwk->outvec,y,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMult_Fwk()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTransporse_Fwk"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_Fwk(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Gather^T */
  ierr = MatMultTranspose(fwk->gather,x,fwk->outvec); CHKERRQ(ierr); 
  ierr = MatMult_FwkAIJ(A, fwk->outvec, fwk->invec); CHKERRQ(ierr);
  /* Scatter^T */
  ierr = MatMultTransposeAdd(fwk->scatter,fwk->invec,y,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMultTranspose_Fwk()

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyBegin_FwkAIJ(Mat A, MatAssemblyType type) {
  Mat_Fwk     *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ  *aij = (Mat_FwkAIJ*)fwk->data;
  PetscInt i,j;
  Mat B;
  Mat_FwkBlock *ap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < fwk->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(j = 0; j < aij->ilen[i]; ++j, ++ap) {
      ierr = Mat_FwkBlock_GetMat(ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyBegin(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatAssemblyBegin_FwkAIJ()

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyEnd_FwkAIJ(Mat A, MatAssemblyType type) {
  Mat_Fwk     *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ  *aij = (Mat_FwkAIJ*)fwk->data;
  PetscInt i,j;
  Mat B;
  Mat_FwkBlock *ap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < fwk->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(j = 0; j < aij->ilen[i]; ++j, ++ap) {
      ierr = Mat_FwkBlock_GetMat(ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatAssemblyEnd_FwkAIJ()



#undef  __FUNCT__
#define __FUNCT__ "MatCreate_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_FwkAIJ(Mat A) {
  Mat_Fwk     *fwk = (Mat_Fwk *)A->data;
  Mat_FwkAIJ  *aij;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,Mat_FwkAIJ,&aij);CHKERRQ(ierr);
  fwk->data             = (void*)aij;

  aij->reallocs         = 0;
  aij->nonew            = 0;
  
  A->ops->assemblybegin = MatAssemblyBegin_FwkAIJ;
  A->ops->assemblyend   = MatAssemblyEnd_FwkAIJ;
  
  PetscFunctionReturn(0);
}/* MatCreate_FwkAIJ() */

#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_FwkAIJ(Mat M) {
  Mat_Fwk     *fwk = (Mat_Fwk *)M->data;
  Mat_FwkAIJ  *aij = (Mat_FwkAIJ *)fwk->data;
  PetscInt    *ai = aij->i, *ailen = aij->ilen;
  Mat_FwkBlock *aa = aij->a, *bp;
  PetscInt    i,n,j;
  Mat         B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy block matrices */
  for(i = 0; i < fwk->rowblockcount; ++i) {
    n = ailen[i];
    bp = aa + ai[i];
    for(j = 0; j < n; ++j) {
      ierr = Mat_FwkBlock_GetMat(bp, &B); CHKERRQ(ierr);
      ierr = MatDestroy(B);
      ++bp;
    }
  }
  /* Dealloc the AIJ structures */
  ierr = MatFwkXAIJFreeAIJ(M, &(aij->a), &(aij->j), &(aij->i)); CHKERRQ(ierr);
  ierr = PetscFree2(aij->imax,aij->ilen);CHKERRQ(ierr);
  ierr = PetscFree(aij);CHKERRQ(ierr);
  fwk->data = 0;
  

  PetscFunctionReturn(0);
}/* MatDestroy_FwkAIJ() */

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Fwk"
PetscErrorCode MatDestroy_Fwk(Mat mat)
{
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)mat->data;

  PetscFunctionBegin;
  ierr = MatDestroy_FwkAIJ(mat); CHKERRQ(ierr);

  ierr = PetscFree(fwk->colblockoffset);         CHKERRQ(ierr);
  ierr = PetscFree(fwk->rowblockoffset);         CHKERRQ(ierr);

  if(fwk->binvec) {
    ierr = VecDestroy(fwk->binvec);                           CHKERRQ(ierr);
  }
  if(fwk->boutvec) {
    ierr = VecDestroy(fwk->boutvec);                          CHKERRQ(ierr);
  }
  
  if(fwk->invec) {
    ierr = VecDestroy(fwk->invec);                            CHKERRQ(ierr);
  }
  if(fwk->outvec) {
    ierr = VecDestroy(fwk->outvec);                           CHKERRQ(ierr);
  }
  
  if(fwk->scatter) {
    ierr = MatDestroy(fwk->scatter);                          CHKERRQ(ierr);
  }
  if(fwk->gather) {
    ierr = MatDestroy(fwk->gather);                           CHKERRQ(ierr);
  }

  ierr      = PetscFree(fwk);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}/* MatDestroy_Fwk() */


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "MatCreate_Fwk"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Fwk(Mat A) {
  /* Assume that this is called after MatSetSizes() */
  Mat_Fwk  *fwk;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->ops->setuppreallocation = MatFwkSetUpPreallocation;
  A->ops->mult          = MatMult_Fwk;
  A->ops->multtranspose = MatMultTranspose_Fwk;
  A->ops->destroy       = MatDestroy_Fwk;

  A->assembled    = PETSC_FALSE;
  A->same_nonzero = PETSC_FALSE;

  ierr = PetscNewLog(A,Mat_Fwk,&fwk);CHKERRQ(ierr);
  A->data = (void*)fwk;
  

  fwk->default_block_type = MATAIJ;
  fwk->scatter = PETSC_NULL;
  fwk->gather  = PETSC_NULL;
  fwk->rowblockcount = 1; 
  fwk->colblockcount = 1; 
  fwk->em = A->rmap->n;
  fwk->en = A->cmap->n;
  ierr = PetscMalloc(sizeof(PetscInt)*(fwk->rowblockcount+1), &(fwk->rowblockoffset)); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(fwk->colblockcount+1), &(fwk->colblockoffset)); CHKERRQ(ierr);
  fwk->rowblockoffset[0] = 0; fwk->rowblockoffset[1] = fwk->em;
  fwk->colblockoffset[0] = 0; fwk->colblockoffset[1] = fwk->en;

  fwk->invec = PETSC_NULL;
  fwk->outvec = PETSC_NULL;

  /*
    HACK: these Vecs are created with minimal size and array information, 
     as we will later directly manipulate those inside MatMult loops to 
     directly place array chunks of variable sizes into these Vecs.
  */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, PETSC_NULL, &(fwk->binvec));  CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, PETSC_NULL, &(fwk->boutvec)); CHKERRQ(ierr);

  ierr = MatCreate_FwkAIJ(A); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATFWK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatCreate_Fwk() */
EXTERN_C_END
