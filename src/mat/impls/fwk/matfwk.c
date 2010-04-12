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


#undef  __FUNCT__
#define __FUNCT__ "Mat_FwkBlockSetMat"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_FwkBlockSetMat(Mat A, PetscInt rowblock, PetscInt colblock, Mat_FwkBlock *block, Mat B) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt              m,n,M,N;
#if defined PETSC_USE_DEBUG
  PetscInt              actualM, actualN;
#endif
  MPI_Comm              subcomm = ((PetscObject)B)->comm;
  PetscMPIInt           subcommsize;
  
  PetscFunctionBegin;
  
  /**/
  m = fwk->lrowblockoffset[rowblock+1]-fwk->lrowblockoffset[rowblock];
  n = fwk->lcolblockoffset[colblock+1]-fwk->lcolblockoffset[colblock];
  M = fwk->growblockoffset[rowblock+1]-fwk->growblockoffset[rowblock];
  N = fwk->gcolblockoffset[colblock+1]-fwk->gcolblockoffset[colblock];

#if defined PETSC_USE_DEBUG
  ierr = MPI_Comm_size(subcomm, &subcommsize); CHKERRQ(ierr);
  /**/
  if(subcommsize == 1) {
    actualM = M; actualN = N;
  }
  else {
    ierr = MPI_Allreduce(&m, &actualM, 1, MPI_INT, MPI_SUM, subcomm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&n, &actualN, 1, MPI_INT, MPI_SUM, subcomm); CHKERRQ(ierr);
  }
  if(actualM != M) {
    SETERRQ4(PETSC_ERR_USER, "Block[%d,%d]'s actual global row size %d doesn't match declared size %d", rowblock, colblock, actualM, M);
  }
  
  if(actualN != N) {
    SETERRQ4(PETSC_ERR_USER, "Block[%d,%d]'s actual global row size %d doesn't match declared size %d", rowblock, colblock, actualN, N);
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
}/* Mat_FwkBlockSetMat() */

#undef  __FUNCT__
#define __FUNCT__ "Mat_FwkBlockInit"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_FwkBlockInit(Mat A, PetscInt rowblock, PetscInt colblock, const MatType blockmattype, MatFwkBlockCommType subcommtype, Mat_FwkBlock *block) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt              m,n,M,N;
  MPI_Comm              comm = ((PetscObject)A)->comm, subcomm;
  PetscMPIInt           commsize, commrank, subcommsize;
  PetscMPIInt           subcommcolor;
  
  PetscFunctionBegin;
  
    /**/
  m = fwk->lrowblockoffset[rowblock+1]-fwk->lrowblockoffset[rowblock];
  n = fwk->lcolblockoffset[colblock+1]-fwk->lcolblockoffset[colblock];
  M = fwk->growblockoffset[rowblock+1]-fwk->growblockoffset[rowblock];
  N = fwk->gcolblockoffset[colblock+1]-fwk->gcolblockoffset[colblock];
  switch(subcommtype) {
  case MATFWK_BLOCK_COMM_SELF:
    subcomm = PETSC_COMM_SELF;
    subcommsize = 1;
    break;
  case MATFWK_BLOCK_COMM_DEFAULT:
    subcomm = comm;
    ierr = MPI_Comm_size(subcomm, &subcommsize); CHKERRQ(ierr);
    break;
  case MATFWK_BLOCK_COMM_DETERMINE:
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
    ierr = MPI_Comm_size(subcomm, &subcommsize);            CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_ERR_USER, "Unknown block comm type: %d", commsize);
    break;
  }/* switch(subcommtype) */
  if(subcomm != MPI_COMM_NULL) {
    ierr = MatCreate(subcomm, &(block->mat)); CHKERRQ(ierr);
    ierr = MatSetSizes(block->mat,m,n,M,N); CHKERRQ(ierr);
    /**/
    if(!blockmattype) {
      ierr = MatSetType(block->mat,fwk->default_block_type); CHKERRQ(ierr);
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
}/* Mat_FwkBlockInit() */


#undef  __FUNCT__
#define __FUNCT__ "Mat_FwkBlockFinit"
PetscErrorCode PETSCMAT_DLLEXPORT Mat_FwkBlockFinit(Mat A, Mat_FwkBlock *block) {
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
}/* Mat_FwkBlockFinit() */



#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetScatters"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetScatter(Mat A, PetscInt blockcount, Mat scatters[]) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt k,j0,n,N;
  PetscScalar *inarr;
  
  PetscFunctionBegin;


  /* check validity of block parameters */
  if(blockcount <= 0) {
    SETERRQ1(PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  }
  if(fwk->scatters) {
    for(k = 0; k < fwk->colblockcount; ++k) {
      if(fwk->scatters[k]) {
        ierr = MatDestroy(fwk->scatters[k]); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->scatters); CHKERRQ(ierr);
  }
  ierr = PetscFree2(fwk->lcolblockoffset, fwk->gcolblockoffset); CHKERRQ(ierr);

  fwk->colblockcount = blockcount;
  ierr = PetscMalloc(sizeof(Mat)*fwk->colblockcount, &(fwk->scatters)); CHKERRQ(ierr);
  ierr = PetscMalloc2(fwk->colblockcount+1,PetscInt,&(fwk->lcolblockoffset), fwk->colblockcount+1,PetscInt,&(fwk->gcolblockoffset)); CHKERRQ(ierr);    
  n = 0; N = 0;
  for(k = 0; k < fwk->colblockcount; ++k) {
    fwk->lcolblockoffset[k] = n;
    fwk->gcolblockoffset[k] = N;
    fwk->scatters[k] = scatters[k];
    if(!scatters[k]) continue;
    ierr = PetscObjectReference((PetscObject)scatters[k]); CHKERRQ(ierr);
    /* make sure the scatter column dimension matches that of MatFwk */
    if(scatters[k]->cmap->N != A->cmap->N){
      SETERRQ3(PETSC_ERR_USER, "Scatter # %d has global column dimension %d, which doesn't match MatFwk's %d", k, scatters[k]->cmap->N, A->cmap->N);
    }
    if(scatters[k]->cmap->n != A->cmap->n) {
      SETERRQ3(PETSC_ERR_USER, "Scatter # %d has local column dimension %d, which doesn't match MatFwk's %d", k, scatters[k]->cmap->n, A->cmap->n);
    }
    n += scatters[k]->rmap->n;
    N += scatters[k]->rmap->N;
  }  
  fwk->lcolblockoffset[fwk->colblockcount] = n;
  fwk->gcolblockoffset[fwk->colblockcount] = N;
  /* Now create invec and invecs.  FIX: get rid of invecs as soon as scatters are merged into one. */
  ierr = VecCreateMPI(((PetscObject)A)->comm, n, N, &(fwk->invec)); CHKERRQ(ierr);
  ierr = VecGetArray(fwk->invec, &inarr); CHKERRQ(ierr);
  for(k = 0; k < fwk->colblockcount; ++k) {
    /* FIX: get rid of invecs into a single Vec, as soon as scatters are merged */
    if(fwk->invecs[k]) {
      ierr = VecResetArray(fwk->invecs[k]);
      ierr = VecDestroy(fwk->invecs[k]); CHKERRQ(ierr);
    }
    j0 = fwk->lcolblockoffset[k];
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, fwk->scatters[k]->rmap->n, fwk->scatters[k]->rmap->N, inarr+j0, &(fwk->invecs[k])); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(fwk->invec, &inarr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
  
}/* MatFwkSetScatters() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkSetGathers"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetGathers(Mat A, PetscInt blockcount, Mat gathers[]) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  PetscInt              k,i0,m,M;
  PetscScalar           *outarr;
  
  PetscFunctionBegin;

  /* check validity of block parameters */
  if(blockcount <= 0) {
    SETERRQ1(PETSC_ERR_USER, "Invalid number of blocks: %d; must be > 0", blockcount);
  }
  if(fwk->gathers) {
    for(k = 0; k < fwk->rowblockcount; ++k) {
      if(fwk->gathers[k]) {
        ierr = MatDestroy(fwk->gathers[k]); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->gathers); CHKERRQ(ierr);
  }
  ierr = PetscFree2(fwk->lcolblockoffset, fwk->gcolblockoffset); CHKERRQ(ierr);

  fwk->rowblockcount = blockcount;
  ierr = PetscMalloc(sizeof(Mat)*fwk->rowblockcount, &(fwk->gathers)); CHKERRQ(ierr);
  ierr = PetscMalloc2(fwk->rowblockcount+1,PetscInt,&(fwk->lrowblockoffset), fwk->rowblockcount+1,PetscInt,&(fwk->growblockoffset)); CHKERRQ(ierr);    
  m = M = 0;
  for(k = 0; k < fwk->rowblockcount; ++k) {
    fwk->lrowblockoffset[k] = m;
    fwk->growblockoffset[k] = M;
    fwk->gathers[k] = gathers[k];
    if(!gathers[k]) continue;
    ierr = PetscObjectReference((PetscObject)gathers[k]); CHKERRQ(ierr);
    /* make sure the gather row dimension matches that of MatFwk */
    if(gathers[k] && gathers[k]->rmap->N != A->rmap->N){
      SETERRQ3(PETSC_ERR_USER, "Gather # %d has global row dimension %d, which doesn't match MatFwk's %d", k, gathers[k]->rmap->N, A->rmap->N);
    }
    if(gathers[k] && gathers[k]->rmap->n != A->rmap->n) {
      SETERRQ3(PETSC_ERR_USER, "Gather # %d has local row dimension %d, which doesn't match MatFwk's %d", k, gathers[k]->rmap->n, A->rmap->n);
    }
    m += gathers[k]->cmap->n;
    M += gathers[k]->cmap->N;
  }  
  fwk->lrowblockoffset[fwk->rowblockcount] = m;
  fwk->growblockoffset[fwk->rowblockcount] = M;
  /* Now create outvec and outvecs.  FIX: get rid of outvecs as soon as gathers are merged into one. */
  ierr = VecCreateMPI(((PetscObject)A)->comm, m,M, &(fwk->outvec)); CHKERRQ(ierr);
  ierr = VecGetArray(fwk->outvec, &outarr); CHKERRQ(ierr);
  for(k = 0; k < fwk->rowblockcount; ++k) {
    /* FIX: get rid of outvecs into a single Vec, as soon as scatters are merged */
    if(fwk->outvecs[k]) {
      ierr = VecResetArray(fwk->outvecs[k]);
      ierr = VecDestroy(fwk->outvecs[k]); CHKERRQ(ierr);
    }
    i0 = fwk->lrowblockoffset[k];
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, fwk->gathers[k]->cmap->n, fwk->gathers[k]->cmap->N, outarr+i0, &(fwk->outvecs[k])); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(fwk->outvec, &outarr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}/* MatFwkSetGathers() */


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
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: block %d value %d",i,nnz[i]);
      if (nnz[i] > fwk->colblockcount) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than the number of column blocks: block %d value %d column block count %d",i,nnz[i],fwk->colblockcount);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = MatFwkSetUpPreallocation_AIJ(A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkSetUpPreallocation() */

#undef  __FUNCT__
#define __FUNCT__ "MatFwkLocateBlock_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkLocateBlock_AIJ(Mat M, PetscInt row, PetscInt col, PetscTruth insert, Mat_FwkBlock **__block) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)M->data;
  Mat_FwkAIJ*           a = (Mat_FwkAIJ*)fwk->data;
  PetscInt       *rp,low,high,t,ii,nrow,i,rmax,N;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  Mat_FwkBlock   *ap,*aa = a->a;
  PetscFunctionBegin;

  *__block = PETSC_NULL;
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
      *__block = ap+i;  
      goto we_are_done;
    }
  } 
  if (!insert || nonew == 1) goto we_are_done;
  if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block at (%D,%D) in the matrix",row,col);
  MatFwkXAIJReallocateAIJ(M,fwk->rowblockcount,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew);
  N = nrow++ - 1; a->nz++; high++;
  /* shift up all the later entries in this row */
  for (ii=N; ii>=i; ii--) {
    rp[ii+1] = rp[ii];
    ap[ii+1] = ap[ii];
  }
  rp[i] = col; 
  *__block = ap+i; 
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
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkAddBlock(Mat A, PetscInt rowblock, PetscInt colblock, const MatType blockmattype, MatFwkBlockCommType blockcommtype, Mat *_B) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  Mat_FwkBlock          *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  ierr = MatFwkLocateBlock_AIJ(A,rowblock, colblock, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = Mat_FwkBlockInit(A,rowblock,colblock,blockmattype, blockcommtype, _block); CHKERRQ(ierr);
  if(_B) {
    ierr = Mat_FwkBlockGetMat(A, rowblock, colblock, _block, _B); CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)(*_B)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatFwkAddBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatFwkAddBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkSetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat B) {
  PetscErrorCode        ierr;
  Mat_Fwk*              fwk = (Mat_Fwk*)A->data;
  Mat_FwkBlock          *_block;
  
  PetscFunctionBegin;
  ierr = MatPreallocated(A); CHKERRQ(ierr);
  if(rowblock < 0 || rowblock > fwk->rowblockcount){
    SETERRQ2(PETSC_ERR_USER, "row block id %d is invalid; must be >= 0 and < %d", rowblock, fwk->rowblockcount);
  }
  if(colblock < 0 || colblock > fwk->colblockcount){
    SETERRQ2(PETSC_ERR_USER, "col block id %d is invalid; must be >= 0 and < %d", colblock, fwk->colblockcount);
  }
  ierr = MatFwkLocateBlock_AIJ(A,rowblock, colblock, PETSC_TRUE, &_block); CHKERRQ(ierr);
  ierr = Mat_FwkBlockSetMat(A,rowblock,colblock,_block, B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(B)); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}/* MatFwkSetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatFwkGetBlock"
PetscErrorCode PETSCMAT_DLLEXPORT MatFwkGetBlock(Mat A, PetscInt rowblock, PetscInt colblock, Mat *_B) {
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
  ierr = MatFwkLocateBlock_AIJ(A, rowblock, colblock, PETSC_FALSE, &_block); CHKERRQ(ierr);
  if(_block == PETSC_NULL) {
    SETERRQ2(PETSC_ERR_USER, "Block not found: row %d col %d", rowblock, colblock);
  }
  ierr = Mat_FwkBlockGetMat(A,rowblock,colblock,_block, _B); CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)(*_B)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatFwkGetBlock() */


#undef  __FUNCT__
#define __FUNCT__ "MatMult_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_FwkAIJ(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ *aij = (Mat_FwkAIJ*)fwk->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  Mat_FwkBlock *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<fwk->rowblockcount; ++i) {
    yoff = fwk->lrowblockoffset[i];
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      xoff = fwk->lcolblockoffset[j];
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
}// MatMult_FwkAIJ()


#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_FwkAIJ(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ *aij = (Mat_FwkAIJ*)fwk->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  Mat_FwkBlock *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<fwk->rowblockcount; ++i) {
    xoff = fwk->lrowblockoffset[i];
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      yoff = fwk->lcolblockoffset[j];
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
}// MatMultTranspose_FwkAIJ()


#undef  __FUNCT__
#define __FUNCT__ "MatMult_Fwk"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_Fwk(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  Vec xx, yy;
  PetscInt i,j;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Scatter */
  /* FIX: merge scatters into a single scatter */
  if(fwk->scatters) {
    for(j = 0; j < fwk->colblockcount; ++j) {
      ierr = MatMult(fwk->scatters[j],x,fwk->invecs[j]); CHKERRQ(ierr); 
    }
    xx = fwk->invec;
  }
  else {
    xx = x;
  }
  if(fwk->gathers) {
    yy = fwk->outvec; 
  }
  else {
    yy = y;
  }
  ierr = MatMult_FwkAIJ(A, xx, yy); CHKERRQ(ierr);
  /* Gather */
  if(fwk->gathers) {
    for(i = 0; i < fwk->rowblockcount; ++i) {
      ierr = MatMultAdd(fwk->gathers[i],fwk->outvecs[i],y,y); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatMult_Fwk()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTransporse_Fwk"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_Fwk(Mat A, Vec x, Vec y) {
  Mat_Fwk  *fwk = (Mat_Fwk*)A->data;
  PetscInt i,j;
  Vec xx,yy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Gather^T */
  if(fwk->gathers) {
    for(i = 0; i < fwk->rowblockcount; ++i) {
      ierr = MatMultTranspose(fwk->gathers[i],x,fwk->outvecs[i]); CHKERRQ(ierr); 
    }
    xx = fwk->outvec;
  }
  else {
    xx = x;
  }
  if(fwk->scatters) {
    yy = fwk->invec;
  }
  else {
    yy = y;
  }
  ierr = MatMultTranspose_FwkAIJ(A, xx, yy); CHKERRQ(ierr);
  /* Scatter^T */
  if(fwk->scatters) {
    for(j = 0; j < fwk->colblockcount; ++j) {
      ierr = MatMultTransposeAdd(fwk->scatters[j],fwk->invecs[j],y,y); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}// MatMultTranspose_Fwk()

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_FwkAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyBegin_FwkAIJ(Mat A, MatAssemblyType type) {
  Mat_Fwk     *fwk = (Mat_Fwk*)A->data;
  Mat_FwkAIJ  *aij = (Mat_FwkAIJ*)fwk->data;
  PetscInt i,j,k;
  Mat B;
  Mat_FwkBlock *ap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < fwk->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = Mat_FwkBlockGetMat(A, i, j, ap,&B); CHKERRQ(ierr);
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
  PetscInt i,j,k;
  Mat B;
  Mat_FwkBlock *ap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < fwk->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = Mat_FwkBlockGetMat(A,i,j,ap,&B); CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy blocks */
  for(i = 0; i < fwk->rowblockcount; ++i) {
    n = ailen[i];
    bp = aa + ai[i];
    for(j = 0; j < n; ++j) {
      ierr = Mat_FwkBlockFinit(M, bp); CHKERRQ(ierr);
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
  PetscInt              k;

  PetscFunctionBegin;
  ierr = MatDestroy_FwkAIJ(mat); CHKERRQ(ierr);

  ierr = PetscFree2(fwk->lcolblockoffset, fwk->gcolblockoffset);         CHKERRQ(ierr);
  ierr = PetscFree2(fwk->lrowblockoffset, fwk->growblockoffset);         CHKERRQ(ierr);

  
  if(fwk->invecs) {
    for(k = 0; k < fwk->colblockcount; ++k) {
      if(fwk->invecs[k]) {
        ierr = VecResetArray(fwk->invecs[k]); CHKERRQ(ierr);
        ierr = VecDestroy(fwk->invecs[k]);    CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->invecs); CHKERRQ(ierr);
  }
  ierr = VecDestroy(fwk->invec); CHKERRQ(ierr);
  if(fwk->outvecs) {
    for(k = 0; k < fwk->rowblockcount; ++k) {
      if(fwk->outvecs[k]) {
        ierr = VecResetArray(fwk->outvecs[k]); CHKERRQ(ierr);
        ierr = VecDestroy(fwk->outvecs[k]);    CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->outvecs); CHKERRQ(ierr);
  }
  ierr = VecDestroy(fwk->outvec); CHKERRQ(ierr);
  if(fwk->scatters) {
    for(k = 0; k < fwk->colblockcount; ++k) {
      if(fwk->scatters[k]) {
        ierr = MatDestroy(fwk->scatters[k]);                          CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->scatters);                                  CHKERRQ(ierr);
  }
  if(fwk->gathers) {
    for(k = 0; k < fwk->rowblockcount; ++k) {
      if(fwk->gathers[k]) {
        ierr = MatDestroy(fwk->gathers[k]);                          CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(fwk->gathers);                                  CHKERRQ(ierr);
  }

  ierr = PetscFree(fwk);CHKERRQ(ierr);
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
  
  /* FIX: gathers and scatters need to be merged */
  fwk->default_block_type = MATAIJ;
  fwk->scatters = PETSC_NULL;
  fwk->gathers  = PETSC_NULL;
  fwk->rowblockcount = 1; 
  fwk->colblockcount = 1; 
  ierr = PetscMalloc2(fwk->rowblockcount+1, PetscInt, &(fwk->lrowblockoffset), fwk->rowblockcount+1, PetscInt, &(fwk->growblockoffset)); CHKERRQ(ierr);
  ierr = PetscMalloc2(fwk->colblockcount+1, PetscInt, &(fwk->lcolblockoffset), fwk->colblockcount+1, PetscInt, &(fwk->gcolblockoffset)); CHKERRQ(ierr);
  fwk->lrowblockoffset[0] = 0; fwk->lrowblockoffset[1] = A->rmap->n;
  fwk->growblockoffset[0] = 0; fwk->growblockoffset[1] = A->rmap->N;
  fwk->lcolblockoffset[0] = 0; fwk->lcolblockoffset[1] = A->cmap->n;
  fwk->gcolblockoffset[0] = 0; fwk->gcolblockoffset[1] = A->cmap->N;

  /* FIX: get rid of invecs and outvecs as soon as gathers and scatters are merged */
  fwk->invec = PETSC_NULL;
  fwk->invecs = PETSC_NULL;
  fwk->outvec = PETSC_NULL;
  fwk->outvecs = PETSC_NULL;

  ierr = MatCreate_FwkAIJ(A); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATFWK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatCreate_Fwk() */
EXTERN_C_END
