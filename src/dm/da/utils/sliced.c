#define PETSCDM_DLL
 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/
#include "private/dmimpl.h"    /*I      "petscmat.h"    I*/


typedef struct _SlicedOps *SlicedOps;
struct _SlicedOps {
  DMOPS(Sliced)
};

/* CSR storage of the nonzero structure of a bs*bs matrix */
typedef struct {
  PetscInt bs,nz,*i,*j;
} SlicedBlockFills;

struct _p_Sliced {
  PETSCHEADER(struct _SlicedOps);
  DMHEADER
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
  SlicedBlockFills *dfill,*ofill;
};

#undef __FUNCT__  
#define __FUNCT__ "SlicedGetMatrix" 
/*@C
    SlicedGetMatrix - Creates a matrix with the correct parallel layout required for 
      computing the Jacobian on a function defined using the informatin in Sliced.

    Collective on Sliced

    Input Parameter:
+   slice - the slice object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

    Output Parameters:
.   J  - matrix with the correct nonzero preallocation
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DASetBlockFills()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedGetMatrix(Sliced slice, const MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  PetscInt               *globals,*sd_nnz,*so_nnz,rstart,bs,i;
  ISLocalToGlobalMapping lmap,blmap;
  void                   (*aij)(void) = PETSC_NULL;

  PetscFunctionBegin;
  bs = slice->bs;
  ierr = MatCreate(((PetscObject)slice)->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,slice->n*bs,slice->n*bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,bs,slice->d_nz,slice->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,bs,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz);CHKERRQ(ierr);
  /* In general, we have to do extra work to preallocate for scalar (AIJ) matrices so we check whether it will do any
  * good before going on with it. */
  ierr = PetscObjectQueryFunction((PetscObject)*J,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  if (!aij) {
    ierr = PetscObjectQueryFunction((PetscObject)*J,"MatSeqAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
  }
  if (aij) {
    if (bs == 1) {
      ierr = MatSeqAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz);CHKERRQ(ierr);
    } else if (!slice->d_nnz) {
      ierr = MatSeqAIJSetPreallocation(*J,slice->d_nz*bs,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(*J,slice->d_nz*bs,PETSC_NULL,slice->o_nz*bs,PETSC_NULL);CHKERRQ(ierr);
    } else {
      /* The user has provided preallocation per block-row, convert it to per scalar-row respecting SlicedSetBlockFills() if applicable */
      ierr = PetscMalloc2(slice->n*bs,PetscInt,&sd_nnz,(!!slice->o_nnz)*slice->n*bs,PetscInt,&so_nnz);CHKERRQ(ierr);
      for (i=0; i<slice->n*bs; i++) {
        sd_nnz[i] = (slice->d_nnz[i/bs]-1) * (slice->ofill ? slice->ofill->i[i%bs+1]-slice->ofill->i[i%bs] : bs)
                                           + (slice->dfill ? slice->dfill->i[i%bs+1]-slice->dfill->i[i%bs] : bs);
        if (so_nnz) {
          so_nnz[i] = slice->o_nnz[i/bs] * (slice->ofill ? slice->ofill->i[i%bs+1]-slice->ofill->i[i%bs] : bs);
        }
      }
      ierr = MatSeqAIJSetPreallocation(*J,slice->d_nz*bs,sd_nnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(*J,slice->d_nz*bs,sd_nnz,slice->o_nz*bs,so_nnz);CHKERRQ(ierr);
      ierr = PetscFree2(sd_nnz,so_nnz);CHKERRQ(ierr);
    }
  }

  ierr = MatSetBlockSize(*J,bs);CHKERRQ(ierr);

  /* Set up the local to global map.  For the scalar map, we have to translate to entry-wise indexing instead of block-wise. */
  ierr = PetscMalloc((slice->n+slice->Nghosts)*bs*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<slice->n*bs; i++) {
    globals[i] = rstart + i;
  }
  for (i=0; i<slice->Nghosts*bs; i++) {
    globals[slice->n*bs+i] = slice->ghosts[i/bs]*bs + i%bs;
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,(slice->n+slice->Nghosts)*bs,globals,&lmap);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(lmap,bs,&blmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,lmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,blmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(blmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedSetGhosts"
/*@C
    SlicedSetGhosts - Sets the global indices of other processes elements that will
      be ghosts on this process

    Not Collective

    Input Parameters:
+    slice - the Sliced object
.    bs - block size
.    nlocal - number of local (owned, non-ghost) blocks
.    Nghosts - number of ghost blocks on this process
-    ghosts - global indices of each ghost block

    Level: advanced

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedSetGhosts(Sliced slice,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(slice,DM_COOKIE,1);
  ierr = PetscFree(slice->ghosts);CHKERRQ(ierr);
  ierr = PetscMalloc(Nghosts*sizeof(PetscInt),&slice->ghosts);CHKERRQ(ierr);
  ierr = PetscMemcpy(slice->ghosts,ghosts,Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  slice->bs      = bs;
  slice->n       = nlocal;
  slice->Nghosts = Nghosts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedSetPreallocation"
/*@C
    SlicedSetPreallocation - sets the matrix memory preallocation for matrices computed by Sliced

    Not Collective

    Input Parameters:
+    slice - the Sliced object
.    d_nz  - number of block nonzeros per block row in diagonal portion of local
           submatrix  (same for all local rows)
.    d_nnz - array containing the number of block nonzeros in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  You must leave room for the diagonal entry even if it is zero.
.    o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-    o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

    Notes:
    See MatMPIBAIJSetPreallocation() for more details on preallocation.  If a scalar matrix (AIJ) is
    obtained with SlicedGetMatrix(), the correct preallocation will be set, respecting SlicedSetBlockFills().

    Level: advanced

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices(), MatMPIAIJSetPreallocation(),
         MatMPIBAIJSetPreallocation(), SlicedGetMatrix(), SlicedSetBlockFills()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedSetPreallocation(Sliced slice,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(slice,DM_COOKIE,1);
  slice->d_nz  = d_nz;
  slice->d_nnz = (PetscInt*)d_nnz;
  slice->o_nz  = o_nz;
  slice->o_nnz = (PetscInt*)o_nnz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedSetBlockFills_Private"
static PetscErrorCode SlicedSetBlockFills_Private(PetscInt bs,const PetscInt *fill,SlicedBlockFills **inf)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,nz,*fi,*fj;
  SlicedBlockFills *f;

  PetscFunctionBegin;
  PetscValidPointer(inf,3);
  if (*inf) {ierr = PetscFree3((*inf)->i,(*inf)->j,*inf);CHKERRQ(ierr);}
  if (!fill) PetscFunctionReturn(0);
  for (i=0,nz=0; i<bs*bs; i++) if (fill[i]) nz++;
  ierr = PetscMalloc3(1,SlicedBlockFills,&f,bs+1,PetscInt,&fi,nz,PetscInt,&fj);CHKERRQ(ierr);
  f->bs = bs;
  f->nz = nz;
  f->i  = fi;
  f->j  = fj;
  for (i=0,nz=0; i<bs; i++) {
    fi[i] = nz;
    for (j=0; j<bs; j++) if (fill[i*bs+j]) fj[nz++] = j;
  }
  fi[i] = nz;
  *inf = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedSetBlockFills"
/*@C
    SlicedSetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by SlicedGetMatrix().

    Collective on Sliced

    Input Parameter:
+   sliced - the Sliced object
.   dfill - the fill pattern in the diagonal block (may be PETSC_NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks

    Notes:
    This only makes sense for multicomponent problems using scalar matrix formats (AIJ).
    See DASetBlockFills() for example usage.

    Level: advanced

.seealso SlicedGetMatrix(), DASetBlockFills()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedSetBlockFills(Sliced slice,const PetscInt *dfill,const PetscInt *ofill)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(slice,DM_COOKIE,1);
  ierr = SlicedSetBlockFills_Private(slice->bs,dfill,&slice->dfill);CHKERRQ(ierr);
  ierr = SlicedSetBlockFills_Private(slice->bs,ofill,&slice->ofill);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedCreate"
/*@C
    SlicedCreate - Creates a DM object, used to manage data for a unstructured problem

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   slice - the slice object

    Level: advanced

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedCreate(MPI_Comm comm,Sliced *slice)
{
  PetscErrorCode ierr;
  Sliced         p;

  PetscFunctionBegin;
  PetscValidPointer(slice,2);
  *slice = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(p,_p_Sliced,struct _SlicedOps,DM_COOKIE,0,"DM",comm,SlicedDestroy,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)p,"Sliced");CHKERRQ(ierr);
  p->ops->createglobalvector = SlicedCreateGlobalVector;
  p->ops->getmatrix          = SlicedGetMatrix;
  p->ops->destroy            = SlicedDestroy;
  *slice = p;
  PetscFunctionReturn(0);
}

extern PetscErrorCode DMDestroy_Private(DM,PetscTruth*);

#undef __FUNCT__  
#define __FUNCT__ "SlicedDestroy"
/*@C
    SlicedDestroy - Destroys a vector slice.

    Collective on Sliced

    Input Parameter:
.   slice - the slice object

    Level: advanced

.seealso SlicedCreate(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedDestroy(Sliced slice)
{
  PetscErrorCode ierr;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = DMDestroy_Private((DM)slice,&done);CHKERRQ(ierr);
  if (!done) PetscFunctionReturn(0);

  if (slice->globalvector) {ierr = VecDestroy(slice->globalvector);CHKERRQ(ierr);}
  ierr = PetscFree(slice->ghosts);CHKERRQ(ierr);
  if (slice->dfill) {ierr = PetscFree3(slice->dfill,slice->dfill->i,slice->dfill->j);CHKERRQ(ierr);}
  if (slice->ofill) {ierr = PetscFree3(slice->ofill,slice->ofill->i,slice->ofill->j);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(slice);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SlicedCreateGlobalVector"
/*@C
    SlicedCreateGlobalVector - Creates a vector of the correct size to be gathered into
        by the slice.

    Collective on Sliced

    Input Parameter:
.    slice - the slice object

    Output Parameters:
.   gvec - the global vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso SlicedDestroy(), SlicedCreate(), SlicedGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedCreateGlobalVector(Sliced slice,Vec *gvec)
{
  PetscErrorCode     ierr;
  PetscInt           bs,cnt,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(slice,DM_COOKIE,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  if (slice->globalvector) {
    ierr = PetscObjectGetReference((PetscObject)slice->globalvector,&cnt);CHKERRQ(ierr);
    if (cnt == 1) {             /* Nobody else has a reference so we can just reference it and give it away */
      *gvec = slice->globalvector;
      ierr = PetscObjectReference((PetscObject)*gvec);CHKERRQ(ierr);
      ierr = VecZeroEntries(*gvec);CHKERRQ(ierr);
    } else {                    /* Someone else has a reference so we duplicate the global vector */
      ierr = VecDuplicate(slice->globalvector,gvec);CHKERRQ(ierr);
    }
  } else {
    bs = slice->bs;
    /* VecCreateGhostBlock requires ghosted blocks to be given in terms of first entry, not block.  Here, we munge the
    * ghost array for this call, then put it back. */
    for (i=0; i<slice->Nghosts; i++) slice->ghosts[i] *= bs;
    ierr = VecCreateGhostBlock(((PetscObject)slice)->comm,bs,slice->n*bs,PETSC_DETERMINE,slice->Nghosts,slice->ghosts,&slice->globalvector);CHKERRQ(ierr);
    for (i=0; i<slice->Nghosts; i++) slice->ghosts[i] /= bs;
    *gvec = slice->globalvector;
    ierr = PetscObjectReference((PetscObject)*gvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlicedGetGlobalIndices"
/*@C
    SlicedGetGlobalIndices - Gets the global indices for all the local entries

    Collective on Sliced

    Input Parameter:
.    slice - the slice object

    Output Parameters:
.    idx - the individual indices for each packed vector/array

    Level: advanced

    Notes:
       The idx parameters should be freed by the calling routine with PetscFree()

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedGetGlobalIndices(Sliced slice,PetscInt *idx[])
{
  PetscFunctionReturn(0);
}


/* Explanation of the missing functions for DA-style handling of the local vector:

   SlicedCreateLocalVector()
   SlicedGlobalToLocalBegin()
   SlicedGlobalToLocalEnd()

 There is no way to get the global form from a local form, so SlicedCreateLocalVector() is a memory leak without
 external accounting for the global vector.  Also, Sliced intends the user to work with the VecGhost interface since the
 ghosts are already ordered after the owned entries.  Contrast this to a DA where the local vector has a special
 ordering described by the structured grid, hence it cannot share memory with the global form.  For this reason, users
 of Sliced should work with the global vector and use

   VecGhostGetLocalForm(), VecGhostRestoreLocalForm()
   VecGhostUpdateBegin(), VecGhostUpdateEnd()

 rather than the missing DA-style functions.  This is conceptually simpler and offers better performance than is
 possible with the DA-style interface.
*/
