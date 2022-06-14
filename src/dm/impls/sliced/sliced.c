#include <petscdmsliced.h>      /*I      "petscdmsliced.h" I*/
#include <petscmat.h>
#include <petsc/private/dmimpl.h>

/* CSR storage of the nonzero structure of a bs*bs matrix */
typedef struct {
  PetscInt bs,nz,*i,*j;
} DMSlicedBlockFills;

typedef struct  {
  PetscInt           bs,n,N,Nghosts,*ghosts;
  PetscInt           d_nz,o_nz,*d_nnz,*o_nnz;
  DMSlicedBlockFills *dfill,*ofill;
} DM_Sliced;

PetscErrorCode  DMCreateMatrix_Sliced(DM dm, Mat *J)
{
  PetscInt               *globals,*sd_nnz,*so_nnz,rstart,bs,i;
  ISLocalToGlobalMapping lmap;
  void                   (*aij)(void) = NULL;
  DM_Sliced              *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  bs   = slice->bs;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm),J));
  PetscCall(MatSetSizes(*J,slice->n*bs,slice->n*bs,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetBlockSize(*J,bs));
  PetscCall(MatSetType(*J,dm->mattype));
  PetscCall(MatSeqBAIJSetPreallocation(*J,bs,slice->d_nz,slice->d_nnz));
  PetscCall(MatMPIBAIJSetPreallocation(*J,bs,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz));
  /* In general, we have to do extra work to preallocate for scalar (AIJ) matrices so we check whether it will do any
  * good before going on with it. */
  PetscCall(PetscObjectQueryFunction((PetscObject)*J,"MatMPIAIJSetPreallocation_C",&aij));
  if (!aij) {
    PetscCall(PetscObjectQueryFunction((PetscObject)*J,"MatSeqAIJSetPreallocation_C",&aij));
  }
  if (aij) {
    if (bs == 1) {
      PetscCall(MatSeqAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz));
      PetscCall(MatMPIAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz));
    } else if (!slice->d_nnz) {
      PetscCall(MatSeqAIJSetPreallocation(*J,slice->d_nz*bs,NULL));
      PetscCall(MatMPIAIJSetPreallocation(*J,slice->d_nz*bs,NULL,slice->o_nz*bs,NULL));
    } else {
      /* The user has provided preallocation per block-row, convert it to per scalar-row respecting DMSlicedSetBlockFills() if applicable */
      PetscCall(PetscMalloc2(slice->n*bs,&sd_nnz,(!!slice->o_nnz)*slice->n*bs,&so_nnz));
      for (i=0; i<slice->n*bs; i++) {
        sd_nnz[i] = (slice->d_nnz[i/bs]-1) * (slice->ofill ? slice->ofill->i[i%bs+1]-slice->ofill->i[i%bs] : bs)
                                           + (slice->dfill ? slice->dfill->i[i%bs+1]-slice->dfill->i[i%bs] : bs);
        if (so_nnz) {
          so_nnz[i] = slice->o_nnz[i/bs] * (slice->ofill ? slice->ofill->i[i%bs+1]-slice->ofill->i[i%bs] : bs);
        }
      }
      PetscCall(MatSeqAIJSetPreallocation(*J,slice->d_nz*bs,sd_nnz));
      PetscCall(MatMPIAIJSetPreallocation(*J,slice->d_nz*bs,sd_nnz,slice->o_nz*bs,so_nnz));
      PetscCall(PetscFree2(sd_nnz,so_nnz));
    }
  }

  /* Set up the local to global map.  For the scalar map, we have to translate to entry-wise indexing instead of block-wise. */
  PetscCall(PetscMalloc1(slice->n+slice->Nghosts,&globals));
  PetscCall(MatGetOwnershipRange(*J,&rstart,NULL));
  for (i=0; i<slice->n; i++) globals[i] = rstart/bs + i;

  for (i=0; i<slice->Nghosts; i++) {
    globals[slice->n+i] = slice->ghosts[i];
  }
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,bs,slice->n+slice->Nghosts,globals,PETSC_OWN_POINTER,&lmap));
  PetscCall(MatSetLocalToGlobalMapping(*J,lmap,lmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&lmap));
  PetscCall(MatSetDM(*J,dm));
  PetscFunctionReturn(0);
}

/*@C
    DMSlicedSetGhosts - Sets the global indices of other processes elements that will
      be ghosts on this process

    Not Collective

    Input Parameters:
+    slice - the DM object
.    bs - block size
.    nlocal - number of local (owned, non-ghost) blocks
.    Nghosts - number of ghost blocks on this process
-    ghosts - global indices of each ghost block

    Level: advanced

.seealso `DMDestroy()`, `DMCreateGlobalVector()`

@*/
PetscErrorCode  DMSlicedSetGhosts(DM dm,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(PetscFree(slice->ghosts));
  PetscCall(PetscMalloc1(Nghosts,&slice->ghosts));
  PetscCall(PetscArraycpy(slice->ghosts,ghosts,Nghosts));
  slice->bs      = bs;
  slice->n       = nlocal;
  slice->Nghosts = Nghosts;
  PetscFunctionReturn(0);
}

/*@C
    DMSlicedSetPreallocation - sets the matrix memory preallocation for matrices computed by DMSliced

    Not Collective

    Input Parameters:
+    slice - the DM object
.    d_nz  - number of block nonzeros per block row in diagonal portion of local
           submatrix  (same for all local rows)
.    d_nnz - array containing the number of block nonzeros in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or NULL.
.    o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-    o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or NULL.

    Notes:
    See MatMPIBAIJSetPreallocation() for more details on preallocation.  If a scalar matrix (AIJ) is
    obtained with DMSlicedGetMatrix(), the correct preallocation will be set, respecting DMSlicedSetBlockFills().

    Level: advanced

.seealso `DMDestroy()`, `DMCreateGlobalVector()`, `MatMPIAIJSetPreallocation()`,
         `MatMPIBAIJSetPreallocation()`, `DMSlicedGetMatrix()`, `DMSlicedSetBlockFills()`

@*/
PetscErrorCode  DMSlicedSetPreallocation(DM dm,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  DM_Sliced *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  slice->d_nz  = d_nz;
  slice->d_nnz = (PetscInt*)d_nnz;
  slice->o_nz  = o_nz;
  slice->o_nnz = (PetscInt*)o_nnz;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSlicedSetBlockFills_Private(PetscInt bs,const PetscInt *fill,DMSlicedBlockFills **inf)
{
  PetscInt           i,j,nz,*fi,*fj;
  DMSlicedBlockFills *f;

  PetscFunctionBegin;
  PetscValidPointer(inf,3);
  if (*inf) PetscCall(PetscFree3(*inf,(*inf)->i,(*inf)->j));
  if (!fill) PetscFunctionReturn(0);
  for (i=0,nz=0; i<bs*bs; i++) if (fill[i]) nz++;
  PetscCall(PetscMalloc3(1,&f,bs+1,&fi,nz,&fj));
  f->bs = bs;
  f->nz = nz;
  f->i  = fi;
  f->j  = fj;
  for (i=0,nz=0; i<bs; i++) {
    fi[i] = nz;
    for (j=0; j<bs; j++) if (fill[i*bs+j]) fj[nz++] = j;
  }
  fi[i] = nz;
  *inf  = f;
  PetscFunctionReturn(0);
}

/*@C
    DMSlicedSetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMSlicedGetMatrix().

    Logically Collective on dm

    Input Parameters:
+   sliced - the DM object
.   dfill - the fill pattern in the diagonal block (may be NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks

    Notes:
    This only makes sense for multicomponent problems using scalar matrix formats (AIJ).
    See DMDASetBlockFills() for example usage.

    Level: advanced

.seealso `DMSlicedGetMatrix()`, `DMDASetBlockFills()`
@*/
PetscErrorCode  DMSlicedSetBlockFills(DM dm,const PetscInt *dfill,const PetscInt *ofill)
{
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMSlicedSetBlockFills_Private(slice->bs,dfill,&slice->dfill));
  PetscCall(DMSlicedSetBlockFills_Private(slice->bs,ofill,&slice->ofill));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMDestroy_Sliced(DM dm)
{
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(slice->ghosts));
  if (slice->dfill) PetscCall(PetscFree3(slice->dfill,slice->dfill->i,slice->dfill->j));
  if (slice->ofill) PetscCall(PetscFree3(slice->ofill,slice->ofill->i,slice->ofill->j));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  PetscCall(PetscFree(slice));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMCreateGlobalVector_Sliced(DM dm,Vec *gvec)
{
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = NULL;
  PetscCall(VecCreateGhostBlock(PetscObjectComm((PetscObject)dm),slice->bs,slice->n*slice->bs,PETSC_DETERMINE,slice->Nghosts,slice->ghosts,gvec));
  PetscCall(VecSetDM(*gvec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMGlobalToLocalBegin_Sliced(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(VecGhostIsLocalForm(g,l,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Local vector is not local form of global vector");
  PetscCall(VecGhostUpdateEnd(g,mode,SCATTER_FORWARD));
  PetscCall(VecGhostUpdateBegin(g,mode,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMGlobalToLocalEnd_Sliced(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(VecGhostIsLocalForm(g,l,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Local vector is not local form of global vector");
  PetscCall(VecGhostUpdateEnd(g,mode,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

/*MC
   DMSLICED = "sliced" - A DM object that is used to manage data for a general graph. Uses VecCreateGhost() ghosted vectors for storing the fields

   See DMCreateSliced() for details.

  Level: intermediate

.seealso: `DMType`, `DMCOMPOSITE`, `DMCreateSliced()`, `DMCreate()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Sliced(DM p)
{
  DM_Sliced      *slice;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(p,&slice));
  p->data = slice;

  p->ops->createglobalvector = DMCreateGlobalVector_Sliced;
  p->ops->creatematrix       = DMCreateMatrix_Sliced;
  p->ops->globaltolocalbegin = DMGlobalToLocalBegin_Sliced;
  p->ops->globaltolocalend   = DMGlobalToLocalEnd_Sliced;
  p->ops->destroy            = DMDestroy_Sliced;
  PetscFunctionReturn(0);
}

/*@C
    DMSlicedCreate - Creates a DM object, used to manage data for a unstructured problem

    Collective

    Input Parameters:
+   comm - the processors that will share the global vector
.   bs - the block size
.   nlocal - number of vector entries on this process
.   Nghosts - number of ghost points needed on this process
.   ghosts - global indices of all ghost points for this process
.   d_nnz - matrix preallocation information representing coupling within this process
-   o_nnz - matrix preallocation information representing coupling between this process and other processes

    Output Parameters:
.   slice - the slice object

    Notes:
        This DM does not support DMCreateLocalVector(), DMGlobalToLocalBegin(), and DMGlobalToLocalEnd() instead one directly uses
        VecGhostGetLocalForm() and VecGhostRestoreLocalForm() to access the local representation and VecGhostUpdateBegin() and VecGhostUpdateEnd() to update
        the ghost points.

        One can use DMGlobalToLocalBegin(), and DMGlobalToLocalEnd() instead of VecGhostUpdateBegin() and VecGhostUpdateEnd().

    Level: advanced

.seealso `DMDestroy()`, `DMCreateGlobalVector()`, `DMSetType()`, `DMSLICED`, `DMSlicedSetGhosts()`, `DMSlicedSetPreallocation()`, `VecGhostUpdateBegin()`, `VecGhostUpdateEnd()`,
         `VecGhostGetLocalForm()`, `VecGhostRestoreLocalForm()`

@*/
PetscErrorCode  DMSlicedCreate(MPI_Comm comm,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[], const PetscInt d_nnz[],const PetscInt o_nnz[],DM *dm)
{
  PetscFunctionBegin;
  PetscValidPointer(dm,8);
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetType(*dm,DMSLICED));
  PetscCall(DMSlicedSetGhosts(*dm,bs,nlocal,Nghosts,ghosts));
  if (d_nnz) PetscCall(DMSlicedSetPreallocation(*dm,0, d_nnz,0,o_nnz));
  PetscFunctionReturn(0);
}
