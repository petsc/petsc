#include <petscdmsliced.h>      /*I      "petscdmsliced.h" I*/
#include <petscmat.h>           /*I      "petscmat.h"      I*/
#include <petsc-private/dmimpl.h>     /*I      "petscdm.h"       I*/

/* CSR storage of the nonzero structure of a bs*bs matrix */
typedef struct {
  PetscInt bs,nz,*i,*j;
} DMSlicedBlockFills;

typedef struct  {
  PetscInt           bs,n,N,Nghosts,*ghosts;
  PetscInt           d_nz,o_nz,*d_nnz,*o_nnz;
  DMSlicedBlockFills *dfill,*ofill;
} DM_Sliced;

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Sliced"
PetscErrorCode  DMCreateMatrix_Sliced(DM dm, MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  PetscInt               *globals,*sd_nnz,*so_nnz,rstart,bs,i;
  ISLocalToGlobalMapping lmap,blmap;
  void                   (*aij)(void) = PETSC_NULL;
  DM_Sliced              *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  bs = slice->bs;
  ierr = MatCreate(((PetscObject)dm)->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,slice->n*bs,slice->n*bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*J,bs);CHKERRQ(ierr);
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
      /* The user has provided preallocation per block-row, convert it to per scalar-row respecting DMSlicedSetBlockFills() if applicable */
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

  /* Set up the local to global map.  For the scalar map, we have to translate to entry-wise indexing instead of block-wise. */
  ierr = PetscMalloc((slice->n+slice->Nghosts)*bs*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<slice->n*bs; i++) {
    globals[i] = rstart + i;
  }
  for (i=0; i<slice->Nghosts*bs; i++) {
    globals[slice->n*bs+i] = slice->ghosts[i/bs]*bs + i%bs;
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,(slice->n+slice->Nghosts)*bs,globals,PETSC_OWN_POINTER,&lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(lmap,bs,&blmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,lmap,lmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,blmap,blmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&blmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSlicedSetGhosts"
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

.seealso DMDestroy(), DMCreateGlobalVector()

@*/
PetscErrorCode  DMSlicedSetGhosts(DM dm,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  PetscErrorCode ierr;
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscFree(slice->ghosts);CHKERRQ(ierr);
  ierr = PetscMalloc(Nghosts*sizeof(PetscInt),&slice->ghosts);CHKERRQ(ierr);
  ierr = PetscMemcpy(slice->ghosts,ghosts,Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  slice->bs      = bs;
  slice->n       = nlocal;
  slice->Nghosts = Nghosts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSlicedSetPreallocation"
/*@C
    DMSlicedSetPreallocation - sets the matrix memory preallocation for matrices computed by DMSliced

    Not Collective

    Input Parameters:
+    slice - the DM object
.    d_nz  - number of block nonzeros per block row in diagonal portion of local
           submatrix  (same for all local rows)
.    d_nnz - array containing the number of block nonzeros in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.
.    o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-    o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

    Notes:
    See MatMPIBAIJSetPreallocation() for more details on preallocation.  If a scalar matrix (AIJ) is
    obtained with DMSlicedGetMatrix(), the correct preallocation will be set, respecting DMSlicedSetBlockFills().

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), MatMPIAIJSetPreallocation(),
         MatMPIBAIJSetPreallocation(), DMSlicedGetMatrix(), DMSlicedSetBlockFills()

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

#undef __FUNCT__
#define __FUNCT__ "DMSlicedSetBlockFills_Private"
static PetscErrorCode DMSlicedSetBlockFills_Private(PetscInt bs,const PetscInt *fill,DMSlicedBlockFills **inf)
{
  PetscErrorCode     ierr;
  PetscInt           i,j,nz,*fi,*fj;
  DMSlicedBlockFills *f;

  PetscFunctionBegin;
  PetscValidPointer(inf,3);
  if (*inf) {ierr = PetscFree3((*inf)->i,(*inf)->j,*inf);CHKERRQ(ierr);}
  if (!fill) PetscFunctionReturn(0);
  for (i=0,nz=0; i<bs*bs; i++) if (fill[i]) nz++;
  ierr = PetscMalloc3(1,DMSlicedBlockFills,&f,bs+1,PetscInt,&fi,nz,PetscInt,&fj);CHKERRQ(ierr);
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
#define __FUNCT__ "DMSlicedSetBlockFills"
/*@C
    DMSlicedSetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMSlicedGetMatrix().

    Logically Collective on DM

    Input Parameter:
+   sliced - the DM object
.   dfill - the fill pattern in the diagonal block (may be PETSC_NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks

    Notes:
    This only makes sense for multicomponent problems using scalar matrix formats (AIJ).
    See DMDASetBlockFills() for example usage.

    Level: advanced

.seealso DMSlicedGetMatrix(), DMDASetBlockFills()
@*/
PetscErrorCode  DMSlicedSetBlockFills(DM dm,const PetscInt *dfill,const PetscInt *ofill)
{
  DM_Sliced      *slice = (DM_Sliced*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSlicedSetBlockFills_Private(slice->bs,dfill,&slice->dfill);CHKERRQ(ierr);
  ierr = DMSlicedSetBlockFills_Private(slice->bs,ofill,&slice->ofill);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Sliced"
static PetscErrorCode  DMDestroy_Sliced(DM dm)
{
  PetscErrorCode ierr;
  DM_Sliced      *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  ierr = PetscFree(slice->ghosts);CHKERRQ(ierr);
  if (slice->dfill) {ierr = PetscFree3(slice->dfill,slice->dfill->i,slice->dfill->j);CHKERRQ(ierr);}
  if (slice->ofill) {ierr = PetscFree3(slice->ofill,slice->ofill->i,slice->ofill->j);CHKERRQ(ierr);}
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(slice);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Sliced"
static PetscErrorCode  DMCreateGlobalVector_Sliced(DM dm,Vec *gvec)
{
  PetscErrorCode     ierr;
  DM_Sliced          *slice = (DM_Sliced*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  ierr = VecCreateGhostBlock(((PetscObject)dm)->comm,slice->bs,slice->n*slice->bs,PETSC_DETERMINE,slice->Nghosts,slice->ghosts,gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*gvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_Sliced"
static PetscErrorCode  DMGlobalToLocalBegin_Sliced(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = VecGhostIsLocalForm(g,l,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Local vector is not local form of global vector");
  ierr = VecGhostUpdateEnd(g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_Sliced"
static PetscErrorCode  DMGlobalToLocalEnd_Sliced(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = VecGhostIsLocalForm(g,l,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Local vector is not local form of global vector");
  ierr = VecGhostUpdateEnd(g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   DMSLICED = "sliced" - A DM object that is used to manage data for a general graph. Uses VecCreateGhost() ghosted vectors for storing the fields

   See DMCreateSliced() for details.

  Level: intermediate

.seealso: DMType, DMCOMPOSITE, DMCreateSliced(), DMCreate()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Sliced"
PetscErrorCode  DMCreate_Sliced(DM p)
{
  PetscErrorCode ierr;
  DM_Sliced      *slice;

  PetscFunctionBegin;
  ierr = PetscNewLog(p,DM_Sliced,&slice);CHKERRQ(ierr);
  p->data = slice;

  ierr = PetscObjectChangeTypeName((PetscObject)p,DMSLICED);CHKERRQ(ierr);
  p->ops->createglobalvector = DMCreateGlobalVector_Sliced;
  p->ops->creatematrix       = DMCreateMatrix_Sliced;
  p->ops->globaltolocalbegin = DMGlobalToLocalBegin_Sliced;
  p->ops->globaltolocalend   = DMGlobalToLocalEnd_Sliced;
  p->ops->destroy            = DMDestroy_Sliced;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMSlicedCreate"
/*@C
    DMSlicedCreate - Creates a DM object, used to manage data for a unstructured problem

    Collective on MPI_Comm

    Input Parameter:
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

.seealso DMDestroy(), DMCreateGlobalVector(), DMSetType(), DMSLICED, DMSlicedSetGhosts(), DMSlicedSetPreallocation(), VecGhostUpdateBegin(), VecGhostUpdateEnd(),
         VecGhostGetLocalForm(), VecGhostRestoreLocalForm()

@*/
PetscErrorCode  DMSlicedCreate(MPI_Comm comm,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[], const PetscInt d_nnz[],const PetscInt o_nnz[],DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMSLICED);CHKERRQ(ierr);
  ierr = DMSlicedSetGhosts(*dm,bs,nlocal,Nghosts,ghosts);CHKERRQ(ierr);
  if (d_nnz) {
    ierr = DMSlicedSetPreallocation(*dm,0, d_nnz,0,o_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

