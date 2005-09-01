#define PETSCDM_DLL
 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/


typedef struct _SlicedOps *SlicedOps;
struct _SlicedOps {
  PetscErrorCode (*view)(Sliced,PetscViewer);
  PetscErrorCode (*createglobalvector)(Sliced,Vec*);
  PetscErrorCode (*getcoloring)(Sliced,ISColoringType,ISColoring*);
  PetscErrorCode (*getmatrix)(Sliced,MatType,Mat*);
  PetscErrorCode (*getinterpolation)(Sliced,Sliced,Mat*,Vec*);
  PetscErrorCode (*refine)(Sliced,MPI_Comm,Sliced*);
};

struct _p_Sliced {
  PETSCHEADER(struct _SlicedOps);
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
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
PetscErrorCode PETSCDM_DLLEXPORT SlicedGetMatrix(Sliced slice, MatType mtype,Mat *J)
{
  PetscErrorCode         ierr,*globals,rstart,i;
  ISLocalToGlobalMapping lmap;

  PetscFunctionBegin;
  ierr = MatCreate(slice->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,slice->n,slice->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*J,slice->bs);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,slice->bs,slice->d_nz,slice->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,slice->bs,slice->d_nz,slice->d_nnz,slice->o_nz,slice->o_nnz);CHKERRQ(ierr);

  ierr = PetscMalloc((slice->n+slice->Nghosts+1)*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<slice->n; i++) {
    globals[i] = rstart + i;
  }
  ierr = PetscMemcpy(globals+slice->n,slice->ghosts,slice->Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,slice->n+slice->Nghosts,globals,&lmap);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(lmap);CHKERRQ(ierr);
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
.    nlocal - number of local (non-ghost) entries
.    Nghosts - number of ghosts on this process
-    ghosts - indices of all the ghost points

    Level: advanced

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedSetGhosts(Sliced slice,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(slice,1);
  if (slice->ghosts) {ierr = PetscFree(slice->ghosts);CHKERRQ(ierr);}
  ierr = PetscMalloc((1+Nghosts)*sizeof(PetscInt),&slice->ghosts);CHKERRQ(ierr);
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
.    d_nz - maximum number of nonzeros in any row of diagonal block
.    d_nnz - number of nonzeros in each row of diagonal block
.    o_nz - maximum number of nonzeros in any row of off-diagonal block
.    o_nnz - number of nonzeros in each row of off-diagonal block


    Level: advanced

.seealso SlicedDestroy(), SlicedCreateGlobalVector(), SlicedGetGlobalIndices(), MatMPIAIJSetPreallocation(),
         MatMPIBAIJSetPreallocation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT SlicedSetPreallocation(Sliced slice,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidPointer(slice,1);
  slice->d_nz  = d_nz;
  slice->d_nnz = (PetscInt*)d_nnz;
  slice->o_nz  = o_nz;
  slice->o_nnz = (PetscInt*)o_nnz;
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

  ierr = PetscHeaderCreate(p,_p_Sliced,struct _SlicedOps,DA_COOKIE,0,"Sliced",comm,SlicedDestroy,0);CHKERRQ(ierr);
  p->ops->createglobalvector = SlicedCreateGlobalVector;
  p->ops->getmatrix          = SlicedGetMatrix;
  *slice = p;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (--slice->refct > 0) PetscFunctionReturn(0);
  if (slice->globalvector) {ierr = VecDestroy(slice->globalvector);CHKERRQ(ierr);}
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


  PetscFunctionBegin;
  if (slice->globalvector) {
    ierr = VecDuplicate(slice->globalvector,gvec);CHKERRQ(ierr);
  } else {
    ierr  = VecCreateGhostBlock(slice->comm,slice->bs,slice->n,PETSC_DETERMINE,slice->Nghosts,slice->ghosts,&slice->globalvector);CHKERRQ(ierr);
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

