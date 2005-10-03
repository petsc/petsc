#define PETSCDM_DLL
 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/


typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  PetscErrorCode (*view)(Mesh,PetscViewer);
  PetscErrorCode (*createglobalvector)(Mesh,Vec*);
  PetscErrorCode (*getcoloring)(Mesh,ISColoringType,ISColoring*);
  PetscErrorCode (*getmatrix)(Mesh,MatType,Mat*);
  PetscErrorCode (*getinterpolation)(Mesh,Mesh,Mat*,Vec*);
  PetscErrorCode (*refine)(Mesh,MPI_Comm,Mesh*);
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  void    *topology;
  void    *boundary;
  void    *bundle;
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
};

#undef __FUNCT__  
#define __FUNCT__ "MeshGetMatrix" 
/*@C
    MeshGetMatrix - Creates a matrix with the correct parallel layout required for 
      computing the Jacobian on a function defined using the informatin in Mesh.

    Collective on Mesh

    Input Parameter:
+   mesh - the mesh object
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh mesh, MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  PetscInt              *globals,rstart,i;
  ISLocalToGlobalMapping lmap;

  PetscFunctionBegin;
  ierr = MatCreate(mesh->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,mesh->n,mesh->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*J,mesh->bs);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,mesh->d_nz,mesh->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,mesh->d_nz,mesh->d_nnz,mesh->o_nz,mesh->o_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,mesh->bs,mesh->d_nz,mesh->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,mesh->bs,mesh->d_nz,mesh->d_nnz,mesh->o_nz,mesh->o_nnz);CHKERRQ(ierr);

  ierr = PetscMalloc((mesh->n+mesh->Nghosts+1)*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<mesh->n; i++) {
    globals[i] = rstart + i;
  }
  ierr = PetscMemcpy(globals+mesh->n,mesh->ghosts,mesh->Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,mesh->n+mesh->Nghosts,globals,&lmap);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(lmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MeshSetGhosts"
/*@C
    MeshSetGhosts - Sets the global indices of other processes elements that will
      be ghosts on this process

    Not Collective

    Input Parameters:
+    mesh - the Mesh object
.    bs - block size
.    nlocal - number of local (non-ghost) entries
.    Nghosts - number of ghosts on this process
-    ghosts - indices of all the ghost points

    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetGhosts(Mesh mesh,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,1);
  if (mesh->ghosts) {ierr = PetscFree(mesh->ghosts);CHKERRQ(ierr);}
  ierr = PetscMalloc((1+Nghosts)*sizeof(PetscInt),&mesh->ghosts);CHKERRQ(ierr);
  ierr = PetscMemcpy(mesh->ghosts,ghosts,Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  mesh->bs      = bs;
  mesh->n       = nlocal;
  mesh->Nghosts = Nghosts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetPreallocation"
/*@C
    MeshSetPreallocation - sets the matrix memory preallocation for matrices computed by Mesh

    Not Collective

    Input Parameters:
+    mesh - the Mesh object
.    d_nz - maximum number of nonzeros in any row of diagonal block
.    d_nnz - number of nonzeros in each row of diagonal block
.    o_nz - maximum number of nonzeros in any row of off-diagonal block
.    o_nnz - number of nonzeros in each row of off-diagonal block


    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices(), MatMPIAIJSetPreallocation(),
         MatMPIBAIJSetPreallocation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetPreallocation(Mesh mesh,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidPointer(mesh,1);
  mesh->d_nz  = d_nz;
  mesh->d_nnz = (PetscInt*)d_nnz;
  mesh->o_nz  = o_nz;
  mesh->o_nnz = (PetscInt*)o_nnz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreate"
/*@C
    MeshCreate - Creates a DM object, used to manage data for an unstructured problem
    described by a Sieve.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   mesh - the mesh object

    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm comm,Mesh *mesh)
{
  PetscErrorCode ierr;
  Mesh         p;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  *mesh = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(p,_p_Mesh,struct _MeshOps,DA_COOKIE,0,"Mesh",comm,MeshDestroy,0);CHKERRQ(ierr);
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->getmatrix          = MeshGetMatrix;
  *mesh = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDestroy"
/*@C
    MeshDestroy - Destroys a mesh.

    Collective on Mesh

    Input Parameter:
.   mesh - the mesh object

    Level: advanced

.seealso MeshCreate(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh mesh)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  if (mesh->globalvector) {ierr = VecDestroy(mesh->globalvector);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateGlobalVector"
/*@C
    MeshCreateGlobalVector - Creates a vector of the correct size to be gathered into 
        by the mesh.

    Collective on Mesh

    Input Parameter:
.    mesh - the mesh object

    Output Parameters:
.   gvec - the global vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso MeshDestroy(), MeshCreate(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh mesh,Vec *gvec)
{
  PetscErrorCode     ierr;


  PetscFunctionBegin;
  if (mesh->globalvector) {
    ierr = VecDuplicate(mesh->globalvector,gvec);CHKERRQ(ierr);
  } else {
    ierr  = VecCreateGhostBlock(mesh->comm,mesh->bs,mesh->n,PETSC_DETERMINE,mesh->Nghosts,mesh->ghosts,&mesh->globalvector);CHKERRQ(ierr);
    *gvec = mesh->globalvector;
    ierr = PetscObjectReference((PetscObject)*gvec);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetGlobalIndices"
/*@C
    MeshGetGlobalIndices - Gets the global indices for all the local entries

    Collective on Mesh

    Input Parameter:
.    mesh - the mesh object

    Output Parameters:
.    idx - the individual indices for each packed vector/array
 
    Level: advanced

    Notes:
       The idx parameters should be freed by the calling routine with PetscFree()

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh mesh,PetscInt *idx[])
{
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetTopology"
/*@C
    MeshGetTopology - Gets the topology Sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    topology - the topology Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetTopology()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetTopology(Mesh mesh,void **topology)
{
  if (topology) {
    PetscValidPointer(topology,2);
    *topology = mesh->topology;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetTopology"
/*@C
    MeshSetTopology - Sets the topology Sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    topology - the topology Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetTopology()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetTopology(Mesh mesh,void *topology)
{
  PetscValidPointer(topology,2);
  mesh->topology = topology;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetBoundary"
/*@C
    MeshGetBoundary - Gets the boundary Sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    boundary - the boundary Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetBoundary()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetBoundary(Mesh mesh,void **boundary)
{
  if (boundary) {
    PetscValidPointer(boundary,2);
    *boundary = mesh->boundary;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetBoundary"
/*@C
    MeshSetBoundary - Sets the boundary Sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    boundary - the boundary Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetBoundary()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetBoundary(Mesh mesh,void *boundary)
{
  PetscValidPointer(boundary,2);
  mesh->boundary = boundary;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetBundle"
/*@C
    MeshGetBundle - Gets the Sieve bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the Sieve bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->bundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetBundle"
/*@C
    MeshSetBundle - Sets the Sieve bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the Sieve bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->bundle = bundle;
  PetscFunctionReturn(0);
}
