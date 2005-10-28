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
  void    *orientation;
  void    *spaceFootprint;
  void    *bundle;
  void    *coordBundle;
  Vec      coordinates;
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
};

#include <ClosureBundle.hh>
PetscErrorCode WriteVTKVertices(Mesh, FILE *);
PetscErrorCode WriteVTKElements(Mesh, FILE *);
PetscErrorCode WriteVTKHeader(Mesh, FILE *);

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(Mesh mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ALE::ClosureBundle *coordBundle;
    FILE               *f;

    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    ierr = WriteVTKHeader(mesh, f);CHKERRQ(ierr);
    ierr = MeshGetCoordinateBundle(mesh, (void **) &coordBundle);CHKERRQ(ierr);
    ierr = WriteVTKVertices(mesh, f);CHKERRQ(ierr);
    ierr = WriteVTKElements(mesh, f);CHKERRQ(ierr);
  } else {
    ALE::Sieve *topology;
    PetscInt dim, d;

    ierr = MeshGetDimension(mesh, &dim);CHKERRQ(ierr);
    ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(d = 0; d < dim; d++) {
      ALE::ClosureBundle dBundle;

      dBundle.setTopology(topology);
      dBundle.setFiberDimensionByDepth(d, 1);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", dBundle.getGlobalSize(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(Mesh mesh, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = MeshView_Sieve_Ascii(mesh, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Mesh");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Mesh");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

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
#define __FUNCT__ "MeshView"
/*@C
   MeshView - Views a Mesh object. 

   Collective on Mesh

   Input Parameters:
+  mesh - the mesh
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the mesh is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs vector to a specified file
.    PetscViewerBinaryOpen() - Outputs vector in binary to a
         specified file; corresponding input uses MeshLoad()
.    PetscViewerDrawOpen() - Outputs vector to an X window display

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints mesh information
-    PETSC_VIEWER_ASCII_VTK - outputs a VTK file describing the mesh

   Level: beginner

   Concepts: mesh^printing
   Concepts: mesh^saving to disk

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerBinaryOpen(),
          MeshLoad(), PetscViewerCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh mesh, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidType(mesh, 1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(mesh->comm);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  ierr = (*mesh->ops->view)(mesh, viewer);CHKERRQ(ierr);
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
  p->ops->view               = MeshView_Sieve;
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->getmatrix          = MeshGetMatrix;

  ierr = PetscObjectChangeTypeName((PetscObject) p, "sieve");CHKERRQ(ierr);

  p->topology       = PETSC_NULL;
  p->boundary       = PETSC_NULL;
  p->orientation    = PETSC_NULL;
  p->spaceFootprint = PETSC_NULL;
  p->bundle         = PETSC_NULL;
  p->coordBundle    = PETSC_NULL;
  p->coordinates    = PETSC_NULL;
  p->globalvector   = PETSC_NULL;
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
  if (mesh->spaceFootprint) {
    /* delete (ALE::Stack *) p->spaceFootprint; */
  }
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

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinateBundle"
/*@C
    MeshGetCoordinateBundle - Gets the coordinate bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the coordinate bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetCoordinateBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinateBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->coordBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetCoordinateBundle"
/*@C
    MeshSetCoordinateBundle - Sets the coordinate bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the coordinate bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetCoordinateBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinateBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->coordBundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetOrientation"
/*@C
    MeshGetOrientation - Gets the orientation sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    orientation - the orientation sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetOrientation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetOrientation(Mesh mesh,void **orientation)
{
  if (orientation) {
    PetscValidPointer(orientation,2);
    *orientation = mesh->orientation;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetOrientation"
/*@C
    MeshOrientation - Sets the orientation sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    orientation - the orientation sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetOrientation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetOrientation(Mesh mesh,void *orientation)
{
  PetscValidPointer(orientation,2);
  mesh->orientation = orientation;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinates"
/*@C
    MeshGetCoordinates - Gets the coordinate vector

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    coordinates - the coordinate vector
 
    Level: advanced

.seealso MeshCreate(), MeshSetCoordinates()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh mesh, Vec *coordinates)
{
  if (coordinates) {
    PetscValidPointer(coordinates,2);
    *coordinates = mesh->coordinates;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetCoordinates"
/*@C
    MeshCoordinates - Sets the coordinate vector

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    coordinates - the coordinate vector
 
    Level: advanced

.seealso MeshCreate(), MeshGetCoordinates()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinates(Mesh mesh, Vec coordinates)
{
  PetscValidPointer(coordinates,2);
  mesh->coordinates = coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSpaceFootprint"
/*@C
    MeshGetSpaceFootprint - Gets the stack endcoding element overlap

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    spaceFootprint - the overlap stack
 
    Level: advanced

.seealso MeshCreate(), MeshSetSpaceFootprint()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetSpaceFootprint(Mesh mesh, void **spaceFootprint)
{
  if (spaceFootprint) {
    PetscValidPointer(spaceFootprint,2);
    *spaceFootprint = mesh->spaceFootprint;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSpaceFootprint"
/*@C
    MeshSpaceFootprint - Sets the stack endcoding element overlap

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    spaceFootprint - the overlap stack
 
    Level: advanced

.seealso MeshCreate(), MeshGetSpaceFootprint()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetSpaceFootprint(Mesh mesh, void *spaceFootprint)
{
  PetscValidPointer(spaceFootprint,2);
  mesh->spaceFootprint = spaceFootprint;
  PetscFunctionReturn(0);
}
