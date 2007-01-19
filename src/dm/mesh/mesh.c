 
#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include <Distribution.hh>
#include <Generator.hh>
#include "src/dm/mesh/meshvtk.h"
#include "src/dm/mesh/meshpcice.h"
#include "src/dm/mesh/meshpylith.h"

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE = 0;
PetscEvent  Mesh_View = 0, Mesh_GetGlobalScatter = 0, Mesh_restrictVector = 0, Mesh_assembleVector = 0,
            Mesh_assembleVectorComplete = 0, Mesh_assembleMatrix = 0, Mesh_updateOperator = 0;

EXTERN PetscErrorCode MeshView_DM(Mesh, PetscViewer);
EXTERN PetscErrorCode MeshRefine_DM(Mesh, MPI_Comm, Mesh *);
EXTERN PetscErrorCode MeshCoarsenHierarchy_DM(Mesh, int, Mesh **);
EXTERN PetscErrorCode MeshGetInterpolation_DM(DM, DM, Mat *, Vec *);

#undef __FUNCT__  
#define __FUNCT__ "MeshFinalize"
PetscErrorCode MeshFinalize()
{
  PetscFunctionBegin;
  ALE::Mesh::NumberingFactory::singleton(0, true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, 0, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, 0, viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH) {
    char *filename;
    char  connectFilename[2048];
    char  coordFilename[2048];

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscStrcpy(connectFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(connectFilename, ".connect");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, connectFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeElements(mesh, mesh->getIntSection("material"), viewer);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(coordFilename, ".coord");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename), PETSC_ERR_FILE_OPEN);
    if (PetscExceptionValue(ierr)) {
      /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
    } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
      ierr = 0;
    } 
    CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH_LOCAL) {
    PetscViewer connectViewer, coordViewer, splitViewer, tractionViewer;
    char       *filename;
    char        localFilename[2048];
    int         rank = mesh->commRank();

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.connect", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(connectViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(connectViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(connectViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeElementsLocal(mesh, mesh->getIntSection("material"), connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(connectViewer);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.coord", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(coordViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(coordViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(coordViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeVerticesLocal(mesh, coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(coordViewer);CHKERRQ(ierr);

    if (mesh->hasPairSection("split")) {
      sprintf(localFilename, "%s.%d.split", filename, rank);
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(splitViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(splitViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(splitViewer, localFilename);CHKERRQ(ierr);
      ierr = ALE::PyLith::Viewer::writeSplitLocal(mesh, mesh->getPairSection("split"), splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(splitViewer);CHKERRQ(ierr);
    }

    if (mesh->hasRealSection("traction")) {
      sprintf(localFilename, "%s.%d.traction", filename, rank);
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &tractionViewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(tractionViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(tractionViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(tractionViewer, localFilename);CHKERRQ(ierr);
      ierr = ALE::PyLith::Viewer::writeTractionsLocal(mesh, mesh->getRealSection("traction"), tractionViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(tractionViewer);CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    char      *filename;
    char       coordFilename[2048];
    PetscTruth isConnect;
    size_t     len;

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
    ierr = PetscStrcmp(&(filename[len-5]), ".lcon", &isConnect);CHKERRQ(ierr);
    if (!isConnect) {
      SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid element connectivity filename: %s", filename);
    }
    ierr = ALE::PCICE::Viewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrncpy(coordFilename, filename, len-5);CHKERRQ(ierr);
    coordFilename[len-5] = '\0';
    ierr = PetscStrcat(coordFilename, ".nodes");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALE::PCICE::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(int d = 0; d <= dim; d++) {
      // FIX: Need to globalize
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->getTopology()->depthStratum(0, d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer)
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

PetscErrorCode PETSCDM_DLLEXPORT MeshView_DM(Mesh mesh, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshView_Sieve(mesh->m, viewer);CHKERRQ(ierr);
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
+    PetscViewerASCIIOpen() - Outputs mesh to a specified file
.    PetscViewerBinaryOpen() - Outputs mesh in binary to a
         specified file; corresponding input uses MeshLoad()
.    PetscViewerDrawOpen() - Outputs mesh to an X window display

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
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidType(mesh, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(mesh->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  ierr = PetscLogEventBegin(Mesh_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*mesh->ops->view)(mesh, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_View,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshLoad" 
/*@C
    MeshLoad - Create a mesh topology from the saved data in a viewer.

    Collective on Viewer

    Input Parameter:
.   viewer - The viewer containing the data

    Output Parameters:
.   mesh - the mesh object

    Level: advanced

.seealso MeshView()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshLoad(PetscViewer viewer, Mesh *mesh)
{
  SETERRQ(PETSC_ERR_SUP, "");
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetMesh"
/*@C
    MeshGetMesh - Gets the internal mesh object

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshSetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh mesh, ALE::Obj<ALE::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  m = mesh->m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetMesh"
/*@C
    MeshSetMesh - Sets the internal mesh object

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshGetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh mesh, const ALE::Obj<ALE::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->m = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateMatrix" 
template<typename Atlas>
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(Mesh mesh, const Obj<Atlas>& atlas, MatType mtype, Mat *J)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::order_type>& order = m->getFactory()->getGlobalOrder(m->getTopology(), 0, "default", atlas);
  int localSize  = order->getLocalSize();
  int globalSize = order->getGlobalSize();

  PetscFunctionBegin;
  ierr = MatCreate(m->comm(), J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, localSize, localSize, globalSize, globalSize);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
  //ierr = MatSetBlockSize(*J, 1);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *J, "mesh", (PetscObject) mesh);

  ierr = preallocateOperator(m->getTopology(), atlas, order, *J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexMatrix" 
PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexMatrix(Mesh mesh, MatType mtype, Mat *J)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::real_section_type> s = new ALE::Mesh::real_section_type(m->getTopology());
  s->setFiberDimensionByDepth(0, 0, 1);
  s->allocate();
  ierr = MeshCreateMatrix(mesh, s->getAtlas(), mtype, J);CHKERRQ(ierr);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh mesh, MatType mtype, Mat *J)
{
  ALE::Obj<ALE::Mesh> m;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(mesh, m->getRealSection("default")->getAtlas(), mtype, J);CHKERRQ(ierr);
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

  ierr = PetscHeaderCreate(p,_p_Mesh,struct _MeshOps,MESH_COOKIE,0,"Mesh",comm,MeshDestroy,0);CHKERRQ(ierr);
  p->ops->view               = MeshView_DM;
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->getcoloring        = PETSC_NULL;
  p->ops->getmatrix          = MeshGetMatrix;
  p->ops->getinterpolation   = MeshGetInterpolation_DM;
  p->ops->getinjection       = PETSC_NULL;
  p->ops->refine             = MeshRefine_DM;
  p->ops->coarsen            = PETSC_NULL;
  p->ops->refinehierarchy    = PETSC_NULL;
  p->ops->coarsenhierarchy   = MeshCoarsenHierarchy_DM;

  ierr = PetscObjectChangeTypeName((PetscObject) p, "sieve");CHKERRQ(ierr);

  p->m             = PETSC_NULL;
  p->globalScatter = PETSC_NULL;
  p->lf            = PETSC_NULL;
  p->lj            = PETSC_NULL;
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
  if (mesh->globalScatter) {ierr = VecScatterDestroy(mesh->globalScatter);CHKERRQ(ierr);}
  mesh->m = PETSC_NULL;
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval"
inline void ExpandInterval(const ALE::Point& interval, int indices[], int& indx)
{
  const int end = interval.prefix + interval.index;
  for(int i = interval.index; i < end; i++) {
    indices[indx++] = i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval_New"
inline void ExpandInterval_New(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.prefix; i++) {
    indices[(*indx)++] = interval.index + i;
  }
  for(int i = 0; i < -interval.prefix; i++) {
    indices[(*indx)++] = -1;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::Mesh::real_section_type::IndexArray> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Mesh::real_section_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    ExpandInterval_New(*i_itor, indices, &k);
  }
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
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh mesh, Vec *gvec)
{
  ALE::Obj<ALE::Mesh> m;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::order_type>& order = m->getFactory()->getGlobalOrder(m->getTopology(), 0, "default", m->getRealSection("default")->getAtlas());

  ierr = VecCreate(m->comm(), gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateLocalVector"
/*@C
  MeshCreateLocalVector - Creates a vector with the local piece of the Section

  Collective on Mesh

  Input Parameters:
+ mesh - the Mesh
- section - the Section  

  Output Parameter:
. localVec - the local vector

  Level: advanced

.seealso MeshDestroy(), MeshCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateLocalVector(Mesh mesh, SectionReal section, Vec *localVec)
{
  ALE::Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, s->size(0), s->restrict(0), localVec);CHKERRQ(ierr);
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
  SETERRQ(PETSC_ERR_SUP, "");
}

#undef __FUNCT__
#define __FUNCT__ "MeshCreateGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(Mesh mesh, SectionReal section, VecScatter *scatter)
{
  typedef ALE::Mesh::real_section_type::index_type index_type;
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::topology_type>&                   topology = m->getTopology();
  const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>&   atlas    = s->getAtlas();
  const ALE::Mesh::real_section_type::patch_type              patch    = 0;
  const ALE::Mesh::real_section_type::atlas_type::chart_type& chart    = atlas->getPatch(patch);
  const ALE::Obj<ALE::Mesh::order_type>& globalOrder = m->getFactory()->getGlobalOrder(topology, patch, name, atlas);
  int *localIndices, *globalIndices;
  int  localSize = s->size(patch);
  int  localIndx = 0, globalIndx = 0;
  Vec  globalVec, localVec;
  IS   localIS, globalIS;

  ierr = VecCreate(m->comm(), &globalVec);CHKERRQ(ierr);
  ierr = VecSetSizes(globalVec, globalOrder->getLocalSize(), PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(globalVec);CHKERRQ(ierr);
  // Loop over all local points
  ierr = PetscMalloc(localSize*sizeof(int), &localIndices); CHKERRQ(ierr);
  ierr = PetscMalloc(localSize*sizeof(int), &globalIndices); CHKERRQ(ierr);
  for(ALE::Mesh::real_section_type::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
    const ALE::Mesh::real_section_type::index_type& idx = atlas->restrictPoint(patch, *p_iter)[0];

    // Map local indices to global indices
    ExpandInterval(idx, localIndices, localIndx);
    ExpandInterval(index_type(idx.prefix, globalOrder->getIndex(*p_iter)), globalIndices, globalIndx);
  }
  if (localIndx  != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of local indices %d, should be %d", localIndx, localSize);
  if (globalIndx != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of global indices %d, should be %d", globalIndx, localSize);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, localIndices,  &localIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, globalIndices, &globalIS);CHKERRQ(ierr);
  ierr = PetscFree(localIndices);CHKERRQ(ierr);
  ierr = PetscFree(globalIndices);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, localSize, s->restrict(patch), &localVec);CHKERRQ(ierr);
  ierr = VecScatterCreate(localVec, localIS, globalVec, globalIS, scatter);CHKERRQ(ierr);
  ierr = ISDestroy(globalIS);CHKERRQ(ierr);
  ierr = ISDestroy(localIS);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = VecDestroy(globalVec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(Mesh mesh, VecScatter *scatter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidPointer(scatter, 2);
  if (!mesh->globalScatter) {
    SectionReal section;

    ierr = MeshGetSectionReal(mesh, "default", &section);CHKERRQ(ierr);
    ierr = MeshCreateGlobalScatter(mesh, section, &mesh->globalScatter);CHKERRQ(ierr);
    ierr = SectionRealDestroy(section);CHKERRQ(ierr);
  }
  *scatter = mesh->globalScatter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetLocalFunction"
PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalFunction(Mesh mesh, PetscErrorCode (*lf)(Mesh, SectionReal, SectionReal, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetLocalJacobian"
PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalJacobian(Mesh mesh, PetscErrorCode (*lj)(Mesh, SectionReal, Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshFormFunction"
PetscErrorCode PETSCDM_DLLEXPORT MeshFormFunction(Mesh mesh, SectionReal X, SectionReal F, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_COOKIE, 2);
  PetscValidHeaderSpecific(F, SECTIONREAL_COOKIE, 3);
  if (mesh->lf) {
    ierr = (*mesh->lf)(mesh, X, F, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshFormJacobian"
PetscErrorCode PETSCDM_DLLEXPORT MeshFormJacobian(Mesh mesh, SectionReal X, Mat J, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_COOKIE, 2);
  PetscValidHeaderSpecific(J, MAT_COOKIE, 3);
  if (mesh->lj) {
    ierr = (*mesh->lj)(mesh, X, J, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshInterpolatePoints"
// Here we assume:
//  - Assumes 3D and tetrahedron
//  - The section takes values on vertices and is P1
//  - Points have the same dimension as the mesh
//  - All values have the same dimension
PetscErrorCode PETSCDM_DLLEXPORT MeshInterpolatePoints(Mesh mesh, SectionReal section, int numPoints, double *points, double **values)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  double        *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  int embedDim = coordinates->getFiberDimension(0, *m->getTopology()->depthStratum(0, 0)->begin());
  int dim      = s->getFiberDimension(0, *m->getTopology()->depthStratum(0, 0)->begin());

  ierr = PetscMalloc3(embedDim,double,&v0,embedDim*embedDim,double,&J,embedDim*embedDim,double,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints*dim * sizeof(double), &values);CHKERRQ(ierr);
  for(int p = 0; p < numPoints; p++) {
    double *point = &points[p*embedDim];
    ALE::Mesh::point_type e = m->locatePoint(0, point);
    const ALE::Mesh::real_section_type::value_type *coeff = s->restrict(0, e);

    m->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    double xi   = (invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]))*0.5;
    double eta  = (invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]))*0.5;
    double zeta = (invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]))*0.5;

    for(int d = 0; d < dim; d++) {
      (*values)[p*dim+d] = coeff[0*dim+d]*(1 - xi - eta - zeta) + coeff[1*dim+d]*xi + coeff[2*dim+d]*eta + coeff[3*dim+d]*zeta;
    }
  }
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode assembleFullField(VecScatter, Vec, Vec, InsertMode);

#undef __FUNCT__
#define __FUNCT__ "restrictVector"
/*@C
  restrictVector - Insert values from a global vector into a local ghosted vector

  Collective on g

  Input Parameters:
+ g - The global vector
. l - The local vector
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode restrictVector(Vec g, Vec l, InsertMode mode)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(g, l, mode, SCATTER_REVERSE, injection);
    ierr = VecScatterEnd(g, l, mode, SCATTER_REVERSE, injection);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(g, l);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(l, 1.0, g);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(Mesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVectorComplete"
/*@C
  assembleVectorComplete - Insert values from a local ghosted vector into a global vector

  Collective on g

  Input Parameters:
+ g - The global vector
. l - The local vector
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleVectorComplete(Vec g, Vec l, InsertMode mode)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
    ierr = VecScatterEnd(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(l, g);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(g, 1.0, l);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(Mesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVector"
/*@C
  assembleVector - Insert values into a vector

  Collective on A

  Input Parameters:
+ b - the vector
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: VecSetOption()
@*/
PetscErrorCode assembleVector(Vec b, PetscInt e, PetscScalar v[], InsertMode mode)
{
  Mesh                     mesh;
  ALE::Obj<ALE::Mesh> m;
  ALE::Mesh::real_section_type::patch_type patch;
  PetscInt                 firstElement;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
  // Must relate b to field
  if (mode == INSERT_VALUES) {
    m->getRealSection(std::string("x"))->update(patch, ALE::Mesh::point_type(e + firstElement), v);
  } else {
    m->getRealSection(std::string("x"))->updateAdd(patch, ALE::Mesh::point_type(e + firstElement), v);
  }
  ierr = PetscLogEventEnd(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<ALE::Mesh::real_section_type>& section, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, const ALE::Mesh::point_type& e, PetscScalar array[], InsertMode mode)
{
  ALE::Mesh::real_section_type::patch_type patch = 0;
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  const ALE::Obj<ALE::Mesh::real_section_type::IndexArray> intervals = section->getIndices(patch, e, globalOrder);

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (section->debug()) {printf("[%d]mat for element %d\n", section->commRank(), e);}
  for(ALE::Mesh::real_section_type::IndexArray::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    numIndices += std::abs(i_iter->prefix);
    if (section->debug()) {
      printf("[%d]mat interval (%d, %d)\n", section->commRank(), i_iter->prefix, i_iter->index);
    }
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  if (section->debug()) {
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]", section->commRank());
      for(int j = 0; j < numIndices; j++) {
        printf(" %g", array[i*numIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: point %d\n", section->commRank(), e);
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperatorGeneral"
PetscErrorCode updateOperatorGeneral(Mat A, const ALE::Obj<ALE::Mesh::real_section_type>& rowSection, const ALE::Obj<ALE::Mesh::order_type>& rowGlobalOrder, const ALE::Mesh::point_type& rowE, const ALE::Obj<ALE::Mesh::real_section_type>& colSection, const ALE::Obj<ALE::Mesh::order_type>& colGlobalOrder, const ALE::Mesh::point_type& colE, PetscScalar array[], InsertMode mode)
{
  ALE::Mesh::real_section_type::patch_type patch = 0;
  static PetscInt  rowIndicesSize = 0;
  static PetscInt *rowIndices = NULL;
  PetscInt         numRowIndices = 0;
  static PetscInt  colIndicesSize = 0;
  static PetscInt *colIndices = NULL;
  PetscInt         numColIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  const ALE::Obj<ALE::Mesh::real_section_type::IndexArray> rowIntervals = rowSection->getIndices(patch, rowE, rowGlobalOrder);
  const ALE::Obj<ALE::Mesh::real_section_type::IndexArray> colIntervals = colSection->getIndices(patch, colE, colGlobalOrder);

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (rowSection->debug()) {printf("[%d]mat for elements %d %d\n", rowSection->commRank(), rowE, colE);}
  for(ALE::Mesh::real_section_type::IndexArray::iterator i_iter = rowIntervals->begin(); i_iter != rowIntervals->end(); ++i_iter) {
    numRowIndices += std::abs(i_iter->prefix);
    if (rowSection->debug()) {
      printf("[%d]mat row interval (%d, %d)\n", rowSection->commRank(), i_iter->prefix, i_iter->index);
    }
  }
  if (rowIndicesSize && (rowIndicesSize != numRowIndices)) {
    ierr = PetscFree(rowIndices); CHKERRQ(ierr);
    rowIndices = NULL;
  }
  if (!rowIndices) {
    rowIndicesSize = numRowIndices;
    ierr = PetscMalloc(rowIndicesSize * sizeof(PetscInt), &rowIndices); CHKERRQ(ierr);
  }
  ierr = ExpandIntervals(rowIntervals, rowIndices); CHKERRQ(ierr);
  if (rowSection->debug()) {
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]mat row indices[%d] = %d\n", rowSection->commRank(), i, rowIndices[i]);
    }
  }
  for(ALE::Mesh::real_section_type::IndexArray::iterator i_iter = colIntervals->begin(); i_iter != colIntervals->end(); ++i_iter) {
    numColIndices += std::abs(i_iter->prefix);
    if (colSection->debug()) {
      printf("[%d]mat col interval (%d, %d)\n", colSection->commRank(), i_iter->prefix, i_iter->index);
    }
  }
  if (colIndicesSize && (colIndicesSize != numColIndices)) {
    ierr = PetscFree(colIndices); CHKERRQ(ierr);
    colIndices = NULL;
  }
  if (!colIndices) {
    colIndicesSize = numColIndices;
    ierr = PetscMalloc(colIndicesSize * sizeof(PetscInt), &colIndices); CHKERRQ(ierr);
  }
  ierr = ExpandIntervals(colIntervals, colIndices); CHKERRQ(ierr);
  if (colSection->debug()) {
    for(int i = 0; i < numColIndices; i++) {
      printf("[%d]mat col indices[%d] = %d\n", colSection->commRank(), i, colIndices[i]);
    }
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]", rowSection->commRank());
      for(int j = 0; j < numColIndices; j++) {
        printf(" %g", array[i*numColIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numRowIndices, rowIndices, numColIndices, colIndices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: points %d %d\n", colSection->commRank(), rowE, colE);
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]mat row indices[%d] = %d\n", rowSection->commRank(), i, rowIndices[i]);
    }
    for(int i = 0; i < numColIndices; i++) {
      printf("[%d]mat col indices[%d] = %d\n", colSection->commRank(), i, colIndices[i]);
    }
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleMatrix"
/*@C
  assembleMatrix - Insert values into a matrix

  Collective on A

  Input Parameters:
+ A - the matrix
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleMatrix(Mat A, PetscInt e, PetscScalar v[], InsertMode mode)
{
  Mesh           mesh;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  try {
    const ALE::Obj<ALE::Mesh::topology_type>&     topology    = m->getTopology();
    const ALE::Obj<ALE::Mesh::numbering_type>&    cNumbering  = m->getFactory()->getLocalNumbering(topology, 0, topology->depth());
    const ALE::Obj<ALE::Mesh::real_section_type>& section     = m->getRealSection("default");
    const ALE::Obj<ALE::Mesh::order_type>&        globalOrder = m->getFactory()->getGlobalOrder(topology, 0, "default", section->getAtlas());

    if (section->debug()) {
      std::cout << "Assembling matrix for element number " << e << " --> point " << cNumbering->getPoint(e) << std::endl;
    }
    ierr = updateOperator(A, section, globalOrder, cNumbering->getPoint(e), v, mode);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscLogEventEnd(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateOperator"
template<typename Atlas>
PetscErrorCode preallocateOperator(const ALE::Obj<ALE::Mesh::topology_type>& topology, const ALE::Obj<Atlas>& atlas, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, Mat A)
{
  typedef ALE::New::NumberingFactory<ALE::Mesh::topology_type> NumberingFactory;
  const ALE::Obj<ALE::Mesh::sieve_type>     adjGraph    = new ALE::Mesh::sieve_type(topology->comm(), topology->debug());
  const ALE::Obj<ALE::Mesh::topology_type>  adjTopology = new ALE::Mesh::topology_type(topology->comm(), topology->debug());
  const ALE::Mesh::real_section_type::patch_type patch  = 0;
  const ALE::Obj<ALE::Mesh::sieve_type>&    sieve       = topology->getPatch(patch);
  PetscInt       numLocalRows, firstRow;
  PetscInt      *dnz, *onz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjTopology->setPatch(patch, adjGraph);
  numLocalRows = globalOrder->getLocalSize();
  firstRow     = globalOrder->getGlobalOffsets()[topology->commRank()];
  ierr = PetscMalloc2(numLocalRows, PetscInt, &dnz, numLocalRows, PetscInt, &onz);CHKERRQ(ierr);
  /* Create local adjacency graph */
  /*   In general, we need to get FIAT info that attaches dual basis vectors to sieve points */
  const ALE::Mesh::real_section_type::atlas_type::chart_type& chart = atlas->getPatch(patch);

  for(ALE::Mesh::real_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const ALE::Mesh::real_section_type::atlas_type::point_type& point = *c_iter;

    adjGraph->addCone(sieve->cone(sieve->support(point)), point);
  }
  /* Distribute adjacency graph */
  topology->constructOverlap(patch);
  const Obj<ALE::Mesh::send_overlap_type>& vertexSendOverlap = topology->getSendOverlap();
  const Obj<ALE::Mesh::recv_overlap_type>& vertexRecvOverlap = topology->getRecvOverlap();
  const Obj<ALE::Mesh::send_overlap_type>  nbrSendOverlap    = new ALE::Mesh::send_overlap_type(topology->comm(), topology->debug());
  const Obj<ALE::Mesh::recv_overlap_type>  nbrRecvOverlap    = new ALE::Mesh::recv_overlap_type(topology->comm(), topology->debug());
  const Obj<ALE::Mesh::send_section_type>  sendSection       = new ALE::Mesh::send_section_type(topology->comm(), topology->debug());
  const Obj<ALE::Mesh::recv_section_type>  recvSection       = new ALE::Mesh::recv_section_type(topology->comm(), sendSection->getTag(), topology->debug());

  ALE::New::Distribution<ALE::Mesh::topology_type>::coneCompletion(vertexSendOverlap, vertexRecvOverlap, adjTopology, sendSection, recvSection);
  /* Distribute indices for new points */
  ALE::New::Distribution<ALE::Mesh::topology_type>::updateOverlap(sendSection, recvSection, nbrSendOverlap, nbrRecvOverlap);
  NumberingFactory::singleton(topology->debug())->completeOrder(globalOrder, nbrSendOverlap, nbrRecvOverlap, patch, true);
  /* Read out adjacency graph */
  const ALE::Obj<ALE::Mesh::sieve_type> graph = adjTopology->getPatch(patch);

  ierr = PetscMemzero(dnz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  for(ALE::Mesh::real_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const ALE::Mesh::real_section_type::atlas_type::point_type& point = *c_iter;

    if (globalOrder->isLocal(point)) {
      const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& adj   = graph->cone(point);
      const ALE::Mesh::order_type::value_type&                rIdx  = globalOrder->restrictPoint(patch, point)[0];
      const int                                               row   = rIdx.prefix;
      const int                                               rSize = rIdx.index;

      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator v_iter = adj->begin(); v_iter != adj->end(); ++v_iter) {
        const ALE::Mesh::real_section_type::atlas_type::point_type& neighbor = *v_iter;
        const ALE::Mesh::order_type::value_type& cIdx     = globalOrder->restrictPoint(patch, neighbor)[0];
        const int&                               cSize    = cIdx.index;

        if (cSize > 0) {
          if (globalOrder->isLocal(neighbor)) {
            for(int r = 0; r < rSize; ++r) {dnz[row - firstRow + r] += cSize;}
          } else {
            for(int r = 0; r < rSize; ++r) {onz[row - firstRow + r] += cSize;}
          }
        }
      }
    }
  }
  if (topology->debug()) {
    int rank = topology->commRank();
    for(int r = 0; r < numLocalRows; r++) {
      std::cout << "["<<rank<<"]: dnz["<<r<<"]: " << dnz[r] << " onz["<<r<<"]: " << onz[r] << std::endl;
    }
  }
  ierr = MatSeqAIJSetPreallocation(A, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateMatrix"
PetscErrorCode preallocateMatrix(const ALE::Obj<ALE::Mesh::topology_type>& topology, const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>& atlas, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, Mat A)
{
  return preallocateOperator(topology, atlas, globalOrder, A);
}

/******************************** C Wrappers **********************************/

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKHeader"
PetscErrorCode WriteVTKHeader(PetscViewer viewer)
{
  return VTKViewer::writeHeader(viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKVertices"
PetscErrorCode WriteVTKVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeVertices(m, 0, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeElements(m, 0, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICERestart"
PetscErrorCode WritePCICERestart(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeRestart(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePCICE"
/*@C
  MeshCreatePCICE - Create a Mesh from PCICE files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. coordFilename - The file containing vertex coordinates
. adjFilename - The file containing the vertices for each element
. interpolate - The flag for construction of intermediate elements
. bcFilename - The file containing the boundary topology and conditions
. numBdFaces - The number of boundary faces (or edges)
- numBdVertices - The number of boundary vertices

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, PCICE
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreatePCICE(MPI_Comm comm, const int dim, const char coordFilename[], const char adjFilename[], PetscTruth interpolate, const char bcFilename[], const int numBdFaces, const int numBdVertices, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PCICE::Builder::readMesh(comm, dim, std::string(coordFilename), std::string(adjFilename), false, interpolate, debug);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
  if (bcFilename) {
    ALE::PCICE::Builder::readBoundary(m, std::string(bcFilename), numBdFaces, numBdVertices);
  }
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePyLith"
/*@C
  MeshCreatePyLith - Create a Mesh from PyLith files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. baseFilename - The basename for mesh files
. zeroBase - Use 0 to start numbering
- interpolate - The flag for mesh interpolation

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, PCICE
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreatePyLith(MPI_Comm comm, const int dim, const char baseFilename[], PetscTruth zeroBase, PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PyLith::Builder::readMesh(comm, dim, std::string(baseFilename), zeroBase, interpolate, debug);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinates"
/*@C
  MeshGetCoordinates - Creates an array holding the coordinates.

  Not Collective

  Input Parameter:
+ mesh - The Mesh object
- columnMajor - Flag for column major order

  Output Parameter:
+ numVertices - The number of vertices
. dim - The embedding dimension
- coords - The array holding local coordinates

  Level: intermediate

.keywords: mesh, coordinates
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetCoordinates(Mesh mesh, PetscTruth columnMajor, PetscInt *numVertices, PetscInt *dim, PetscReal *coords[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputVerticesLocal(m, numVertices, dim, coords, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetElements"
/*@C
  MeshGetElements - Creates an array holding the vertices on each element.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- columnMajor - Flag for column major order

  Output Parameters:
+ numElements - The number of elements
. numCorners - The number of vertices per element
- vertices - The array holding vertices on each local element

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetElements(Mesh mesh, PetscTruth columnMajor, PetscInt *numElements, PetscInt *numCorners, PetscInt *vertices[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputElementsLocal(m, numElements, numCorners, vertices, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDistribute"
/*@C
  MeshDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
. serialMesh - The original Mesh object

  Output Parameter:
. parallelMesh - The distributed Mesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshDistribute(Mesh serialMesh, Mesh *parallelMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh> newMesh = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeMesh(oldMesh);
  ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGenerate"
/*@C
  MeshGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The Mesh boundary object
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The Mesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshRefine()
@*/
PetscErrorCode MeshGenerate(Mesh boundary, PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> mB;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(boundary, mB);CHKERRQ(ierr);
  ierr = MeshCreate(mB->comm(), mesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh> m = ALE::Generator::generateMesh(mB, interpolate);
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine"
/*@C
  MeshRefine - Refines the mesh.

  Not Collective

  Input Parameters:
+ mesh - The original Mesh object
. refinementLimit - The maximum size of any cell
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. refinedMesh - The refined Mesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshGenerate()
@*/
PetscErrorCode MeshRefine(Mesh mesh, double refinementLimit, PetscTruth interpolate, Mesh *refinedMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, interpolate);
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine_DM"
PetscErrorCode MeshRefine_DM(Mesh mesh, MPI_Comm comm, Mesh *refinedMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  double              refinementLimit;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(comm, refinedMesh);CHKERRQ(ierr);
  refinementLimit = oldMesh->getMaxVolume()/2.0;
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, false);
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type>& s = newMesh->getRealSection("default");

  newMesh->setDiscretization(oldMesh->getDiscretization());
  newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
  newMesh->setupField(s);
  PetscFunctionReturn(0);
}

#include "src/dm/mesh/examples/tutorials/Coarsener.h"
#include "src/dm/mesh/examples/tutorials/fast_coarsen.h"

#undef __FUNCT__  
#define __FUNCT__ "MeshCoarsenHierarchy"
/*@C
  MeshCoarsenHierarchy - Coarsens the mesh into a hierarchy.

  Not Collective

  Input Parameters:
+ mesh - The original Mesh object
. numLevels - The number of 
. coarseningFactor - The expansion factor for coarse meshes
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. coarseHierarchy - The coarse Mesh objects

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshGenerate()
@*/
PetscErrorCode MeshCoarsenHierarchy(Mesh mesh, int numLevels, double coarseningFactor, PetscTruth interpolate, Mesh **coarseHierarchy)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (numLevels < 1) {
    *coarseHierarchy = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  if (oldMesh->getDimension() != 2) SETERRQ(PETSC_ERR_SUP, "Coarsening only works in two dimensions right now");
  ierr = ALE::Coarsener::IdentifyBoundary(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::make_coarsest_boundary(oldMesh, 2, numLevels+1);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateSpacingFunction(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateCoarsenedHierarchyNew(oldMesh, 2, numLevels, coarseningFactor);CHKERRQ(ierr);
  ierr = PetscMalloc(numLevels * sizeof(Mesh),coarseHierarchy);CHKERRQ(ierr);
  for(int l = 0; l < numLevels; l++) {
    ALE::Obj<ALE::Mesh> newMesh = new ALE::Mesh(oldMesh->comm(), oldMesh->debug());
    const ALE::Obj<ALE::Mesh::real_section_type>& s = newMesh->getRealSection("default");

    ierr = MeshCreate(oldMesh->comm(), &(*coarseHierarchy)[l]);CHKERRQ(ierr);
    newMesh->getTopology()->setPatch(0, oldMesh->getTopology()->getPatch(l+1));
    newMesh->setDiscretization(oldMesh->getDiscretization());
    newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
    newMesh->setupField(s);
    ierr = MeshSetMesh((*coarseHierarchy)[l], newMesh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MeshCoarsenHierarchy_DM(Mesh mesh, int numLevels, Mesh **coarseHierarchy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCoarsenHierarchy(mesh, numLevels, 1.41, PETSC_FALSE, coarseHierarchy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetInterpolation_DM"
/*
  This method only handle P_1 discretizations at present.
*/
PetscErrorCode MeshGetInterpolation_DM(DM dmFine, DM dmCoarse, Mat *interpolation, Vec *scaling)
{
  ALE::Obj<ALE::Mesh> coarse;
  ALE::Obj<ALE::Mesh> fine;
  Mesh coarseMesh = (Mesh) dmCoarse;
  Mesh fineMesh   = (Mesh) dmFine;
  const ALE::Mesh::patch_type patch = 0;
  Mat  P;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(fineMesh,   fine);CHKERRQ(ierr);
  ierr = MeshGetMesh(coarseMesh, coarse);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type>&             coarseCoordinates = coarse->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>&             fineCoordinates   = fine->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& vertices          = fine->getTopology()->depthStratum(patch, 0);
  const ALE::Obj<ALE::Mesh::real_section_type>&             sCoarse           = coarse->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type>&             sFine             = fine->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>& coarseAtlas       = sCoarse->getAtlas();
  const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>& fineAtlas         = sFine->getAtlas();
  const ALE::Obj<ALE::Mesh::order_type>& coarseOrder = coarse->getFactory()->getGlobalOrder(coarse->getTopology(), patch, "default", coarseAtlas);
  const ALE::Obj<ALE::Mesh::order_type>& fineOrder   = fine->getFactory()->getGlobalOrder(fine->getTopology(), patch, "default", fineAtlas);
  const int dim = coarse->getDimension();
  double *v0, *J, *invJ, detJ, *refCoords, *values;

  ierr = MatCreate(fine->comm(), &P);CHKERRQ(ierr);
  ierr = MatSetSizes(P, sFine->size(patch), sCoarse->size(patch), PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ,dim,double,&refCoords,dim+1,double,&values);CHKERRQ(ierr);
  for(ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const ALE::Mesh::real_section_type::value_type *coords     = fineCoordinates->restrict(patch, *v_iter);
    const ALE::Mesh::point_type                     coarseCell = coarse->locatePoint(patch, coords);

    coarse->computeElementGeometry(coarseCoordinates, coarseCell, v0, J, invJ, detJ);
    for(int d = 0; d < dim; ++d) {
      refCoords[d] = 0.0;
      for(int e = 0; e < dim; ++e) {
        refCoords[d] += invJ[d*dim+e]*(coords[e] - v0[e]);
      }
      refCoords[d] -= 1.0;
    }
    values[0] = 1.0/3.0 - (refCoords[0] + refCoords[1])/3.0;
    values[1] = 0.5*(refCoords[0] + 1.0);
    values[2] = 0.5*(refCoords[1] + 1.0);
    ierr = updateOperatorGeneral(P, sFine, fineOrder, *v_iter, sCoarse, coarseOrder, coarseCell, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree5(v0,J,invJ,refCoords,values);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *interpolation = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshHasSectionReal"
/*@C
  MeshHasSectionReal - Determines whether this mesh has a SectionReal with the given name.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionReal is present in the Mesh

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshHasSectionReal(Mesh mesh, const char name[], PetscTruth *flag)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *flag = (PetscTruth) m->hasRealSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionReal"
/*@C
  MeshGetSectionReal - Returns a SectionReal of the given name from the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. section - The SectionReal

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetSectionReal(Mesh mesh, const char name[], SectionReal *section)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSectionReal"
/*@C
  MeshSetSectionReal - Puts a SectionReal of the given name into the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- section - The SectionReal

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshSetSectionReal(Mesh mesh, SectionReal section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  m->setRealSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshHasSectionInt"
/*@C
  MeshHasSectionInt - Determines whether this mesh has a SectionInt with the given name.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionInt is present in the Mesh

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshHasSectionInt(Mesh mesh, const char name[], PetscTruth *flag)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *flag = (PetscTruth) m->hasIntSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionInt"
/*@C
  MeshGetSectionInt - Returns a SectionInt of the given name from the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. section - The SectionInt

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetSectionInt(Mesh mesh, const char name[], SectionInt *section)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionIntSetSection(*section, m->getIntSection(std::string(name)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSectionInt"
/*@C
  MeshSetSectionInt - Puts a SectionInt of the given name into the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- section - The SectionInt

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshSetSectionInt(Mesh mesh, SectionInt section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  m->setIntSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshHasSectionPair"
/*@C
  MeshHasSectionPair - Determines whether this mesh has a SectionPair with the given name.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionPair is present in the Mesh

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshHasSectionPair(Mesh mesh, const char name[], PetscTruth *flag)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *flag = (PetscTruth) m->hasPairSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionPair"
/*@C
  MeshGetSectionPair - Returns a SectionPair of the given name from the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. section - The SectionPair

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetSectionPair(Mesh mesh, const char name[], SectionPair *section)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionPairCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionPairSetSection(*section, m->getPairSection(std::string(name)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSectionPair"
/*@C
  MeshSetSectionPair - Puts a SectionPair of the given name into the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- section - The SectionPair

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshSetSectionPair(Mesh mesh, SectionPair section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::pair_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionPairGetSection(section, s);CHKERRQ(ierr);
  m->setPairSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionGetArray"
/*@C
  SectionGetArray - Returns the array underlying the Section.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode SectionGetArray(Mesh mesh, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscScalar *array[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>&        section = m->getRealSection(std::string(name));
  const ALE::Mesh::real_section_type::patch_type  patch   = 0;
  if (!section->hasPatch(patch)) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const ALE::Mesh::real_section_type::chart_type& chart   = section->getPatch(patch);
/*   const int                                  depth   = m->getTopology()->depth(patch, *chart.begin()); */
/*   *numElements = m->getTopology()->depthStratum(patch, depth)->size(); */
/*   *fiberDim    = section->getFiberDimension(patch, *chart.begin()); */
/*   *array       = (PetscScalar *) section->restrict(patch); */
  int fiberDimMin = section->getFiberDimension(patch, *chart.begin());
  int numElem     = 0;

  for(ALE::Mesh::real_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(patch, *c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(ALE::Mesh::real_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(patch, *c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscScalar *) section->restrict(patch);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "BCSectionGetArray"
/*@C
  BCSectionGetArray - Returns the array underlying the BCSection.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode BCSectionGetArray(Mesh mesh, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscInt *array[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::int_section_type>&       section = m->getIntSection(std::string(name));
  const ALE::Mesh::int_section_type::patch_type patch   = 0;
  if (!section->hasPatch(patch)) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const ALE::Mesh::int_section_type::chart_type& chart = section->getPatch(patch);
  int fiberDimMin = section->getFiberDimension(patch, *chart.begin());
  int numElem     = 0;

  for(ALE::Mesh::int_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(patch, *c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(ALE::Mesh::int_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(patch, *c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscInt *) section->restrict(patch);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "BCFUNCGetArray"
PetscErrorCode BCFUNCGetArray(Mesh mesh, PetscInt *numElements, PetscInt *fiberDim, PetscScalar *array[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::Mesh::bc_values_type& bcValues = m->getBCValues();
  *numElements = bcValues.size();
  *fiberDim    = 4;
  *array       = new PetscScalar[(*numElements)*(*fiberDim)];
  for(int bcf = 1; bcf <= (int) bcValues.size(); ++bcf) {
    (*array)[(bcf-1)*4+0] = bcValues[bcf].rho;
    (*array)[(bcf-1)*4+1] = bcValues[bcf].u;
    (*array)[(bcf-1)*4+2] = bcValues[bcf].v;
    (*array)[(bcf-1)*4+3] = bcValues[bcf].p;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(material, s);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElements(m, s, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVerticesLocal(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(material, s);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElementsLocal(m, s, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithTractionsLocal"
PetscErrorCode WritePyLithTractionsLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeTractionsLocal(m, m->getRealSection("tractions"), viewer);
}
