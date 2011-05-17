#include <private/meshimpl.h>   /*I      "petscdmmesh.h"   I*/
#include <petscdmmesh_viewers.hh>
#include <petscdmmesh_formats.hh>

/* Logging support */
PetscLogEvent DMMesh_View, DMMesh_GetGlobalScatter, DMMesh_restrictVector, DMMesh_assembleVector, DMMesh_assembleVectorComplete, DMMesh_assembleMatrix, DMMesh_updateOperator;

ALE::MemoryLogger Petsc_MemoryLogger;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMMesh_DelTag"
/*
   Private routine to delete internal tag storage when a communicator is freed.

   This is called by MPI, not by users.

   Note: this is declared extern "C" because it is passed to MPI_Keyval_create

         we do not use PetscFree() since it is unsafe after PetscFinalize()
*/
PetscMPIInt DMMesh_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state)
{
  free(attr_val);
  return(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMMeshFinalize"
PetscErrorCode DMMeshFinalize()
{
  PetscFunctionBegin;
  PETSC_MESH_TYPE::MeshNumberingFactory::singleton(0, 0, true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetMesh"
/*@C
  DMMeshGetMesh - Gets the internal mesh object

  Not collective

  Input Parameter:
. mesh - the mesh object

  Output Parameter:
. m - the internal mesh object

  Level: advanced

.seealso DMMeshCreate(), DMMeshSetMesh()
@*/
PetscErrorCode DMMeshGetMesh(DM dm, ALE::Obj<PETSC_MESH_TYPE>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  m = ((DM_Mesh*) dm->data)->m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetMesh"
/*@C
  DMMeshSetMesh - Sets the internal mesh object

  Not collective

  Input Parameters:
+ mesh - the mesh object
- m - the internal mesh object

  Level: advanced

.seealso DMMeshCreate(), DMMeshGetMesh()
@*/
PetscErrorCode DMMeshSetMesh(DM dm, const ALE::Obj<PETSC_MESH_TYPE>& m)
{
  DM_Mesh        *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->m = m;
  ierr = VecScatterDestroy(&mesh->globalScatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshView_Sieve_Ascii"
PetscErrorCode DMMeshView_Sieve_Ascii(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = VTKViewer::writeHeader(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    const ALE::Obj<PETSC_MESH_TYPE::int_section_type>& p     = mesh->getIntSection("Partition");
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&   cells = mesh->heightStratum(0);
    const PETSC_MESH_TYPE::label_sequence::iterator    end   = cells->end();
    const int                                          rank  = mesh->commRank();

    p->setChart(PETSC_MESH_TYPE::int_section_type::chart_type(*cells));
    p->setFiberDimension(cells, 1);
    p->allocatePoint();
    for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
      p->updatePoint(*c_iter, &rank);
    }
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(mesh, p, "Partition", viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    char      *filename;
    char       coordFilename[2048];
    PetscBool  isConnect;
    size_t     len;

    ierr = PetscViewerFileGetName(viewer, (const char **) &filename);CHKERRQ(ierr);
    ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
    ierr = PetscStrcmp(&(filename[len-5]), ".lcon", &isConnect);CHKERRQ(ierr);
    if (!isConnect) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid element connectivity filename: %s", filename);
    }
    ierr = ALE::PCICE::Viewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrncpy(coordFilename, filename, len-5);CHKERRQ(ierr);
    coordFilename[len-5] = '\0';
    ierr = PetscStrcat(coordFilename, ".nodes");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALE::PCICE::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    mesh->view("");
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(int d = 0; d <= dim; d++) {
      // FIX: Need to globalize
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->depthStratum(d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMeshView_Sieve_Binary"
PetscErrorCode DMMeshView_Sieve_Binary(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
{
  char           *filename;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileGetName(viewer, (const char **) &filename);CHKERRQ(ierr);
  ALE::MeshSerializer::writeMesh(filename, *mesh);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshView_Sieve"
PetscErrorCode DMMeshView_Sieve(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
{
  PetscBool      iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = DMMeshView_Sieve_Ascii(mesh, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = DMMeshView_Sieve_Binary(mesh, viewer);CHKERRQ(ierr);
  } else if (isdraw){
    SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP, "Draw viewer not implemented for DMMesh");
  } else {
    SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Mesh"
PetscErrorCode DMView_Mesh(DM dm, PetscViewer viewer)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshView_Sieve(mesh->m, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshLoad"
/*@
  DMMeshLoad - Create a mesh topology from the saved data in a viewer.

  Collective on Viewer

  Input Parameter:
. viewer - The viewer containing the data

  Output Parameters:
. mesh - the mesh object

  Level: advanced

.seealso DMView()
@*/
PetscErrorCode DMMeshLoad(PetscViewer viewer, DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  char          *filename;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mesh->m) {
    MPI_Comm comm;

    ierr = PetscObjectGetComm((PetscObject) viewer, &comm);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE> m = new PETSC_MESH_TYPE(comm, 1);
    ierr = DMMeshSetMesh(dm, m);CHKERRQ(ierr);
  }
  ierr = PetscViewerFileGetName(viewer, (const char **) &filename);CHKERRQ(ierr);
  ALE::MeshSerializer::loadMesh(filename, *mesh->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateMatrix"
/*@C
  DMMeshCreateMatrix - Creates a matrix with the correct parallel layout required for
    computing the Jacobian on a function defined using the information in the Section.

  Collective on DMMesh

  Input Parameters:
+ mesh    - the mesh object
. section - the section which determines data layout
- mtype   - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

  Output Parameter:
. J  - matrix with the correct nonzero preallocation
       (obviously without the correct Jacobian values)

  Level: advanced

  Notes: This properly preallocates the number of nonzeros in the sparse matrix so you
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DMDASetBlockFills()
@*/
PetscErrorCode DMMeshCreateMatrix(DM dm, SectionReal section, const MatType mtype, Mat *J)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  if (!mtype) mtype = MATAIJ;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  try {
    ierr = DMMeshCreateMatrix(m, s, mtype, J, -1, !dm->prealloc_only);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, e.message());
  }
  ierr = PetscObjectCompose((PetscObject) *J, "DMMesh", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetVertexMatrix"
PetscErrorCode DMMeshGetVertexMatrix(DM dm, const MatType mtype, Mat *J)
{
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetVertexSectionReal(dm, "default", 1, &section);CHKERRQ(ierr);
  ierr = DMMeshCreateMatrix(dm, section, mtype, J);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCellMatrix"
PetscErrorCode DMMeshGetCellMatrix(DM dm, const MatType mtype, Mat *J)
{
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetCellSectionReal(dm, "default", 1, &section);CHKERRQ(ierr);
  ierr = DMMeshCreateMatrix(dm, section, mtype, J);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetMatrix_Mesh"
PetscErrorCode DMGetMatrix_Mesh(DM dm, const MatType mtype, Mat *J)
{
  SectionReal            section;
  ISLocalToGlobalMapping ltog;
  PetscBool              flag;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = DMMeshGetSectionReal(dm, "default", &section);CHKERRQ(ierr);
  ierr = DMMeshCreateMatrix(dm, section, mtype, J);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm, &ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J, ltog, ltog);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Mesh"
PetscErrorCode DMDestroy_Mesh(DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mesh->m = PETSC_NULL;
  ierr = VecScatterDestroy(&mesh->globalScatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Mesh"
PetscErrorCode DMCreateGlobalVector_Mesh(DM dm, Vec *gvec)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(((PetscObject) dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", m->getRealSection("default"));

  ierr = VecCreate(((PetscObject) dm)->comm, gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *gvec, "DMMesh", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateVector"
/*@
  DMMeshCreateVector - Creates a global vector matching the input section

  Collective on DMMesh

  Input Parameters:
+ mesh - the DMMesh
- section - the Section

  Output Parameter:
. vec - the global vector

  Level: advanced

  Notes: The vector can safely be destroyed using VecDestroy().
.seealso DMMeshCreate()
@*/
PetscErrorCode DMMeshCreateVector(DM mesh, SectionReal section, Vec *vec)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, s->getName(), s);

  ierr = VecCreate(m->comm(), vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Mesh"
PetscErrorCode DMCreateLocalVector_Mesh(DM dm, Vec *lvec)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(((PetscObject) dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const int size = m->getRealSection("default")->getStorageSize();

  ierr = VecCreate(PETSC_COMM_SELF, lvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*lvec, size, size);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*lvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *lvec, "DMMesh", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalToGlobalMapping_Mesh"
PetscErrorCode DMCreateLocalToGlobalMapping_Mesh(DM dm)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  SectionReal    section;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(((PetscObject) dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = DMMeshGetSectionReal(dm ,"default", &section);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder = m->getFactory()->getGlobalOrder(m, s->getName(), s);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& localOrder  = m->getFactory()->getLocalOrder(m, s->getName(), s);
  PetscInt *ltog;

  ierr = PetscMalloc(localOrder->getLocalSize() * sizeof(PetscInt), &ltog);CHKERRQ(ierr);
  for(PetscInt p = s->getChart().min(); p <= s->getChart().max(); ++p) {
    PetscInt l = localOrder->getIndex(p);
    PetscInt g = globalOrder->getIndex(p);

    for(PetscInt c = 0; c < s->getFiberDimension(p); ++c) {
      ltog[l+c] = g+c;
    }
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, localOrder->getLocalSize(), ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(dm, dm->ltogmap);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateGlobalScatter"
/*@
  DMMeshCreateGlobalScatter - Create a VecScatter which maps from local, overlapping
  storage in the Section to a global Vec

  Collective on DMMesh

  Input Parameters:
+ mesh - the mesh object
- section - The Scetion which determines data layout

  Output Parameter:
. scatter - the VecScatter

  Level: advanced

.seealso DMDestroy(), DMMeshCreateGlobalRealVector(), DMMeshCreate()
@*/
PetscErrorCode DMMeshCreateGlobalScatter(DM dm, SectionReal section, VecScatter *scatter)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  if (m->hasLabel("marker")) {
    ierr = DMMeshCreateGlobalScatter(m, s, m->getLabel("marker"), scatter);CHKERRQ(ierr);
  } else {
    ierr = DMMeshCreateGlobalScatter(m, s, scatter);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetGlobalScatter"
/*@
  DMMeshGetGlobalScatter - Retrieve the VecScatter which maps from local, overlapping storage in the default Section to a global Vec

  Collective on DMMesh

  Input Parameters:
. mesh - the mesh object

  Output Parameter:
. scatter - the VecScatter

  Level: advanced

.seealso MeshDestroy(), DMMeshCreateGlobalrealVector(), DMMeshCreate()
@*/
PetscErrorCode DMMeshGetGlobalScatter(DM dm, VecScatter *scatter)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(scatter, 2);
  if (!mesh->globalScatter) {
    SectionReal section;

    ierr = DMMeshGetSectionReal(dm, "default", &section);CHKERRQ(ierr);
    ierr = DMMeshCreateGlobalScatter(dm, section, &mesh->globalScatter);CHKERRQ(ierr);
    ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  }
  *scatter = mesh->globalScatter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_Mesh"
PetscErrorCode  DMGlobalToLocalBegin_Mesh(DM dm, Vec g, InsertMode mode, Vec l)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetGlobalScatter(dm, &injection);CHKERRQ(ierr);
  ierr = VecScatterBegin(injection, g, l, mode, SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_Mesh"
PetscErrorCode  DMGlobalToLocalEnd_Mesh(DM dm, Vec g, InsertMode mode, Vec l)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetGlobalScatter(dm, &injection);CHKERRQ(ierr);
  ierr = VecScatterEnd(injection, g, l, mode, SCATTER_REVERSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_Mesh"
PetscErrorCode  DMLocalToGlobalBegin_Mesh(DM dm, Vec l, InsertMode mode, Vec g)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetGlobalScatter(dm, &injection);CHKERRQ(ierr);
  ierr = VecScatterBegin(injection, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_Mesh"
PetscErrorCode  DMLocalToGlobalEnd_Mesh(DM dm, Vec l, InsertMode mode, Vec g)
{
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetGlobalScatter(dm, &injection);CHKERRQ(ierr);
  ierr = VecScatterEnd(injection, l, g, mode, SCATTER_FORWARD);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLocalFunction"
PetscErrorCode DMMeshGetLocalFunction(DM dm, PetscErrorCode (**lf)(DM, SectionReal, SectionReal, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lf) *lf = mesh->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetLocalFunction"
PetscErrorCode DMMeshSetLocalFunction(DM dm, PetscErrorCode (*lf)(DM, SectionReal, SectionReal, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLocalJacobian"
PetscErrorCode DMMeshGetLocalJacobian(DM dm, PetscErrorCode (**lj)(DM, SectionReal, Mat, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lj) *lj = mesh->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetLocalJacobian"
PetscErrorCode DMMeshSetLocalJacobian(DM dm, PetscErrorCode (*lj)(DM, SectionReal, Mat, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshFormFunction"
PetscErrorCode DMMeshFormFunction(DM dm, SectionReal X, SectionReal F, void *ctx)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_CLASSID, 2);
  PetscValidHeaderSpecific(F, SECTIONREAL_CLASSID, 3);
  if (mesh->lf) {
    ierr = (*mesh->lf)(dm, X, F, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshFormJacobian"
PetscErrorCode DMMeshFormJacobian(DM dm, SectionReal X, Mat J, void *ctx)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_CLASSID, 2);
  PetscValidHeaderSpecific(J, MAT_CLASSID, 3);
  if (mesh->lj) {
    ierr = (*mesh->lj)(dm, X, J, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolatePoints"
// Here we assume:
//  - Assumes 3D and tetrahedron
//  - The section takes values on vertices and is P1
//  - Points have the same dimension as the mesh
//  - All values have the same dimension
PetscErrorCode DMMeshInterpolatePoints(DM dm, SectionReal section, int numPoints, PetscReal *points, PetscScalar **values)
{
  Obj<PETSC_MESH_TYPE> m;
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  double        *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  int embedDim = coordinates->getFiberDimension(*m->depthStratum(0)->begin());
  int dim      = s->getFiberDimension(*m->depthStratum(0)->begin());

  ierr = PetscMalloc3(embedDim,double,&v0,embedDim*embedDim,double,&J,embedDim*embedDim,double,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints*dim * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  for(int p = 0; p < numPoints; p++) {
    PetscReal *point = &points[p*embedDim];

    PETSC_MESH_TYPE::point_type e = m->locatePoint(point);
    const PETSC_MESH_TYPE::real_section_type::value_type *coeff = s->restrictPoint(e);

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

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetMaximumDegree"
/*@C
  DMMeshGetMaximumDegree - Return the maximum degree of any mesh vertex

  Collective on mesh

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:
. maxDegree - The maximum number of edges at any vertex

   Level: beginner

.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetMaximumDegree(DM dm, PetscInt *maxDegree)
{
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type>&     sieve    = m->getSieve();
  PetscInt                                         maxDeg   = -1;

  for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    maxDeg = PetscMax(maxDeg, (PetscInt) sieve->getSupportSize(*v_iter));
  }
  *maxDegree = maxDeg;
  PetscFunctionReturn(0);
}

extern PetscErrorCode assembleFullField(VecScatter, Vec, Vec, InsertMode);

#undef __FUNCT__
#define __FUNCT__ "restrictVector"
/*@
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
  ierr = PetscLogEventBegin(DMMesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(injection, g, l, mode, SCATTER_REVERSE);
    ierr = VecScatterEnd(injection, g, l, mode, SCATTER_REVERSE);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(g, l);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(l, 1.0, g);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(DMMesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVectorComplete"
/*@
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
  ierr = PetscLogEventBegin(DMMesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(injection, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(injection, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(l, g);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(g, 1.0, l);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(DMMesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVector"
/*@
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
  DM             dm;
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) b, "DMMesh", (PetscObject *) &dm);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dm, "x", &section);CHKERRQ(ierr);
  ierr = assembleVector(b, dm, section, e, v, mode);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode assembleVector(Vec b, DM dm, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt                  firstElement;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMMesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(((PetscObject)mesh)->comm,PETSC_ERR_SUP, "SectionReal does not support complex update");
#else
  if (mode == INSERT_VALUES) {
    m->update(s, PETSC_MESH_TYPE::point_type(e + firstElement), v);
  } else {
    m->updateAdd(s, PETSC_MESH_TYPE::point_type(e + firstElement), v);
  }
#endif
  ierr = PetscLogEventEnd(DMMesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesTopology"
/*@C
  MatSetValuesTopology - Sets values in a matrix using DM Mesh points rather than indices

  Not Collective

  Input Parameters:
+ mat - the matrix
. dmr - The row DM
. nrow, rowPoints - number of rows and their local Sieve points
. dmc - The column DM
. ncol, colPoints - number of columns and their local Sieve points
. v -  a logically two-dimensional array of values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: intermediate

.seealso: DMMeshCreate(), MatSetValuesStencil()
@*/
PetscErrorCode MatSetValuesTopology(Mat mat, DM dmr, PetscInt nrow, const PetscInt rowPoints[], DM dmc, PetscInt ncol, const PetscInt colPoints[], const PetscScalar v[], InsertMode mode)
{
  ALE::Obj<PETSC_MESH_TYPE> mr;
  ALE::Obj<PETSC_MESH_TYPE> mc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(dmr,DM_CLASSID,2);
  PetscValidIntPointer(rowPoints,4);
  PetscValidHeaderSpecific(dmc,DM_CLASSID,5);
  PetscValidIntPointer(colPoints,7);
  if (v) PetscValidDoublePointer(v,8);
  ierr = DMMeshGetMesh(dmr, mr);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(dmc, mc);CHKERRQ(ierr);
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type rV(*mr->getRealSection("default"), *mr->getFactory()->getLocalOrder(mr, "default", mr->getRealSection("default")),
                  (int) pow((double) mr->getSieve()->getMaxConeSize(), mr->depth())*mr->getMaxDof()*nrow, mr->depth() > 1);
  visitor_type cV(*mc->getRealSection("default"), *mc->getFactory()->getLocalOrder(mc, "default", mc->getRealSection("default")),
                  (int) pow((double) mc->getSieve()->getMaxConeSize(), mc->depth())*mc->getMaxDof()*ncol, mc->depth() > 1);

  try {
    for(PetscInt r = 0; r < nrow; ++r) {
      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mr->getSieve(), rowPoints[r], rV);
    }
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, e.message());
  }
  const PetscInt *rowIndices    = rV.getValues();
  const int       numRowIndices = rV.getSize();
  try {
    for(PetscInt c = 0; c < ncol; ++c) {
      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mc->getSieve(), colPoints[c], cV);
    }
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, e.message());
  }
  const PetscInt *colIndices    = cV.getValues();
  const int       numColIndices = cV.getSize();

  ierr = MatSetValuesLocal(mat, numRowIndices, rowIndices, numColIndices, colIndices, v, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& section, const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder, const PETSC_MESH_TYPE::point_type& e, PetscScalar array[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type iV(*section, *globalOrder, (int) pow((double) m->getSieve()->getMaxConeSize(), m->depth())*m->getMaxDof(), m->depth() > 1);

  ierr = updateOperator(A, *m->getSieve(), iV, e, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperatorGeneral"
PetscErrorCode updateOperatorGeneral(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& rowM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& rowSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& rowGlobalOrder, const PETSC_MESH_TYPE::point_type& rowE, const ALE::Obj<PETSC_MESH_TYPE>& colM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& colSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& colGlobalOrder, const PETSC_MESH_TYPE::point_type& colE, PetscScalar array[], InsertMode mode)
{
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type iVr(*rowSection, *rowGlobalOrder, (int) pow((double) rowM->getSieve()->getMaxConeSize(), rowM->depth())*rowM->getMaxDof(), rowM->depth() > 1);
  visitor_type iVc(*colSection, *colGlobalOrder, (int) pow((double) colM->getSieve()->getMaxConeSize(), colM->depth())*colM->getMaxDof(), colM->depth() > 1);

  PetscErrorCode ierr = updateOperator(A, *rowM->getSieve(), iVr, rowE, *colM->getSieve(), iVc, colE, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetMaxDof"
/*@
  DMMeshSetMaxDof - Sets the maximum number of degrees of freedom on any sieve point

  Logically Collective on A

  Input Parameters:
+ A - the matrix
. mesh - DMMesh needed for orderings
. section - A Section which describes the layout
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: This is used by routines like updateOperator() to bound buffer sizes

   Level: developer

.seealso: updateOperator(), assembleMatrix()
@*/
PetscErrorCode DMMeshSetMaxDof(DM dm, PetscInt maxDof)
{
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  m->setMaxDof(maxDof);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleMatrix"
/*@
  assembleMatrix - Insert values into a matrix

  Collective on A

  Input Parameters:
+ A - the matrix
. dm - DMMesh needed for orderings
. section - A Section which describes the layout
. e - The element
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleMatrix(Mat A, DM dm, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMMesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  try {
    Obj<PETSC_MESH_TYPE> m;
    Obj<PETSC_MESH_TYPE::real_section_type> s;

    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder = m->getFactory()->getGlobalOrder(m, s->getName(), s);

    if (m->debug()) {
      std::cout << "Assembling matrix for element number " << e << " --> point " << e << std::endl;
    }
    ierr = updateOperator(A, m, s, globalOrder, e, v, mode);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscLogEventEnd(DMMesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/******************************** C Wrappers **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLabelSize"
/*@C
  DMMeshGetLabelSize - Get the number of different integer ids in a Label

  Not Collective

  Input Parameters:
+ dm   - The DMMesh object
- name - The label name

  Output Parameter:
. size - The label size (number of different integer ids)

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreateExodus()
@*/
PetscErrorCode DMMeshGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *size = m->getLabel(name)->getCapSize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLabelIds"
/*@C
  DMMeshGetLabelIds - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
. name - The label name
- ids - The id storage array

  Output Parameter:
. ids - The integer ids

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreateExodus()
@*/
PetscErrorCode DMMeshGetLabelIds(DM dm, const char name[], PetscInt *ids)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_type::capSequence>&      labelIds = m->getLabel(name)->cap();
  const PETSC_MESH_TYPE::label_type::capSequence::const_iterator iEnd     = labelIds->end();
  PetscInt                                                       i        = 0;

  for(PETSC_MESH_TYPE::label_type::capSequence::const_iterator i_iter = labelIds->begin(); i_iter != iEnd; ++i_iter, ++i) {
    ids[i] = *i_iter;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetStratumSize"
/*@C
  DMMeshGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMMesh object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The stratum size

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreateExodus()
@*/
PetscErrorCode DMMeshGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *size = m->getLabelStratum(name, value)->size();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetStratum"
/*@C
  DMMeshGetStratum - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMMesh object
. name - The label name
. value - The stratum value
- points - The stratum points storage array

  Output Parameter:
. points - The stratum points

  Level: beginner

.keywords: mesh, ExodusII
.seealso: DMMeshCreateExodus()
@*/
PetscErrorCode DMMeshGetStratum(DM dm, const char name[], PetscInt value, PetscInt *points)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& stratum = m->getLabelStratum(name, value);
  const PETSC_MESH_TYPE::label_sequence::iterator  sEnd    = stratum->end();
  PetscInt                                         s       = 0;

  for(PETSC_MESH_TYPE::label_sequence::iterator s_iter = stratum->begin(); s_iter != sEnd; ++s_iter, ++s) {
    points[s] = *s_iter;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteVTKHeader"
PetscErrorCode WriteVTKHeader(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return VTKViewer::writeHeader(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "WriteVTKVertices"
PetscErrorCode WriteVTKVertices(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return VTKViewer::writeVertices(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return VTKViewer::writeElements(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCoordinates"
/*@C
  DMMeshGetCoordinates - Creates an array holding the coordinates.

  Not Collective

  Input Parameter:
+ dm - The DMMesh object
- columnMajor - Flag for column major order

  Output Parameter:
+ numVertices - The number of vertices
. dim - The embedding dimension
- coords - The array holding local coordinates

  Level: intermediate

.keywords: mesh, coordinates
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetCoordinates(DM dm, PetscBool  columnMajor, PetscInt *numVertices, PetscInt *dim, PetscReal *coords[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputVerticesLocal(m, numVertices, dim, coords, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetElements"
/*@C
  DMMeshGetElements - Creates an array holding the vertices on each element.

  Not Collective

  Input Parameters:
+ dm - The DMMesh object
- columnMajor - Flag for column major order

  Output Parameters:
+ numElements - The number of elements
. numCorners - The number of vertices per element
- vertices - The array holding vertices on each local element

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetElements(DM dm, PetscBool  columnMajor, PetscInt *numElements, PetscInt *numCorners, PetscInt *vertices[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputElementsLocal(m, numElements, numCorners, vertices, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCone"
/*@C
  DMMeshGetCone - Creates an array holding the cone of a given point

  Not Collective

  Input Parameters:
+ dm - The DMMesh object
- p - The mesh point

  Output Parameters:
+ numPoints - The number of points in the cone
- points - The array holding the cone points

  Level: intermediate

.keywords: mesh, cone
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetCone(DM dm, PetscInt p, PetscInt *numPoints, PetscInt *points[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *numPoints = m->getSieve()->getConeSize(p);
  ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> v(*numPoints);

  m->getSieve()->cone(p, v);
  *points = const_cast<PetscInt*>(v.getPoints());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshDistribute"
/*@C
  DMMeshDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ serialMesh  - The original DMMesh object
- partitioner - The partitioning package, or NULL for the default

  Output Parameter:
. parallelMesh - The distributed DMMesh object

  Level: intermediate

.keywords: mesh, elements

.seealso: DMMeshCreate(), DMMeshDistributeByFace()
@*/
PetscErrorCode DMMeshDistribute(DM serialMesh, const char partitioner[], DM *parallelMesh)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscMPIInt         commSize;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject) serialMesh)->comm, &commSize);CHKERRQ(ierr);
  if (commSize == 1) PetscFunctionReturn(0);
  ierr = DMMeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = DMMeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE>             newMesh  = new PETSC_MESH_TYPE(oldMesh->comm(), oldMesh->getDimension(), oldMesh->debug());
  const Obj<PETSC_MESH_TYPE::sieve_type> newSieve = new PETSC_MESH_TYPE::sieve_type(oldMesh->comm(), oldMesh->debug());

  newMesh->setSieve(newSieve);
  ALE::DistributionNew<PETSC_MESH_TYPE>::distributeMeshAndSectionsV(oldMesh, newMesh);
  ierr = DMMeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshDistributeByFace"
/*@C
  DMMeshDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ serialMesh  - The original DMMesh object
- partitioner - The partitioning package, or NULL for the default

  Output Parameter:
. parallelMesh - The distributed DMMesh object

  Level: intermediate

.keywords: mesh, elements

.seealso: DMMeshCreate(), DMMeshDistribute()
@*/
PetscErrorCode DMMeshDistributeByFace(DM serialMesh, const char partitioner[], DM *parallelMesh)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = DMMeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "I am being lazy, bug me.");
#if 0
  ALE::DistributionNew<PETSC_MESH_TYPE>::distributeMeshAndSectionsV(oldMesh, newMesh, height = 1);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGenerate"
/*@C
  DMMeshGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The DMMesh boundary object
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The DMMesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate(), DMMeshRefine()
@*/
PetscErrorCode DMMeshGenerate(DM boundary, PetscBool  interpolate, DM *mesh)
{
  ALE::Obj<PETSC_MESH_TYPE> mB;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(boundary, mB);CHKERRQ(ierr);
  ierr = DMMeshCreate(mB->comm(), mesh);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> m = ALE::Generator<PETSC_MESH_TYPE>::generateMeshV(mB, interpolate);
  ierr = DMMeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshRefine"
/*@C
  DMMeshRefine - Refines the mesh.

  Not Collective

  Input Parameters:
+ mesh - The original DMMesh object
. refinementLimit - The maximum size of any cell
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. refinedMesh - The refined DMMesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate(), DMMeshGenerate()
@*/
PetscErrorCode DMMeshRefine(DM mesh, double refinementLimit, PetscBool  interpolate, DM *refinedMesh)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (refinementLimit == 0.0) PetscFunctionReturn(0);
  ierr = DMMeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = DMMeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, refinementLimit, interpolate);
  ierr = DMMeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefine_Mesh"
PetscErrorCode DMRefine_Mesh(DM dm, MPI_Comm comm, DM *dmRefined)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  double                    refinementLimit;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, oldMesh);CHKERRQ(ierr);
  ierr = DMMeshCreate(comm, dmRefined);CHKERRQ(ierr);
  refinementLimit = oldMesh->getMaxVolume()/2.0;
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, refinementLimit, true);
  ierr = DMMeshSetMesh(*dmRefined, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_Mesh"
PetscErrorCode DMCoarsenHierarchy_Mesh(DM mesh, int numLevels, DM *coarseHierarchy)
{
  PetscReal      cfactor = 1.5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsReal("-dmmg_coarsen_factor", "The coarsening factor", PETSC_NULL, cfactor, &cfactor, PETSC_NULL);CHKERRQ(ierr);
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Peter needs to incorporate his code.");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetInterpolation_Mesh"
PetscErrorCode DMGetInterpolation_Mesh(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling) {
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Peter needs to incorporate his code.");
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshMarkBoundaryCells"
PetscErrorCode DMMeshMarkBoundaryCells(DM dm, const char labelName[], PetscInt marker, PetscInt newMarker) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  mesh->markBoundaryCells(labelName, marker, newMarker);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetDepthStratum"
PetscErrorCode DMMeshGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->depthStratum(stratumValue);
    if (start) *start = *stratum->begin();
    if (end)   *end   = *stratum->rbegin()+1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetHeightStratum"
PetscErrorCode DMMeshGetHeightStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->heightStratum(stratumValue);
    if (start) *start = *stratum->begin();
    if (end)   *end   = *stratum->rbegin()+1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetSection"
PetscErrorCode DMMeshGetSection(DM dm, const char name[], PetscSection *section) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    const Obj<PETSC_MESH_TYPE::real_section_type>& s = mesh->getRealSection(name);
    const PetscInt pStart = s->getChart().min();
    const PetscInt pEnd   = s->getChart().max();

    ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetDof(*section, p, s->getFiberDimension(p));CHKERRQ(ierr);
      ierr = PetscSectionSetConstraintDof(*section, p, s->getConstraintDimension(p));CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetConstraintIndices(*section, p, (PetscInt *) s->getConstraintDof(p));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetDefaultSection"
PetscErrorCode DMMeshGetDefaultSection(DM dm, PetscSection *section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetSection(dm, "default", section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCoordinateSection"
PetscErrorCode DMMeshGetCoordinateSection(DM dm, PetscSection *section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetSection(dm, "coordinates", section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCoordinateVec"
PetscErrorCode DMMeshGetCoordinateVec(DM dm, Vec *coordinates) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    const Obj<PETSC_MESH_TYPE::real_section_type>& coords = mesh->getRealSection("coordinates");
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, coords->getStorageSize(), coords->restrictSpace(), coordinates);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshComputeCellGeometry"
PetscErrorCode DMMeshComputeCellGeometry(DM dm, PetscInt cell, PetscReal *v0, PetscReal *J, PetscReal *invJ, PetscReal *detJ) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> coordinates = mesh->getRealSection("coordinates");

    mesh->computeElementGeometry(coordinates, cell, v0, J, invJ, *detJ);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshVecGetClosure"
PetscErrorCode DMMeshVecGetClosure(DM dm, Vec v, PetscInt point, const PetscScalar *values[]) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "DMMesh does not support complex closure");
#else
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  /* Peeling back IMesh::restrictClosure() */
  try {
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s = mesh->getRealSection("default");
    const PETSC_MESH_TYPE::real_section_type::value_type *oldStorage = s->restrictSpace();
    const PetscInt size = mesh->sizeWithBC(s, point);
    ALE::ISieveVisitor::RestrictVisitor<PETSC_MESH_TYPE::real_section_type> rV(*s, size, s->getRawArray(size));
    PetscScalar *array;

    ierr = VecGetArray(v, &array);CHKERRQ(ierr);
    s->setStorage(array);
    if (mesh->depth() == 1) {
      rV.visitPoint(point, 0);
      // Cone is guarateed to be ordered correctly
      mesh->getSieve()->orientedCone(point, rV);
    } else {
      ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type,ALE::ISieveVisitor::RestrictVisitor<PETSC_MESH_TYPE::real_section_type> > pV((int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, rV, true);

      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), point, pV);
    }
    s->setStorage((PETSC_MESH_TYPE::real_section_type::value_type *) oldStorage);
    ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
    *values = rV.getValues();
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshVecSetClosure"
PetscErrorCode DMMeshVecSetClosure(DM dm, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "DMMesh does not support complex closure");
#else
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  /* Peeling back IMesh::update() and IMesh::updateAdd() */
  try {
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s = mesh->getRealSection("default");
    const PETSC_MESH_TYPE::real_section_type::value_type *oldStorage = s->restrictSpace();
    PetscScalar *array;

    ierr = VecGetArray(v, &array);CHKERRQ(ierr);
    s->setStorage(array);

    if (mode == INSERT_VALUES) {
      ALE::ISieveVisitor::UpdateVisitor<PETSC_MESH_TYPE::real_section_type> uV(*s, values);
      if (mesh->depth() == 1) {
        uV.visitPoint(point, 0);
        // Cone is guarateed to be ordered correctly
        mesh->getSieve()->orientedCone(point, uV);
      } else {
        ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type,ALE::ISieveVisitor::UpdateVisitor<PETSC_MESH_TYPE::real_section_type> > pV((int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, uV, true);

        ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), point, pV);
      }
    } else {
      ALE::ISieveVisitor::UpdateAddVisitor<PETSC_MESH_TYPE::real_section_type> uV(*s, values);
      if (mesh->depth() == 1) {
        uV.visitPoint(point, 0);
        // Cone is guarateed to be ordered correctly
        mesh->getSieve()->orientedCone(point, uV);
      } else {
        ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type,ALE::ISieveVisitor::UpdateAddVisitor<PETSC_MESH_TYPE::real_section_type> > pV((int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, uV, true);

        ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), point, pV);
      }
    }

    s->setStorage((PETSC_MESH_TYPE::real_section_type::value_type *) oldStorage);
    ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshMatSetClosure"
PetscErrorCode DMMeshMatSetClosure(DM dm, Mat A, PetscInt point, PetscScalar values[], InsertMode mode) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "DMMesh does not support complex closure");
#else
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  /* Copying from updateOperator() */
  try {
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s = mesh->getRealSection("default");
    const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder = mesh->getFactory()->getGlobalOrder(mesh, s->getName(), s);
    typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
    visitor_type iV(*s, *globalOrder, (int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())*mesh->getMaxDof(), mesh->depth() > 1);

    ierr = updateOperator(A, *mesh->getSieve(), iV, point, values, mode);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshHasSectionReal"
/*@C
  DMMeshHasSectionReal - Determines whether this mesh has a SectionReal with the given name.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionReal is present in the DMMesh

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshHasSectionReal(DM dm, const char name[], PetscBool  *flag)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *flag = (PetscBool) m->hasRealSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetSectionReal"
/*@C
  DMMeshGetSectionReal - Returns a SectionReal of the given name from the DMMesh.

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameter:
. section - The SectionReal

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements

.seealso: DMMeshCreate(), SectionRealDestroy()
@*/
PetscErrorCode DMMeshGetSectionReal(DM dm, const char name[], SectionReal *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  bool                      has;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  has  = m->hasRealSection(std::string(name));
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
  if (!has) {
    m->getRealSection(std::string(name))->setChart(m->getSieve()->getChart());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetSectionReal"
/*@C
  DMMeshSetSectionReal - Puts a SectionReal of the given name into the DMMesh.

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- section - The SectionReal

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshSetSectionReal(DM dm, SectionReal section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  m->setRealSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshHasSectionInt"
/*@C
  DMMeshHasSectionInt - Determines whether this mesh has a SectionInt with the given name.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionInt is present in the DMMesh

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshHasSectionInt(DM dm, const char name[], PetscBool  *flag)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *flag = (PetscBool) m->hasIntSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetSectionInt"
/*@C
  DMMeshGetSectionInt - Returns a SectionInt of the given name from the DMMesh.

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameter:
. section - The SectionInt

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetSectionInt(DM dm, const char name[], SectionInt *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  bool                      has;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  has  = m->hasIntSection(std::string(name));
  ierr = SectionIntSetSection(*section, m->getIntSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*section, m);CHKERRQ(ierr);
  if (!has) {
    m->getIntSection(std::string(name))->setChart(m->getSieve()->getChart());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetSectionInt"
/*@C
  DMMeshSetSectionInt - Puts a SectionInt of the given name into the DMMesh.

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- section - The SectionInt

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshSetSectionInt(DM dm, SectionInt section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  m->setIntSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionGetArray"
/*@C
  SectionGetArray - Returns the array underlying the Section.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionGetArray(DM dm, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscScalar *array[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& section = m->getRealSection(std::string(name));
  if (section->size() == 0) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const PETSC_MESH_TYPE::real_section_type::chart_type& chart = section->getChart();
/*   const int                                  depth   = m->depth(*chart.begin()); */
/*   *numElements = m->depthStratum(depth)->size(); */
/*   *fiberDim    = section->getFiberDimension(*chart.begin()); */
/*   *array       = (PetscScalar *) m->restrict(section); */
  int fiberDimMin = section->getFiberDimension(*chart.begin());
  int numElem     = 0;

  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscScalar *) section->restrictSpace();
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
