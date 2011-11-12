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
    PetscInt  dim   = mesh->getDimension();
    PetscInt  size  = mesh->commSize();
    PetscInt  depth = mesh->depth();
    PetscInt  num   = 0;
    PetscInt *sizes;

    ierr = PetscMalloc(size * sizeof(PetscInt), &sizes);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&depth, &depth, 1, MPIU_INT, MPI_MAX, mesh->comm());CHKERRQ(ierr);
    if (depth == 1) {
      num  = mesh->depthStratum(0)->size();
      ierr = MPI_Gather(&num, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, mesh->comm());CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", 0);CHKERRQ(ierr);
      for(PetscInt p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      num  = mesh->heightStratum(0)->size();
      ierr = MPI_Gather(&num, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, mesh->comm());CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", dim);CHKERRQ(ierr);
      for(PetscInt p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for(int d = 0; d <= dim; d++) {
        num  = mesh->depthStratum(d)->size();
        ierr = MPI_Gather(&num, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, mesh->comm());CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", d);CHKERRQ(ierr);
        for(PetscInt p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(sizes);CHKERRQ(ierr);
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
  ierr = PetscObjectCompose((PetscObject) *J, "DM", (PetscObject) dm);CHKERRQ(ierr);
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
  ierr = PetscObjectCompose((PetscObject) *gvec, "DM", (PetscObject) dm);CHKERRQ(ierr);
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
  ierr = PetscObjectCompose((PetscObject) *lvec, "DM", (PetscObject) dm);CHKERRQ(ierr);
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
  PetscInt *ltog;

  ierr = PetscMalloc(s->size() * sizeof(PetscInt), &ltog);CHKERRQ(ierr); // We want the local+overlap size
  for(PetscInt p = s->getChart().min(), l = 0; p < s->getChart().max(); ++p) {
    PetscInt g = globalOrder->getIndex(p);

    for(PetscInt c = 0; c < s->getConstrainedFiberDimension(p); ++c, ++l) {
      ltog[l] = g+c;
    }
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, s->size(), ltog, PETSC_OWN_POINTER, &dm->ltogmap);CHKERRQ(ierr);
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
PetscErrorCode DMMeshGetLocalFunction(DM dm, PetscErrorCode (**lf)(DM, Vec, Vec, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lf) *lf = mesh->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetLocalFunction"
PetscErrorCode DMMeshSetLocalFunction(DM dm, PetscErrorCode (*lf)(DM, Vec, Vec, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLocalJacobian"
PetscErrorCode DMMeshGetLocalJacobian(DM dm, PetscErrorCode (**lj)(DM, Vec, Mat, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (lj) *lj = mesh->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetLocalJacobian"
PetscErrorCode DMMeshSetLocalJacobian(DM dm, PetscErrorCode (*lj)(DM, Vec, Mat, void *))
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationCreate"
PetscErrorCode DMMeshInterpolationCreate(DM dm, DMMeshInterpolationInfo *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  ierr = PetscMalloc(sizeof(struct _DMMeshInterpolationInfo), ctx);CHKERRQ(ierr);
  (*ctx)->dim    = -1;
  (*ctx)->nInput = 0;
  (*ctx)->points = PETSC_NULL;
  (*ctx)->cells  = PETSC_NULL;
  (*ctx)->n      = -1;
  (*ctx)->coords = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetDim"
PetscErrorCode DMMeshInterpolationSetDim(DM dm, PetscInt dim, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if ((dim < 1) || (dim > 3)) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %d", dim);}
  ctx->dim = dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetDim"
PetscErrorCode DMMeshInterpolationGetDim(DM dm, PetscInt *dim, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  *dim = ctx->dim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetDof"
PetscErrorCode DMMeshInterpolationSetDof(DM dm, PetscInt dof, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dof < 1) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %d", dof);}
  ctx->dof = dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetDof"
PetscErrorCode DMMeshInterpolationGetDof(DM dm, PetscInt *dof, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dof, 2);
  *dof = ctx->dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationAddPoints"
PetscErrorCode DMMeshInterpolationAddPoints(DM dm, PetscInt n, PetscReal points[], DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ctx->dim < 0) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  }
  if (ctx->points) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  }
  ctx->nInput = n;
  ierr = PetscMalloc(n*ctx->dim * sizeof(PetscReal), &ctx->points);CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->points, points, n*ctx->dim * sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationSetUp"
PetscErrorCode DMMeshInterpolationSetUp(DM dm, DMMeshInterpolationInfo ctx) {
  Obj<PETSC_MESH_TYPE> m;
  PetscScalar   *a;
  PetscInt       p, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  if (ctx->dim < 0) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  }
  // Locate points
  PetscInt N, found;

  ierr = MPI_Allreduce(&ctx->nInput, &N, 1, MPIU_INT, MPI_SUM, ((PetscObject) dm)->comm);CHKERRQ(ierr);
  // Communicate all points to all processes
  ctx->n = 0;
  ierr = PetscMalloc(N * sizeof(PetscInt), &ctx->cells);CHKERRQ(ierr);
  for(p = 0; p < N; ++p) {
    ctx->cells[p] = m->locatePoint(&ctx->points[p*ctx->dim]);
    if (ctx->cells[p] >= 0) ctx->n++;
  }
  // Check that exactly this many points were found
  ierr = MPI_Allreduce(&ctx->n, &found, 1, MPIU_INT, MPI_SUM, ((PetscObject) dm)->comm);CHKERRQ(ierr);
  if (found != N) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Invalid number of points located %d should be %d", found, N);}
  // Create coordinates vector
  ierr = VecCreate(((PetscObject) dm)->comm, &ctx->coords);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->coords, ctx->dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->coords);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &a);CHKERRQ(ierr);
  for(p = 0, i = 0; p < N; ++p) {
    if (ctx->cells[p] >= 0) {
      PetscInt d;

      for(d = 0; d < ctx->dim; ++d, ++i) {
        a[i] = ctx->points[p*ctx->dim+d];
      }
    }
  }
  ierr = VecRestoreArray(ctx->coords, &a);CHKERRQ(ierr);
  // Compress cells array
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetCoordinates"
PetscErrorCode DMMeshInterpolationGetCoordinates(DM dm, Vec *coordinates, DMMeshInterpolationInfo ctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coordinates, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  *coordinates = ctx->coords;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationGetVector"
PetscErrorCode DMMeshInterpolationGetVector(DM dm, Vec *v, DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(v, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  ierr = VecCreate(((PetscObject) dm)->comm, v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v, ctx->n*ctx->dof, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*v, ctx->dof);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationRestoreVector"
PetscErrorCode DMMeshInterpolationRestoreVector(DM dm, Vec *v, DMMeshInterpolationInfo ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(v, 2);
  if (!ctx->coords) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");}
  ierr = VecDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationEvaluate"
PetscErrorCode DMMeshInterpolationEvaluate(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx) {
  Obj<PETSC_MESH_TYPE> m;
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt       p, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, SECTIONREAL_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  if (n != ctx->n*ctx->dof) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %d should be %d", n, ctx->n*ctx->dof);}
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(x, s);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  PetscReal   *v0, *J, *invJ, detJ;
  PetscScalar *a, *coords;

  ierr = PetscMalloc3(ctx->dim,PetscReal,&v0,ctx->dim*ctx->dim,PetscReal,&J,ctx->dim*ctx->dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for(p = 0; p < ctx->n; ++p) {
    PetscInt           e = ctx->cells[p];
    const PetscScalar *c = m->restrictClosure(s, e);
    PetscReal          xi[4];
    PetscInt           d, f, comp;

    if ((ctx->dim+1)*ctx->dof != m->sizeWithBC(s, e)) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Invalid restrict size %d should be %d", m->sizeWithBC(s, e), (ctx->dim+1)*ctx->dof);}
    m->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    for(comp = 0; comp < ctx->dof; ++comp) {
      a[p*ctx->dof+comp] = c[0*ctx->dof+comp];
    }
    for(d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for(f = 0; f < ctx->dim; ++f) {
        xi[d] += invJ[d*ctx->dim+f]*0.5*(coords[p*ctx->dim+f] - v0[f]);
      }
      for(comp = 0; comp < ctx->dof; ++comp) {
        a[p*ctx->dof+comp] += (c[d*ctx->dof+comp] - c[0*ctx->dof+comp])*xi[d];
      }
    }
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshInterpolationDestroy"
PetscErrorCode DMMeshInterpolationDestroy(DM dm, DMMeshInterpolationInfo *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  ierr = VecDestroy(&(*ctx)->coords);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->points);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->cells);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  *ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetDimension"
/*@
  DMMeshGetDimension - Return the topological mesh dimension

  Collective on mesh

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:
. dim - The topological mesh dimension

  Level: beginner

.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetDimension(DM dm, PetscInt *dim)
{
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  *dim = m->getDimension();
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
#define __FUNCT__ "DMMeshRestrictVector"
/*@
  DMMeshRestrictVector - Insert values from a global vector into a local ghosted vector

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
PetscErrorCode DMMeshRestrictVector(Vec g, Vec l, InsertMode mode)
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
#define __FUNCT__ "DMMeshAssembleVectorComplete"
/*@
  DMMeshAssembleVectorComplete - Insert values from a local ghosted vector into a global vector

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
PetscErrorCode DMMeshAssembleVectorComplete(Vec g, Vec l, InsertMode mode)
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
#define __FUNCT__ "DMMeshAssembleVector"
/*@
  DMMeshAssembleVector - Insert values into a vector

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
PetscErrorCode DMMeshAssembleVector(Vec b, PetscInt e, PetscScalar v[], InsertMode mode)
{
  DM             dm;
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) b, "DM", (PetscObject *) &dm);CHKERRQ(ierr);
  ierr = DMMeshGetSectionReal(dm, "x", &section);CHKERRQ(ierr);
  ierr = DMMeshAssembleVector(b, dm, section, e, v, mode);CHKERRQ(ierr);
  ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMMeshAssembleVector(Vec b, DM dm, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
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
#define __FUNCT__ "DMMeshUpdateOperator"
PetscErrorCode DMMeshUpdateOperator(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& section, const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder, const PETSC_MESH_TYPE::point_type& e, PetscScalar array[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type iV(*section, *globalOrder, (int) pow((double) m->getSieve()->getMaxConeSize(), m->depth())*m->getMaxDof(), m->depth() > 1);

  ierr = updateOperator(A, *m->getSieve(), iV, e, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshUpdateOperatorGeneral"
PetscErrorCode DMMeshUpdateOperatorGeneral(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& rowM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& rowSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& rowGlobalOrder, const PETSC_MESH_TYPE::point_type& rowE, const ALE::Obj<PETSC_MESH_TYPE>& colM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& colSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& colGlobalOrder, const PETSC_MESH_TYPE::point_type& colE, PetscScalar array[], InsertMode mode)
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

   Notes: This is used by routines like DMMeshUpdateOperator() to bound buffer sizes

   Level: developer

.seealso: DMMeshUpdateOperator(), DMMeshAssembleMatrix()
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
#define __FUNCT__ "DMMeshAssembleMatrix"
/*@
  DMMeshAssembleMatrix - Insert values into a matrix

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
PetscErrorCode DMMeshAssembleMatrix(Mat A, DM dm, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
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
    ierr = DMMeshUpdateOperator(A, m, s, globalOrder, e, v, mode);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMeshGetStratumIS"
PetscErrorCode DMMeshGetStratumIS(DM dm, const char labelName[], PetscInt stratumValue, IS *is) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    if (mesh->hasLabel(labelName)) {
      const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->getLabelStratum(labelName, stratumValue);
      PetscInt *idx, i = 0;

      ierr = PetscMalloc(stratum->size() * sizeof(PetscInt), &idx);CHKERRQ(ierr);
      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = stratum->begin(); e_iter != stratum->end(); ++e_iter, ++i) {
        idx[i] = *e_iter;
      }
      ierr = ISCreateGeneral(((PetscObject)dm)->comm, stratum->size(), idx, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
    } else {
      *is = PETSC_NULL;
    }
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
  ALE::Obj<PETSC_MESH_TYPE> m = ALE::Generator<PETSC_MESH_TYPE>::generateMeshV(mB, interpolate, false, true);
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
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, refinementLimit, interpolate, false, true);
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
    if (stratumValue < 0) {
      if (start) *start = mesh->getSieve()->getChart().min();
      if (end)   *end   = mesh->getSieve()->getChart().max();
    } else {
      const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->depthStratum(stratumValue);
      if (start) *start = *stratum->begin();
      if (end)   *end   = *stratum->rbegin()+1;
    }
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
    if (mesh->getLabel("height")->size()) {
      const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->heightStratum(stratumValue);
      if (start) *start = *stratum->begin();
      if (end)   *end   = *stratum->rbegin()+1;
    } else {
      if (start) *start = 0;
      if (end)   *end   = 0;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateSection"
PetscErrorCode DMMeshCreateSection(DM dm, PetscInt dim, PetscInt numFields, PetscInt numComp[], PetscInt numDof[], PetscInt numBC, PetscInt bcField[], IS bcPoints[], PetscSection *section) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscInt      *numDofTot, *maxConstraints;
  PetscInt       pStart = 0, pEnd = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(dim+1,PetscInt,&numDofTot,numFields+1,PetscInt,&maxConstraints);CHKERRQ(ierr);
  for(PetscInt d = 0; d <= dim; ++d) {
    numDofTot[d] = 0;
    for(PetscInt f = 0; f < numFields; ++f) {
      numDofTot[d] += numDof[f*(dim+1)+d];
    }
  }
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
  if (numFields > 1) {
    ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
    if (numComp) {
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
      }
    }
  } else {
    numFields = 0;
  }
  ierr = DMMeshGetDepthStratum(dm, -1, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
  for(PetscInt d = 0; d <= dim; ++d) {
    ierr = DMMeshGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(*section, p, f, numDof[f*(dim+1)+d]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(*section, p, numDofTot[d]);CHKERRQ(ierr);
    }
  }
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  if (numBC) {
    for(PetscInt f = 0; f <= numFields; ++f) {maxConstraints[f] = 0;}
    for(PetscInt bc = 0; bc < numBC; ++bc) {
      PetscInt        field = 0;
      const PetscInt *idx;
      PetscInt        n;

      if (numFields) {field = bcField[bc];}
      ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
      ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
      for(PetscInt i = 0; i < n; ++i) {
        const PetscInt p        = idx[i];
        const PetscInt numConst = numDof[field*(dim+1)+mesh->depth(p)];

        maxConstraints[field] = PetscMax(maxConstraints[field], numConst);
        if (numFields) {
          ierr = PetscSectionSetFieldConstraintDof(*section, p, field, numConst);CHKERRQ(ierr);
        }
        ierr = PetscSectionAddConstraintDof(*section, p, numConst);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    }
    for(PetscInt f = 0; f < numFields; ++f) {
      maxConstraints[numFields] += maxConstraints[f];
    }
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  if (maxConstraints[numFields]) {
    PetscInt *indices;

    ierr = PetscMalloc(maxConstraints[numFields] * sizeof(PetscInt), &indices);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(*section, &pStart, &pEnd);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      PetscInt cDof;

      ierr = PetscSectionGetConstraintDof(*section, p, &cDof);CHKERRQ(ierr);
      if (cDof) {
        if (cDof > maxConstraints[numFields]) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB, "Likely memory corruption, point %d cDof %d > maxConstraints %d", p, cDof, maxConstraints);}
        if (numFields) {
          PetscInt numConst = 0, fOff = 0;

          for(PetscInt f = 0; f < numFields; ++f) {
            PetscInt cfDof, fDof;

            ierr = PetscSectionGetFieldDof(*section, p, f, &fDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldConstraintDof(*section, p, f, &cfDof);CHKERRQ(ierr);
            for(PetscInt d = 0; d < cfDof; ++d) {
              indices[numConst+d] = fOff+d;
            }
            ierr = PetscSectionSetFieldConstraintIndices(*section, p, f, &indices[numConst]);CHKERRQ(ierr);
            numConst += cfDof;
            fOff     += fDof;
          }
          if (cDof != numConst) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of field constraints %d should be %d", numConst, cDof);}
        } else {
          for(PetscInt d = 0; d < cDof; ++d) {
            indices[d] = d;
          }
        }
        ierr = PetscSectionSetConstraintIndices(*section, p, indices);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }
  ierr = PetscFree2(numDofTot,maxConstraints);CHKERRQ(ierr);
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
    const PetscInt pStart    = s->getChart().min();
    const PetscInt pEnd      = s->getChart().max();
    PetscInt       numFields = s->getNumSpaces();

    ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
    if (numFields) {
      ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, s->getSpaceComponents(f));CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetDof(*section, p, s->getFiberDimension(p));CHKERRQ(ierr);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(*section, p, f, s->getFiberDimension(p, f));CHKERRQ(ierr);
      }
      ierr = PetscSectionSetConstraintDof(*section, p, s->getConstraintDimension(p));CHKERRQ(ierr);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldConstraintDof(*section, p, f, s->getConstraintDimension(p, f));CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
    for(PetscInt p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetConstraintIndices(*section, p, (PetscInt *) s->getConstraintDof(p));CHKERRQ(ierr);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldConstraintIndices(*section, p, f, (PetscInt *) s->getConstraintDof(p, f));CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetSection"
PetscErrorCode DMMeshSetSection(DM dm, const char name[], PetscSection section) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  {
    const Obj<PETSC_MESH_TYPE::real_section_type>& s = mesh->getRealSection(name);
    PetscInt pStart, pEnd, numFields;

    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    s->setChart(PETSC_MESH_TYPE::real_section_type::chart_type(pStart, pEnd));
    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    for(PetscInt f = 0; f < numFields; ++f) {
      PetscInt comp;
      ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRQ(ierr);
      s->addSpace(comp);
    }
    for(PetscInt p = pStart; p < pEnd; ++p) {
      PetscInt fDim, cDim;

      ierr = PetscSectionGetDof(section, p, &fDim);CHKERRQ(ierr);
      s->setFiberDimension(p, fDim);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldDof(section, p, f, &fDim);CHKERRQ(ierr);
        s->setFiberDimension(p, fDim, f);
      }
      ierr = PetscSectionGetConstraintDof(section, p, &cDim);CHKERRQ(ierr);
      if (cDim) {
        s->setConstraintDimension(p, cDim);
        for(PetscInt f = 0; f < numFields; ++f) {
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &cDim);CHKERRQ(ierr);
          s->setConstraintDimension(p, cDim, f);
        }
      }
    }
    s->allocatePoint();
    for(PetscInt p = pStart; p < pEnd; ++p) {
      PetscInt *indices;

      ierr = PetscSectionGetConstraintIndices(section, p, &indices);CHKERRQ(ierr);
      s->setConstraintDof(p, indices);
      for(PetscInt f = 0; f < numFields; ++f) {
        ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &indices);CHKERRQ(ierr);
        s->setConstraintDof(p, indices, f);
      }
    }
    {
      PetscBool isDefault;

      ierr = PetscStrcmp(name, "default", &isDefault);CHKERRQ(ierr);
      if (isDefault) {
        PetscInt maxDof = 0;

        for(PetscInt p = pStart; p < pEnd; ++p) {
          PetscInt fDim;

          ierr = PetscSectionGetDof(section, p, &fDim);CHKERRQ(ierr);
          maxDof = PetscMax(maxDof, fDim);
        }
        mesh->setMaxDof(maxDof);
      }
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
    typedef ALE::ISieveVisitor::RestrictVecVisitor<PetscScalar> visitor_type;
    PetscSection section;
    PetscScalar *array;
    PetscInt     numFields;

    ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    const PetscInt size = mesh->sizeWithBC(section, point); /* OPT: This can be precomputed */
    ierr = DMGetWorkArray(dm, 2*size+numFields+1, &array);CHKERRQ(ierr);
    visitor_type rV(v, section, size, array, (PetscInt *) &array[2*size], (PetscInt *) &array[size]);
    if (mesh->depth() == 1) {
      rV.visitPoint(point, 0);
      // Cone is guarateed to be ordered correctly
      mesh->getSieve()->orientedCone(point, rV);
    } else {
      ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type,visitor_type> pV((int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, rV, true);

      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), point, pV);
    }
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
    typedef ALE::ISieveVisitor::UpdateVecVisitor<PetscScalar> visitor_type;
    PetscSection section;
    PetscInt    *fieldSize;
    PetscInt     numFields;

    ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFields, (PetscScalar **) &fieldSize);CHKERRQ(ierr);
    mesh->sizeWithBC(section, point, fieldSize); /* OPT: This can be precomputed */
    visitor_type uV(v, section, values, mode, numFields, fieldSize);
    if (mesh->depth() == 1) {
      uV.visitPoint(point, 0);
      // Cone is guarateed to be ordered correctly
      mesh->getSieve()->orientedCone(point, uV);
    } else {
      ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type,visitor_type> pV((int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())+1, uV, true);

      ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*mesh->getSieve(), point, pV);
    }
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
    typedef ALE::ISieveVisitor::IndicesVisitor<PetscSection,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s = mesh->getRealSection("default");
    const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder = mesh->getFactory()->getGlobalOrder(mesh, s->getName(), s);
    PetscSection section;
    PetscInt     numFields;
    PetscInt    *fieldSize = PETSC_NULL;

    ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    if (numFields) {
      ierr = DMGetWorkArray(dm, numFields, (PetscScalar **) &fieldSize);CHKERRQ(ierr);
      mesh->sizeWithBC(section, point, fieldSize); /* OPT: This can be precomputed */
    }
    visitor_type iV(section, *globalOrder, (int) pow((double) mesh->getSieve()->getMaxConeSize(), mesh->depth())*mesh->getMaxDof(), mesh->depth() > 1, fieldSize);

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
PetscErrorCode DMMeshSetSectionReal(DM dm, const char name[], SectionReal section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
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


/******************************** FEM Support **********************************/

#undef __FUNCT__
#define __FUNCT__ "DMMeshPrintCellVector"
PetscErrorCode DMMeshPrintCellVector(PetscInt c, const char name[], PetscInt len, const PetscScalar x[]) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %d Element %s\n", c, name);CHKERRQ(ierr);
  for(PetscInt f = 0; f < len; ++f) {
    PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", x[f]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshPrintCellMatrix"
PetscErrorCode DMMeshPrintCellMatrix(PetscInt c, const char name[], PetscInt rows, PetscInt cols, const PetscScalar A[]) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %d Element %s\n", c, name);CHKERRQ(ierr);
  for(int f = 0; f < rows; ++f) {
    PetscPrintf(PETSC_COMM_SELF, "  |");
    for(int g = 0; g < cols; ++g) {
      PetscPrintf(PETSC_COMM_SELF, " % 9.5g", A[f*cols+g]);
    }
    PetscPrintf(PETSC_COMM_SELF, " |\n");
  }
  PetscFunctionReturn(0);
}
