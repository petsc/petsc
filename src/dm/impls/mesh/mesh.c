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
  DM_Mesh *mesh = (DM_Mesh*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This method is only valid for C++ implementation meshes.");}
  m = mesh->m;
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
    mesh->view("Mesh");
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
#define __FUNCT__ "DMMeshView_Mesh_Ascii"
PetscErrorCode DMMeshView_Mesh_Ascii(DM dm, PetscViewer viewer)
{
  DM_Mesh          *mesh = (DM_Mesh *) dm->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const char *name;
    PetscInt    maxConeSize, maxSupportSize;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMMeshGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh '%s':\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "Max sizes cone: %d support: %d\n", maxConeSize, maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "orientation is missing\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "cap --> base:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for(s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d ----> %d\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "base <-- cap:\n", name);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for(c = off; c < off+dof; ++c) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d <---- %d\n", rank, p, mesh->cones[c]);CHKERRQ(ierr);
        /* ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %d <---- %d: %d\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]);CHKERRQ(ierr); */
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscSectionVecView(mesh->coordSection, mesh->coordinates, viewer);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm = ((PetscObject) dm)->comm;
    PetscInt   *sizes;
    PetscInt    depth, dim, d;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt size;

    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMMeshGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&depth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &sizes);CHKERRQ(ierr);
    ierr = DMMeshGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
    if (depth == 2) {
      ierr = DMMeshGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", 0);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMMeshGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", dim);CHKERRQ(ierr);
      for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      for(d = 0; d <= dim; d++) {
        ierr = DMMeshGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd = pEnd - pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", d);CHKERRQ(ierr);
        for(p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %d", sizes[p]);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(sizes);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (mesh->useNewImpl) {
    PetscBool      iascii, isbinary;

    ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);

    if (iascii) {
      ierr = DMMeshView_Mesh_Ascii(dm, viewer);CHKERRQ(ierr);
#if 0
    } else if (isbinary) {
      ierr = DMMeshView_Mesh_Binary(dm, viewer);CHKERRQ(ierr);
#endif
    } else {
      SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
    }
  } else {
    ierr = DMMeshView_Sieve(mesh->m, viewer);CHKERRQ(ierr);
  }
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

#if 0
#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateAllocationVectors"
PetscErrorCode DMMeshCreateAllocationVectors(DM dm, PetscInt bs, PetscSF sf, PetscSection const ALE::Obj<Order>& globalOrder, const ALE::Obj<ALE::Mesh<PetscInt,PetscScalar>::sieve_type>& adjGraph, PetscBool isSymmetric, PetscInt dnz[], PetscInt onz[])
{
  PetscInt                          numLocalRows = globalOrder->getLocalSize();
  PetscInt                          firstRow     = globalOrder->getGlobalOffsets()[atlas->commRank()];
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMemzero(dnz, numLocalRows/bs * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows/bs * sizeof(PetscInt));CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    PetscBool isOwned;

    ierr = PetscSF(sf, p, isOwned);CHKERRQ(ierr);
    if (isOwned) {
      const ALE::Obj<typename FlexMesh::sieve_type::coneSequence>& adj   = adjGraph->cone(point);
      const typename Order::value_type& rIdx  = globalOrder->restrictPoint(point)[0];
      const int                         row   = rIdx.prefix;
      const int                         rSize = rIdx.index/bs;

      if ((atlas->debug() > 1) && ((bs == 1) || (rIdx.index%bs == 0))) std::cout << "["<<adjGraph->commRank()<<"]: row "<<row<<": size " << rIdx.index << " bs "<<bs<<std::endl;
      if (rSize == 0) continue;
      for(typename FlexMesh::sieve_type::coneSequence::iterator v_iter = adj->begin(); v_iter != adj->end(); ++v_iter) {
        const typename Atlas::point_type& neighbor = *v_iter;
        const typename Order::value_type& cIdx     = globalOrder->restrictPoint(neighbor)[0];
        const int                         col      = cIdx.prefix>=0 ? cIdx.prefix : -(cIdx.prefix+1);
        const int&                        cSize    = cIdx.index/bs;

        if ((atlas->debug() > 1) && ((bs == 1) || (cIdx.index%bs == 0))) std::cout << "["<<adjGraph->commRank()<<"]:   col "<<col<<": size " << cIdx.index << " bs "<<bs<<std::endl;
        if (cSize > 0) {
          if (isSymmetric && (col < row)) {
            if (atlas->debug() > 1) {std::cout << "["<<adjGraph->commRank()<<"]: Rejecting row "<<row<<" col " << col <<std::endl;}
            continue;
          }
          if (globalOrder->isLocal(neighbor)) {
            for(int r = 0; r < rSize; ++r) {dnz[(row - firstRow)/bs + r] += cSize;}
          } else {
            for(int r = 0; r < rSize; ++r) {onz[(row - firstRow)/bs + r] += cSize;}
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshPreallocateOperator"
PetscErrorCode DMMeshPreallocateOperator(DM dm, PetscInt bs, PetscSection section, PetscInt dnz[], PetscInt onz[], PetscBool isSymmetric, Mat A, PetscBool fillMatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create local adjacency graph */
  ierr = createLocalAdjacencyGraph(mesh, atlas, adjGraph);CHKERRQ(ierr);
  if (debug) adjGraph->view("Adjacency Graph");
  // Complete adjacency graph
  typedef ALE::ConeSection<FlexMesh::sieve_type>              cones_wrapper_type;
  typedef ALE::Section<ALE::Pair<int, point_type>, point_type> cones_type;
  Obj<cones_wrapper_type>       cones          = new cones_wrapper_type(adjGraph);
  Obj<cones_type>               overlapCones   = new cones_type(adjGraph->comm(), adjGraph->debug());
  const Obj<send_overlap_type>& sendOverlap    = mesh->getSendOverlap();
  const Obj<recv_overlap_type>& recvOverlap    = mesh->getRecvOverlap();
  const Obj<send_overlap_type>  nbrSendOverlap = new send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<recv_overlap_type>  nbrRecvOverlap = new recv_overlap_type(mesh->comm(), mesh->debug());

  // Now overlapCones will have the neighbors for any point in the overlap, in the remote numbering
  ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, cones, overlapCones);
  if (debug) overlapCones->view("Overlap Cones");
  // TODO Copy overlaps
  sendOverlap->copy(nbrSendOverlap.ptr());
  recvOverlap->copy(nbrRecvOverlap.ptr());
  if (debug) nbrSendOverlap->view("Initial Send Overlap");
  if (debug) nbrRecvOverlap->view("Initial Recv Overlap");
  // TODO Update neighbor send overlap from local adjacency
  //   For each localPoint in sendOverlap
  //     For each rank receiving this point
  //       For each adjPoint in adjGraph->cone(point)
  //         If recvOverlap does not contain an arrow (rank, adjPoint, *), meaning the point is not interior to the domain
  //           nbrSendOverlap->addArrow(adjPoint, rank, -1)
  const typename send_overlap_type::baseSequence::iterator sBegin = sendOverlap->baseBegin();
  const typename send_overlap_type::baseSequence::iterator sEnd   = sendOverlap->baseEnd();

  for(typename send_overlap_type::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
    const typename send_overlap_type::target_type            rank   = *r_iter;
    const typename send_overlap_type::coneSequence::iterator pBegin = sendOverlap->coneBegin(*r_iter);
    const typename send_overlap_type::coneSequence::iterator pEnd   = sendOverlap->coneEnd(*r_iter);

    for(typename send_overlap_type::coneSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
      const typename send_overlap_type::source_type               localPoint = *p_iter;
      const typename FlexMesh::sieve_type::coneSequence::iterator adjBegin   = adjGraph->cone(localPoint)->begin();
      const typename FlexMesh::sieve_type::coneSequence::iterator adjEnd     = adjGraph->cone(localPoint)->end();

      for(typename FlexMesh::sieve_type::coneSequence::iterator a_iter = adjBegin; a_iter != adjEnd; ++a_iter) {
        const typename FlexMesh::sieve_type::coneSequence::iterator::value_type adjPoint = *a_iter;

        // Deal with duplication at the assembly stage
        nbrSendOverlap->addArrow(adjPoint, rank, -1);
      }
    }
  }
  nbrSendOverlap->assemble();
  nbrSendOverlap->assemblePoints();
  if (debug) nbrSendOverlap->view("Modified Send Overlap");
  //   Let maxPoint be the first point not contained in adjGraph
  point_type maxPoint = std::max(*std::max_element(adjGraph->cap()->begin(),  adjGraph->cap()->end()),
                                 *std::max_element(adjGraph->base()->begin(), adjGraph->base()->end())) + 1;
  // TODO Update neighbor recv overlap and local adjacency
  //   For each point in recvOverlap
  //     For each rank sending this point
  //       For each adjPoint in the overlap cone from adjGraph for this point
  //         If adjPoint is interior, meaning sendOverlap has no arrow (rank, *, adjPoint) CAN THIS EVER HAPPEN???
  //           If nbrRevOverlap has arrow (rank, newPoint, adjPoint)
  //             Let newPoint = maxPoint, increment maxPoint
  //             Add arrows (point, newPoint) and (newPoint, point) to adjGraph
  //           Else
  //             Add arrows (point, newPoint) and (newPoint, point) to adjGraph
  //         Else
  //           Why would we see a new connection for an old point??? Need an example
  //           We have the arrow (rank, oldPoint, adjPoint)
  //           Add arrows (point, oldPoint) and (oldPoint, point) to adjGraph
  const typename recv_overlap_type::capSequence::iterator rBegin = recvOverlap->capBegin();
  const typename recv_overlap_type::capSequence::iterator rEnd   = recvOverlap->capEnd();

  for(typename recv_overlap_type::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
    const int                                                   rank   = *r_iter;
    const typename recv_overlap_type::supportSequence::iterator pBegin = recvOverlap->supportBegin(*r_iter);
    const typename recv_overlap_type::supportSequence::iterator pEnd   = recvOverlap->supportEnd(*r_iter);

    for(typename recv_overlap_type::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
      const point_type&                      localPoint  = *p_iter;
      const point_type&                      remotePoint = p_iter.color();
      const int                              size        = overlapCones->getFiberDimension(typename cones_type::point_type(rank, remotePoint));
      const typename cones_type::value_type *values      = overlapCones->restrictPoint(typename cones_type::point_type(rank, remotePoint));

      for(int i = 0; i < size; ++i) {
        const typename recv_overlap_type::supportSequence::iterator newPointsBegin = nbrRecvOverlap->supportBegin(rank, values[i]);
        const int                                                   numNewPoints   = nbrRecvOverlap->getSupportSize(rank, values[i]);
        point_type                                                  newPoint;

        if (!numNewPoints) {
          typename Mesh::order_type::value_type value(-1, 0);

          newPoint = maxPoint++;
          globalOrder->updatePoint(newPoint, &value); // Mark the new point as nonlocal
          nbrRecvOverlap->addArrow(rank, newPoint, values[i]);
        } else {
          newPoint = *newPointsBegin;
        }
        adjGraph->addArrow(newPoint,   localPoint);
        adjGraph->addArrow(localPoint, newPoint);
      }
    }
  }
  nbrRecvOverlap->assemble();
  nbrRecvOverlap->assemblePoints();
  if (debug) nbrRecvOverlap->view("Modified Recv Overlap");
  if (debug) adjGraph->view("Modified Adjacency Graph");
  /* Update global order */
  mesh->getFactory()->completeOrder(globalOrder, nbrSendOverlap, nbrRecvOverlap);
  if (debug) globalOrder->view("Modified Global Order");


  /* Read out adjacency graph */
  ierr = createAllocationVectors(bs, atlas, globalOrder, adjGraph, isSymmetric, dnz, onz);
  /* Set matrix pattern */
  ierr = MatSeqAIJSetPreallocation(A, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A, bs, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A, bs, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A, bs, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A, bs, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* Fill matrix with zeros */
  if (fillMatrix) {
    ierr = fillMatrixWithZero(A, bs, atlas, globalOrder, adjGraph, isSymmetric, dnz, onz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Mesh"
PetscErrorCode DMCreateMatrix_Mesh(DM dm, const MatType mtype, Mat *J)
{
  DM_Mesh               *mesh = (DM_Mesh *) dm->data;
  ISLocalToGlobalMapping ltog;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  if (!mtype) mtype = MATAIJ;
  if (mesh->useNewImpl) {
    PetscSection   section;
    PetscInt       bs = -1;
    PetscInt       localSize  = order->getLocalSize();
    PetscInt       globalSize = order->getGlobalSize();
    PetscBool      isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock, isSymmetric;
    PetscErrorCode ierr;

    ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = MatCreate(((PetscObject) dm)->comm, J);CHKERRQ(ierr);
    ierr = MatSetSizes(*J, localSize, localSize, globalSize, globalSize);CHKERRQ(ierr);
    ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSHELL, &isShell);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATBAIJ, &isBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSBAIJ, &isSymBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock);CHKERRQ(ierr);
    ierr = PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock);CHKERRQ(ierr);
    /* Check for symmetric storage */
    isSymmetric = (PetscBool) (isSymBlock || isSymSeqBlock || isSymMPIBlock);
    if (isSymmetric) {
      ierr = MatSetOption(*J, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE);CHKERRQ(ierr);
    }
    if (!isShell) {
      PetscInt *dnz, *onz, bsLocal;

      if (bs < 0) {
        if (isBlock || isSeqBlock || isMPIBlock || isSymBlock || isSymSeqBlock || isSymMPIBlock) {
          const typename Section::chart_type& chart = section->getChart();

          ierr = DMMeshGetChart();CHKERRQ(ierr);
          for(typename Section::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            if (section->getFiberDimension(*c_iter)) {
              bs = section->getFiberDimension(*c_iter);
              break;
            }
          }
        } else {
          bs = 1;
        }
        /* Must have same blocksize on all procs (some might have no points) */
        bsLocal = bs;
        ierr = MPI_Allreduce(&bsLocal, &bs, 1, MPIU_INT, MPI_MAX, ((PetscObject) dm)->comm);CHKERRQ(ierr);
      }
      ierr = PetscMalloc2(localSize/bs, PetscInt, &dnz, localSize/bs, PetscInt, &onz);CHKERRQ(ierr);
      //ierr = DMMeshPreallocateOperator(dm, bs, section, dnz, onz, isSymmetric, *J, !dm->prealloc_only);CHKERRQ(ierr);
      ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
    SectionReal section;
    PetscBool   flag;
    ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
    ierr = DMMeshGetSectionReal(dm, "default", &section);CHKERRQ(ierr);
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    try {
      ierr = DMMeshCreateMatrix(m, s, mtype, J, -1, !dm->prealloc_only);CHKERRQ(ierr);
    } catch(ALE::Exception e) {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_LIB, e.message());
    }
    ierr = SectionRealDestroy(&section);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject) *J, "DM", (PetscObject) dm);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm, &ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J, ltog, ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Mesh"
PetscErrorCode DMDestroy_Mesh(DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  SieveLabel     next = mesh->labels;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mesh->m = PETSC_NULL;
  ierr = PetscSectionDestroy(&mesh->defaultSection);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mesh->globalScatter);CHKERRQ(ierr);

  /* NEW_MESH_IMPL */
  ierr = PetscSFDestroy(&mesh->sf);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->cones);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->coordSection);CHKERRQ(ierr);
  ierr = VecDestroy(&mesh->coordinates);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->meetTmpA,mesh->meetTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->joinTmpA,mesh->joinTmpB);CHKERRQ(ierr);
  ierr = PetscFree2(mesh->closureTmpA,mesh->closureTmpB);CHKERRQ(ierr);
  while(next) {
    SieveLabel tmp;

    ierr = PetscFree(next->name);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumValues);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumOffsets);CHKERRQ(ierr);
    ierr = PetscFree(next->stratumSizes);CHKERRQ(ierr);
    ierr = PetscFree(next->points);CHKERRQ(ierr);
    tmp  = next->next;
    ierr = PetscFree(next);CHKERRQ(ierr);
    next = tmp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Mesh"
PetscErrorCode DMCreateGlobalVector_Mesh(DM dm, Vec *gvec)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscInt       localSize, globalSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mesh->useNewImpl) {
    PetscSection s;

    ierr = DMMeshGetDefaultSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscSectionGetOwnedStorageSize(s, mesh->sf, &localSize);CHKERRQ(ierr);
    globalSize = PETSC_DETERMINE;
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    PetscBool                 flag;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(((PetscObject) dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
    const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", m->getRealSection("default"));

    localSize  = order->getLocalSize();
    globalSize = order->getGlobalSize();
  }
  ierr = VecCreate(((PetscObject) dm)->comm, gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, localSize, globalSize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *gvec, "DM", (PetscObject) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Mesh"
PetscErrorCode DMCreateLocalVector_Mesh(DM dm, Vec *lvec)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscInt       size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mesh->useNewImpl) {
    PetscSection s;

    ierr = DMMeshGetDefaultSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(s, &size);CHKERRQ(ierr);
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    PetscBool                 flag;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ierr = DMMeshHasSectionReal(dm, "default", &flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(((PetscObject) dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
    size = m->getRealSection("default")->getStorageSize();
  }
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

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:
. dim - The topological mesh dimension

  Level: beginner

.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshGetDimension(DM dm, PetscInt *dim)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dim, 2);
  if (mesh->useNewImpl) {
    *dim = mesh->dim;
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *dim = m->getDimension();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetDimension"
/*@
  DMMeshSetDimension - Set the topological mesh dimension

  Collective on mesh

  Input Parameters:
+ mesh - The DMMesh
- dim - The topological mesh dimension

  Level: beginner

.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshSetDimension(DM dm, PetscInt dim)
{
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  if (mesh->useNewImpl) {
    mesh->dim = dim;
  } else {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Cannot reset dimension of C++ mesh");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetChart"
/*@
  DMMeshGetChart - Return the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameters:
+ pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetChart()
@*/
PetscErrorCode DMMeshGetChart(DM dm, PetscInt *pStart, PetscInt *pEnd)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    ierr = PetscSectionGetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    if (pStart) {*pStart = m->getSieve()->getChart().min();}
    if (pEnd)   {*pEnd   = m->getSieve()->getChart().max();}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetChart"
/*@
  DMMeshSetChart - Set the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameters:
+ mesh - The DMMesh
. pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Output Parameters:

  Level: beginner

.seealso: DMMeshCreate(), DMMeshGetChart()
@*/
PetscErrorCode DMMeshSetChart(DM dm, PetscInt pStart, PetscInt pEnd)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    ierr = PetscSectionSetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    m->getSieve()->setChart(PETSC_MESH_TYPE::sieve_type::chart_type(pStart, pEnd));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetConeSize"
/*@
  DMMeshGetConeSize - Return the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
- p - The Sieve point, which must lie in the chart set with DMMeshSetChart()

  Output Parameter:
. size - The cone size for point p

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetConeSize(), DMMeshSetChart()
@*/
PetscErrorCode DMMeshGetConeSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  if (mesh->useNewImpl) {
    ierr = PetscSectionGetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *size = m->getSieve()->getConeSize(p);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetConeSize"
/*@
  DMMeshSetConeSize - Set the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
. p - The Sieve point, which must lie in the chart set with DMMeshSetChart()
- size - The cone size for point p

  Output Parameter:

  Note:
  This should be called after DMMeshSetChart().

  Level: beginner

.seealso: DMMeshCreate(), DMMeshGetConeSize(), DMMeshSetChart()
@*/
PetscErrorCode DMMeshSetConeSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    ierr = PetscSectionSetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
    mesh->maxConeSize = PetscMax(mesh->maxConeSize, size);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    m->getSieve()->setConeSize(p, size);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCone"
/*@C
  DMMeshGetCone - Return the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
- p - The Sieve point, which must lie in the chart set with DMMeshSetChart()

  Output Parameter:
. cone - An array of points which are on the in-edges for point p

  Level: beginner

  Note:
  This routine is not available in Fortran.

.seealso: DMMeshCreate(), DMMeshSetCone(), DMMeshSetChart()
@*/
PetscErrorCode DMMeshGetCone(DM dm, PetscInt p, const PetscInt *cone[])
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  if (mesh->useNewImpl) {
    PetscInt off;

    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    *cone = &mesh->cones[off];
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> v(m->getSieve()->getConeSize(p));

    m->getSieve()->cone(p, v);
    if (!mesh->meetTmpA) {ierr = PetscMalloc2(m->getSieve()->getMaxConeSize(),PetscInt,&mesh->meetTmpA,m->getSieve()->getMaxConeSize(),PetscInt,&mesh->meetTmpB);CHKERRQ(ierr);}
    for(size_t c = 0; c < v.getSize(); ++c) {
      mesh->meetTmpA[c] = v.getPoints()[c];
    }
    *cone = mesh->meetTmpA;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetCone"
/*@
  DMMeshSetCone - Set the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
. p - The Sieve point, which must lie in the chart set with DMMeshSetChart()
- cone - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMMeshSetConeSize() and DMMeshSetUp().

  Level: beginner

.seealso: DMMeshCreate(), DMMeshGetCone(), DMMeshSetChart(), DMMeshSetConeSize(), DMMeshSetUp()
@*/
PetscErrorCode DMMeshSetCone(DM dm, PetscInt p, const PetscInt cone[])
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  if (mesh->useNewImpl) {
    PetscInt pStart, pEnd;
    PetscInt dof, off, c;

    ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for(c = 0; c < dof; ++c) {
      if ((cone[c] < pStart) || (cone[c] >= pEnd)) {SETERRQ3(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone point %d is not in the valid range [%d. %d)", cone[c], pStart, pEnd);}
      mesh->cones[off+c] = cone[c];
    }
  } else {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "This method does not make sense for the C++ Sieve implementation");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetSupportSize"
/*@
  DMMeshGetSupportSize - Return the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
- p - The Sieve point, which must lie in the chart set with DMMeshSetChart()

  Output Parameter:
. size - The support size for point p

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetConeSize(), DMMeshSetChart(), DMMeshGetConeSize()
@*/
PetscErrorCode DMMeshGetSupportSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  if (mesh->useNewImpl) {
    ierr = PetscSectionGetDof(mesh->supportSection, p, size);CHKERRQ(ierr);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *size = m->getSieve()->getSupportSize(p);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetSupport"
/*@C
  DMMeshGetSupport - Return the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
- p - The Sieve point, which must lie in the chart set with DMMeshSetChart()

  Output Parameter:
. support - An array of points which are on the out-edges for point p

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetCone(), DMMeshSetChart(), DMMeshGetCone()
@*/
PetscErrorCode DMMeshGetSupport(DM dm, PetscInt p, const PetscInt *support[])
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  if (mesh->useNewImpl) {
    PetscInt off;

    ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
    *support = &mesh->supports[off];
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> v(m->getSieve()->getSupportSize(p));

    m->getSieve()->support(p, v);
    if (!mesh->joinTmpA) {ierr = PetscMalloc2(m->getSieve()->getMaxSupportSize(),PetscInt,&mesh->joinTmpA,m->getSieve()->getMaxSupportSize(),PetscInt,&mesh->joinTmpB);CHKERRQ(ierr);}
    for(size_t s = 0; s < v.getSize(); ++s) {
      mesh->joinTmpA[s] = v.getPoints()[s];
    }
    *support = mesh->joinTmpA;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetTransitiveClosure"
/*@C
  DMMeshGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMMesh
. p - The Sieve point, which must lie in the chart set with DMMeshSetChart()
- useCone - PETSC_TRUE for in-edges,  otherwise use out-edges

  Output Parameters:
+ numPoints - The number of points in the closure
- points - The points

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetCone(), DMMeshSetChart(), DMMeshGetCone()
@*/
PetscErrorCode DMMeshGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, const PetscInt *points[])
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->closureTmpA) {
    PetscInt maxSize;
    if (mesh->useNewImpl) {
      maxSize = PetscMax(mesh->maxConeSize, mesh->maxSupportSize)+1;
    } else {
      Obj<PETSC_MESH_TYPE> m;
      ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
      maxSize = PetscMax(m->getSieve()->getMaxConeSize(), m->getSieve()->getMaxSupportSize())+1;
    }
    ierr = PetscMalloc2(maxSize,PetscInt,&mesh->closureTmpA,maxSize,PetscInt,&mesh->closureTmpB);CHKERRQ(ierr);
  }
  if (mesh->useNewImpl) {
    const PetscInt *tmp;
    PetscInt        tmpSize, t;
    PetscInt        closureSize = 1;

    mesh->closureTmpA[0] = p;
    /* This is only 1-level */
    if (useCone) {
      ierr = DMMeshGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
      ierr = DMMeshGetCone(dm, p, &tmp);CHKERRQ(ierr);
    } else {
      ierr = DMMeshGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
      ierr = DMMeshGetSupport(dm, p, &tmp);CHKERRQ(ierr);
    }
    for(t = 0; t < tmpSize; ++t) {
      mesh->closureTmpA[closureSize++] = tmp[t];
    }
    if (numPoints) *numPoints = closureSize;
    if (points)    *points    = mesh->closureTmpA;
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    typedef ALE::ISieveVisitor::TransitiveClosureVisitor<PETSC_MESH_TYPE::sieve_type> visitor_type;
    visitor_type::visitor_type nV;
    visitor_type               cV(*m->getSieve(), nV);

    if (useCone) {
      m->getSieve()->cone(p, cV);
    } else {
      cV.setIsCone(false);
      m->getSieve()->support(p, cV);
    }
    int i = 0;

    for(std::set<PETSC_MESH_TYPE::point_type>::const_iterator p_iter = cV.getPoints().begin(); p_iter != cV.getPoints().end(); ++p_iter, ++i) {
      mesh->closureTmpA[i] = *p_iter;
    }
    if (numPoints) *numPoints = cV.getPoints().size();
    if (points)    *points    = mesh->closureTmpA;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetMaxSizes"
/*@
  DMMeshGetMaxSizes - Return the maximum number of in-edges (cone) and out-edges (support) for any point in the Sieve DAG

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameters:
+ maxConeSize - The maximum number of in-edges
- maxSupportSize - The maximum number of out-edges

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetConeSize(), DMMeshSetChart()
@*/
PetscErrorCode DMMeshGetMaxSizes(DM dm, PetscInt *maxConeSize, PetscInt *maxSupportSize)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    if (maxConeSize)    *maxConeSize    = mesh->maxConeSize;
    if (maxSupportSize) *maxSupportSize = mesh->maxSupportSize;
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    if (maxConeSize)    *maxConeSize    = m->getSieve()->getMaxConeSize();
    if (maxSupportSize) *maxSupportSize = m->getSieve()->getMaxSupportSize();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetUp"
/*@
  DMMeshSetUp - Allocate space for the Sieve DAG

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:

  Note:
  This should be called after DMMeshSetChart() and all calls to DMMeshSetConeSize()

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetChart(), DMMeshSetConeSize()
@*/
PetscErrorCode DMMeshSetUp(DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    PetscInt size;

    ierr = PetscSectionSetUp(mesh->coneSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->coneSection, &size);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &mesh->cones);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSymmetrize"
/*@
  DMMeshSymmetrize - Creates support (out-edge) information from cone (in-edge) inoformation

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:

  Note:
  This should be called after all calls to DMMeshSetCone()

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSetChart(), DMMeshSetConeSize(), DMMeshSetCone()
@*/
PetscErrorCode DMMeshSymmetrize(DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    PetscInt *offsets;
    PetscInt  supportSize;
    PetscInt  pStart, pEnd, p;

    /* Calculate support sizes */
    ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(mesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for(c = off; c < off+dof; ++c) {
        ierr = PetscSectionAddDof(mesh->supportSection, mesh->cones[c], 1);CHKERRQ(ierr);
      }
    }
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, dof);
    }
    ierr = PetscSectionSetUp(mesh->supportSection);CHKERRQ(ierr);
    /* Calculate supports */
    ierr = PetscSectionGetStorageSize(mesh->supportSection, &supportSize);CHKERRQ(ierr);
    ierr = PetscMalloc(supportSize * sizeof(PetscInt), &mesh->supports);CHKERRQ(ierr);
    ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &offsets);CHKERRQ(ierr);
    ierr = PetscMemzero(offsets, (pEnd - pStart) * sizeof(PetscInt));CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for(c = off; c < off+dof; ++c) {
        const PetscInt q = mesh->cones[c];
        PetscInt       offS;

        ierr = PetscSectionGetOffset(mesh->supportSection, q, &offS);CHKERRQ(ierr);
        mesh->supports[offS+offsets[q]] = p;
        ++offsets[q];
      }
    }
    ierr = PetscFree(offsets);CHKERRQ(ierr);
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    m->getSieve()->symmetrize();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshStratify"
/*@
  DMMeshStratify - The Sieve DAG for most topologies is a graded poset (http://en.wikipedia.org/wiki/Graded_poset), and
  can be illustrated by Hasse Diagram (a http://en.wikipedia.org/wiki/Hasse_diagram). The strata group all points of the
  same grade, and this function calculates the strata. This grade can be seen as the height (or depth) of the point in
  the DAG.

  Not collective

  Input Parameter:
. mesh - The DMMesh

  Output Parameter:

  Notes:
  The normal association for the point grade is element dimension (or co-dimension). For instance, all vertices would
  have depth 0, and all edges depth 1. Likewise, all cells heights would have height 0, and all faces height 1.

  This should be called after all calls to DMMeshSymmetrize()

  Level: beginner

.seealso: DMMeshCreate(), DMMeshSymmetrize()
@*/
PetscErrorCode DMMeshStratify(DM dm)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    PetscInt pStart, pEnd, p;
    PetscInt numRoots = 0, numLeaves = 0;

    /* Calculate depth */
    ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    /* Initialize roots and count leaves */
    for(p = pStart; p < pEnd; ++p) {
      PetscInt coneSize, supportSize;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &coneSize);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(mesh->supportSection, p, &supportSize);CHKERRQ(ierr);
      if (!coneSize && supportSize) {
        ++numRoots;
        ierr = DMMeshSetLabelValue(dm, "depth", p, 0);CHKERRQ(ierr);
      } else if (!supportSize && coneSize) {
        ++numLeaves;
      }
    }
    if (numRoots + numLeaves == (pEnd - pStart)) {
      for(p = pStart; p < pEnd; ++p) {
        PetscInt coneSize, supportSize;

        ierr = PetscSectionGetDof(mesh->coneSection, p, &coneSize);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(mesh->supportSection, p, &supportSize);CHKERRQ(ierr);
        if (!supportSize && coneSize) {
          ierr = DMMeshSetLabelValue(dm, "depth", p, 1);CHKERRQ(ierr);
        }
      }
    } else {
      SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Have not yet coded general stratification");
    }
  } else {
    Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    m->stratify();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLabelValue"
/*@C
  DMMeshGetLabelValue - Get the value in a Sieve Label for the given point, with 0 as the default

  Not Collective

  Input Parameters:
+ dm   - The DMMesh object
. name - The label name
- point - The mesh point

  Output Parameter:
. value - The label value for this point, or 0 if the point is not in the label

  Level: beginner

.keywords: mesh
.seealso: DMMeshSetLabelValue(), DMMeshGetLabelStratum()
@*/
PetscErrorCode DMMeshGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg  = PETSC_FALSE;
    PetscInt   v, p;

    *value = 0;
    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) break;
      next = next->next;
    }
    if (!flg) {SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);CHKERRQ(ierr);}
    /* Find, or add, label value */
    for(v = 0; v < next->numStrata; ++v) {
      for(p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
        if (next->points[p] == point) {
          *value = next->stratumValues[v];
          break;
        }
      }
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *value = m->getValue(m->getLabel(name), point);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetLabelValue"
/*@C
  DMMeshSetLabelValue - Add a point to a Sieve Label with given value

  Not Collective

  Input Parameters:
+ dm   - The DMMesh object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.keywords: mesh
.seealso: DMMeshGetLabelStratum()
@*/
PetscErrorCode DMMeshSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg  = PETSC_FALSE;
    PetscInt   v, p;

    /* Find, or create, label */
    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) break;
      next = next->next;
    }
    if (!flg) {
      SieveLabel tmpLabel = mesh->labels;
      ierr = PetscNew(struct Sieve_Label, &mesh->labels);CHKERRQ(ierr);
      mesh->labels->next = tmpLabel;
      next = mesh->labels;
      ierr = PetscStrallocpy(name, &next->name);CHKERRQ(ierr);
    }
    /* Find, or add, label value */
    for(v = 0; v < next->numStrata; ++v) {
      if (next->stratumValues[v] == value) break;
    }
    if (v >= next->numStrata) {
      PetscInt *tmpV, *tmpO, *tmpS;
      ierr = PetscMalloc3(next->numStrata+1,PetscInt,&tmpV,next->numStrata+2,PetscInt,&tmpO,next->numStrata+1,PetscInt,&tmpS);CHKERRQ(ierr);
      for(v = 0; v < next->numStrata; ++v) {
        tmpV[v] = next->stratumValues[v];
        tmpO[v] = next->stratumOffsets[v];
        tmpS[v] = next->stratumSizes[v];
      }
      tmpV[v] = value;
      tmpO[v] = v == 0 ? 0 : next->stratumOffsets[v];
      tmpS[v] = 0;
      tmpO[v+1] = tmpO[v];
      ++next->numStrata;
      ierr = PetscFree3(next->stratumValues,next->stratumOffsets,next->stratumSizes);CHKERRQ(ierr);
      next->stratumValues  = tmpV;
      next->stratumOffsets = tmpO;
      next->stratumSizes   = tmpS;
    }
    /* Check whether point exists */
    for(p = next->stratumOffsets[v]; p < next->stratumOffsets[v]+next->stratumSizes[v]; ++p) {
      if (next->points[p] == point) {
        break;
      }
    }
    /* Add point: NEED TO OPTIMIZE */
    if (p >= next->stratumOffsets[v]+next->stratumSizes[v]) {
      /* Check for reallocation */
      if (next->stratumSizes[v] >= next->stratumOffsets[v+1]-next->stratumOffsets[v]) {
        PetscInt  oldSize   = next->stratumOffsets[v+1]-next->stratumOffsets[v];
        PetscInt  newSize   = PetscMax(10, 2*oldSize); /* Double the size, since 2 is the optimal base for this online algorithm */
        PetscInt  shift     = newSize - oldSize;
        PetscInt  allocSize = next->stratumOffsets[next->numStrata] + shift;
        PetscInt *newPoints;
        PetscInt  w, q;

        ierr = PetscMalloc(allocSize * sizeof(PetscInt), &newPoints);CHKERRQ(ierr);
        for(q = 0; q < next->stratumOffsets[v]+next->stratumSizes[v]; ++q) {
          newPoints[q] = next->points[q];
        }
        for(w = v+1; w < next->numStrata; ++w) {
          for(q = next->stratumOffsets[w]; q < next->stratumOffsets[w]+next->stratumSizes[w]; ++q) {
              newPoints[q+shift] = next->points[q];
          }
          next->stratumOffsets[w] += shift;
        }
        next->stratumOffsets[next->numStrata] += shift;
        ierr = PetscFree(next->points);CHKERRQ(ierr);
        next->points = newPoints;
      }
      /* Insert point and resort */
      next->points[next->stratumOffsets[v]+next->stratumSizes[v]] = point;
      ++next->stratumSizes[v];
      ierr = PetscSortInt(next->stratumSizes[v], &next->points[next->stratumOffsets[v]]);CHKERRQ(ierr);
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    m->setValue(m->getLabel(name), point, value);
  }
  PetscFunctionReturn(0);
}

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

.keywords: mesh
.seealso: DMMeshSetLabelValue()
@*/
PetscErrorCode DMMeshGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 3);
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg;

    *size = 0;
    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) {
        *size = next->numStrata;
        break;
      }
      next = next->next;
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *size = m->getLabel(name)->getCapSize();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetLabelIdIS"
/*@C
  DMMeshGetLabelIdIS - Get the integer ids in a label

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The label name

  Output Parameter:
. ids - The integer ids

  Level: beginner

.keywords: mesh
.seealso: DMMeshGetLabelSize()
@*/
PetscErrorCode DMMeshGetLabelIdIS(DM dm, const char name[], IS *ids)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscInt      *values;
  PetscInt       size, i = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ids, 3);
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg;

    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) {
        size = next->numStrata;
        ierr = PetscMalloc(size * sizeof(PetscInt), &values);CHKERRQ(ierr);
        for(i = 0; i < next->numStrata; ++i) {
          values[i] = next->stratumValues[i];
        }
        break;
      }
      next = next->next;
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    const ALE::Obj<PETSC_MESH_TYPE::label_type::capSequence>&      labelIds = m->getLabel(name)->cap();
    const PETSC_MESH_TYPE::label_type::capSequence::const_iterator iEnd     = labelIds->end();

    size = labelIds->size();
    ierr = PetscMalloc(size * sizeof(PetscInt), &values);CHKERRQ(ierr);
    for(PETSC_MESH_TYPE::label_type::capSequence::const_iterator i_iter = labelIds->begin(); i_iter != iEnd; ++i_iter, ++i) {
      values[i] = *i_iter;
    }
  }
  ierr = ISCreateGeneral(((PetscObject) dm)->comm, size, values, PETSC_OWN_POINTER, ids);CHKERRQ(ierr);
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

.keywords: mesh
.seealso: DMMeshGetLabelSize(), DMMeshGetLabelIds()
@*/
PetscErrorCode DMMeshGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(size, 4);
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg;

    *size = 0;
    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) {
        PetscInt v;

        for(v = 0; v < next->numStrata; ++v) {
          if (next->stratumValues[v] == value) {
            *size = next->stratumSizes[v];
            break;
          }
        }
        break;
      }
      next = next->next;
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    *size = m->getLabelStratum(name, value)->size();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetStratumIS"
/*@C
  DMMeshGetStratumIS - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The DMMesh object
. name - The label name
- value - The stratum value

  Output Parameter:
. is - The stratum points

  Level: beginner

.keywords: mesh
.seealso: DMMeshGetStratumSize()
@*/
PetscErrorCode DMMeshGetStratumIS(DM dm, const char name[], PetscInt value, IS *is) {
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(is, 4);
  *is = PETSC_NULL;
  if (mesh->useNewImpl) {
    SieveLabel next = mesh->labels;
    PetscBool  flg;

    while(next) {
      ierr = PetscStrcmp(name, next->name, &flg);CHKERRQ(ierr);
      if (flg) {
        PetscInt v;

        for(v = 0; v < next->numStrata; ++v) {
          if (next->stratumValues[v] == value) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF, next->stratumSizes[v], &next->points[next->stratumOffsets[v]], PETSC_COPY_VALUES, is);CHKERRQ(ierr);
            break;
          }
        }
        break;
      }
      next = next->next;
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
    if (mesh->hasLabel(name)) {
      const Obj<PETSC_MESH_TYPE::label_sequence>& stratum = mesh->getLabelStratum(name, value);
      PetscInt *idx, i = 0;

      ierr = PetscMalloc(stratum->size() * sizeof(PetscInt), &idx);CHKERRQ(ierr);
      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = stratum->begin(); e_iter != stratum->end(); ++e_iter, ++i) {
        idx[i] = *e_iter;
      }
      ierr = ISCreateGeneral(((PetscObject) dm)->comm, stratum->size(), idx, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshJoinPoints"
/* This is a 1-level join */
PetscErrorCode DMMeshJoinPoints(DM dm, const PetscInt points[], PetscInt *coveredPoint)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(coveredPoint, 3);
  if (mesh->useNewImpl) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not yet supported");
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    /* const Obj<typename Mesh::sieve_type::supportSet> edge = m->getSieve()->nJoin(points[0], points[1], 1); */
    *coveredPoint = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshMeetPoints"
/* This is a 1-level meet */
PetscErrorCode DMMeshMeetPoints(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveringPoints, const PetscInt **coveringPoints)
{
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscInt      *meet[2];
  PetscInt       meetSize, i = 0;
  PetscInt       dof, off, p, c, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveringPoints, 3);
  PetscValidPointer(coveringPoints, 4);
  if (mesh->useNewImpl) {
    if (!mesh->meetTmpA) {ierr = PetscMalloc2(mesh->maxConeSize,PetscInt,&mesh->meetTmpA,mesh->maxConeSize,PetscInt,&mesh->meetTmpB);CHKERRQ(ierr);}
    meet[0] = mesh->meetTmpA; meet[1] = mesh->meetTmpB;
    /* Copy in cone of first point */
    ierr = PetscSectionGetDof(mesh->coneSection, points[0], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, points[0], &off);CHKERRQ(ierr);
    for(meetSize = 0; meetSize < dof; ++meetSize) {
      meet[i][meetSize] = mesh->cones[off+meetSize];
    }
    /* Check each successive cone */
    for(p = 1; p < numPoints; ++p) {
      PetscInt newMeetSize = 0;

      ierr = PetscSectionGetDof(mesh->coneSection, points[p], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, points[p], &off);CHKERRQ(ierr);
      for(c = 0; c < dof; ++c) {
        const PetscInt point = mesh->cones[off+c];

        for(m = 0; m < meetSize; ++m) {
          if (point == meet[i][m]) {
            meet[1-i][newMeetSize++] = point;
            break;
          }
        }
      }
      meetSize = newMeetSize;
      i = 1-i;
    }
    *numCoveringPoints = meetSize;
    *coveringPoints    = meet[1-i];
  } else {
    ALE::Obj<PETSC_MESH_TYPE> m;
    ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
    /* const Obj<typename Mesh::sieve_type::supportSet> edge = m->getSieve()->nJoin(points[0], points[1], 1); */
    *numCoveringPoints = 0;
    *coveringPoints    = PETSC_NULL;
  }
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
  PetscValidHeaderSpecific(serialMesh, DM_CLASSID, 1);
  PetscValidPointer(parallelMesh,3);
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

#ifdef PETSC_HAVE_TRIANGLE
/* Already included since C++ is turned on #include <triangle.h> */

#undef __FUNCT__
#define __FUNCT__ "TriangleInitInput"
PetscErrorCode TriangleInitInput(struct triangulateio *inputCtx) {
  PetscFunctionBegin;
  inputCtx->numberofpoints = 0;
  inputCtx->numberofpointattributes = 0;
  inputCtx->pointlist = PETSC_NULL;
  inputCtx->pointattributelist = PETSC_NULL;
  inputCtx->pointmarkerlist = PETSC_NULL;
  inputCtx->numberofsegments = 0;
  inputCtx->segmentlist = PETSC_NULL;
  inputCtx->segmentmarkerlist = PETSC_NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->trianglelist = PETSC_NULL;
  inputCtx->numberofholes = 0;
  inputCtx->holelist = PETSC_NULL;
  inputCtx->numberofregions = 0;
  inputCtx->regionlist = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TriangleInitOutput"
PetscErrorCode TriangleInitOutput(struct triangulateio *outputCtx) {
  PetscFunctionBegin;
  outputCtx->numberofpoints = 0;
  outputCtx->pointlist = PETSC_NULL;
  outputCtx->pointattributelist = PETSC_NULL;
  outputCtx->pointmarkerlist = PETSC_NULL;
  outputCtx->numberoftriangles = 0;
  outputCtx->trianglelist = PETSC_NULL;
  outputCtx->triangleattributelist = PETSC_NULL;
  outputCtx->neighborlist = PETSC_NULL;
  outputCtx->segmentlist = PETSC_NULL;
  outputCtx->segmentmarkerlist = PETSC_NULL;
  outputCtx->edgelist = PETSC_NULL;
  outputCtx->edgemarkerlist = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TriangleFiniOutput"
PetscErrorCode TriangleFiniOutput(struct triangulateio *outputCtx) {
  PetscFunctionBegin;
  free(outputCtx->pointmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGenerate_Triangle"
PetscErrorCode DMMeshGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm             comm = ((PetscObject) boundary)->comm;
  DM_Mesh             *bd   = (DM_Mesh *) boundary->data;
  PetscInt             dim              = 2;
  const PetscBool      createConvexHull = PETSC_FALSE;
  const PetscBool      constrained      = PETSC_FALSE;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             vStart, vEnd, v, eStart, eEnd, e;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = TriangleInitInput(&in);CHKERRQ(ierr);
  ierr = TriangleInitOutput(&out);CHKERRQ(ierr);
  ierr  = DMMeshGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in.numberofpoints*dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(bd->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(bd->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMMeshGetLabelValue(boundary, "marker", v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(bd->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMMeshGetHeightStratum(boundary, 0, &eStart, &eEnd);CHKERRQ(ierr);
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    ierr = PetscMalloc(in.numberofsegments*2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofsegments   * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
    for(e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;

      ierr = DMMeshGetCone(boundary, e, &cone);CHKERRQ(ierr);
      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;
      ierr = DMMeshGetLabelValue(boundary, "marker", e, &in.segmentmarkerlist[idx]);CHKERRQ(ierr);
    }
  }
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMMeshGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
  if (in.numberofholes > 0) {
    ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
    for(h = 0; h < in.numberofholes; ++h) {
      for(d = 0; d < dim; ++d) {
        in.holelist[h*dim+d] = holeCoords[h*dim+d];
      }
    }
  }
#endif
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    if (createConvexHull) {
      ierr = PetscStrcat(args, "c");CHKERRQ(ierr);
    }
    if (constrained) {
      ierr = PetscStrcpy(args, "zepDQ");CHKERRQ(ierr);
    }
    triangulate(args, &in, &out, PETSC_NULL);
  }
  ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
  ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.holelist);CHKERRQ(ierr);

  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMMESH);CHKERRQ(ierr);
  ierr = DMMeshSetDimension(*dm, dim);CHKERRQ(ierr);
  {
    DM_Mesh       *mesh        = (DM_Mesh *) (*dm)->data;
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    int           *cells       = out.trianglelist;
    double        *meshCoords  = out.pointlist;
    PetscInt       coordSize, c;
    PetscScalar   *coords;

    ierr = DMMeshSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMMeshSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMMeshSetUp(*dm);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      PetscInt cone[numCorners] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells};

      ierr = DMMeshSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMMeshSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMMeshStratify(*dm);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(mesh->coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
    for(v = numCells; v < numCells+numVertices; ++v) {
      ierr = PetscSectionSetDof(mesh->coordSection, v, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      coords[v*dim+0] = meshCoords[v*dim+0];
      coords[v*dim+1] = meshCoords[v*dim+1];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        ierr = DMMeshSetLabelValue(*dm, "marker", v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      for(e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          PetscInt       edge;

          ierr = DMMeshJoinPoints(*dm, vertices, &edge);CHKERRQ(ierr); /* 1-level join */
          ierr = DMMeshSetLabelValue(*dm, "marker", edge, out.edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
    }
  }
#if 0 /* Do not currently support holes */
  ierr = DMMeshCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = TriangleFiniOutput(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

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
  DM_Mesh       *bd = (DM_Mesh *) boundary->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(boundary, DM_CLASSID, 1);
  /* PetscValidLogicalCollectiveLogical(dm, interpolate, 2); */
  if (bd->useNewImpl) {
    if (interpolate) {SETERRQ(((PetscObject) boundary)->comm, PETSC_ERR_SUP, "Interpolation (creation of faces and edges) is not yet supported.");}
#ifdef PETSC_HAVE_TRIANGLE
    ierr = DMMeshGenerate_Triangle(boundary, interpolate, mesh);CHKERRQ(ierr);
#endif
  } else {
    ALE::Obj<PETSC_MESH_TYPE> mB;

    ierr = DMMeshGetMesh(boundary, mB);CHKERRQ(ierr);
    ierr = DMMeshCreate(mB->comm(), mesh);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE> m = ALE::Generator<PETSC_MESH_TYPE>::generateMeshV(mB, interpolate, false, true);
    ierr = DMMeshSetMesh(*mesh, m);CHKERRQ(ierr);
  }
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
#define __FUNCT__ "DMCreateInterpolation_Mesh"
PetscErrorCode DMCreateInterpolation_Mesh(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling) {
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
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    SieveLabel next  = mesh->labels;
    PetscBool  flg   = PETSC_FALSE;
    PetscInt   depth;

    if (stratumValue < 0) {
      ierr = DMMeshGetChart(dm, start, end);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else {
      PetscInt pStart, pEnd;

      if (start) {*start = 0;}
      if (end)   {*end   = 0;}
      ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
      if (pStart == pEnd) {PetscFunctionReturn(0);}
    }
    while(next) {
      ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
      if (flg) break;
      next = next->next;
    }
    if (!flg) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);}
    /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
    depth = stratumValue;
    if ((depth < 0) || (depth >= next->numStrata)) {
      if (start) {*start = 0;}
      if (end)   {*end   = 0;}
    } else {
      if (start) {*start = next->points[next->stratumOffsets[depth]];}
      if (end)   {*end   = next->points[next->stratumOffsets[depth]+next->stratumSizes[depth]-1]+1;}
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
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
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->useNewImpl) {
    SieveLabel next  = mesh->labels;
    PetscBool  flg   = PETSC_FALSE;
    PetscInt   depth;

    if (stratumValue < 0) {
      ierr = DMMeshGetChart(dm, start, end);CHKERRQ(ierr);
    } else {
      PetscInt pStart, pEnd;

      if (start) {*start = 0;}
      if (end)   {*end   = 0;}
      ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
      if (pStart == pEnd) {PetscFunctionReturn(0);}
    }
    while(next) {
      ierr = PetscStrcmp("depth", next->name, &flg);CHKERRQ(ierr);
      if (flg) break;
      next = next->next;
    }
    if (!flg) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);}
    /* Strata are sorted and contiguous -- In addition, depth/height is either full or 1-level */
    depth = next->stratumValues[next->numStrata-1] - stratumValue;
    if ((depth < 0) || (depth >= next->numStrata)) {
      if (start) {*start = 0;}
      if (end)   {*end   = 0;}
    } else {
      if (start) {*start = next->points[next->stratumOffsets[depth]];}
      if (end)   {*end   = next->points[next->stratumOffsets[depth]+next->stratumSizes[depth]-1]+1;}
    }
  } else {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
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
  ierr = DMMeshGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
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
        const PetscInt p = idx[i];
        PetscInt       depth, numConst;

        ierr = DMMeshGetLabelValue(dm, "depth", p, &depth);CHKERRQ(ierr);
        numConst              = numDof[field*(dim+1)+depth];
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
  {
    PetscBool view = PETSC_FALSE;

    ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-section_view", &view);CHKERRQ(ierr);
    if (view) {
      ierr = PetscSectionView(*section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
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
/*
  Note: This gets a borrowed reference, so the user should not destroy this PetscSection.
*/
PetscErrorCode DMMeshGetDefaultSection(DM dm, PetscSection *section) {
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mesh->defaultSection && !mesh->useNewImpl) {
    ierr = DMMeshGetSection(dm, "default", &mesh->defaultSection);CHKERRQ(ierr);
  }
  *section = mesh->defaultSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetDefaultSection"
/*
  Note: This reference will be stolen, so the user should not destroy this PetscSection.
*/
PetscErrorCode DMMeshSetDefaultSection(DM dm, PetscSection section) {
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mesh->defaultSection = section;
  if (!mesh->useNewImpl) {
    ierr = DMMeshSetSection(dm, "default", mesh->defaultSection);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCoordinateSection"
PetscErrorCode DMMeshGetCoordinateSection(DM dm, PetscSection *section) {
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (mesh->useNewImpl) {
    *section = mesh->coordSection;
  } else {
    ierr = DMMeshGetSection(dm, "coordinates", section);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetCoordinateSection"
PetscErrorCode DMMeshSetCoordinateSection(DM dm, PetscSection section) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshSetSection(dm, "coordinates", section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetConeSection"
PetscErrorCode DMMeshGetConeSection(DM dm, PetscSection *section) {
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->useNewImpl) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This method is only valid for C implementation meshes.");}
  if (section) *section = mesh->coneSection;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCones"
PetscErrorCode DMMeshGetCones(DM dm, PetscInt *cones[]) {
  DM_Mesh *mesh = (DM_Mesh *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->useNewImpl) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "This method is only valid for C implementation meshes.");}
  if (cones) *cones = mesh->cones;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateConeSection"
PetscErrorCode DMMeshCreateConeSection(DM dm, PetscSection *section) {
  ALE::Obj<PETSC_MESH_TYPE> mesh;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, mesh->getSieve()->getChart().min(), mesh->getSieve()->getChart().max());CHKERRQ(ierr);
  for(p = mesh->getSieve()->getChart().min(); p < mesh->getSieve()->getChart().max(); ++p) {
    ierr = PetscSectionSetDof(*section, p, mesh->getSieve()->getConeSize(p));CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCoordinateVec"
PetscErrorCode DMMeshGetCoordinateVec(DM dm, Vec *coordinates) {
  DM_Mesh       *mesh = (DM_Mesh *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coordinates, 2);
  if (mesh->useNewImpl) {
    *coordinates = mesh->coordinates;
  } else {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
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
  } catch(PETSc::Exception e) {
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
