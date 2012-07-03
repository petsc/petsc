#if !defined(__PETSCDMMESH_VIEWERS_HH)
#define __PETSCDMMESH_VIEWERS_HH

#include <petscdmmesh.hh>
#include <iomanip>

using ALE::Obj;

class VTKViewer {
 public:
  VTKViewer() {};
  virtual ~VTKViewer() {};

  #undef __FUNCT__  
  #define __FUNCT__ "writeHeader"
  template<typename Mesh>
  static PetscErrorCode writeHeader(const Obj<Mesh>& mesh, PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Simplicial Mesh Example\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "writeVertices"
  template<typename Mesh>
  static PetscErrorCode writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer) {
    Obj<typename Mesh::real_section_type> coordinates;

    if (mesh->hasRealSection("coordinates_dimensioned")) {
      coordinates = mesh->getRealSection("coordinates_dimensioned");
    } else if (mesh->hasRealSection("coordinates")) {
      coordinates = mesh->getRealSection("coordinates");
    } else {
      throw ALE::Exception("Missing coordinates in mesh");
    }
    const int                                    embedDim    = coordinates->getFiberDimension(*mesh->depthStratum(0)->begin());
    Obj<typename Mesh::numbering_type>           vNumbering;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (mesh->hasLabel("censored depth")) {
      vNumbering = mesh->getFactory()->getNumbering(mesh, "censored depth", 0);
    } else {
      vNumbering = mesh->getFactory()->getNumbering(mesh, 0);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", vNumbering->getGlobalSize());CHKERRQ(ierr);
    ierr = writeSection(coordinates, embedDim, vNumbering, viewer, 3);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "writeField"
  template<typename Section>
    static PetscErrorCode writeField(const Obj<Section>& field, const std::string& name, const int fiberDim, const Obj<PETSC_MESH_TYPE::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1, int precision = 6) {
    int            dim = enforceDim > 0 ? enforceDim : fiberDim;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (enforceDim == -4) dim = 4;
    if (dim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name.c_str());CHKERRQ(ierr);
    } else {
      if (name == "") {
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS Unknown double %d\n", dim);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name.c_str(), dim);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    ierr = writeSection(field, fiberDim, numbering, viewer, enforceDim, precision);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "writeSection"
  template<typename Section>
  static PetscErrorCode writeSection(const Obj<Section>& field, const int fiberDim, const Obj<PETSC_MESH_TYPE::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1, int precision = 6) {
    typedef typename Section::value_type value_type;
    const typename Section::chart_type& chart   = field->getChart();
    const MPI_Datatype                  mpiType = ALE::New::ParallelFactory<value_type>::singleton(field->debug())->getMPIType();
    const bool                          verify  = enforceDim == -4;
    const bool                          opt2    = enforceDim == 3 && fiberDim == 2;
    const bool                          opt3    = std::max(fiberDim, enforceDim) == fiberDim && fiberDim == 3;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (verify) enforceDim = 3;
    if (field->commRank() == 0) {
      const typename Section::chart_type::const_iterator cEnd = chart.end();
      std::ostringstream formatString;

      if (opt2) {
        formatString << "%." << precision << "e %." << precision << "e 0.0\n";
      } else if (opt3) {
        formatString << "%." << precision << "e %." << precision << "e %." << precision << "e\n";
      }
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != cEnd; ++p_iter) {
        if (!numbering->hasPoint(*p_iter)) continue;
        const value_type *array = field->restrictPoint(*p_iter);
        const int&        dim   = field->getFiberDimension(*p_iter);

        // Perhaps there should be a flag for excluding boundary values
        if (dim != 0) {
          if (opt2) {
            ierr = PetscViewerASCIIPrintf(viewer, formatString.str().c_str(), array[0], array[1]);CHKERRQ(ierr);
          } else if (opt3) {
            ierr = PetscViewerASCIIPrintf(viewer, formatString.str().c_str(), array[0], array[1], array[2]);CHKERRQ(ierr);
          } else {
            ostringstream line;

            line << std::resetiosflags(std::ios::fixed)
                 << std::setiosflags(std::ios::scientific)
                 << std::setprecision(precision);
            if (verify) {line << *p_iter << " ";}
            for(int d = 0; d < fiberDim; d++) {
              if (d > 0) {
                line << " ";
              }
              line << array[d];
            }
            for(int d = fiberDim; d < enforceDim; d++) {
              line << " 0.0";
            }
            line << std::endl;
            ierr = PetscViewerASCIIPrintf(viewer, "%s", line.str().c_str());CHKERRQ(ierr);
          }
        }
      }
      for(int p = 1; p < field->commSize(); p++) {
        value_type *remoteValues;
        int         numLocalElementsAndFiberDim[2];
        int         size;
        MPI_Status  status;

        ierr = MPI_Recv(numLocalElementsAndFiberDim, 2, MPI_INT, p, 1, field->comm(), &status);CHKERRQ(ierr);
        size = numLocalElementsAndFiberDim[0]*numLocalElementsAndFiberDim[1];
        ierr = PetscMalloc(size * sizeof(value_type), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, size, mpiType, p, 1, field->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElementsAndFiberDim[0]; e++) {
          if (opt2) {
            ierr = PetscViewerASCIIPrintf(viewer, formatString.str().c_str(), remoteValues[e*numLocalElementsAndFiberDim[1]+0], remoteValues[e*numLocalElementsAndFiberDim[1]+1]);CHKERRQ(ierr);
          } else if (opt3) {
            ierr = PetscViewerASCIIPrintf(viewer, formatString.str().c_str(), remoteValues[e*numLocalElementsAndFiberDim[1]+0], remoteValues[e*numLocalElementsAndFiberDim[1]+1], remoteValues[e*numLocalElementsAndFiberDim[1]+2]);CHKERRQ(ierr);
          } else {
            ostringstream line;

            line << std::resetiosflags(std::ios::fixed)
                 << std::setiosflags(std::ios::scientific)
                 << std::setprecision(precision);
            if (verify) {line << ((int) remoteValues[e*numLocalElementsAndFiberDim[1]+0]);}
            for(int d = verify; d < numLocalElementsAndFiberDim[1]; d++) {
              if (d > (int) verify) {              line << " ";
              }
              line << remoteValues[e*numLocalElementsAndFiberDim[1]+d];
            }
            for(int d = numLocalElementsAndFiberDim[1]; d < enforceDim; d++) {
              line << " 0.0";
            }
            line << std::endl;
            ierr = PetscViewerASCIIPrintf(viewer, "%s", line.str().c_str());CHKERRQ(ierr);
          }
        }
        ierr = PetscFree(remoteValues);CHKERRQ(ierr);
      }
    } else {
      const typename Section::chart_type::const_iterator cEnd = chart.end();
      int         numLocalElements = numbering->getLocalSize();
      const int   size = numLocalElements*(fiberDim+verify);
      int         k = 0;
      value_type *localValues;

      ierr = PetscMalloc(size * sizeof(value_type), &localValues);CHKERRQ(ierr);
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != cEnd; ++p_iter) {
        if (!numbering->hasPoint(*p_iter)) continue;
        if (numbering->isLocal(*p_iter)) {
          const value_type *array = field->restrictPoint(*p_iter);
          const int&        dim   = field->getFiberDimension(*p_iter);

          if (verify) localValues[k++] = *p_iter;
          for(int i = 0; i < dim; ++i) {
            localValues[k++] = array[i];
          }
        }
      }
      if (k != size) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, size);
      }
      int numLocalElementsAndFiberDim[2] = {numLocalElements, fiberDim+verify};
      ierr = MPI_Send(numLocalElementsAndFiberDim, 2, MPI_INT, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, size, mpiType, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = PetscFree(localValues);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  static int getCellType(const int dim, const int corners) {
    int cellType = -1;
    switch(dim) {
    case 0:
      switch(corners) {
      case 1:
        // VTK_VERTEX
        cellType = 1;
        break;
      default:
        break;
      }
      break;
    case 1:
      switch(corners) {
      case 2:
        // VTK_LINE
        cellType = 3;
        break;
      case 3:
        // VTJ_QUADRATIC_EDGE
        cellType = 21;
        break;
      default:
        break;
      }
      break;
    case 2:
      switch(corners) {
      case 3:
        // VTK_TRIANGLE
        cellType = 5;
        break;
      case 4:
        // VTK_QUAD
        cellType = 9;
        break;
      case 6:
        // VTK_QUADRATIC_TRIANGLE
        cellType = 22;
        break;
      case 9:
        // VTK_QUADRATIC_QUAD
        cellType = 23;
        break;
      default:
        break;
      }
      break;
    case 3:
      switch(corners) {
      case 4:
        // VTK_TETRA
        cellType = 10;
        break;
      case 8:
        // VTK_HEXAHEDRON
        cellType = 12;
        break;
      case 10:
        // VTK_QUADRATIC_TETRA
        cellType = 24;
        break;
      case 27:
        // VTK_QUADRATIC_HEXAHEDRON
        cellType = 29;
        break;
      default:
        break;
      }
    }
    return cellType;
  };

  #undef __FUNCT__
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, PetscViewer viewer)
  {
    int                                depth = mesh->depth();
    Obj<typename Mesh::label_sequence> elements;
    Obj<typename Mesh::numbering_type> cNumbering;
    Obj<typename Mesh::numbering_type> vNumbering;
    PetscErrorCode                     ierr;

    /* Empty portions of the mesh can give rise to 0 for 'depth', and we need the same value */
    ierr = MPI_Allreduce(&depth, &depth, 1, MPI_INT, MPI_MAX, mesh->comm());CHKERRQ(ierr);
    if (mesh->hasLabel("censored depth")) {
      elements   = mesh->getLabelStratum("censored depth", depth);
      cNumbering = mesh->getFactory()->getNumbering(mesh, "censored depth", depth);
      vNumbering = mesh->getFactory()->getNumbering(mesh, "censored depth", 0);
    } else {
      elements   = mesh->heightStratum(0);
      cNumbering = mesh->getFactory()->getNumbering(mesh, depth);
      vNumbering = mesh->getFactory()->getNumbering(mesh, 0);
    }
    return writeElements(mesh, elements, cNumbering, vNumbering, viewer);
  };
  #undef __FUNCT__  
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const std::string& cLabelName, const int cLabelValue, const std::string& vLabelName, const int vLabelValue, PetscViewer viewer)
  {
    Obj<typename Mesh::label_sequence> elements;
    Obj<typename Mesh::numbering_type> cNumbering;
    Obj<typename Mesh::numbering_type> vNumbering;

    if (mesh->hasLabel(cLabelName)) {
      elements   = mesh->getLabelStratum(cLabelName, cLabelValue);
      cNumbering = mesh->getFactory()->getNumbering(mesh, cLabelName, cLabelValue);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid label name: %s", cLabelName.c_str());
    }
    if (mesh->hasLabel(vLabelName)) {
      vNumbering = mesh->getFactory()->getNumbering(mesh, vLabelName, vLabelValue);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid label name: %s", vLabelName.c_str());
    }
    return writeElements(mesh, elements, cNumbering, vNumbering, viewer);
  };
  #undef __FUNCT__  
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const Obj<typename Mesh::label_sequence>& elements, const Obj<typename Mesh::numbering_type>& cNumbering, const Obj<typename Mesh::numbering_type>& vNumbering, PetscViewer viewer)
  {
    typedef typename Mesh::sieve_type                      sieve_type;
    typedef ALE::ISieveVisitor::NConeRetriever<sieve_type> visitor_type;
    const Obj<sieve_type>& sieve        = mesh->getSieve();
    int                    localCorners = 0;
    int                    corners;
    int                    numElements;
    PetscErrorCode         ierr;
    visitor_type           ncV(*sieve, (size_t) pow((double) sieve->getMaxConeSize(), std::max(0, mesh->depth())));

    PetscFunctionBegin;
    if (elements->size()) localCorners = mesh->getNumCellCorners(*elements->begin());
    corners     = localCorners;
    numElements = cNumbering->getGlobalSize();
    ierr = MPI_Reduce(&localCorners, &corners, 1, MPI_INT, MPI_MAX, 0, mesh->comm());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      const typename Mesh::label_sequence::iterator eEnd = elements->end();
      const bool                                    opt3 = corners == 3;
      const bool                                    opt4 = corners == 4;

      for(typename Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != eEnd; ++e_iter) {
        ALE::ISieveTraversal<sieve_type>::orientedClosure(*sieve, *e_iter, ncV);
        const typename visitor_type::oriented_point_type *cone = ncV.getOrientedPoints();

        const int coneSize = ncV.getOrientedSize();
        if (coneSize != corners) {
          std::ostringstream msg;
	  msg << "Inconsistency in topology found for mesh '"
	      << mesh->getName() << "' during output.\n"
	      << "Number of vertices (" << coneSize << ") in cell '"
	      << *e_iter << "' does not expected number of vertices ("
	      << corners << ").";
	  throw ALE::Exception(msg.str());
        } // if

        if (opt3) {
          ierr = PetscViewerASCIIPrintf(viewer, "3 %d %d %d\n", vNumbering->getIndex(cone[0].first), vNumbering->getIndex(cone[1].first), vNumbering->getIndex(cone[2].first));CHKERRQ(ierr);
        } else if (opt4) {
          ierr = PetscViewerASCIIPrintf(viewer, "4 %d %d %d %d\n", vNumbering->getIndex(cone[0].first), vNumbering->getIndex(cone[1].first), vNumbering->getIndex(cone[2].first), vNumbering->getIndex(cone[3].first));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
          for(int c = 0; c < coneSize; ++c) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(cone[c].first));CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ncV.clear();
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        int        numLocalElementsAndCorners[2];
        int       *remoteVertices;
        MPI_Status status;

        ierr = MPI_Recv(numLocalElementsAndCorners, 2, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1] * sizeof(int), &remoteVertices);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteVertices, numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1], MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElementsAndCorners[0]; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%d ", numLocalElementsAndCorners[1]);CHKERRQ(ierr);
          for(int c = 0; c < numLocalElementsAndCorners[1]; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*numLocalElementsAndCorners[1]+c]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
      }
    } else {
      const typename Mesh::label_sequence::iterator eEnd = elements->end();
      int  numLocalElements = cNumbering->getLocalSize();
      int *localVertices;
      int  k = 0;

      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
      for(typename Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != eEnd; ++e_iter) {
        if (cNumbering->isLocal(*e_iter)) {
          ALE::ISieveTraversal<sieve_type>::orientedClosure(*sieve, *e_iter, ncV);
          const typename visitor_type::oriented_point_type *cone = ncV.getOrientedPoints();

          for(int c = 0; c < ncV.getOrientedSize(); ++c) {
            localVertices[k++] = vNumbering->getIndex(cone[c].first);
          }
          ncV.clear();
        }
      }
      if (k != numLocalElements*corners) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
      }
      int numLocalElementsAndCorners[2] = {numLocalElements, corners};
      ierr = MPI_Send(numLocalElementsAndCorners, 2, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localVertices, numLocalElements*corners, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localVertices);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "CELL_TYPES %d\n", numElements);CHKERRQ(ierr);
    const int cellType = getCellType(mesh->getDimension(), corners);
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%d\n", cellType);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };
  #undef __FUNCT__  
  #define __FUNCT__ "writeElements"
  static PetscErrorCode writeElements(const Obj<PETSC_MESH_TYPE>& mesh, const Obj<PETSC_MESH_TYPE::label_sequence>& elements, const Obj<PETSC_MESH_TYPE::numbering_type>& cNumbering, const Obj<PETSC_MESH_TYPE::numbering_type>& vNumbering, PetscViewer viewer)
  {
    typedef ALE::SieveAlg<PETSC_MESH_TYPE>  sieve_alg_type;
    const Obj<PETSC_MESH_TYPE::sieve_type>& sieve        = mesh->getSieve();
    int                               depth        = mesh->depth();
    int                               localCorners = 0;
    int                               corners;
    int                               numElements;
    PetscErrorCode                    ierr;

    PetscFunctionBegin;
    if (elements->size()) localCorners = mesh->getNumCellCorners();
    corners     = localCorners;
    numElements = cNumbering->getGlobalSize();
    ierr = MPI_Reduce(&localCorners, &corners, 1, MPI_INT, MPI_MAX, 0, mesh->comm());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
#ifdef PETSC_OPT_SIEVE
      ALE::ISieveVisitor::NConeRetriever<PETSC_MESH_TYPE::sieve_type> ncV(*sieve, (size_t) pow((double) sieve->getMaxConeSize(), std::max(0, depth)));
#endif
      const PETSC_MESH_TYPE::label_sequence::iterator eEnd = elements->end();

      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = elements->begin(); e_iter != eEnd; ++e_iter) {
        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
        ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*sieve, *e_iter, ncV);
        const int                          coneSize = ncV.getSize();
        const PETSC_MESH_TYPE::point_type *cone     = ncV.getPoints();

        for(int v = 0; v < coneSize; ++v) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(cone[v]));CHKERRQ(ierr);
        }
        ncV.clear();
#else
        const Obj<sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(mesh, *e_iter, depth);
        const sieve_alg_type::coneArray::iterator cEnd = cone->end();

        for(sieve_alg_type::coneArray::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(*c_iter));CHKERRQ(ierr);
        }
#endif
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        int        numLocalElementsAndCorners[2];
        int       *remoteVertices;
        MPI_Status status;

        ierr = MPI_Recv(numLocalElementsAndCorners, 2, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1] * sizeof(int), &remoteVertices);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteVertices, numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1], MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElementsAndCorners[0]; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%d ", numLocalElementsAndCorners[1]);CHKERRQ(ierr);
          for(int c = 0; c < numLocalElementsAndCorners[1]; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*numLocalElementsAndCorners[1]+c]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
      }
    } else {
      int  numLocalElements = cNumbering->getLocalSize();
      int *localVertices;
      int  k = 0;
#ifdef PETSC_OPT_SIEVE
      ALE::ISieveVisitor::NConeRetriever<PETSC_MESH_TYPE::sieve_type> ncV(*sieve, (size_t) pow((double) sieve->getMaxConeSize(), std::max(0, depth)));
#endif
      const PETSC_MESH_TYPE::label_sequence::iterator eEnd = elements->end();

      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
      for(PETSC_MESH_TYPE::label_sequence::iterator e_iter = elements->begin(); e_iter != eEnd; ++e_iter) {
#ifdef PETSC_OPT_SIEVE
        ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*sieve, *e_iter, ncV);
        const int                          coneSize = ncV.getSize();
        const PETSC_MESH_TYPE::point_type *cone     = ncV.getPoints();

        if (cNumbering->isLocal(*e_iter)) {
          for(int v = 0; v < coneSize; ++v) {
            localVertices[k++] = vNumbering->getIndex(cone[v]);
          }
        }
        ncV.clear();
#else
        const Obj<sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(mesh, *e_iter, depth);

        if (cNumbering->isLocal(*e_iter)) {
          const PETSC_MESH_TYPE::sieve_type::coneArray::iterator cEnd = cone->end();

          for(PETSC_MESH_TYPE::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
            localVertices[k++] = vNumbering->getIndex(*c_iter);
          }
        }
#endif
      }
      if (k != numLocalElements*corners) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
      }
      int numLocalElementsAndCorners[2] = {numLocalElements, corners};
      ierr = MPI_Send(numLocalElementsAndCorners, 2, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localVertices, numLocalElements*corners, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localVertices);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "CELL_TYPES %d\n", numElements);CHKERRQ(ierr);
    const int cellType = getCellType(mesh->getDimension(), corners);
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%d\n", cellType);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };
};

class VTKXMLViewer {
 public:
  VTKXMLViewer() {};
  virtual ~VTKXMLViewer() {};

  #undef __FUNCT__
  #define __FUNCT__ "getVertexNumbering"
  template<typename Mesh>
  static Obj<typename Mesh::numbering_type> getVertexNumbering(const Obj<Mesh>& mesh) {
    if (mesh->hasLabel("censored depth")) {
      return mesh->getFactory()->getNumbering(mesh, "censored depth", 0);
    }
    return mesh->getFactory()->getNumbering(mesh, 0);
  };

  #undef __FUNCT__
  #define __FUNCT__ "getCellNumbering"
  template<typename Mesh>
  static Obj<typename Mesh::numbering_type> getCellNumbering(const Obj<Mesh>& mesh) {
    const int depth = mesh->depth();

    if (mesh->hasLabel("censored depth")) {
      return mesh->getFactory()->getNumbering(mesh, "censored depth", depth);
    }
    return mesh->getFactory()->getNumbering(mesh, depth);
  };

  #undef __FUNCT__
  #define __FUNCT__ "getCells"
  template<typename Mesh>
  static Obj<typename Mesh::label_sequence> getCells(const Obj<Mesh>& mesh) {
    const int depth = mesh->depth();

    if (mesh->hasLabel("censored depth")) {
      return mesh->getLabelStratum("censored depth", depth);
    }
    return mesh->heightStratum(0);
  };

  #undef __FUNCT__
  #define __FUNCT__ "writeHeader"
  template<typename Mesh>
  static PetscErrorCode writeHeader(const Obj<Mesh>& mesh, PetscViewer viewer) {
    Obj<typename Mesh::numbering_type> vNumbering = getVertexNumbering(mesh);
    Obj<typename Mesh::numbering_type> cNumbering = getCellNumbering(mesh);
    PetscErrorCode                     ierr;

    PetscFunctionBegin;
#ifdef PETSC_WORDS_BIGENDIAN
    ierr = PetscViewerASCIIPrintf(viewer,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer,"  <UnstructuredGrid>\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", vNumbering->getGlobalSize(), cNumbering->getGlobalSize());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "writeVertices"
  template<typename Mesh>
  static PetscErrorCode writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer) {
    Obj<typename Mesh::real_section_type> coordinates;

    if (mesh->hasRealSection("coordinates_dimensioned")) {
      coordinates = mesh->getRealSection("coordinates_dimensioned");
    } else if (mesh->hasRealSection("coordinates")) {
      coordinates = mesh->getRealSection("coordinates");
    } else {
      throw ALE::Exception("Missing coordinates in mesh");
    }
    const int                          embedDim   = coordinates->getFiberDimension(*mesh->depthStratum(0)->begin());
    Obj<typename Mesh::numbering_type> vNumbering = getVertexNumbering(mesh);
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer, "<Points>\n");CHKERRQ(ierr);
    ierr = writeSection(coordinates, embedDim, vNumbering, viewer, 3);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "</Points>\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "writeSection"
  template<typename Section>
  static PetscErrorCode writeSection(const Obj<Section>& field, const int fiberDim, const Obj<PETSC_MESH_TYPE::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1, int precision = 6) {
    typedef typename Section::value_type value_type;
    const typename Section::chart_type&                chart   = field->getChart();
    const typename Section::chart_type::const_iterator cEnd    = chart.end();
    const MPI_Datatype                                 mpiType = ALE::New::ParallelFactory<value_type>::singleton(field->debug())->getMPIType();
    PetscErrorCode ierr;

    PetscFunctionBegin;
    int         numLocalElements = numbering->getLocalSize();
    const int   size             = numLocalElements*std::max(fiberDim, enforceDim);
    int         k                = 0;
    value_type *localValues;

    ierr = PetscMalloc(size * sizeof(value_type), &localValues);CHKERRQ(ierr);
    for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != cEnd; ++p_iter) {
      if (!numbering->hasPoint(*p_iter)) continue;
      if (numbering->isLocal(*p_iter)) {
        const value_type *array = field->restrictPoint(*p_iter);
        const int&        dim   = field->getFiberDimension(*p_iter);

        for(int i = 0; i < dim; ++i) {
          localValues[k++] = array[i];
        }
        for(int d = dim; d < enforceDim; d++) {
          localValues[k++] = 0.0;
        }
      }
    }
    if (k != size) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, size);
    }

    if (field->commRank() == 0) {
      ierr = PetscViewerASCIIPrintf(viewer, "<DataArray type=\"Float32\" Name=\"%s\" format=\"binary\" Number of Components=\"%d\">\n", field->getName());CHKERRQ(ierr);
      // Encode and write data
      for(int p = 1; p < field->commSize(); p++) {
        value_type *remoteValues;
        int         numLocalElementsAndFiberDim[2];
        int         remoteSize;
        MPI_Status  status;

        ierr = MPI_Recv(numLocalElementsAndFiberDim, 2, MPI_INT, p, 1, field->comm(), &status);CHKERRQ(ierr);
        remoteSize = numLocalElementsAndFiberDim[0]*numLocalElementsAndFiberDim[1];
        ierr = PetscMalloc(remoteSize * sizeof(value_type), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, remoteSize, mpiType, p, 1, field->comm(), &status);CHKERRQ(ierr);
        // Encode and write data
        ierr = PetscFree(remoteValues);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "</DataArray>\n");CHKERRQ(ierr);
    } else {
      int numLocalElementsAndFiberDim[2] = {numLocalElements, fiberDim};
      ierr = MPI_Send(numLocalElementsAndFiberDim, 2, MPI_INT, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, size, mpiType, 0, 1, field->comm());CHKERRQ(ierr);
    }
    ierr = PetscFree(localValues);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, PetscViewer viewer)
  {
    Obj<typename Mesh::label_sequence> elements   = getCells(mesh);
    Obj<typename Mesh::numbering_type> cNumbering = getCellNumbering(mesh);
    Obj<typename Mesh::numbering_type> vNumbering = getVertexNumbering(mesh);

    return writeElements(mesh, elements, cNumbering, vNumbering, viewer);
  };
  #undef __FUNCT__  
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const std::string& cLabelName, const int cLabelValue, const std::string& vLabelName, const int vLabelValue, PetscViewer viewer)
  {
    Obj<typename Mesh::label_sequence> elements;
    Obj<typename Mesh::numbering_type> cNumbering;
    Obj<typename Mesh::numbering_type> vNumbering;

    if (mesh->hasLabel(cLabelName)) {
      elements   = mesh->getLabelStratum(cLabelName, cLabelValue);
      cNumbering = mesh->getFactory()->getNumbering(mesh, cLabelName, cLabelValue);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid label name: %s", cLabelName.c_str());
    }
    if (mesh->hasLabel(vLabelName)) {
      vNumbering = mesh->getFactory()->getNumbering(mesh, vLabelName, vLabelValue);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid label name: %s", vLabelName.c_str());
    }
    return writeElements(mesh, elements, cNumbering, vNumbering, viewer);
  };
  #undef __FUNCT__  
  #define __FUNCT__ "writeElements"
  template<typename Mesh>
  static PetscErrorCode writeElements(const Obj<Mesh>& mesh, const Obj<typename Mesh::label_sequence>& elements, const Obj<typename Mesh::numbering_type>& cNumbering, const Obj<typename Mesh::numbering_type>& vNumbering, PetscViewer viewer)
  {
    typedef typename Mesh::sieve_type                      sieve_type;
    typedef ALE::ISieveVisitor::NConeRetriever<sieve_type> visitor_type;
    const Obj<sieve_type>&                        sieve            = mesh->getSieve();
    const typename Mesh::label_sequence::iterator eEnd             = elements->end();
    int                                           localCorners     = elements->size() ? mesh->getNumCellCorners(*elements->begin()) : 0;
    int                                           corners          = localCorners;
    int                                           numElements      = cNumbering->getGlobalSize();
    int                                           numLocalElements = cNumbering->getLocalSize();
    int                                           k                = 0;
    int                                          *localVertices;
    PetscErrorCode                                ierr;
    visitor_type                                  ncV(*sieve, (size_t) pow((double) sieve->getMaxConeSize(), std::max(0, mesh->depth())));

    PetscFunctionBegin;
    ierr = MPI_Reduce(&localCorners, &corners, 1, MPI_INT, MPI_MAX, 0, mesh->comm());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"    <Cells>\n");CHKERRQ(ierr);
    ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
    for(typename Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != eEnd; ++e_iter) {
      if (cNumbering->isLocal(*e_iter)) {
        ALE::ISieveTraversal<sieve_type>::orientedClosure(*sieve, *e_iter, ncV);
        const typename visitor_type::oriented_point_type *cone = ncV.getOrientedPoints();

        for(int c = 0; c < ncV.getOrientedSize(); ++c) {
          localVertices[k++] = vNumbering->getIndex(cone[c].first);
        }
        ncV.clear();
      }
    }
    if (k != numLocalElements*corners) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
    }

    if (mesh->commRank() == 0) {
      ierr = PetscViewerASCIIPrintf(viewer, "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");CHKERRQ(ierr);
      for(int e = 0; e < numLocalElements; e++) {
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", localVertices[e*corners+c]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        int        numLocalElementsAndCorners[2];
        int       *remoteVertices;
        MPI_Status status;

        ierr = MPI_Recv(numLocalElementsAndCorners, 2, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1] * sizeof(int), &remoteVertices);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteVertices, numLocalElementsAndCorners[0]*numLocalElementsAndCorners[1], MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElementsAndCorners[0]; e++) {
          for(int c = 0; c < numLocalElementsAndCorners[1]; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*numLocalElementsAndCorners[1]+c]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "      </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");CHKERRQ(ierr);
      for(int e = 0; e < numElements*corners; e += corners) {
        ierr = PetscViewerASCIIPrintf(viewer, "  %d", e);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "      </DataArray>\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");CHKERRQ(ierr);
      const int cellType = getCellType(mesh->getDimension(), corners);
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "  %d", cellType);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "      </DataArray>\n");CHKERRQ(ierr);
    } else {
      int numLocalElementsAndCorners[2] = {numLocalElements, corners};
      ierr = MPI_Send(numLocalElementsAndCorners, 2, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localVertices, numLocalElements*corners, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "    </Cells>\n");CHKERRQ(ierr);
    ierr = PetscFree(localVertices);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__
  #define __FUNCT__ "writeFooter"
  static PetscErrorCode writeFooter(PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"    </Piece>\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  </UnstructuredGrid>\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"</VTKFile>\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };
};

#undef __FUNCT__  
#define __FUNCT__ "SectionView_Sieve_Ascii"
template<typename Bundle, typename Section>
  PetscErrorCode SectionView_Sieve_Ascii(const Obj<Bundle>& bundle, const Obj<Section>& s, const char name[], PetscViewer viewer, int enforceDim = -1)
{
  // state 0: No header has been output
  // state 1: Only POINT_DATA has been output
  // state 2: Only CELL_DATA has been output
  // state 3: Output both, POINT_DATA last
  // state 4: Output both, CELL_DATA last
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
    static PetscInt   stateId     = -1;
    PetscInt          doOutput    = 0;
    PetscInt          outputState = 0;
    PetscBool         hasState;

    if (stateId < 0) {
      ierr = PetscObjectComposedDataRegister(&stateId);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, 0);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_VTK) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) {
        doOutput = 0;
      } else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) {
        doOutput = 0;
      } else if (outputState == 4) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
      const ALE::Obj<PETSC_MESH_TYPE::numbering_type>& numbering = bundle->getFactory()->getNumbering(bundle, 0);
      PetscInt fiberDim = std::abs(s->getFiberDimension(*bundle->depthStratum(0)->begin()));

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", numbering->getGlobalSize());CHKERRQ(ierr);
      }
      VTKViewer::writeField(s, std::string(name), fiberDim, numbering, viewer, enforceDim);
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput = 1;
      } else if (outputState == 2) {
        doOutput = 0;
      } else if (outputState == 3) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
      const ALE::Obj<PETSC_MESH_TYPE::numbering_type>& numbering = bundle->getFactory()->getNumbering(bundle, bundle->depth());
      PetscInt fiberDim = s->getFiberDimension(*bundle->heightStratum(0)->begin());

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", numbering->getGlobalSize());CHKERRQ(ierr);
      }
      VTKViewer::writeField(s, std::string(name), fiberDim, numbering, viewer, enforceDim);
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
  } else {
    s->view(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMeshView_Sieve_Ascii"
template<typename Mesh, typename Section>
PetscErrorCode DMMeshView_Sieve_Ascii(const Obj<Mesh>& mesh, const Obj<Section>& partition, PetscViewer viewer)
{
  typedef ALE::IUniformSection<typename Mesh::point_type, typename Section::point_type> partitionMap_type;
  const int      numLocalPoints = partition->size();
  int            numPoints;
  PetscErrorCode ierr;

  ierr = MPI_Allreduce((void *) &numLocalPoints, (void *) &numPoints, 1, MPI_INT, MPI_SUM, partition->comm());CHKERRQ(ierr);
  Obj<partitionMap_type> partitionMap = new partitionMap_type(mesh->comm(), 0, numPoints, mesh->debug());

  PetscFunctionBegin;
  ALE::Partitioner<>::createPartitionMap(partition, partitionMap);
  ierr = VTKViewer::writeHeader(mesh, viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionView_Sieve_Ascii(mesh, partitionMap, "Partition", viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif // __PETSCDMMESH_VIEWERS_HH
