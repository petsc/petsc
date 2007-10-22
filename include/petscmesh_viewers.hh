#if !defined(__PETSCMESH_VIEWERS_HH)
#define __PETSCMESH_VIEWERS_HH

#include <petscmesh.hh>
#include <petscdmmg.h>

using ALE::Obj;

class VTKViewer {
 public:
  VTKViewer() {};
  virtual ~VTKViewer() {};

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteHeader"
  static PetscErrorCode writeHeader(PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Simplicial Mesh Example\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteVertices"
  static PetscErrorCode writeVertices(const Obj<ALE::Mesh>& mesh, PetscViewer viewer) {
    const Obj<ALE::Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
    const Obj<ALE::Mesh::numbering_type>&    vNumbering  = mesh->getFactory()->getNumbering(mesh, 0);
    const int      embedDim = coordinates->getFiberDimension(*mesh->depthStratum(0)->begin());
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", vNumbering->getGlobalSize());CHKERRQ(ierr);
    ierr = writeSection(coordinates, embedDim, vNumbering, viewer, 3);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteField"
  template<typename Section>
    static PetscErrorCode writeField(const Obj<Section>& field, const std::string& name, const int fiberDim, const Obj<ALE::Mesh::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1) {
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
    ierr = writeSection(field, fiberDim, numbering, viewer, enforceDim);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteSection"
  template<typename Section>
  static PetscErrorCode writeSection(const Obj<Section>& field, const int fiberDim, const Obj<ALE::Mesh::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1) {
    typedef typename Section::value_type value_type;
    const typename Section::chart_type& chart   = field->getChart();
    const MPI_Datatype                  mpiType = ALE::New::ParallelFactory<value_type>::singleton(field->debug())->getMPIType();
    const bool                          verify  = enforceDim == -4;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (verify) enforceDim = 3;
    if (field->commRank() == 0) {
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (!numbering->hasPoint(*p_iter)) continue;
        const value_type *array = field->restrictPoint(*p_iter);
        const int&        dim   = field->getFiberDimension(*p_iter);

        // Perhaps there should be a flag for excluding boundary values
        if (dim != 0) {
          if (verify) {ierr = PetscViewerASCIIPrintf(viewer, "%d ", *p_iter);CHKERRQ(ierr);}
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            if (mpiType == MPI_INT) {
              ierr = PetscViewerASCIIPrintf(viewer, "%d", array[d]);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer, "%G", array[d]);CHKERRQ(ierr);
            }
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
      for(int p = 1; p < field->commSize(); p++) {
        value_type *remoteValues;
        int         numLocalElements;
        int         size;
        const int   newDim = fiberDim+verify;
        MPI_Status  status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, field->comm(), &status);CHKERRQ(ierr);
        size = numLocalElements*newDim;
        ierr = PetscMalloc(size * sizeof(value_type), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, size, mpiType, p, 1, field->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElements; e++) {
          if (verify) {ierr = PetscViewerASCIIPrintf(viewer, "%d ", (int) remoteValues[e*newDim+0]);CHKERRQ(ierr);}
          for(int d = verify; d < newDim; d++) {
            if (d > verify) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            if (mpiType == MPI_INT) {
              ierr = PetscViewerASCIIPrintf(viewer, "%d", remoteValues[e*newDim+d]);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer, "%G", remoteValues[e*newDim+d]);CHKERRQ(ierr);
            }
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ierr = PetscFree(remoteValues);CHKERRQ(ierr);
      }
    } else {
      value_type *localValues;
      int         numLocalElements = numbering->getLocalSize();
      const int   size = numLocalElements*(fiberDim+verify);
      int         k = 0;

      ierr = PetscMalloc(size * sizeof(value_type), &localValues);CHKERRQ(ierr);
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
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
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, size);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, size, mpiType, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = PetscFree(localValues);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  static int getCellType(const int dim, const int corners) {
    if (corners == 2) {
      // VTK_LINE
      return 3;
    } else if (corners == 3) {
      // VTK_TRIANGLE
      return 5;
    } else if (corners == 4) {
      if (dim == 3) {
        // VTK_TETRA
        return 10;
      } else if (dim == 2) {
        // VTK_QUAD
        return 9;
      }
    } else if (corners == 8) {
      // VTK_HEXAHEDRON
      return 12;
    }
    return -1;
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteElements"
  static PetscErrorCode writeElements(const Obj<ALE::Mesh>& mesh, PetscViewer viewer)
  {
    typedef ALE::SieveAlg<ALE::Mesh> sieve_alg_type;
    const Obj<ALE::Mesh::sieve_type>&     sieve      = mesh->getSieve();
    int                                   depth      = mesh->depth();
    const Obj<ALE::Mesh::numbering_type>& vNumbering = mesh->getFactory()->getNumbering(mesh, 0);
    Obj<ALE::Mesh::numbering_type> cNumbering;
    Obj<ALE::Mesh::label_sequence> elements;
    if (mesh->hasLabel("censored depth")) {
      cNumbering = mesh->getFactory()->getNumbering(mesh, "censored depth", depth);
      elements   = mesh->getLabelStratum("censored depth", depth);
    } else {
      cNumbering = mesh->getFactory()->getNumbering(mesh, depth);
      elements   = mesh->heightStratum(0);
    }
    int            corners = sieve->nCone(*elements->begin(), depth)->size();
    int            numElements;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    numElements = cNumbering->getGlobalSize();
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(mesh, *e_iter, depth);

        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(sieve_alg_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(*c_iter));CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        int        numLocalElements;
        int       *remoteVertices;
        MPI_Status status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &remoteVertices);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteVertices, numLocalElements*corners, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElements; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
          for(int c = 0; c < corners; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
      }
    } else {
      int  numLocalElements = cNumbering->getLocalSize();
      int *localVertices;
      int  k = 0;

      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
      for(ALE::Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(mesh, *e_iter, depth);

        if (cNumbering->isLocal(*e_iter)) {
          for(ALE::Mesh::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            localVertices[k++] = vNumbering->getIndex(*c_iter);
          }
        }
      }
      if (k != numLocalElements*corners) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
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
  #define __FUNCT__ "VTKWriteHierarchyVertices"
  static PetscErrorCode writeHierarchyVertices(DMMG * dmmg, PetscViewer viewer, double * offset) {
    PetscErrorCode ierr;
    ALE::Obj<ALE::Mesh> mesh;
    int nlevels = dmmg[0]->nlevels;
    PetscFunctionBegin;
    int totalpoints = 0;
    int embedDim = -1;
    for(int i = 0; i < nlevels; i++) {
      ierr = MeshGetMesh((Mesh)dmmg[i]->dm, mesh);CHKERRQ(ierr);
      if (embedDim == -1) {
        embedDim = mesh->getDimension();
      }
      totalpoints += mesh->depthStratum(0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", totalpoints);CHKERRQ(ierr);
    for (int i = 0; i < nlevels; i++) {
      ierr = MeshGetMesh((Mesh)dmmg[i]->dm, mesh);CHKERRQ(ierr);
      Obj<ALE::Mesh::label_sequence> vertices = mesh->depthStratum(0);
      Obj<ALE::Mesh::real_section_type> coordinates = mesh->getRealSection("coordinates");
      for(ALE::Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const ALE::Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
        for(int d = 0; d < embedDim; d++) {
          if (d > 0) {
            ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "%G", array[d]+(double)i*offset[d]);CHKERRQ(ierr);
        }
        for(int d = embedDim; d < 3; d++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %G", (double) i*offset[d]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteHierarchyElements"
  static PetscErrorCode writeHierarchyElements(DMMG * dmmg, PetscViewer viewer) {
    int            numElements = 0;
    int            numVertices = 0;
    int            embedDim = -1;
    int            corners = -1;
    int            depth;
    ALE::Obj<ALE::Mesh> mesh;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    int nlevels = dmmg[0]->nlevels;
    //PetscPrintf(PETSC_COMM_WORLD, "LEVELS: %d\n", nlevels);
    for(int i = 0; i < nlevels; i++) {
      ierr = MeshGetMesh((Mesh)dmmg[i]->dm, mesh);CHKERRQ(ierr);
      if (embedDim == -1) {
        embedDim = mesh->getDimension();
      }
      if (corners == -1) {
        ALE::Mesh::point_type firstcell = *mesh->heightStratum(0)->begin();
        depth = mesh->depth(firstcell);
        corners = mesh->getSieve()->nCone(firstcell, depth)->size();
      }
      numElements += mesh->heightStratum(0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(int i = 0; i < nlevels; i++) {
        ierr = MeshGetMesh((Mesh)dmmg[i]->dm, mesh);CHKERRQ(ierr);
        Obj<ALE::Mesh::sieve_type>     sieve      = mesh->getSieve();
        Obj<ALE::Mesh::label_sequence> elements   = mesh->heightStratum(0);
        Obj<ALE::Mesh::numbering_type> vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);

        for(ALE::Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          Obj<ALE::Mesh::sieve_type::coneArray> cone = sieve->nCone(*e_iter, depth);

          ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
          for(ALE::Mesh::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(*c_iter) + numVertices);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        numVertices += mesh->depthStratum(0)->size();
      }
    }
    ierr = PetscViewerASCIIPrintf(viewer, "CELL_TYPES %d\n", numElements);CHKERRQ(ierr);
    const int cellType = getCellType(mesh->getDimension(), corners);
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%d\n", cellType);CHKERRQ(ierr);
    }
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
    PetscTruth        hasState;

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
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = bundle->getFactory()->getNumbering(bundle, 0);
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
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = bundle->getFactory()->getNumbering(bundle, bundle->depth());
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

#endif // __PETSCMESH_VIEWERS_HH
