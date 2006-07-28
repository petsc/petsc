#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

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
  static PetscErrorCode writeVertices(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::section_type>   coordinates = mesh->getSection("coordinates");
    ALE::Mesh::section_type::patch_type patch       = 0;
    int embedDim = coordinates->getAtlas()->size(patch, *mesh->getTopologyNew()->depthStratum(patch, 0)->begin());
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", mesh->getTopologyNew()->depthStratum(patch, 0)->size());CHKERRQ(ierr);
    ierr = writeSection(mesh, coordinates, embedDim, viewer, 3);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteField"
  static PetscErrorCode writeField(const ALE::Obj<ALE::Mesh>& mesh, const ALE::Obj<ALE::Mesh::section_type>& field, const std::string& name, const int fiberDim, PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (fiberDim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name.c_str());CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name.c_str(), fiberDim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    ierr = writeSection(mesh, field, fiberDim, viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteSection"
  static PetscErrorCode writeSection(const ALE::Obj<ALE::Mesh>& mesh, const ALE::Obj<ALE::Mesh::section_type>& field, const int fiberDim, PetscViewer viewer, int enforceDim = -1) {
    const ALE::Mesh::section_type::patch_type  patch = 0;
    const ALE::Mesh::section_type::value_type *array = field->restrict(patch);
    const ALE::Mesh::atlas_type::chart_type&   chart = field->getAtlas()->getChart(patch);
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const ALE::Mesh::atlas_type::index_type& idx = p_iter->second;

        if (idx.index > 0) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", array[idx.prefix+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        double    *remoteValues;
        int        numLocalElements;
        MPI_Status status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(double), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, numLocalElements*fiberDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElements; e++) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", remoteValues[e*fiberDim+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    } else {
      //int     numLocalElements = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/fiberDim;
      int     numLocalElements = chart.size();
      double *localValues;
      int     k = 0;

      ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(ALE::Mesh::section_type::value_type), &localValues);CHKERRQ(ierr);
      for(ALE::Mesh::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        int offset = p_iter->second.prefix;
        int dim    = p_iter->second.index;

        for(int i = offset; i < offset+dim; ++i) {
          localValues[k++] = array[i];
        }
      }
      if (k != numLocalElements*fiberDim) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, numLocalElements*fiberDim);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, numLocalElements*fiberDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localValues);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteElements"
  static PetscErrorCode writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer)
  {
    ALE::Mesh::topology_type::patch_type   patch    = 0;
    const ALE::Obj<ALE::Mesh::sieve_type>& topology = mesh->getTopologyNew()->getPatch(patch);
    const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& elements = mesh->getTopologyNew()->heightStratum(patch, 0);
    int            corners = topology->nCone(*elements->begin(), mesh->getTopologyNew()->depth())->size();
    int            numElements = mesh->getTopologyNew()->heightStratum(patch, 0)->size();
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = topology->cone(*e_iter);

        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", (*c_iter).index - numElements);CHKERRQ(ierr);
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
      int  numLocalElements = elements->size();
      int *localVertices;
      int  k = 0;

      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
      for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = topology->cone(*e_iter);

        for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          localVertices[k++] = (*c_iter).index - numElements;
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
    if (corners == 2) {
      // VTK_LINE
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "3\n");CHKERRQ(ierr);
      }
    } else if (corners == 3) {
      // VTK_TRIANGLE
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "5\n");CHKERRQ(ierr);
      }
    } else if (corners == 4) {
      if (mesh->getDimension() == 3) {
        // VTK_TETRA
        for(int e = 0; e < numElements; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "10\n");CHKERRQ(ierr);
        }
      } else if (mesh->getDimension() == 2) {
        // VTK_QUAD
        for(int e = 0; e < numElements; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "9\n");CHKERRQ(ierr);
        }
      }
    } else if (corners == 8) {
      // VTK_HEXAHEDRON
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "12\n");CHKERRQ(ierr);
      }
    }
    PetscFunctionReturn(0);
  };
};
