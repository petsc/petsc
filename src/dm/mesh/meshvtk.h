#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

#include <Distribution.hh>

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
  static PetscErrorCode writeVertices(const Obj<ALE::Mesh>& mesh, const ALE::Mesh::topology_type::patch_type& patch, PetscViewer viewer) {
    const Obj<ALE::Mesh::section_type>&   coordinates = mesh->getSection("coordinates");
    const Obj<ALE::Mesh::topology_type>&  topology    = coordinates->getTopology();
    const Obj<ALE::Mesh::numbering_type>& vNumbering  = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getNumbering(topology, patch, 0);
    const int      embedDim = coordinates->getFiberDimension(patch, *topology->depthStratum(patch, 0)->begin());
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", vNumbering->getGlobalSize());CHKERRQ(ierr);
    ierr = writeSection(coordinates, patch, embedDim, vNumbering, viewer, 3);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteField"
  static PetscErrorCode writeField(const Obj<ALE::Mesh::section_type>& field, const std::string& name, const int fiberDim, const Obj<ALE::Mesh::numbering_type>& numbering, PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (fiberDim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name.c_str());CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name.c_str(), fiberDim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    ierr = writeSection(field, 0, fiberDim, numbering, viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteSection"
  static PetscErrorCode writeSection(const Obj<ALE::Mesh::section_type>& field, const ALE::Mesh::topology_type::patch_type& patch, const int fiberDim, const Obj<ALE::Mesh::numbering_type>& numbering, PetscViewer viewer, int enforceDim = -1) {
    const ALE::Mesh::section_type::value_type *array = field->restrict(patch);
    const ALE::Obj<ALE::Mesh::atlas_type>&     atlas = field->getAtlas();
    const ALE::Mesh::atlas_type::chart_type&   chart = atlas->getPatch(patch);
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (field->commRank() == 0) {
      for(ALE::Mesh::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const ALE::Mesh::atlas_type::value_type& idx = atlas->restrict(patch, *p_iter)[0];

        if (idx.prefix > 0) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", array[idx.index+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
      for(int p = 1; p < field->commSize(); p++) {
        ALE::Mesh::section_type::value_type *remoteValues;
        int        numLocalElements;
        MPI_Status status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, field->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(ALE::Mesh::section_type::value_type), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, numLocalElements*fiberDim, MPI_DOUBLE, p, 1, field->comm(), &status);CHKERRQ(ierr);
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
      ALE::Mesh::section_type::value_type *localValues;
      int numLocalElements = numbering->getLocalSize();
      int k = 0;

      ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(ALE::Mesh::section_type::value_type), &localValues);CHKERRQ(ierr);
      for(ALE::Mesh::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const ALE::Mesh::atlas_type::value_type& idx = atlas->restrict(patch, *p_iter)[0];
        const int& dim    = idx.prefix;
        const int& offset = idx.index;

        if (numbering->isLocal(*p_iter)) {
          for(int i = offset; i < offset+dim; ++i) {
            localValues[k++] = array[i];
          }
        }
      }
      if (k != numLocalElements*fiberDim) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, numLocalElements*fiberDim);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, field->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, numLocalElements*fiberDim, MPI_DOUBLE, 0, 1, field->comm());CHKERRQ(ierr);
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
  static PetscErrorCode writeElements(const Obj<ALE::Mesh>& mesh, const ALE::Mesh::topology_type::patch_type& patch, PetscViewer viewer)
  {
    const Obj<ALE::Mesh::topology_type>&                 topology   = mesh->getTopologyNew();
    const Obj<ALE::Mesh::sieve_type>&                    sieve   = mesh->getTopologyNew()->getPatch(patch);
    const Obj<ALE::Mesh::topology_type::label_sequence>& elements   = mesh->getTopologyNew()->heightStratum(patch, 0);
    const Obj<ALE::Mesh::numbering_type>&                vNumbering = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getNumbering(topology, patch, 0);
    const Obj<ALE::Mesh::numbering_type>&                cNumbering = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getNumbering(topology, patch, topology->depth());
    //int            corners = sieve->nCone(*elements->begin(), mesh->getTopologyNew()->depth())->size();
    int            corners = sieve->cone(*elements->begin())->size();
    int            numElements;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    numElements = cNumbering->getGlobalSize();
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);

        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
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
      for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);

        if (cNumbering->isLocal(*e_iter)) {
          for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
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
  static PetscErrorCode writeHierarchyVertices(const Obj<ALE::Mesh>& mesh, PetscViewer viewer, double zScale = 1.0) {
    const Obj<ALE::Mesh::topology_type>&        topology    = mesh->getTopologyNew();
    const ALE::Mesh::topology_type::sheaf_type& patches     = topology->getPatches();
    const ALE::Mesh::topology_type::patch_type  firstPatch  = patches.begin()->first;
    const Obj<ALE::Mesh::section_type>&    coordinates = mesh->getSection("coordinates");
    const int      embedDim    = coordinates->getFiberDimension(firstPatch, *topology->depthStratum(firstPatch, 0)->begin());
    int            totalPoints = 0;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    for(ALE::Mesh::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
      // This should be a Numbering, not just the stratum
      totalPoints += topology->depthStratum(p_iter->first, 0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", totalPoints);CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const ALE::Mesh::topology_type::patch_type           patch    = p_iter->first;
        const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);

        for(ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          const ALE::Mesh::section_type::value_type *array = coordinates->restrictPoint(firstPatch, *v_iter);

          for(int d = 0; d < embedDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", array[d]);CHKERRQ(ierr);
          }
          for(int d = embedDim; d < 3; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %G", (double) patch*zScale);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteHierarchyElements"
  static PetscErrorCode writeHierarchyElements(const Obj<ALE::Mesh>& mesh, PetscViewer viewer) {
    const Obj<ALE::Mesh::topology_type>&        topology    = mesh->getTopologyNew();
    const ALE::Mesh::topology_type::sheaf_type& patches     = topology->getPatches();
    const ALE::Mesh::topology_type::patch_type  firstPatch  = patches.begin()->first;
    const int      corners = topology->getPatch(firstPatch)->nCone(*topology->heightStratum(firstPatch, 0)->begin(), topology->depth())->size();
    int            numElements = 0;
    int            numVertices = 0;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    for(ALE::Mesh::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
      // This should be a Numbering, not just the stratum
      numElements += topology->heightStratum(p_iter->first, 0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const ALE::Mesh::topology_type::patch_type           patch      = p_iter->first;
        const Obj<ALE::Mesh::topology_type::sieve_type>&     sieve      = topology->getPatch(patch);
        const Obj<ALE::Mesh::topology_type::label_sequence>& elements   = topology->heightStratum(patch, 0);
        const Obj<ALE::Mesh::numbering_type>&                vNumbering = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getLocalNumbering(topology, patch, 0);
        const int                                            depth      = topology->depth(patch);

        for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<ALE::Mesh::sieve_type::coneArray>& cone = sieve->nCone(*e_iter, depth);

          ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
          for(ALE::Mesh::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", vNumbering->getIndex(patch, *c_iter) + numVertices);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        numVertices += topology->depthStratum(p_iter->first, 0)->size();
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
