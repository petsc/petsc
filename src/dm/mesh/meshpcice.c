#include "src/dm/mesh/meshpcice.h"   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace PCICE {
    //
    // Builder methods
    //
    void Builder::readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numCells, cellCount = 0;
      PetscInt      *verts;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      if (fgets(buf, 2048, f) == NULL) {
        throw ALE::Exception("Invalid connectivity file: Missing number of elements");
      }
      const char *sizes = strtok(buf, " ");
      numCells = atoi(sizes);
      sizes = strtok(NULL, " ");
      if (sizes != NULL) {
        corners = atoi(sizes);
        std::cout << "Reset corners to " << corners << std::endl;
      }
      ierr = PetscMalloc(numCells*corners * sizeof(PetscInt), &verts);
      while(fgets(buf, 2048, f) != NULL) {
        const char *v = strtok(buf, " ");
      
        /* Ignore cell number */
        v = strtok(NULL, " ");
        for(c = 0; c < corners; c++) {
          int vertex = atoi(v);
        
          if (!useZeroBase) vertex -= 1;
          verts[cellCount*corners+c] = vertex;
          v = strtok(NULL, " ");
        }
        cellCount++;
      }
      ierr = PetscViewerDestroy(viewer);
      numElements = numCells;
      *vertices = verts;
    };
    void Builder::readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numVerts, vertexCount = 0;
      PetscScalar   *coords;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      numVerts = atoi(fgets(buf, 2048, f));
      ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);
      while(fgets(buf, 2048, f) != NULL) {
        const char *x = strtok(buf, " ");
      
        /* Ignore vertex number */
        x = strtok(NULL, " ");
        for(c = 0; c < dim; c++) {
          coords[vertexCount*dim+c] = atof(x);
          x = strtok(NULL, " ");
        }
        vertexCount++;
      }
      ierr = PetscViewerDestroy(viewer);
      numVertices = numVerts;
      *coordinates = coords;
    };
    void Builder::buildCoordinates(const Obj<section_type>& coords, const int embedDim, const double coordinates[]) {
      const section_type::patch_type            patch    = 0;
      const Obj<topology_type::label_sequence>& vertices = coords->getTopology()->depthStratum(patch, 0);
      const int numCells = coords->getTopology()->heightStratum(patch, 0)->size();

      coords->setFiberDimensionByDepth(patch, 0, embedDim);
      coords->allocate();
      for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        coords->update(patch, *v_iter, &(coordinates[(*v_iter - numCells)*embedDim]));
      }
    };
    Obj<Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      return readMesh(comm, dim, basename+".nodes", basename+".lcon", useZeroBase, interpolate, debug);
    };
    Obj<Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      Obj<Mesh>          mesh     = Mesh(comm, dim, debug);
      Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
      Obj<topology_type> topology = new topology_type(comm, debug);
      int    *cells;
      double *coordinates;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;

      ALE::PCICE::Builder::readConnectivity(comm, adjFilename, numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(comm, coordFilename, dim, numVertices, &coordinates);
      ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
      sieve->stratify();
      topology->setPatch(0, sieve);
      topology->stratify();
      mesh->setTopologyNew(topology);
      buildCoordinates(mesh->getSection("coordinates"), dim, coordinates);
      return mesh;
    };
    // Creates boundary sections:
    //   IBC[NBFS,2]:     ALL
    //     BL[NBFS,1]:
    //     BNVEC[NBFS,2]:
    //   BCFUNC[NBCF,NV]: ALL
    //   IBNDFS[NBN,2]:   STILL NEED 4-5
    //     BNNV[NBN,2]
    void Builder::readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename, const int numBdFaces, const int numBdVertices) {
      const Mesh::topology_type::patch_type patch = 0;
      PetscViewer    viewer;
      FILE          *f;
      char           buf[2048];
      PetscErrorCode ierr;

      const Obj<Mesh::bc_section_type>& ibc    = mesh->getBCSection("IBC");
      const Obj<Mesh::bc_section_type>& ibndfs = mesh->getBCSection("IBNDFS");
      const Obj<Mesh::bc_section_type>& ibcnum = mesh->getBCSection("IBCNUM");
      const Obj<Mesh::section_type>&    bl     = mesh->getSection("BL");
      const Obj<Mesh::section_type>&    bnvec  = mesh->getSection("BNVEC");
      const Obj<Mesh::section_type>&    bnnv   = mesh->getSection("BNNV");
      const Obj<Mesh::section_type>&    bcvec  = mesh->getSection("BCVEC");
      if (mesh->commRank() != 0) {
        mesh->distributeBCValues();
        return;
      }
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, bcFilename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      // Create IBC section
      int *tmpIBC = new int[numBdFaces*4];
      std::map<int,std::set<int> > elem2Idx;
      std::map<int,int> bfReorder;
      for(int bf = 0; bf < numBdFaces; bf++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");

        // Ignore boundary face number
        x = strtok(NULL, " ");
        tmpIBC[bf*4+0] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+1] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+2] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+3] = atoi(x);
        const int elem = tmpIBC[bf*4+0]-1;

        ibc->addFiberDimension(patch, elem, 4);
        ibcnum->addFiberDimension(patch, elem, 1);
        bl->addFiberDimension(patch, elem, 1);
        bnvec->addFiberDimension(patch, elem, 2);
        bcvec->addFiberDimension(patch, elem, 4);
        elem2Idx[elem].insert(bf);
      }
      ibc->allocate();
      ibcnum->allocate();
      bl->allocate();
      bnvec->allocate();
      bcvec->allocate();
      const Mesh::bc_section_type::chart_type& chart = ibc->getPatch(patch);
      int num = 1;

      for(Mesh::bc_section_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const int elem = *p_iter;
        int bfNum[2];
        int k = 0;

        for(std::set<int>::const_iterator i_iter = elem2Idx[elem].begin(); i_iter != elem2Idx[elem].end(); ++i_iter) {
          bfReorder[(*i_iter)+1] = num;
          bfNum[k++] = num;
          num++;
        }
        ibcnum->update(patch, elem, bfNum);
      }
      for(int bf = 0; bf < numBdFaces; bf++) {
        const int elem = tmpIBC[bf*4]-1;

        if (elem2Idx[elem].size() > 1) {
          if (*elem2Idx[elem].begin() == bf) {
            int values[8];
            int k = 0;

            for(std::set<int>::const_iterator i_iter = elem2Idx[elem].begin(); i_iter != elem2Idx[elem].end(); ++i_iter) {
              for(int v = 0; v < 4; ++v) {
                values[k*4+v] = tmpIBC[*i_iter*4+v];
              }
              k++;
            }
            ibc->update(patch, elem, values);
          }
        } else {
          ibc->update(patch, elem, &tmpIBC[bf*4]);
        }
      }
      delete [] tmpIBC;
      // Create BCFUNC section
      int numBcFunc = atoi(strtok(fgets(buf, 2048, f), " "));
      for(int bc = 0; bc < numBcFunc; bc++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");
        Mesh::bc_value_type value;

        // Ignore function number
        x = strtok(NULL, " ");
        value.rho = atof(x);
        x = strtok(NULL, " ");
        value.u   = atof(x);
        x = strtok(NULL, " ");
        value.v   = atof(x);
        x = strtok(NULL, " ");
        value.p   = atof(x);
        mesh->setBCValue(bc+1, value);
      }
      mesh->distributeBCValues();
      // Create IBNDFS section
      const int numElements = mesh->getTopologyNew()->heightStratum(patch, 0)->size();
      int      *tmpIBNDFS   = new int[numBdVertices*3];

      for(int bv = 0; bv < numBdVertices; bv++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");

        // Ignore boundary node number
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+0] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+1] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+2] = atoi(x);
        ibndfs->setFiberDimension(patch, tmpIBNDFS[bv*3+0]-1+numElements, 5);
      }
      ibndfs->allocate();
      for(int bv = 0; bv < numBdVertices; bv++) {
        int values[5];

        values[0] = tmpIBNDFS[bv*3+0];
        // Covert to new boundary face numbers
        values[1] = bfReorder[tmpIBNDFS[bv*3+1]];
        values[2] = bfReorder[tmpIBNDFS[bv*3+2]];
        values[3] = 0;
        values[4] = 0;
        ibndfs->update(patch, values[0]-1+numElements, values);
      }
      ierr = PetscViewerDestroy(viewer);
      // Create BNNV[NBN,2]
      const int dim = mesh->getDimension();

      for(int bv = 0; bv < numBdVertices; bv++) {
        bnnv->setFiberDimension(patch, tmpIBNDFS[bv*3+0]-1+numElements, dim);
      }
      bnnv->allocate();
      delete [] tmpIBNDFS;
    };
    void Builder::outputVerticesLocal(const Obj<Mesh>& mesh, int *numVertices, int *dim, double *coordinates[], const bool columnMajor) {
      const Mesh::section_type::patch_type            patch      = 0;
      const Obj<Mesh::section_type>&                  coordSec   = mesh->getSection("coordinates");
      if (!coordSec->hasPatch(patch)) {
        *numVertices = 0;
        *dim         = 0;
        *coordinates = NULL;
        return;
      }
      const Obj<Mesh::topology_type::label_sequence>& vertices   = mesh->getTopologyNew()->depthStratum(patch, 0);
      const Obj<Mesh::numbering_type>&                vNumbering = mesh->getLocalNumbering(0);
      int            size     = vertices->size();
      int            embedDim = coordSec->getFiberDimension(patch, *vertices->begin());
      double        *coords;
      PetscErrorCode ierr;

      ierr = PetscMalloc(vertices->size()*embedDim * sizeof(double), &coords);
      for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::section_type::value_type *array = coordSec->restrict(patch, *v_iter);
        const int                             row   = vNumbering->getIndex(*v_iter);

        if (columnMajor) {
          for(int d = 0; d < embedDim; d++) {
            coords[d*size + row] = array[d];
          }
        } else {
          for(int d = 0; d < embedDim; d++) {
            coords[row*embedDim + d] = array[d];
          }
        }
      }
      *numVertices = size;
      *dim         = embedDim;
      *coordinates = coords;
    };
    void Builder::outputElementsLocal(const Obj<Mesh>& mesh, int *numElements, int *numCorners, int *vertices[], const bool columnMajor) {
      const Mesh::topology_type::patch_type           patch      = 0;
      const Obj<Mesh::topology_type>&                 topology   = mesh->getTopologyNew();
      if (!topology->hasPatch(patch)) {
        *numElements = 0;
        *numCorners  = 0;
        *vertices    = NULL;
        return;
      }
      const Obj<Mesh::sieve_type>&                    sieve      = topology->getPatch(patch);
      const Obj<Mesh::topology_type::label_sequence>& elements   = topology->heightStratum(patch, 0);
      const Obj<Mesh::numbering_type>&                eNumbering = mesh->getLocalNumbering(topology->depth());
      const Obj<Mesh::numbering_type>&                vNumbering = mesh->getLocalNumbering(0);
      int            size         = elements->size();
      //int            corners      = sieve->nCone(*elements->begin(), topology->depth())->size();
      int            corners      = sieve->cone(*elements->begin())->size();
      int           *v;
      PetscErrorCode ierr;

      ierr = PetscMalloc(elements->size()*corners * sizeof(int), &v);
      for(Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<Mesh::sieve_type::traits::coneSequence> cone  = sieve->cone(*e_iter);
        Mesh::sieve_type::traits::coneSequence::iterator  begin = cone->begin();
        Mesh::sieve_type::traits::coneSequence::iterator  end   = cone->end();

        const int row = eNumbering->getIndex(*e_iter);
        int       c   = -1;
        if (columnMajor) {
          for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[(++c)*size + row] = vNumbering->getIndex(*c_iter)+1;
          }
        } else {
          for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[row*corners + ++c] = vNumbering->getIndex(*c_iter)+1;
          }
        }
      }
      *numElements = size;
      *numCorners  = corners;
      *vertices    = v;
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteVertices"
    PetscErrorCode Viewer::writeVertices(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer) {
      ALE::Obj<ALE::Mesh::section_type> coordinates = mesh->getSection("coordinates");
#if 0
      ALE::Mesh::field_type::patch_type patch;
      const double  *array = coordinates->restrict(patch);
      int            numVertices;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      //FIX:
      if (vertexBundle->getGlobalOffsets()) {
        numVertices = vertexBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numVertices = mesh->getTopology()->depthStratum(0)->size();
      }
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", numVertices);CHKERRQ(ierr);
      if (mesh->commRank() == 0) {
        int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
        int embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int vertexCount = 1;

        for(int v = 0; v < numLocalVertices; v++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%7D   ", vertexCount++);CHKERRQ(ierr);
          for(int d = 0; d < embedDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", array[v*embedDim+d]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          double    *remoteCoords;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalVertices, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteCoords, numLocalVertices*embedDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int v = 0; v < numLocalVertices; v++) {
            ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
            for(int d = 0; d < embedDim; d++) {
              if (d > 0) {
                ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", remoteCoords[v*embedDim+d]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
        }
      } else {
        ALE::Obj<ALE::Mesh::bundle_type>                           globalOrder = coordinates->getGlobalOrder();
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone        = globalOrder->getPatch(patch);
        const int *offsets = coordinates->getGlobalOffsets();
        int        embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/embedDim;
        double    *localCoords;
        int        k = 0;

        ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &localCoords);CHKERRQ(ierr);
        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          int dim = globalOrder->getFiberDimension(patch, *p_iter);

          if (dim > 0) {
            int offset = coordinates->getFiberOffset(patch, *p_iter);

            for(int i = offset; i < offset+dim; ++i) {
              localCoords[k++] = array[i];
            }
          }
        }
        if (k != numLocalVertices*embedDim) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalVertices*embedDim);
        }
        ierr = MPI_Send(&numLocalVertices, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localCoords, numLocalVertices*embedDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localCoords);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteElements"
    PetscErrorCode Viewer::writeElements(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer) {
      ALE::Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
#if 0
      ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      ALE::Obj<ALE::Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
      ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
      ALE::Obj<ALE::Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
      ALE::Obj<ALE::Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
      ALE::Mesh::bundle_type::patch_type patch;
      std::string    orderName("element");
      int            dim  = mesh->getDimension();
      int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
      int            numElements;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (corners != dim+1) {
        SETERRQ(PETSC_ERR_SUP, "PCICE only supports simplicies");
      }
      if (!globalVertex) {
        globalVertex = vertexBundle;
      }
      if (elementBundle->getGlobalOffsets()) {
        numElements = elementBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numElements = mesh->getTopology()->heightStratum(0)->size();
      }
      if (mesh->commRank() == 0) {
        int elementCount = 1;

        ierr = PetscViewerASCIIPrintf(viewer, "%d\n", numElements);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
          for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
            ierr = PetscViewerASCIIPrintf(viewer, " %7d", globalVertex->getIndex(patch, *c_itor).prefix);CHKERRQ(ierr);
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
            ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
            for(int c = 0; c < corners; c++) {
              ierr = PetscViewerASCIIPrintf(viewer, " %7d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
          ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
        }
      } else {
        const int *offsets = elementBundle->getGlobalOffsets();
        int        numLocalElements = offsets[mesh->commRank()+1] - offsets[mesh->commRank()];
        int       *localVertices;
        int        k = 0;

        ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
            for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              localVertices[k++] = globalVertex->getIndex(patch, *c_itor).prefix;
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
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteVerticesLocal"
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      const Mesh::section_type::patch_type            patch       = 0;
      Obj<Mesh::section_type>                         coordinates = mesh->getSection("coordinates");
      const Obj<Mesh::topology_type>&                 topology    = mesh->getTopologyNew();
      const Obj<Mesh::topology_type::label_sequence>& vertices    = topology->depthStratum(patch, 0);
      const Obj<Mesh::numbering_type>&                vNumbering  = mesh->getLocalNumbering(0);
      int            embedDim = coordinates->getFiberDimension(patch, *vertices->begin());
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", vertices->size());CHKERRQ(ierr);
      for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::section_type::value_type *array = coordinates->restrict(patch, *v_iter);

        PetscViewerASCIIPrintf(viewer, "%7D   ", vNumbering->getIndex(*v_iter)+1);
        for(int d = 0; d < embedDim; d++) {
          if (d > 0) {
            PetscViewerASCIIPrintf(viewer, " ");
          }
          PetscViewerASCIIPrintf(viewer, "% 12.5E", array[d]);
        }
        PetscViewerASCIIPrintf(viewer, "\n");
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteRestart"
    PetscErrorCode Viewer::writeRestart(const Obj<Mesh>& mesh, PetscViewer viewer) {
      const Mesh::section_type::patch_type patch = 0;
      const Obj<Mesh::section_type>&   velocity    = mesh->getSection("VELN");
      const Obj<Mesh::section_type>&   pressure    = mesh->getSection("PN");
      const Obj<Mesh::section_type>&   temperature = mesh->getSection("TN");
      const Obj<Mesh::numbering_type>& vNumbering  = mesh->getNumbering(0);
      const Obj<Mesh::numbering_type>& cNumbering  = mesh->getNumbering(mesh->getTopologyNew()->depth());
      const int                        numCells    = cNumbering->getGlobalSize();
      PetscErrorCode ierr;

      PetscFunctionBegin;
      int          blen[2];
      MPI_Aint     indices[2];
      MPI_Datatype oldtypes[2], newtype;
      blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
      blen[1] = 4; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
      ierr = MPI_Type_struct(2, blen, indices, oldtypes, &newtype);CHKERRQ(ierr);
      ierr = MPI_Type_commit(&newtype);CHKERRQ(ierr);

      if (mesh->commRank() == 0) {
        const Obj<Mesh::topology_type::label_sequence>& vertices = mesh->getTopologyNew()->depthStratum(patch, 0);

        for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const ALE::Mesh::section_type::value_type *veln = velocity->restrictPoint(patch, *v_iter);
            const ALE::Mesh::section_type::value_type *pn   = pressure->restrictPoint(patch, *v_iter);
            const ALE::Mesh::section_type::value_type *tn   = temperature->restrictPoint(patch, *v_iter);

            ierr = PetscViewerASCIIPrintf(viewer, "%6d% 16.8E% 16.8E% 16.8E% 16.8E\n", *v_iter-numCells+1, veln[0], veln[1], pn[0], tn[0]);CHKERRQ(ierr);
          }
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          RestartType *remoteValues;
          int          numLocalElements;
          MPI_Status   status;

          ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalElements * sizeof(RestartType), &remoteValues);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteValues, numLocalElements, newtype, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int e = 0; e < numLocalElements; e++) {
            ierr = PetscViewerASCIIPrintf(viewer, "%6d% 16.8E% 16.8E% 16.8E% 16.8E\n", remoteValues[e].vertex-numCells+1, remoteValues[e].veln_x, remoteValues[e].veln_y, remoteValues[e].pn, remoteValues[e].tn);CHKERRQ(ierr);
          }
        }
      } else {
        const Obj<Mesh::topology_type::label_sequence>& vertices = mesh->getTopologyNew()->depthStratum(patch, 0);
        RestartType *localValues;
        int numLocalElements = vNumbering->getLocalSize();
        int k = 0;

        ierr = PetscMalloc(numLocalElements * sizeof(RestartType), &localValues);CHKERRQ(ierr);
        for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const ALE::Mesh::section_type::value_type *veln = velocity->restrictPoint(patch, *v_iter);
            const ALE::Mesh::section_type::value_type *pn   = pressure->restrictPoint(patch, *v_iter);
            const ALE::Mesh::section_type::value_type *tn   = temperature->restrictPoint(patch, *v_iter);

            localValues[k].vertex = *v_iter;
            localValues[k].veln_x = veln[0];
            localValues[k].veln_y = veln[1];
            localValues[k].pn     = pn[0];
            localValues[k].tn     = tn[0];
            k++;
          }
        }
        if (k != numLocalElements) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, numLocalElements);
        }
        ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localValues, numLocalElements, newtype, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localValues);CHKERRQ(ierr);
      }
      ierr = MPI_Type_free(&newtype);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    };
  };
};
