#include <petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#include<list>
#include<map>

namespace ALE {
  namespace PyLith {
    //
    // Builder methods
    //
    inline void Builder::ignoreComments(char *buf, PetscInt bufSize, FILE *f) {
      while((fgets(buf, bufSize, f) != NULL) && ((buf[0] == '#') || (buf[0] == '\0'))) {}
    };
    void Builder::readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[], int *materials[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       maxCells = 1024, cellCount = 0;
      PetscInt      *verts;
      PetscInt      *mats;
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
      /* Ignore comments */
      ignoreComments(buf, 2048, f);
      do {
        const char *v = strtok(buf, " ");
        int         elementType;

        if (cellCount == maxCells) {
          PetscInt *vtmp, *mtmp;

          vtmp = verts;
          mtmp = mats;
          ierr = PetscMalloc2(maxCells*2*corners,PetscInt,&verts,maxCells*2,PetscInt,&mats);
          ierr = PetscMemcpy(verts, vtmp, maxCells*corners * sizeof(PetscInt));
          ierr = PetscMemcpy(mats,  mtmp, maxCells         * sizeof(PetscInt));
          ierr = PetscFree2(vtmp,mtmp);
          maxCells *= 2;
        }
        /* Ignore cell number */
        v = strtok(NULL, " ");
        /* Get element type */
        elementType = atoi(v);
        if (elementType == 1) {
          corners = 8;
        } else if (elementType == 5) {
          corners = 4;
        } else {
          ostringstream msg;

          msg << "We do not accept element type " << elementType << " right now";
          throw ALE::Exception(msg.str().c_str());
        }
        if (cellCount == 0) {
          ierr = PetscMalloc2(maxCells*corners,PetscInt,&verts,maxCells,PetscInt,&mats);
        }
        v = strtok(NULL, " ");
        /* Store material type */
        mats[cellCount] = atoi(v);
        v = strtok(NULL, " ");
        /* Ignore infinite domain element code */
        v = strtok(NULL, " ");
        for(c = 0; c < corners; c++) {
          int vertex = atoi(v);
        
          if (!useZeroBase) vertex -= 1;
          verts[cellCount*corners+c] = vertex;
          v = strtok(NULL, " ");
        }
        cellCount++;
      } while(fgets(buf, 2048, f) != NULL);
      ierr = PetscViewerDestroy(viewer);
      numElements = cellCount;
      *vertices   = verts;
      *materials  = mats;
    };
    void Builder::readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       maxVerts = 1024, vertexCount = 0;
      PetscReal     *coords;
      double         scaleFactor = 1.0;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);
      if (commRank == 0) {
        ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
        ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
        ierr = PetscViewerFileSetName(viewer, filename.c_str());
        ierr = PetscViewerASCIIGetPointer(viewer, &f);
        /* Ignore comments */
        ignoreComments(buf, 2048, f);
        ierr = PetscMalloc(maxVerts*dim * sizeof(PetscReal), &coords);
        /* Read units */
        const char *units = strtok(buf, " ");
        if (strcmp(units, "coord_units")) {
          throw ALE::Exception("Invalid coordinate units line");
        }
        units = strtok(NULL, " ");
        if (strcmp(units, "=")) {
          throw ALE::Exception("Invalid coordinate units line");
        }
        units = strtok(NULL, " ");
        if (!strcmp(units, "km")) {
          /* Should use Pythia to do units conversion */
          scaleFactor = 1000.0;
        }
        /* Ignore comments */
        ignoreComments(buf, 2048, f);
        do {
          const char *x = strtok(buf, " ");

          if (vertexCount == maxVerts) {
            PetscReal *ctmp;

            ctmp = coords;
            ierr = PetscMalloc(maxVerts*2*dim * sizeof(PetscReal), &coords);
            ierr = PetscMemcpy(coords, ctmp, maxVerts*dim * sizeof(PetscReal));
            ierr = PetscFree(ctmp);
            maxVerts *= 2;
          }
          /* Ignore vertex number */
          x = strtok(NULL, " ");
          for(c = 0; c < dim; c++) {
            coords[vertexCount*dim+c] = atof(x)*scaleFactor;
            x = strtok(NULL, " ");
          }
          vertexCount++;
        } while(fgets(buf, 2048, f) != NULL);
        ierr = PetscViewerDestroy(viewer);
        numVertices = vertexCount;
        *coordinates = coords;
      }
    };
    // numSplit is the number of split node entries (lines in the file)
    // splitInd[] is an array of numSplit pairs, <element, vertex>
    // splitValues[] is an array of numSplit*dim displacements
    void Builder::readSplit(MPI_Comm comm, const std::string& filename, const int dim, const bool useZeroBase, int& numSplit, int *splitInd[], int *loadHistory[], double *splitValues[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       maxSplit = 1024, splitCount = 0;
      PetscInt      *splitId;
      PetscInt      *loadHist;
      PetscReal     *splitVal;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);
      if (dim != 3) {
        throw ALE::Exception("PyLith only works in 3D");
      }
      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename.c_str()), PETSC_ERR_FILE_OPEN);
      if (PetscExceptionValue(ierr)) {
        // this means that a caller above me has also tryed this exception so I don't handle it here, pass it up
      } else if (PetscExceptionCaught(ierr,PETSC_ERR_FILE_OPEN)) {
        // File does not exist
        return;
      } 
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      /* Ignore comments */
      ignoreComments(buf, 2048, f);
      ierr = PetscMalloc3(maxSplit*2,PetscInt,&splitId,maxSplit,PetscInt,&loadHist,maxSplit*dim,PetscReal,&splitVal);
      do {
        const char *s = strtok(buf, " ");

        if (splitCount == maxSplit) {
          PetscInt    *sitmp;
          PetscInt    *lhtmp;
          PetscReal   *svtmp;

          sitmp = splitId;
          lhtmp = loadHist;
          svtmp = splitVal;
          ierr = PetscMalloc3(maxSplit*2*2,PetscInt,&splitId,maxSplit*2,PetscInt,&loadHist,maxSplit*dim*2,PetscScalar,&splitVal);
          ierr = PetscMemcpy(splitId,  sitmp, maxSplit*2   * sizeof(PetscInt));
          ierr = PetscMemcpy(loadHist, lhtmp, maxSplit     * sizeof(PetscInt));
          ierr = PetscMemcpy(splitVal, svtmp, maxSplit*dim * sizeof(PetscReal));
          ierr = PetscFree3(sitmp,lhtmp,svtmp);
          maxSplit *= 2;
        }
        /* Get element number */
        int elem = atoi(s);
        if (!useZeroBase) elem -= 1;
        splitId[splitCount*2+0] = elem;
        s = strtok(NULL, " ");
        /* Get node number */
        int node = atoi(s);
        if (!useZeroBase) node -= 1;
        splitId[splitCount*2+1] = node;
        s = strtok(NULL, " ");
        /* Ignore load history number */
        loadHist[splitCount] = atoi(s);
        s = strtok(NULL, " ");
        /* Get split values */
        for(c = 0; c < dim; c++) {
          splitVal[splitCount*dim+c] = atof(s);
          s = strtok(NULL, " ");
        }
        splitCount++;
      } while(fgets(buf, 2048, f) != NULL);
      ierr = PetscViewerDestroy(viewer);
      numSplit     = splitCount;
      *splitInd    = splitId;
      *loadHistory = loadHist;
      *splitValues = splitVal;
    };
#if 0
    void Builder::buildSplit(const Obj<pair_section_type>& splitField, const Obj<int_section_type>& loadField, int numCells, int numSplit, int splitInd[], int loadHistory[], double splitVals[]) {
      const pair_section_type::patch_type                     patch = 0;
      pair_section_type::value_type                          *values;
      int_section_type::value_type                           *history;
      std::map<pair_section_type::point_type, std::set<int> > elem2index;
      int                                                     numValues = 0;

      splitField->setName("split");
      for(int e = 0; e < numSplit; e++) {
        splitField->addFiberDimension(patch, splitInd[e*2+0], 1);
        loadField->addFiberDimension(patch, splitInd[e*2+0], 1);
        elem2index[splitInd[e*2+0]].insert(e);
      }
      splitField->allocate();
      loadField->allocate();
      for(std::map<pair_section_type::point_type, std::set<int> >::const_iterator e_iter = elem2index.begin(); e_iter != elem2index.end(); ++e_iter) {
        numValues = std::max(numValues, (int) e_iter->second.size());
      }
      values  = new pair_section_type::value_type[numValues];
      history = new int_section_type::value_type[numValues];
      for(std::map<pair_section_type::point_type, std::set<int> >::const_iterator e_iter = elem2index.begin(); e_iter != elem2index.end(); ++e_iter) {
        const pair_section_type::point_type& e = e_iter->first;
        int                                  k = 0;

        for(std::set<int>::const_iterator i_iter = e_iter->second.begin(); i_iter != e_iter->second.end(); ++i_iter, ++k) {
          const int& i = *i_iter;

          if (k >= numValues) {throw ALE::Exception("Invalid split node input");}
          values[k].first    = splitInd[i*2+1] + numCells;
          values[k].second.x = splitVals[i*3+0];
          values[k].second.y = splitVals[i*3+1];
          values[k].second.z = splitVals[i*3+2];
          history[k]         = loadHistory[i];
        }
        splitField->updatePoint(patch, e, values);
        loadField->updatePoint(patch, e, history);
      }
      delete [] values;
    };
#endif
#ifdef PETSC_OPT_SIEVE
    void Builder::readTractions(MPI_Comm comm, const std::string& filename, const int dim, const int& corners, const bool useZeroBase, int& numTractions, int& vertsPerFace, int *tractionVertices[], double *tractionValues[]) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    void Builder::buildTractions(const Obj<real_section_type>& tractionField, const Obj<Mesh>& boundaryMesh, int numCells, int numTractions, int vertsPerFace, int tractionVertices[], double tractionValues[]) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    void Builder::buildMaterials(const Obj<Mesh>& mesh, const Obj<int_section_type>& matField, const int materials[]) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    Obj<Builder::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = false, const bool interpolate = false, const int debug = 0) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    Obj<Builder::Mesh> Builder::createTraction(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase = false) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    void Builder::createCohesiveElements(const Obj<Mesh>& mesh, const std::set<Mesh::point_type>& faultVertices) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeElements(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeElementsLocal(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeTractionsLocal(const Obj<Mesh>& mesh, const Obj<Mesh>& tractionMesh, const Obj<Builder::real_section_type>& tractionField, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
#else
    void Builder::readTractions(MPI_Comm comm, const std::string& filename, const int dim, const int& corners, const bool useZeroBase, int& numTractions, int& vertsPerFace, int *tractionVertices[], double *tractionValues[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       maxTractions = 1024, tractionCount = 0;
      PetscInt      *tractionVerts;
      PetscScalar   *tractionVals;
      double         scaleFactor = 1.0;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);
      if (dim != 3) {
        throw ALE::Exception("PyLith only works in 3D");
      }
      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename.c_str()), PETSC_ERR_FILE_OPEN);
      if (PetscExceptionValue(ierr)) {
        // this means that a caller above me has also tryed this exception so I don't handle it here, pass it up
      } else if (PetscExceptionCaught(ierr,PETSC_ERR_FILE_OPEN)) {
        // File does not exist
        return;
      } 
      /* Logic right now is only good for linear tets and hexes, and should be fixed in the future. */
      if (corners == 4) {
        vertsPerFace = 3;
      } else if (corners == 8) {
        vertsPerFace = 4;
      } else {
        throw ALE::Exception("Unrecognized element type");
      }

      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      /* Ignore comments */
      ignoreComments(buf, 2048, f);
      /* Read units */
      const char *units = strtok(buf, " ");
      if (strcmp(units, "traction_units")) {
        throw ALE::Exception("Invalid traction units line");
      }
      units = strtok(NULL, " ");
      if (strcmp(units, "=")) {
        throw ALE::Exception("Invalid traction units line");
      }
      units = strtok(NULL, " ");
      if (!strcmp(units, "MPa")) {
        /* Should use Pythia to do units conversion */
        scaleFactor = 1.0e6;
      }
      /* Ignore comments */
      ignoreComments(buf, 2048, f);
      // Allocate memory.
      ierr = PetscMalloc2(maxTractions*vertsPerFace,PetscInt,&tractionVerts,maxTractions*dim,PetscScalar,&tractionVals);
      do {
        const char *s = strtok(buf, " ");

        if (tractionCount == maxTractions) {
          PetscInt    *titmp;
          PetscScalar *tvtmp;

          titmp = tractionVerts;
          tvtmp = tractionVals;
          ierr = PetscMalloc2(maxTractions*vertsPerFace*2,PetscInt,&tractionVerts,maxTractions*dim*2,PetscScalar,&tractionVals);
          ierr = PetscMemcpy(tractionVerts,  titmp, maxTractions*vertsPerFace   * sizeof(PetscInt));
          ierr = PetscMemcpy(tractionVals, tvtmp, maxTractions*dim * sizeof(PetscScalar));
          ierr = PetscFree2(titmp,tvtmp);
          maxTractions *= 2;
        }
        /* Get vertices */
        int v1 = atoi(s);
        if (!useZeroBase) v1 -= 1;
        tractionVerts[tractionCount*vertsPerFace+0] = v1;
        s = strtok(NULL, " ");
        int v2 = atoi(s);
        if (!useZeroBase) v2 -= 1;
        tractionVerts[tractionCount*vertsPerFace+1] = v2;
        s = strtok(NULL, " ");
        int v3 = atoi(s);
        if (!useZeroBase) v3 -= 1;
        tractionVerts[tractionCount*vertsPerFace+2] = v3;
        s = strtok(NULL, " ");
        if (vertsPerFace > 3) {
          int v4 = atoi(s);
          if (!useZeroBase) v4 -= 1;
          tractionVerts[tractionCount*vertsPerFace+3] = v4;
          s = strtok(NULL, " ");
        }
        /* Get traction values */
        for(c = 0; c < dim; c++) {
          tractionVals[tractionCount*dim+c] = atof(s);
          s = strtok(NULL, " ");
        }
        tractionCount++;
      } while(fgets(buf, 2048, f) != NULL);
      ierr = PetscViewerDestroy(viewer);
      numTractions      = tractionCount;
      *tractionVertices = tractionVerts;
      *tractionValues   = tractionVals;
    };
    void Builder::buildTractions(const Obj<real_section_type>& tractionField, const Obj<Mesh>& boundaryMesh, int numCells, int numTractions, int vertsPerFace, int tractionVertices[], double tractionValues[]) {
      real_section_type::value_type values[3];
      // Make boundary topology
      Obj<sieve_type> boundarySieve = new sieve_type(tractionField->comm(), tractionField->debug());

      ALE::SieveBuilder<Mesh>::buildTopology(boundarySieve, 2, numTractions, tractionVertices, 0, false, vertsPerFace, numCells);
      boundaryMesh->setSieve(boundarySieve);
      boundaryMesh->stratify();
      // Make traction field
      tractionField->setName("traction");
      tractionField->setFiberDimension(boundaryMesh->heightStratum(0), 3);
      boundaryMesh->allocate(tractionField);
      const Obj<Mesh::label_sequence>& faces = boundaryMesh->heightStratum(0);
      int k = 0;

      for(Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
        const Mesh::point_type& face = *f_iter;

        values[0] = tractionValues[k*3+0];
        values[1] = tractionValues[k*3+1];
        values[2] = tractionValues[k*3+2];
        k++;
        tractionField->updatePoint(face, values);
      }
    };
    void Builder::buildMaterials(const Obj<Mesh>& mesh, const Obj<int_section_type>& matField, const int materials[]) {
      const Obj<Mesh::label_sequence>& elements = mesh->heightStratum(0);

      matField->setName("material");
      matField->setFiberDimension(elements, 1);
      mesh->allocate(matField);
      for(Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        matField->updatePoint(*e_iter, &materials[*e_iter]);
      }
    };
    Obj<Builder::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = false, const bool interpolate = false, const int debug = 0) {
      Obj<Mesh>       mesh  = new Mesh(comm, dim, debug);
      Obj<sieve_type> sieve = new sieve_type(comm, debug);
      int    *cells, *materials;
      double *coordinates;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;

      ALE::PyLith::Builder::readConnectivity(comm, basename+".connect", numCorners, useZeroBase, numCells, &cells, &materials);
      ALE::PyLith::Builder::readCoordinates(comm, basename+".coord", dim, numVertices, &coordinates);
      ALE::SieveBuilder<Mesh>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
      mesh->setSieve(sieve);
      mesh->stratify();
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coordinates);
      Obj<int_section_type> material = mesh->getIntSection("material");
      buildMaterials(mesh, material, materials);
      Obj<PETSC_MESH_TYPE> tractionMesh = createTraction(mesh, basename, useZeroBase);
      if (!tractionMesh.isNull()) {/* Have to carry tractions around somehow */}
      return mesh;
    };
    Obj<Builder::Mesh> Builder::createTraction(const Obj<Mesh>& mesh, const std::string& basename, const bool useZeroBase = false) {
      Obj<Mesh> tractionMesh = NULL;
      MPI_Comm comm       = mesh->comm();
      int      debug      = mesh->debug();
      int      dim        = mesh->getDimension();
      int      numCells   = mesh->heightStratum(0)->size();
      int      numCorners = mesh->getSieve()->cone(*mesh->heightStratum(0)->begin())->size();
      int     *tractionVertices;
      double  *tractionValues;
      int      numTractions = 0, vertsPerFace = 0, hasTractions;

      ALE::PyLith::Builder::readTractions(comm, basename+".traction", dim, numCorners, useZeroBase, numTractions, vertsPerFace, &tractionVertices, &tractionValues);
      MPI_Allreduce(&numTractions, &hasTractions, 1, MPI_INT, MPI_MAX, comm);
      if (hasTractions) {
        tractionMesh = new Mesh(comm, debug);

        const Obj<Mesh::real_section_type>& traction = tractionMesh->getRealSection("traction");
        buildTractions(traction, tractionMesh, numCells, numTractions, vertsPerFace, tractionVertices, tractionValues);
      }
      return tractionMesh;
    };
    void Builder::createCohesiveElements(const Obj<Mesh>& mesh, const std::set<Mesh::point_type>& faultVertices) {
      typedef std::vector<Mesh::point_type> PointArray;
      const Obj<Mesh::sieve_type>               sieve      = mesh->getSieve();
      const Obj<Mesh>                           fault      = new Mesh(mesh->comm(), mesh->debug());
      const Obj<Mesh::sieve_type>               faultSieve = new Mesh::sieve_type(sieve->comm(), sieve->debug());
      const std::set<Mesh::point_type>::const_iterator fvBegin = faultVertices.begin();
      const std::set<Mesh::point_type>::const_iterator fvEnd   = faultVertices.end();
      // There should be logic here to determine this
      const unsigned int                        faceSize   = 3;
      int                                       f          = 0;
      int                                       debug      = mesh->debug();
      Obj<PointArray>                           face       = new PointArray();
      std::set<Mesh::point_type>                faultCells;

      // Create a sieve which captures the fault
      for(std::set<int>::const_iterator fv_iter = fvBegin; fv_iter != fvEnd; ++fv_iter) {
        const Obj<Mesh::sieve_type::traits::supportSequence>&     cells  = sieve->support(*fv_iter);
        const Mesh::sieve_type::traits::supportSequence::iterator cBegin = cells->begin();
        const Mesh::sieve_type::traits::supportSequence::iterator cEnd   = cells->end();

        if (debug) {std::cout << "Checking fault vertex " << *fv_iter << std::endl;}
        for(Mesh::sieve_type::traits::supportSequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
          if (debug) {std::cout << "  Checking cell " << *c_iter << std::endl;}
          if (faultCells.find(*c_iter) != faultCells.end()) continue;
          const Obj<Mesh::sieve_type::traits::coneSequence>& cone   = sieve->cone(*c_iter);
          const Mesh::sieve_type::traits::coneSequence::iterator  vBegin = cone->begin();
          const Mesh::sieve_type::traits::coneSequence::iterator  vEnd   = cone->end();

          face->clear();
          for(Mesh::sieve_type::traits::coneSequence::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
            if (faultVertices.find(*v_iter) != fvEnd) {
              if (debug) {std::cout << "    contains fault vertex " << *v_iter << std::endl;}
              face->insert(face->end(), *v_iter);
            }
          }
          if (face->size() > faceSize) throw ALE::Exception("Invalid fault mesh: Too many vertices of an element on the fault");
          if (face->size() == faceSize) {
            if (debug) {std::cout << "  Contains a face on the fault" << std::endl;}
            const Obj<sieve_type::supportSet> preFace = faultSieve->nJoin1(face);

            if (preFace->size() > 1) {
              throw ALE::Exception("Invalid fault sieve: Multiple faces from vertex set");
            } else if (preFace->size() == 1) {
              faultSieve->addArrow(*preFace->begin(), *c_iter);
            } else if (preFace->size() == 0) {
              if (debug) {std::cout << "  Adding face " << f << std::endl;}
              int color = 0;
              for(PointArray::const_iterator f_iter = face->begin(); f_iter != face->end(); ++f_iter) {
                if (debug) {std::cout << "    vertex " << *f_iter << std::endl;}
                faultSieve->addArrow(*f_iter, f, color++);
              }
              faultSieve->addArrow(f, *c_iter);
              f++;
            }
            faultCells.insert(*c_iter);
          }
        }
      }
      fault->setSieve(faultSieve);
      fault->stratify();
      faultCells.clear();
      if (debug) {fault->view("Fault mesh");}
      // Add new shadow vertices
      const Obj<Mesh::label_sequence>& fVertices = fault->depthStratum(0);
      const Obj<Mesh::label_sequence>& vertices  = mesh->depthStratum(0);
      Mesh::point_type                 newVertex = *vertices->begin() + vertices->size();
      std::map<int,int>                vertexRenumber;

      for(Mesh::label_sequence::iterator v_iter = fVertices->begin(); v_iter != fVertices->end(); ++v_iter) {
        if (debug) {std::cout << "Duplicating " << *v_iter << " to " << vertexRenumber[*v_iter] << std::endl;}
        vertexRenumber[*v_iter] = newVertex++;
      }
      // Split the mesh along the fault sieve and create cohesive elements
      const Obj<Mesh::label_sequence>& faces = fault->depthStratum(1);
      PointArray                       newVertices;

      for(Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
        if (debug) {std::cout << "Considering fault face " << *f_iter << std::endl;}
        const Obj<Mesh::sieve_type::traits::supportSequence>& cells = faultSieve->support(*f_iter);
        Mesh::point_type                                      cell  = std::max(*cells->begin(), *(++cells->begin()));
        const Obj<Mesh::sieve_type::traits::coneSequence>&    cone  = sieve->cone(cell);

        if (debug) {std::cout << "  Replacing cell " << cell << std::endl;}
        newVertices.clear();
        for(PETSC_MESH_TYPE::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
          if (vertexRenumber.find(*v_iter) != vertexRenumber.end()) {
            if (debug) {std::cout << "    vertex " << vertexRenumber[*v_iter] << std::endl;}
            newVertices.insert(newVertices.end(), vertexRenumber[*v_iter]);
          } else {
            if (debug) {std::cout << "    vertex " << *v_iter << std::endl;}
            newVertices.insert(newVertices.end(), *v_iter);
          }
        }
        sieve->clearCone(cell);
        int color = 0;
        for(PointArray::const_iterator v_iter = newVertices.begin(); v_iter != newVertices.end(); ++v_iter) {
          sieve->addArrow(*v_iter, cell, color++);
        }
      }
      // Fix coordinates
      const Obj<Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
      const Obj<Mesh::label_sequence>&    fVertices2  = fault->depthStratum(0);

      for(Mesh::label_sequence::iterator v_iter = fVertices2->begin(); v_iter != fVertices2->end(); ++v_iter) {
        coordinates->addPoint(vertexRenumber[*v_iter], coordinates->getFiberDimension(*v_iter));
      }
      mesh->reallocate(coordinates);
      for(Mesh::label_sequence::iterator v_iter = fVertices2->begin(); v_iter != fVertices2->end(); ++v_iter) {
        coordinates->updatePoint(vertexRenumber[*v_iter], coordinates->restrictPoint(*v_iter));
      }
    };
    //
    // Viewer methods
    //
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteVertices"
    PetscErrorCode Viewer::writeVertices(const Obj<Mesh>& mesh, PetscViewer viewer) {
      Obj<Builder::real_section_type> coordinates  = mesh->getRealSection("coordinates");
      //Mesh::section_type::patch_type patch;
      //const double  *array = coordinates->restrict(Mesh::section_type::patch_type());
      //int            dim = mesh->getDimension();
      //int            numVertices;
      //PetscErrorCode ierr;

      PetscFunctionBegin;
#if 0
      //FIX:
      if (vertexBundle->getGlobalOffsets()) {
        numVertices = vertexBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numVertices = mesh->getTopology()->depthStratum(0)->size();
      }
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"coord_units = m\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      if (mesh->commRank() == 0) {
        int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
        int vertexCount = 1;

        for(int v = 0; v < numLocalVertices; v++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%7D ", vertexCount++);CHKERRQ(ierr);
          for(int d = 0; d < dim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"% 16.8E", array[v*dim+d]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          double    *remoteCoords;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalVertices, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalVertices*dim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteCoords, numLocalVertices*dim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int v = 0; v < numLocalVertices; v++) {
            ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
            for(int d = 0; d < dim; d++) {
              if (d > 0) {
                ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer, "% 16.8E", remoteCoords[v*dim+d]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
        }
      } else {
        Obj<Mesh::bundle_type> globalOrder = coordinates->getGlobalOrder();
        Obj<Mesh::field_type::order_type::coneSequence> cone = globalOrder->getPatch(patch);
        const int *offsets = coordinates->getGlobalOffsets();
        int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/dim;
        double    *localCoords;
        int        k = 0;

        ierr = PetscMalloc(numLocalVertices*dim * sizeof(double), &localCoords);CHKERRQ(ierr);
        for(Mesh::field_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          int dim = globalOrder->getFiberDimension(patch, *p_iter);

          if (dim > 0) {
            int offset = coordinates->getFiberOffset(patch, *p_iter);

            for(int i = offset; i < offset+dim; ++i) {
              localCoords[k++] = array[i];
            }
          }
        }
        if (k != numLocalVertices*dim) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalVertices*dim);
        }
        ierr = MPI_Send(&numLocalVertices, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localCoords, numLocalVertices*dim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localCoords);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteElements"
    PetscErrorCode Viewer::writeElements(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer) {
#if 0
      Obj<Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      Obj<Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
      Obj<Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
      Obj<Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
      Obj<Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
      Mesh::bundle_type::patch_type patch;
      std::string    orderName("element");
      bool           hasMaterial  = !materialField.isNull();
      int            dim  = mesh->getDimension();
      int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
      int            elementType = -1;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (dim != 3) {
        SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
      }
      if (corners == 4) {
        // Linear tetrahedron
        elementType = 5;
      } else if (corners == 8) {
        // Linear hexahedron
        elementType = 1;
      } else {
        SETERRQ1(PETSC_ERR_SUP, "PyLith Error: Unsupported number of elements vertices: %d", corners);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#     N ETP MAT INF     N1     N2     N3     N4     N5     N6     N7     N8\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      if (mesh->commRank() == 0) {
        int elementCount = 1;

        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d", elementCount++, elementType);CHKERRQ(ierr);
          if (hasMaterial) {
            // No infinite elements
            ierr = PetscViewerASCIIPrintf(viewer, " %3d %3d", (int) materialField->restrict(patch, *e_itor)[0], 0);CHKERRQ(ierr);
          } else {
            // No infinite elements
            ierr = PetscViewerASCIIPrintf(viewer, " %3d %3d", 1, 0);CHKERRQ(ierr);
          }
          for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
            ierr = PetscViewerASCIIPrintf(viewer, " %6d", globalVertex->getIndex(patch, *c_itor).prefix+1);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          int         numLocalElements;
          int        *remoteVertices;
          MPI_Status  status;

          ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalElements*(corners+1) * sizeof(int), &remoteVertices);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteVertices, numLocalElements*(corners+1), MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int e = 0; e < numLocalElements; e++) {
            // Only linear tetrahedra, material, no infinite elements
            int mat = remoteVertices[e*(corners+1)+corners];

            ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, elementType, mat, 0);CHKERRQ(ierr);
            for(int c = 0; c < corners; c++) {
              ierr = PetscViewerASCIIPrintf(viewer, " %6d", remoteVertices[e*(corners+1)+c]);CHKERRQ(ierr);
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

        ierr = PetscMalloc(numLocalElements*(corners+1) * sizeof(int), &localVertices);CHKERRQ(ierr);
        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
            for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              localVertices[k++] = globalVertex->getIndex(patch, *c_itor).prefix;
            }
            if (hasMaterial) {
              localVertices[k++] = (int) materialField->restrict(patch, *e_itor)[0];
            } else {
              localVertices[k++] = 1;
            }
          }
        }
        if (k != numLocalElements*corners) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
        }
        ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localVertices, numLocalElements*(corners+1), MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localVertices);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteVerticesLocal"
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      const Obj<Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
      const Obj<Mesh::label_sequence>&    vertices    = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>&    vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            embedDim = coordinates->getFiberDimension(*vertices->begin());
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"coord_units = m\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);

      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);

        PetscViewerASCIIPrintf(viewer, "%7D ", vNumbering->getIndex(*v_iter)+1);
        for(int d = 0; d < embedDim; d++) {
          if (d > 0) {
            PetscViewerASCIIPrintf(viewer, " ");
          }
          PetscViewerASCIIPrintf(viewer, "% 16.8E", array[d]);
        }
        PetscViewerASCIIPrintf(viewer, "\n");
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteElementsLocal"
    PetscErrorCode Viewer::writeElementsLocal(const Obj<Mesh>& mesh, const Obj<Builder::int_section_type>& materialField, PetscViewer viewer) {
      const Obj<Mesh::sieve_type>&     sieve      = mesh->getSieve();
      const Obj<Mesh::label_sequence>& elements   = mesh->heightStratum(0);
      const Obj<Mesh::numbering_type>& eNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            dim          = mesh->getDimension();
      //int            corners      = sieve->nCone(*elements->begin(), topology->depth())->size();
      int            corners      = sieve->cone(*elements->begin())->size();
      bool           hasMaterial  = !materialField.isNull();
      int            elementType  = -1;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (dim != 3) {
        SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
      }
      if (corners == 4) {
        // Linear tetrahedron
        elementType = 5;
      } else if (corners == 8) {
        // Linear hexahedron
        elementType = 1;
      } else {
        SETERRQ1(PETSC_ERR_SUP, "PyLith Error: Unsupported number of elements vertices: %d", corners);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#     N ETP MAT INF     N1     N2     N3     N4     N5     N6     N7     N8\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      for(Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<Mesh::sieve_type::traits::coneSequence> cone  = sieve->cone(*e_iter);
        Mesh::sieve_type::traits::coneSequence::iterator  begin = cone->begin();
        Mesh::sieve_type::traits::coneSequence::iterator  end   = cone->end();

        ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d", eNumbering->getIndex(*e_iter)+1, elementType);CHKERRQ(ierr);
        if (hasMaterial) {
          // No infinite elements
          ierr = PetscViewerASCIIPrintf(viewer, " %3d %3d", (int) materialField->restrictPoint(*e_iter)[0], 0);CHKERRQ(ierr);
        } else {
          // No infinite elements
          ierr = PetscViewerASCIIPrintf(viewer, " %3d %3d", 1, 0);CHKERRQ(ierr);
        }
        for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
          //FIX: Need a global ordering here
          ierr = PetscViewerASCIIPrintf(viewer, " %6d", vNumbering->getIndex(*c_iter)+1);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    };
#if 0
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteSplitLocal"
    // The elements seem to be implicitly numbered by appearance, which makes it impossible to
    //   number here by bundle, but we can fix it by traversing the elements like the vertices
    PetscErrorCode Viewer::writeSplitLocal(const Obj<Mesh>& mesh, const Obj<Builder::pair_section_type>& splitField, PetscViewer viewer) {
      const Obj<Mesh::topology_type>&        topology   = mesh->getTopology();
      Builder::pair_section_type::patch_type patch      = 0;
      const Obj<Mesh::numbering_type>&       eNumbering = mesh->getFactory()->getLocalNumbering(topology, patch, topology->depth());
      const Obj<Mesh::numbering_type>&       vNumbering = mesh->getFactory()->getLocalNumbering(topology, patch, 0);
      PetscErrorCode ierr;

      PetscFunctionBegin;
      const Builder::pair_section_type::atlas_type::chart_type& chart = splitField->getPatch(patch);

      for(Builder::pair_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const Builder::pair_section_type::point_type& e      = *c_iter;
        const Builder::pair_section_type::value_type *values = splitField->restrict(patch, e);
        const int                                     size   = splitField->getFiberDimension(patch, e);

        for(int i = 0; i < size; i++) {
          const Builder::pair_section_type::point_type& v     = values[i].first;
          const PETSC_MESH_TYPE::base_type::split_value&      split = values[i].second;

          // No time history
          ierr = PetscViewerASCIIPrintf(viewer, "%6d %6d 0 %15.9g %15.9g %15.9g\n", eNumbering->getIndex(e)+1, vNumbering->getIndex(v)+1, split.x, split.y, split.z);CHKERRQ(ierr);
        }
      }
      PetscFunctionReturn(0);
    };
#endif
    #undef __FUNCT__  
    #define __FUNCT__ "PyLithWriteTractionsLocal"
    PetscErrorCode Viewer::writeTractionsLocal(const Obj<Mesh>& mesh, const Obj<Mesh>& tractionMesh, const Obj<Builder::real_section_type>& tractionField, PetscViewer viewer) {
      const Obj<Mesh::sieve_type>&     sieve = tractionMesh->getSieve();
      const Obj<Mesh::label_sequence>& faces = tractionMesh->heightStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"traction_units = Pa\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
      for(Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
        const Mesh::point_type& face = *f_iter;
        const Obj<Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(face);

        for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          const Mesh::point_type& vertex = *c_iter;

          ierr = PetscViewerASCIIPrintf(viewer, "%6d", vNumbering->getIndex(vertex)+1);CHKERRQ(ierr);
          std::cout << vNumbering->getIndex(vertex) << " ("<<vertex<<") ";
        }
        const Mesh::real_section_type::value_type *values = tractionField->restrictPoint(face);

        for(int i = 0; i < mesh->getDimension(); ++i) {
          if (i > 0) {
            ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            std::cout << " ";
          }
          ierr = PetscViewerASCIIPrintf(viewer, "%15.9g", values[i]);CHKERRQ(ierr);
          std::cout << values[i];
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        std::cout << std::endl;
      }
      PetscFunctionReturn(0);
    };
#endif
  };
};
