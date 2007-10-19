#include <petscmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#if defined(PETSC_HAVE_HDF5)
#include <hdf5.h>
#endif

namespace ALE {
  namespace PFLOTRAN {
    //
    // Builder methods
    //
    void Builder::readConnectivity(MPI_Comm comm, const std::string& filename, 
                                   int& corners, const bool useZeroBase, 
                                   int& numElements, int *vertices[]) {
      PetscViewer    viewer;
      PetscInt       numCells;
      PetscInt      *verts;
      PetscInt       commRank;
      PetscErrorCode ierr;
#if defined(PETSC_HAVE_HDF5)
      herr_t         status;
      hid_t          file_id;
      hid_t          group_id;
      hid_t          dataset_id;
      hid_t          prop_id;
      hid_t          type_id;
      hid_t          attribute_id;
      hid_t          string_id;
      H5T_class_t    class_type;
      char           element_type[33];
#endif

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;

#if defined(PETSC_HAVE_HDF5)
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_HDF5);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      if (ierr) {
        ostringstream txt;
        txt << "Could not open PFLOTRAN connectivity file: " << filename;
        throw ALE::Exception(txt.str().c_str());
      }
      ierr = PetscViewerHDF5GetFileId(viewer, &file_id);

      group_id = H5Gopen(file_id, "Connectivity");

// read in number of cells
      attribute_id = H5Aopen_name(group_id, "Number of Elements");
      type_id = H5Aget_type(attribute_id);
      class_type = H5Tget_class(type_id);
      if (class_type != H5T_INTEGER) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "ERROR: 'Number of Elements' attribute should be an int\n");
      }
      status = H5Tclose(type_id);
      status = H5Aread(attribute_id, H5T_NATIVE_INT, &numCells);
      status = H5Aclose(attribute_id);

// read in cell type

      attribute_id = H5Aopen_name(group_id, "Element Type");
      type_id = H5Aget_type(attribute_id);
      class_type = H5Tget_class(type_id);
      if (class_type != H5T_STRING) {
        // print error message
      }
      size_t strsize = H5Tget_size(type_id);
      status = H5Tclose(type_id);
      if (strsize != 32) {  // right now, 32 is arbitrary
        PetscPrintf(PETSC_COMM_WORLD,
                    "ERROR: Size of attribute string should be 32\n");
      }
      string_id = H5Tcopy(H5T_C_S1);
      status = H5Tset_strpad(string_id, H5T_STR_NULLTERM);
      status = H5Tset_size(string_id, strsize+1);
      status = H5Aread(attribute_id, string_id, element_type);
      status = H5Aclose(attribute_id);


      if (!strncmp(element_type, "tri", 3)) {
        corners = 3;
      }
      else if (!strncmp(element_type, "quad", 4) || 
              !strncmp(element_type, "tet", 3)) {
        corners = 4;
      }
      else if (!strncmp(element_type, "hex", 3)) {
        corners = 8;
      }
      ierr = PetscMalloc(numCells*corners * sizeof(PetscInt), &verts);

      dataset_id = H5Dopen(group_id, "Connectivity");
      prop_id = H5Pcreate(H5P_DATASET_XFER);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
      H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_INDEPENDENT);
#endif
      H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, prop_id, verts);
      H5Pclose(prop_id); 
      H5Dclose(dataset_id); 
      H5Gclose(group_id); 

      if (!useZeroBase) {
        for (int i=0; i<numCells*corners; i++)
          verts[i] -= 1;
      }
      ierr = PetscViewerDestroy(viewer);
      numElements = numCells;
      *vertices = verts;
      PetscPrintf(PETSC_COMM_WORLD,"%d %s elements read.\n",numCells,
                  element_type);
#else
      SETERRABORT(comm,PETSC_ERR_SUP,"PETSc has not been compiled with hdf5 enabled.");
#endif
    };
    void Builder::readCoordinates(MPI_Comm comm, const std::string& filename, 
                                  const int dim, int& numVertices, 
                                  double *coordinates[]) {
      PetscViewer    viewer;
      PetscInt       numVerts;
      PetscScalar   *coords, *coord;
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;
#if defined(PETSC_HAVE_HDF5)
      herr_t         status;
      hid_t          file_id;
      hid_t          group_id;
      hid_t          dataset_id;
      hid_t          prop_id;
      hid_t          type_id;
      hid_t          attribute_id;
      H5T_class_t    class_type;
#endif

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;

#if defined(PETSC_HAVE_HDF5)
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_HDF5);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      if (ierr) {
        ostringstream txt;
        txt << "Could not open PFLOTRAN connectivity file: " << filename;
        throw ALE::Exception(txt.str().c_str());
      }
      ierr = PetscViewerHDF5GetFileId(viewer, &file_id);

      group_id = H5Gopen(file_id, "Coordinates");

// read in number of vertices
      attribute_id = H5Aopen_name(group_id, "Number of Vertices");
      type_id = H5Aget_type(attribute_id);
      class_type = H5Tget_class(type_id);
      if (type_id != H5T_INTEGER) {
        // print error message
      }
      status = H5Tclose(type_id);
      status = H5Aread(attribute_id, H5T_NATIVE_INT, &numVerts);
      status = H5Aclose(attribute_id);

      ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);
      ierr = PetscMalloc(numVerts * sizeof(PetscScalar), &coord);

      dataset_id = H5Dopen(group_id, "X-Coordinates");
      prop_id = H5Pcreate(H5P_DATASET_XFER);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
      H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_INDEPENDENT);
#endif
      H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, prop_id, coord);
      H5Pclose(prop_id); 
      H5Dclose(dataset_id); 
      c = 0;
      for (int i=0; i<numVerts; i++) {
        coords[c] = coord[i];
        c += dim;
      }

      if (dim > 1) {
        dataset_id = H5Dopen(group_id, "Y-Coordinates");
        prop_id = H5Pcreate(H5P_DATASET_XFER);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
        H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_INDEPENDENT);
#endif
        H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, prop_id, 
                coord);
        H5Pclose(prop_id); 
        H5Dclose(dataset_id); 
        c = 1;
        for (int i=0; i<numVerts; i++) {
          coords[c] = coord[i];
          c += dim;
        }
      }

      if (dim > 2) {
        dataset_id = H5Dopen(group_id, "Z-Coordinates");
        prop_id = H5Pcreate(H5P_DATASET_XFER);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
        H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_INDEPENDENT);
#endif
        H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, prop_id,
                coord);
        H5Pclose(prop_id); 
        H5Dclose(dataset_id); 
        c = 2;
        for (int i=0; i<numVerts; i++) {
          coords[c] = coord[i];
          c += dim;
        }
      }
      H5Gclose(group_id); 
      ierr  = PetscFree(coord);

      ierr = PetscViewerDestroy(viewer);
      numVertices = numVerts;
      *coordinates = coords;
      PetscPrintf(PETSC_COMM_WORLD,"%d vertices read.\n",numVerts);
#else
      SETERRABORT(comm,PETSC_ERR_SUP,"PETSc has not been compiled with hdf5 enabled.");
#endif
    };
    Obj<ALE::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, 
                                     const std::string& basename, 
                                     const bool useZeroBase = true, 
                                     const bool interpolate = true, 
                                     const int debug = 0) {
      return readMesh(comm, dim, basename+".h5", basename+".h5", 
                      useZeroBase, interpolate, debug);
    };
    Obj<ALE::Mesh> Builder::readMesh(MPI_Comm comm, const int dim, 
                                     const std::string& coordFilename, 
                                     const std::string& adjFilename, 
                                     const bool useZeroBase = true, 
                                     const bool interpolate = true, 
                                     const int debug = 0) {
      Obj<Mesh>          mesh     = new Mesh(comm, dim, debug);
      Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
      int    *cells = NULL;
      double *coordinates = NULL;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode ierr;

      ALE::PFLOTRAN::Builder::readConnectivity(comm, adjFilename, numCorners, 
                                               useZeroBase, numCells, &cells);
      ALE::PFLOTRAN::Builder::readCoordinates(comm, coordFilename, dim, 
                                              numVertices, &coordinates);
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(sieve, dim, numCells, cells, 
                                                  numVertices, interpolate, 
                                                  numCorners, -1, 
                                          mesh->getArrowSection("orientation"));
      mesh->setSieve(sieve);
      mesh->stratify();
      ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(mesh, dim, coordinates);
      if (cells) {ierr = PetscFree(cells);}
      if (coordinates) {ierr = PetscFree(coordinates);}
      return mesh;
    };
    // Creates boundary sections:
    //   IBC[NBFS,2]:     ALL
    //     BL[NBFS,1]:
    //     BNVEC[NBFS,2]:
    //   BCFUNC[NBCF,NV]: ALL
    //   IBNDFS[NBN,2]:   STILL NEED 4-5
    //     BNNV[NBN,2]
    void Builder::readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename) {
      PetscViewer    viewer;
      FILE          *f;
      char           buf[2048];
      PetscErrorCode ierr;

      const Obj<Mesh::int_section_type>&  ibc    = mesh->getIntSection("IBC");
      const Obj<Mesh::int_section_type>&  ibndfs = mesh->getIntSection("IBNDFS");
      const Obj<Mesh::int_section_type>&  ibcnum = mesh->getIntSection("IBCNUM");
      const Obj<Mesh::int_section_type>&  ibfcon = mesh->getIntSection("IBFCON");
      const Obj<Mesh::real_section_type>& bl     = mesh->getRealSection("BL");
      const Obj<Mesh::real_section_type>& bnvec  = mesh->getRealSection("BNVEC");
      const Obj<Mesh::real_section_type>& bnnv   = mesh->getRealSection("BNNV");
      if (mesh->commRank() != 0) {
#if 0
        mesh->distributeBCValues();
#endif
        return;
      }
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, bcFilename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      // Create IBC section
      int  numBdFaces = atoi(strtok(fgets(buf, 2048, f), " "));
      int *tmpIBC     = new int[numBdFaces*4];
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

        ibc->addFiberDimension(elem, 4);
        ibcnum->addFiberDimension(elem, 1);
        ibfcon->addFiberDimension(elem, 2);
        bl->addFiberDimension(elem, 1);
        bnvec->addFiberDimension(elem, 2);
        elem2Idx[elem].insert(bf);
      }
      mesh->allocate(ibc);
      mesh->allocate(ibcnum);
      mesh->allocate(ibfcon);
      mesh->allocate(bl);
      mesh->allocate(bnvec);
      const Mesh::int_section_type::chart_type& chart = ibc->getChart();
      int num = 1;

      for(Mesh::int_section_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const int elem = *p_iter;
        int bfNum[2];
        int k = 0;

        for(std::set<int>::const_iterator i_iter = elem2Idx[elem].begin(); i_iter != elem2Idx[elem].end(); ++i_iter) {
          bfReorder[(*i_iter)+1] = num;
          bfNum[k++] = num;
          num++;
        }
        ibcnum->updatePoint(elem, bfNum);
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
            ibc->updatePoint(elem, values);
          }
        } else {
          ibc->updatePoint(elem, &tmpIBC[bf*4]);
        }
      }
      delete [] tmpIBC;
      // Create BCFUNC section
      int numBcFunc = atoi(strtok(fgets(buf, 2048, f), " "));
      if (numBcFunc != 0) {throw ALE::Exception("Cannot handle BCFUNCS after rewrite");}
      for(int bc = 0; bc < numBcFunc; bc++) {
#if 0
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
#endif
      }
#if 0
      mesh->distributeBCValues();
#endif
      // Create IBNDFS section
      int       numBdVertices = atoi(strtok(fgets(buf, 2048, f), " "));
      const int numElements   = mesh->heightStratum(0)->size();
      int      *tmpIBNDFS     = new int[numBdVertices*3];

      for(int bv = 0; bv < numBdVertices; bv++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");

        // Ignore boundary node number
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+0] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+1] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+2] = atoi(x);
        ibndfs->setFiberDimension(tmpIBNDFS[bv*3+0]-1+numElements, 6);
      }
      mesh->allocate(ibndfs);
      for(int bv = 0; bv < numBdVertices; bv++) {
        int values[5];

        values[0] = tmpIBNDFS[bv*3+0];
        // Covert to new boundary face numbers
        values[1] = bfReorder[tmpIBNDFS[bv*3+1]];
        values[2] = bfReorder[tmpIBNDFS[bv*3+2]];
        values[3] = 0;
        values[4] = 0;
        ibndfs->updatePoint(values[0]-1+numElements, values);
      }
      ierr = PetscViewerDestroy(viewer);
      // Create BNNV[NBN,2]
      const int dim = mesh->getDimension();

      for(int bv = 0; bv < numBdVertices; bv++) {
        bnnv->setFiberDimension(tmpIBNDFS[bv*3+0]-1+numElements, dim);
      }
      mesh->allocate(bnnv);
      delete [] tmpIBNDFS;
    };
    void Builder::outputVerticesLocal(const Obj<Mesh>& mesh, int *numVertices, int *dim, double *coordinates[], const bool columnMajor) {
      const Obj<Mesh::real_section_type>& coordSec = mesh->getRealSection("coordinates");
      if (!coordSec->size()) {
        *numVertices = 0;
        *dim         = 0;
        *coordinates = NULL;
        return;
      }
      const Obj<Mesh::label_sequence>& vertices   = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size     = vertices->size();
      int            embedDim = coordSec->getFiberDimension(*vertices->begin());
      double        *coords;
      PetscErrorCode ierr;

      ierr = PetscMalloc(vertices->size()*embedDim * sizeof(double), &coords);
      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::real_section_type::value_type *array = coordSec->restrictPoint(*v_iter);
        const int                                  row   = vNumbering->getIndex(*v_iter);

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
      if (!mesh->heightStratum(0)->size()) {
        *numElements = 0;
        *numCorners  = 0;
        *vertices    = NULL;
        return;
      }
      const Obj<Mesh::sieve_type>&     sieve      = mesh->getSieve();
      const Obj<Mesh::label_sequence>& elements   = mesh->heightStratum(0);
      const Obj<Mesh::numbering_type>& eNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size         = elements->size();
      //int            corners      = sieve->nCone(*elements->begin(), topology->depth())->size();
      int            corners      = sieve->cone(*elements->begin())->size();
      int           *v;
      PetscErrorCode ierr;

      ierr = PetscMalloc(elements->size()*corners * sizeof(int), &v);
      for(Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
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
    #define __FUNCT__ "PFLOTRANWriteVertices"
    PetscErrorCode Viewer::writeVertices(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
      ALE::Obj<Mesh::real_section_type> coordinates = mesh->getRealSection("coordinates");
#if 0
      Mesh::field_type::patch_type patch;
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
        ALE::Obj<Mesh::bundle_type>                           globalOrder = coordinates->getGlobalOrder();
        ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone        = globalOrder->getPatch(patch);
        const int *offsets = coordinates->getGlobalOffsets();
        int        embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/embedDim;
        double    *localCoords;
        int        k = 0;

        ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &localCoords);CHKERRQ(ierr);
        for(Mesh::bundle_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
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
    #define __FUNCT__ "PFLOTRANWriteElements"
    PetscErrorCode Viewer::writeElements(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
#if 0
      ALE::Obj<Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      ALE::Obj<Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
      ALE::Obj<Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
      ALE::Obj<Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
      ALE::Obj<Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
      Mesh::bundle_type::patch_type patch;
      std::string    orderName("element");
      int            dim  = mesh->getDimension();
      int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
      int            numElements;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (corners != dim+1) {
        SETERRQ(PETSC_ERR_SUP, "PFLOTRAN only supports simplicies");
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
        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
          for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
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
        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
            for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
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
    #define __FUNCT__ "PFLOTRANWriteVerticesLocal"
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      Obj<Mesh::real_section_type>     coordinates = mesh->getRealSection("coordinates");
      const Obj<Mesh::label_sequence>& vertices    = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            embedDim = coordinates->getFiberDimension(*vertices->begin());
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", vertices->size());CHKERRQ(ierr);
      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);

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
    #define __FUNCT__ "PFLOTRANWriteRestart"
    PetscErrorCode Viewer::writeRestart(const Obj<Mesh>& mesh, PetscViewer viewer) {
      const Obj<Mesh::real_section_type>&   velocity    = mesh->getRealSection("VELN");
      const Obj<Mesh::real_section_type>&   pressure    = mesh->getRealSection("PN");
      const Obj<Mesh::real_section_type>&   temperature = mesh->getRealSection("TN");
      const Obj<Mesh::numbering_type>& cNumbering  = mesh->getFactory()->getNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering  = mesh->getFactory()->getNumbering(mesh, 0);
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
        const Obj<Mesh::label_sequence>& vertices = mesh->depthStratum(0);

        for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const Mesh::real_section_type::value_type *veln = velocity->restrictPoint(*v_iter);
            const Mesh::real_section_type::value_type *pn   = pressure->restrictPoint(*v_iter);
            const Mesh::real_section_type::value_type *tn   = temperature->restrictPoint(*v_iter);

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
        const Obj<Mesh::label_sequence>& vertices = mesh->depthStratum(0);
        RestartType *localValues;
        int numLocalElements = vNumbering->getLocalSize();
        int k = 0;

        ierr = PetscMalloc(numLocalElements * sizeof(RestartType), &localValues);CHKERRQ(ierr);
        for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const Mesh::real_section_type::value_type *veln = velocity->restrictPoint(*v_iter);
            const Mesh::real_section_type::value_type *pn   = pressure->restrictPoint(*v_iter);
            const Mesh::real_section_type::value_type *tn   = temperature->restrictPoint(*v_iter);

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

    //   This class reconstructs the local pieces of the boundary that distributed PFLOTRAN needs.
    // The boundary along with the boundary conditions is encoded in a collection of sections
    // over the PFLOTRAN mesh.  These sections contain a description of the boundary topology 
    // using elements' global names.  This is unacceptable for PFLOTRAN, since it interprets 
    // elements of the connectivity data arrays as local offsets into (some other of) these arrays.
    //   This subroutine performs the renumbering based on the local numbering on the distributed
    // mesh.  Essentially, we replace each global element id by its local number.
    //
    // Note: Any vertex or element number from PFLOTRAN is 1-based, but in Sieve we are 0-based. Thus
    //       we add and subtract 1 during conversion. Also, Sieve vertices are numbered after cells.
    void fuseBoundary(const ALE::Obj<ALE::Mesh>& mesh) {
      // Extract PFLOTRAN boundary sections
      ALE::Obj<ALE::Mesh::int_section_type> IBCsec    = mesh->getIntSection("IBC");
      ALE::Obj<ALE::Mesh::int_section_type> IBNDFSsec = mesh->getIntSection("IBNDFS");
      ALE::Obj<ALE::Mesh::int_section_type> IBCNUMsec = mesh->getIntSection("IBCNUM");

      // Look at the sections, if debugging
      if (mesh->debug()) {
        IBCsec->view("IBCsec", mesh->comm());
        IBNDFSsec->view("IBNDFSsec", mesh->comm());
      }

      // Extract the local numberings
      ALE::Obj<ALE::Mesh::numbering_type> vertexNum  = mesh->getFactory()->getLocalNumbering(mesh, 0);
      ALE::Obj<ALE::Mesh::numbering_type> elementNum = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const int numElements = mesh->getFactory()->getNumbering(mesh, mesh->depth())->getGlobalSize();
      std::map<int,int> bfMap;
      // Declare points to the extracted numbering data
      const ALE::Mesh::numbering_type::value_type *vNum, *eNum;

      // Create map from serial bdFace numbers to local bdFace numbers
      {
        const ALE::Mesh::int_section_type::chart_type     chart = IBCNUMsec->getChart();
        ALE::Mesh::int_section_type::chart_type::iterator begin = chart.begin();
        ALE::Mesh::int_section_type::chart_type::iterator end   = chart.end();
        int num = 0;

        for(ALE::Mesh::int_section_type::chart_type::iterator p_iter = begin; p_iter != end; ++p_iter) {
          const int  fiberDim  = IBCNUMsec->getFiberDimension(*p_iter);
          const int *globalNum = IBCNUMsec->restrictPoint(*p_iter);

          for(int n = 0; n < fiberDim; ++n) {
            bfMap[globalNum[n]] = ++num;
          }
        }
      }
      // Renumber vertex section IBC
      {
        const ALE::Mesh::int_section_type::chart_type     IBCchart = IBCsec->getChart();
        ALE::Mesh::int_section_type::chart_type::iterator begin    = IBCchart.begin();
        ALE::Mesh::int_section_type::chart_type::iterator end      = IBCchart.end();
        for(ALE::Mesh::int_section_type::chart_type::iterator p_iter = begin; p_iter != end; ++p_iter) {
          ALE::Mesh::point_type p = *p_iter;
          const ALE::Mesh::int_section_type::value_type *ibc_in = IBCsec->restrictPoint(p);
          int fiberDimension = IBCsec->getFiberDimension(p);
          ALE::Mesh::int_section_type::value_type ibc_out[8];
          // k controls the update of edge connectivity for each containing element;
          // if fiberDimension is 4, only one boundary face is connected to the element, and that edge's data
          // are contained in entries 0 - 3 of the section over the element p;
          // if fiberDimension is 8, two boundary faces are connected to the element, so the second edge's data
          // are contained in entries 4 - 7
          for(int k = 0; k < fiberDimension/4; k++) {
            // Extract IBC entry 1 (entry kk*4) for edge kk connected to element p.
            // This is the entry that needs renumbering for renumbering (2,3 & 4 are invariant under distribution), 
            // see IBC's description.
            // Here we assume that elementNum's domain contains all boundary elements found in IBC, 
            // so we can restrict to the extracted entry.
            eNum = elementNum->restrictPoint((ALE::Mesh::numbering_type::point_type) ibc_in[k*4]-1);
            ibc_out[k*4+0] = eNum[0]+1;
            // Copy the other entries right over
            ibc_out[k*4+1] = ibc_in[k*4+1];
            ibc_out[k*4+2] = ibc_in[k*4+2];
            ibc_out[k*4+3] = ibc_in[k*4+3];
          }
          // Update IBC
          IBCsec->updatePoint(p, ibc_out);
        }
      }
      {
        // Renumber vertex section IBNDFS
        const ALE::Mesh::int_section_type::chart_type     IBNDFSchart = IBNDFSsec->getChart();
        ALE::Mesh::int_section_type::chart_type::iterator begin       = IBNDFSchart.begin();
        ALE::Mesh::int_section_type::chart_type::iterator end         = IBNDFSchart.end();
        for(ALE::Mesh::int_section_type::chart_type::iterator p_iter = begin; p_iter != end; ++p_iter) {
          ALE::Mesh::point_type p = *p_iter;
          const ALE::Mesh::int_section_type::value_type *ibndfs_in = IBNDFSsec->restrictPoint(p);
          // Here we assume the fiber dimension is 5
          ALE::Mesh::int_section_type::value_type ibndfs_out[5];
          // Renumber entries 1,2 & 3 (4 & 5 are invariant under distribution), see IBNDFS's description
          // Here we assume that vertexNum's domain contains all boundary verticies found in IBNDFS, so we can restrict to the first extracted entry
          vNum= vertexNum->restrictPoint((ALE::Mesh::numbering_type::point_type)ibndfs_in[0]-1+numElements);
          ibndfs_out[0] = vNum[0]+1;
          // Map serial bdFace numbers to local bdFace numbers
          ibndfs_out[1] = bfMap[ibndfs_in[1]];
          ibndfs_out[2] = bfMap[ibndfs_in[2]];
          // Copy the other entries right over
          ibndfs_out[3] = ibndfs_in[3];
          ibndfs_out[4] = ibndfs_in[4];
          // Update IBNDFS
          IBNDFSsec->updatePoint(p,ibndfs_out);
        }
      }
      if (mesh->debug()) {
        IBCsec->view("Renumbered IBCsec", mesh->comm());
        IBNDFSsec->view("Renumbered IBNDFSsec", mesh->comm());
      }
      // It's not clear whether IBFCON needs to be renumbered (what does it mean that its entries are not "GLOBAL NODE NUMBER(s)" -- see IBFCON's descriptions
    };
  };
};
