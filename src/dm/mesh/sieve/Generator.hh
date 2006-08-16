#ifndef included_ALE_Generator_hh
#define included_ALE_Generator_hh

#ifndef  included_ALE_Distribution_hh
#include <Distribution.hh>
#endif
#include <src/dm/mesh/meshpylith.h>

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>
#endif
#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#endif

namespace ALE {
#ifdef PETSC_HAVE_TRIANGLE
  namespace Triangle {
    class Generator {
    public:
      static void initInput(struct triangulateio *inputCtx) {
        inputCtx->numberofpoints = 0;
        inputCtx->numberofpointattributes = 0;
        inputCtx->pointlist = NULL;
        inputCtx->pointattributelist = NULL;
        inputCtx->pointmarkerlist = NULL;
        inputCtx->numberofsegments = 0;
        inputCtx->segmentlist = NULL;
        inputCtx->segmentmarkerlist = NULL;
        inputCtx->numberoftriangleattributes = 0;
        inputCtx->numberofholes = 0;
        inputCtx->holelist = NULL;
        inputCtx->numberofregions = 0;
        inputCtx->regionlist = NULL;
      };
      static void initOutput(struct triangulateio *outputCtx) {
        outputCtx->pointlist = NULL;
        outputCtx->pointattributelist = NULL;
        outputCtx->pointmarkerlist = NULL;
        outputCtx->trianglelist = NULL;
        outputCtx->triangleattributelist = NULL;
        outputCtx->neighborlist = NULL;
        outputCtx->segmentlist = NULL;
        outputCtx->segmentmarkerlist = NULL;
        outputCtx->edgelist = NULL;
        outputCtx->edgemarkerlist = NULL;
      };
      static void finiOutput(struct triangulateio *outputCtx) {
        free(outputCtx->pointmarkerlist);
        free(outputCtx->edgelist);
        free(outputCtx->edgemarkerlist);
        free(outputCtx->trianglelist);
        free(outputCtx->neighborlist);
      };
      static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false) {
        int                  dim  = 2;
        Obj<Mesh>            mesh = Mesh(boundary->comm(), dim, boundary->debug);
        const Mesh::topology_type::patch_type patch    = 0;
        const Obj<Mesh::topology_type>&       topology = boundary->getTopologyNew();
        const Obj<Mesh::sieve_type>&          sieve    = topology->getPatch(patch);
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode       ierr;

        initInput(&in);
        initOutput(&out);
        if (mesh->commRank() == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          const Obj<Mesh::topology_type::label_sequence>&   vertices = topology->depthStratum(patch, 0);
          const Obj<Mesh::topology_type::patch_label_type>& markers  = topology->getLabel(patch, "marker");

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            const Obj<Mesh::section_type>&   coordinates = boundary->getSection("coordinates");
            const Obj<Mesh::numbering_type>& vNumbering  = boundary->getLocalNumbering(0);

            ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
            ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
            for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
              const Mesh::section_type::value_type *array = coordinates->restrict(patch, *v_iter);
              const int                             idx   = vNumbering->getIndex(*v_iter);

              for(int d = 0; d < dim; d++) {
                in.pointlist[idx*dim + d] = array[d];
              }
              in.pointmarkerlist[idx] = topology->getValue(markers, *v_iter);
            }
          }
          const Obj<Mesh::topology_type::label_sequence>&   edges = topology->depthStratum(patch, 1);

          in.numberofsegments = edges->size();
          if (in.numberofsegments > 0) {
            const Obj<Mesh::numbering_type>& vNumbering = boundary->getLocalNumbering(0);
            const Obj<Mesh::numbering_type>& eNumbering = boundary->getLocalNumbering(1);

            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            for(Mesh::topology_type::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
              const Obj<Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);
              const int                                          idx  = eNumbering->getIndex(*e_iter);
              int                                                v    = 0;
        
              for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                in.segmentlist[idx*dim + (v++)] = vNumbering->getIndex(*c_iter);
              }
              in.segmentmarkerlist[idx] = topology->getValue(markers, *e_iter);
            }
          }

          in.numberofholes = 0;
          if (in.numberofholes > 0) {
            ierr = PetscMalloc(in.numberofholes*dim * sizeof(int), &in.holelist);
          }
          if (createConvexHull) {
            args += "c";
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);

          ierr = PetscFree(in.pointlist);
          ierr = PetscFree(in.pointmarkerlist);
          ierr = PetscFree(in.segmentlist);
          ierr = PetscFree(in.segmentmarkerlist);
        }
        const Obj<Mesh::topology_type>& newTopology = new Mesh::topology_type(mesh->comm(), mesh->debug);
        const Obj<Mesh::sieve_type>     newSieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug);
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coordinates = out.pointlist;

        ALE::New::SieveBuilder<Mesh::sieve_type>::buildTopology(newSieve, dim, numCells, cells, numVertices, interpolate, numCorners);
        newSieve->stratify();
        newTopology->setPatch(patch, newSieve);
        newTopology->stratify();
        mesh->setTopologyNew(newTopology);
        ALE::PyLith::Builder::buildCoordinates(mesh->getSection("coordinates"), dim, coordinates);
        if (mesh->commRank() == 0) {
          const Obj<Mesh::topology_type::patch_label_type>& newMarkers = newTopology->createLabel(patch, "marker");

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<Mesh::sieve_type::supportSet> edge = sieve->nJoin(vertexA, vertexB, 1);

                topology->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
              }
            }
          }
        }
        finiOutput(&out);
        return mesh;
      };
    };
    class Refiner {
    public:
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
        int                  dim     = 2;
        Obj<Mesh>            refMesh = Mesh(mesh->comm(), dim, mesh->debug);
        return mesh;
      };
    };
  };
#endif
#ifdef PETSC_HAVE_TETGEN
  namespace TetGen {
    class Generator {
    public:
      static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false) {
        int                  dim  = 3;
        Obj<Mesh>            mesh = Mesh(boundary->comm(), dim, boundary->debug);
        return mesh;
      };
    };
    class Refiner {
    public:
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
        int                  dim     = 3;
        Obj<Mesh>            refMesh = Mesh(mesh->comm(), dim, mesh->debug);
        return mesh;
      };
    };
  }
#endif
  class Generator {
  public:
    static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false) {
      int dim = boundary->getDimension();

      if (dim == 1) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Generator::generateMesh(boundary, interpolate);
#else
        throw ALE::Exception("Mesh generation currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 2) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Generator::generateMesh(boundary, interpolate);
#else
        throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner::refineMesh(mesh, maxVolume, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner::refineMesh(mesh, maxVolume, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
  };
}

#endif

#if 0
    class Generator {
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> generate_TetGen(Obj<Mesh> boundary, bool interpolate) {
        ::tetgenio             in;
        ::tetgenio             out;
        int                    dim = 3;
        Obj<Mesh>              m = Mesh(boundary->comm(), dim, boundary->debug);
        Obj<Mesh::sieve_type>  bdTopology = boundary->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = boundary->getBundle(0);
        Obj<Mesh::bundle_type> facetBundle = boundary->getBundle(bdTopology->depth());
        PetscMPIInt            rank;
        PetscErrorCode         ierr;

        ierr = MPI_Comm_rank(boundary->comm(), &rank);

        if (rank == 0) {
          std::string args("pqenzQ");
          bool        createConvexHull = false;
          Obj<Mesh::sieve_type::traits::depthSequence> vertices = bdTopology->depthStratum(0);
          Mesh::field_type::patch_type         patch;

          in.numberofpoints = vertices->size();
          if (in.numberofpoints > 0) {
            Obj<Mesh::field_type> coordinates = boundary->getCoordinates();

            in.pointlist       = new double[in.numberofpoints*dim];
            in.pointmarkerlist = new int[in.numberofpoints];
            for(Mesh::sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
              const Mesh::field_type::index_type& interval = coordinates->getIndex(patch, *v_itor);
              const Mesh::field_type::value_type *array = coordinates->restrict(patch, *v_itor);

              for(int d = 0; d < interval.index; d++) {
                in.pointlist[interval.prefix + d] = array[d];
              }
              const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *v_itor);
              in.pointmarkerlist[vInterval.prefix] = v_itor.marker();
            }
          }

          Obj<Mesh::sieve_type::traits::heightSequence> facets = bdTopology->heightStratum(0);

          in.numberoffacets = facets->size();
          if (in.numberoffacets > 0) {
            in.facetlist       = new tetgenio::facet[in.numberoffacets];
            in.facetmarkerlist = new int[in.numberoffacets];
            for(Mesh::sieve_type::traits::heightSequence::iterator f_itor = facets->begin(); f_itor != facets->end(); ++f_itor) {
              const Mesh::field_type::index_type& interval = facetBundle->getIndex(patch, *f_itor);
              //Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch("element", *f_itor);
              Obj<Mesh::bundle_type::order_type::coneSequence> cone;

              in.facetlist[interval.prefix].numberofpolygons = 1;
              in.facetlist[interval.prefix].polygonlist = new tetgenio::polygon[in.facetlist[interval.prefix].numberofpolygons];
              in.facetlist[interval.prefix].numberofholes = 0;
              in.facetlist[interval.prefix].holelist = NULL;

              tetgenio::polygon *poly = in.facetlist[interval.prefix].polygonlist;
              int                c = 0;

              poly->numberofvertices = cone->size();
              poly->vertexlist = new int[poly->numberofvertices];
              // The "element" reorder should be fused with the structural order
              for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
                const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *c_itor);

                poly->vertexlist[c++] = vInterval.prefix;
              }
              in.facetmarkerlist[interval.prefix] = f_itor.marker();
            }
          }

          in.numberofholes = 0;
          if (createConvexHull) args += "c";
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist, interpolate);
  
        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(v + out.numberoftetrahedra), out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            if (out.edgemarkerlist) {
              for(int e = 0; e < out.numberofedges; e++) {
                if (out.edgemarkerlist[e]) {
                  Mesh::point_type endpointA(out.edgelist[e*2+0] + out.numberoftetrahedra);
                  Mesh::point_type endpointB(out.edgelist[e*2+1] + out.numberoftetrahedra);
                  Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(endpointA, endpointB, 1);

                  topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
                }
              }
            }
            if (out.trifacemarkerlist) {
              for(int f = 0; f < out.numberoftrifaces; f++) {
                if (out.trifacemarkerlist[f]) {
                  Obj<Mesh::sieve_type::supportSet> point = Mesh::sieve_type::supportSet();
                  Obj<Mesh::sieve_type::supportSet> edge = Mesh::sieve_type::supportSet();
                  Mesh::point_type cornerA(out.trifacelist[f*3+0] + out.numberoftetrahedra);
                  Mesh::point_type cornerB(out.trifacelist[f*3+1] + out.numberoftetrahedra);
                  Mesh::point_type cornerC(out.trifacelist[f*3+2] + out.numberoftetrahedra);
                  point->insert(cornerA);
                  edge->insert(cornerB);
                  edge->insert(cornerC);
                  Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(point, edge, 2);

                  topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
                }
              }
            }
          }
        }
        return m;
      };
#endif
    private:
#ifdef PETSC_HAVE_TRIANGLE
      static Obj<Mesh> refine_Triangle(Obj<Mesh> serialMesh, const double maxAreas[], bool interpolate) {
        struct triangulateio in;
        struct triangulateio out;
        int                  dim = 2;
        Obj<Mesh>            m = Mesh(serialMesh->comm(), dim, serialMesh->debug);
        PetscInt             numElements = serialMesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt          rank;
        PetscErrorCode       ierr;

        ierr = MPI_Comm_rank(serialMesh->comm(), &rank);
        initInput_Triangle(&in);
        initOutput_Triangle(&out);
        if (rank == 0) {
          ierr = PetscMalloc(numElements * sizeof(double), &in.trianglearealist);
          for(int i = 0; i < numElements; i++) {
            in.trianglearealist[i] = maxAreas[i];
          }
        }

        Obj<Mesh::sieve_type>  serialTopology = serialMesh->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = serialMesh->getBundle(0);

        if (rank == 0) {
          std::string args("pqenzQra");
          Mesh::field_type::patch_type          patch;
          std::string                           orderName("element");
          Obj<Mesh::sieve_type::traits::heightSequence> faces = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::traits::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::field_type>                 coordinates = serialMesh->getCoordinates();
          const double                         *array = coordinates->restrict(patch);
          int                                   f = 0;

          in.numberofpoints = vertices->size();
          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);
          for(int v = 0; v < (int) vertices->size(); ++v) {
            for(int d = 0; d < 2; d++) {
              in.pointlist[v*2 + d] = array[v*2 + d];
              in.pointmarkerlist[v] = 0;
            }
          }

          in.numberofcorners = 3;
          in.numberoftriangles = faces->size();
          ierr = PetscMalloc(in.numberoftriangles * in.numberofcorners * sizeof(int), &in.trianglelist);
          for(Mesh::sieve_type::traits::heightSequence::iterator f_itor = faces->begin(); f_itor != faces->end(); f_itor++) {
            //Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *f_itor);
            Obj<Mesh::bundle_type::order_type::coneSequence> cone;
            int                                              v = 0;

            for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              in.trianglelist[f * in.numberofcorners + v++] = vertexBundle->getIndex(patch, *c_itor).prefix;
            }
            f++;
          }

          Obj<Mesh::field_type> boundary = serialMesh->getBoundary();
          Obj<Mesh::sieve_type::traits::depthSequence> segments = serialTopology->depthStratum(1);
          Obj<Mesh::field_type::order_type::baseSequence> patches = boundary->getPatches();

          in.numberofsegments = 0;
          for(Mesh::field_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
            for(Mesh::sieve_type::traits::depthSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
              if (boundary->getIndex(*p_iter, *s_itor).index > 0) {
                in.numberofsegments++;
              }
            }
          }

          if (in.numberofsegments > 0) {
            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);
            int s = 0;
            for(Mesh::field_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
              for(Mesh::sieve_type::traits::depthSequence::iterator s_itor = segments->begin(); s_itor != segments->end(); s_itor++) {
                const Mesh::field_type::index_type& interval = boundary->getIndex(*p_iter, *s_itor);

                if (interval.index > 0) {
                  Obj<Mesh::sieve_type::traits::coneSequence> cone = serialTopology->cone(*s_itor);
                  int                                         p    = 0;

                  for(Mesh::sieve_type::traits::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
                    int v = vertexBundle->getIndex(patch, *c_itor).prefix;

                    in.segmentlist[s*2 + (p++)] = v;
                    in.pointmarkerlist[v] = (*p_iter).index;
                  }
                  in.segmentmarkerlist[s++] = (*p_iter).index;
                }
              }
            }
          }

          in.numberofholes = 0;
          if (in.numberofholes > 0) {
            ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);
          ierr = PetscFree(in.trianglearealist);
          ierr = PetscFree(in.pointlist);
          ierr = PetscFree(in.pointmarkerlist);
          ierr = PetscFree(in.segmentlist);
          ierr = PetscFree(in.segmentmarkerlist);
        }
        m->populate(out.numberoftriangles, out.trianglelist, out.numberofpoints, out.pointlist, interpolate);

        // Make boundary
        Obj<Mesh::sieve_type> topology = m->getTopology();
        Obj<Mesh::field_type> boundary = m->getBoundary();
        Mesh::sieve_type::markerSet markers;

        if (rank == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              markers.insert(out.pointmarkerlist[v]);
            }
          }
          for(int e = 0; e < out.numberofedges; e++) {
            if (out.edgemarkerlist[e]) {
              markers.insert(out.edgemarkerlist[e]);
            }
          }
          boundary->setTopology(topology);
          for(Mesh::sieve_type::markerSet::iterator m_iter = markers.begin(); m_iter != markers.end(); ++m_iter) {
            boundary->setPatch(topology->leaves(), Mesh::field_type::patch_type(0, *m_iter));
          }
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              boundary->setFiberDimension(Mesh::field_type::patch_type(0, out.pointmarkerlist[v]), Mesh::point_type(v+out.numberoftriangles), 1);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(vertexA, vertexB, 1);

                boundary->setFiberDimension(Mesh::field_type::patch_type(0, out.edgemarkerlist[e]), *(join->begin()), 1);
              }
            }
          }
        }
        boundary->orderPatches();

        m = m->distribute();

        finiOutput_Triangle(&out);
        return m;
      };
#endif
#ifdef PETSC_HAVE_TETGEN
      static Obj<Mesh> refine_TetGen(Obj<Mesh> mesh, const double maxAreas[], bool interpolate) {
        ::tetgenio     in;
        ::tetgenio     out;
        int            dim = 3;
        Obj<Mesh>      m = Mesh(mesh->comm(), dim, mesh->debug);
        // FIX: Need to globalize
        PetscInt       numElements = mesh->getTopology()->heightStratum(0)->size();
        PetscMPIInt    rank;
        PetscErrorCode ierr;

        ierr = MPI_Comm_rank(mesh->comm(), &rank);

        if (rank == 0) {
          in.tetrahedronvolumelist = new double[numElements];
        }
        {
          // Scatter in local area constraints
#ifdef PARALLEL
          Vec        locAreas;
          VecScatter areaScatter;

          ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, numElements, areas, &locAreas);CHKERRQ(ierr);
          ierr = MeshCreateMapping(oldMesh, elementBundle, partitionTypes, serialElementBundle, &areaScatter);CHKERRQ(ierr);
          ierr = VecScatterBegin(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
          ierr = VecScatterEnd(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
          ierr = VecDestroy(locAreas);CHKERRQ(ierr);
          ierr = VecScatterDestroy(areaScatter);CHKERRQ(ierr);
#else
          for(int i = 0; i < numElements; i++) {
            in.tetrahedronvolumelist[i] = maxAreas[i];
          }
#endif
        }

#ifdef PARALLEL
        Obj<Mesh> serialMesh = this->unify(mesh);
#else
        Obj<Mesh> serialMesh = mesh;
#endif
        Obj<Mesh::sieve_type>  serialTopology = serialMesh->getTopology();
        Obj<Mesh::bundle_type> vertexBundle = serialMesh->getBundle(0);

        if (rank == 0) {
          std::string args("qenzQra");
          Obj<Mesh::sieve_type::traits::heightSequence> cells = serialTopology->heightStratum(0);
          Obj<Mesh::sieve_type::traits::depthSequence>  vertices = serialTopology->depthStratum(0);
          Obj<Mesh::field_type>                 coordinates = serialMesh->getCoordinates();
          Mesh::field_type::patch_type          patch;
          int                                   c = 0;

          in.numberofpoints = vertices->size();
          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(Mesh::sieve_type::traits::depthSequence::iterator v_itor = vertices->begin(); v_itor != vertices->end(); ++v_itor) {
            const Mesh::field_type::index_type& interval = coordinates->getIndex(patch, *v_itor);
            const Mesh::field_type::value_type *array = coordinates->restrict(patch, *v_itor);

            for(int d = 0; d < interval.index; d++) {
              in.pointlist[interval.prefix + d] = array[d];
            }
            const Mesh::field_type::index_type& vInterval = vertexBundle->getIndex(patch, *v_itor);
            in.pointmarkerlist[vInterval.prefix] = v_itor.marker();
          }

          in.numberofcorners = 4;
          in.numberoftetrahedra = cells->size();
          in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
          for(Mesh::sieve_type::traits::heightSequence::iterator c_itor = cells->begin(); c_itor != cells->end(); ++c_itor) {
            //Obj<Mesh::field_type::IndexArray> intervals = vertexBundle->getIndices("element", *c_itor);
            Obj<Mesh::field_type::IndexArray> intervals;
            int                               v = 0;

            for(Mesh::field_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
              if (i_itor->index) {
                in.tetrahedronlist[c * in.numberofcorners + v++] = i_itor->prefix;
              }
            }
            c++;
          }

          in.numberofholes = 0;
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        m->populate(out.numberoftetrahedra, out.tetrahedronlist, out.numberofpoints, out.pointlist, interpolate);
  
        if (rank == 0) {
          Obj<Mesh::sieve_type> topology = m->getTopology();

          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              topology->setMarker(Mesh::point_type(v + out.numberoftetrahedra), out.pointmarkerlist[v]);
            }
          }
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                Mesh::point_type endpointA(out.edgelist[e*2+0] + out.numberoftetrahedra);
                Mesh::point_type endpointB(out.edgelist[e*2+1] + out.numberoftetrahedra);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(endpointA, endpointB, 1);

                topology->setMarker(*join->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                Obj<Mesh::sieve_type::supportSet> point = Mesh::sieve_type::supportSet();
                Obj<Mesh::sieve_type::supportSet> edge = Mesh::sieve_type::supportSet();
                Mesh::point_type cornerA(out.edgelist[f*3+0] + out.numberoftetrahedra);
                Mesh::point_type cornerB(out.edgelist[f*3+1] + out.numberoftetrahedra);
                Mesh::point_type cornerC(out.edgelist[f*3+2] + out.numberoftetrahedra);
                point->insert(cornerA);
                edge->insert(cornerB);
                edge->insert(cornerC);
                Obj<Mesh::sieve_type::supportSet> join = topology->nJoin(point, edge, 2);

                topology->setMarker(*join->begin(), out.trifacemarkerlist[f]);
              }
            }
          }
        }
        m = m->distribute();
        return m;
      };
#endif
      static Obj<Mesh::field_type> getSerialConstraints(Obj<Mesh> serialMesh, Obj<Mesh> parallelMesh, Obj<Mesh::field_type> parallelConstraints);
    public:
      static Obj<Mesh> refine(const Obj<Mesh>& mesh, double maxArea, bool interpolate = true) {
        Obj<Mesh::field_type> constraints = Mesh::field_type(mesh->comm(), mesh->debug);
        int             numElements = mesh->getTopology()->heightStratum(0)->size();
        Mesh::field_type::patch_type patch;

        constraints->setTopology(mesh->getTopology());
        constraints->setPatch(mesh->getTopology()->leaves(), patch);
        constraints->setFiberDimensionByHeight(patch, 0, 1);
        constraints->orderPatches();

        double *maxAreas = new double[numElements];
        for(int e = 0; e < numElements; e++) {
          maxAreas[e] = maxArea;
        }
        constraints->update(patch, maxAreas);
        delete maxAreas;
        Obj<Mesh> refinedMesh = refine(mesh, constraints, interpolate);

        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, double (*maxArea)(const double centroid[], void *ctx), void *ctx, bool interpolate = true) {
        Obj<Mesh::sieve_type>                         topology = mesh->getTopology();
        Obj<Mesh::field_type>                         constraints = Mesh::field_type(mesh->comm(), mesh->debug);
        Obj<Mesh::field_type>                         coordinates = mesh->getCoordinates();
        Obj<Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
        Mesh::field_type::patch_type                  patch;
        int                                           corners = topology->nCone(*elements->begin(), topology->depth())->size();
        int                                           embedDim = coordinates->getFiberDimension(patch, *topology->depthStratum(0)->begin());
        double                                       *centroid = new double[embedDim];
        std::string                                   orderName("element");

        constraints->setTopology(topology);
        constraints->setPatch(topology->leaves(), patch);
        constraints->setFiberDimensionByHeight(patch, 0, 1);
        constraints->orderPatches();

        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          //const double *coords = coordinates->restrict(orderName, *e_itor);
          const double *coords = NULL;

          for(int d = 0; d < embedDim; d++) {
            centroid[d] = 0.0;
            for(int c = 0; c < corners; c++) {
              centroid[d] += coords[c*embedDim+d];
            }
            centroid[d] /= corners;
          }
          double area = maxArea(centroid, ctx);
          constraints->update(patch, *e_itor, &area);
        }
        delete [] centroid;
        Obj<Mesh> refinedMesh = refine(mesh, constraints, interpolate);

        return refinedMesh;
      };
      static Obj<Mesh> refine(Obj<Mesh> mesh, Obj<Mesh::field_type> parallelConstraints, bool interpolate = true) {
        Obj<Mesh> refinedMesh;
        Obj<Mesh> serialMesh;
        int       dim = mesh->getDimension();

        if (mesh->distributed) {
          serialMesh = mesh->unify();
        } else {
          serialMesh = mesh;
        }
        Obj<Mesh::field_type> serialConstraints = getSerialConstraints(serialMesh, mesh, parallelConstraints);
        Mesh::field_type::patch_type patch;

        if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
          refinedMesh = refine_Triangle(serialMesh, serialConstraints->restrict(patch), interpolate);
#else
          throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
        } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
          refinedMesh = refine_TetGen(serialMesh, serialConstraints->restrict(patch), interpolate);
#else
          throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
        }
        return refinedMesh;
      };
    };
    inline
    Obj<Mesh::field_type> Generator::getSerialConstraints(Obj<Mesh> serialMesh, Obj<Mesh> parallelMesh, Obj<Mesh::field_type> parallelConstraints) {
      Obj<Mesh::field_type> serialConstraints = Mesh::field_type(parallelMesh->comm(), parallelMesh->debug);

//       if (parallelMesh->distributed) {
//         Mesh::field_type::patch_type patch;
//         PetscErrorCode ierr;

//         serialConstraints->setTopology(serialMesh->getTopology());
//         serialConstraints->setPatch(serialMesh->getTopology()->leaves(), patch);
//         serialConstraints->setFiberDimensionByHeight(patch, 0, 1);
//         serialConstraints->orderPatches();

//         Obj<ALE::Distributer<ALE::Mesh::sieve_type>::coneDelta_type::bioverlap_type> partitionOverlap = ALE::Distributer<ALE::Mesh::sieve_type>::coneDelta_type::overlap(serialMesh->getTopology(), parallelMesh->getTopology());
//         Obj<Flip<ALE::Distributer<ALE::Mesh::sieve_type>::coneDelta_type::bioverlap_type> > overlapFlip = Flip<ALE::Distributer<ALE::Mesh::sieve_type>::coneDelta_type::bioverlap_type>(partitionOverlap);
//         VecScatter scatter = ALE::Distributer<ALE::Mesh::sieve_type>::createMappingStoP(serialConstraints, parallelConstraints, overlapFlip, false);
//         Vec        serialVec, parallelVec;

//         ierr = VecCreateMPIWithArray(serialMesh->comm(), serialConstraints->getSize(patch), PETSC_DETERMINE, serialConstraints->restrict(patch), &serialVec);CHKERROR(ierr, "Error in VecCreate");
//         ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, parallelConstraints->getSize(patch), parallelConstraints->restrict(patch), &parallelVec);CHKERROR(ierr, "Error in VecCreate");
//         ierr = VecScatterBegin(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
//         ierr = VecScatterEnd(parallelVec, serialVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERROR(ierr, "Error in VecScatter");
//         ierr = VecDestroy(serialVec);CHKERROR(ierr, "Error in VecDestroy");
//         ierr = VecDestroy(parallelVec);CHKERROR(ierr, "Error in VecDestroy");
//         ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");
//       } else {
//         serialConstraints = parallelConstraints;
//       }
      return serialConstraints;
    };
#endif
