#ifndef included_ALE_Generator_hh
#define included_ALE_Generator_hh

#ifndef  included_ALE_Distribution_hh
#include <sieve/Distribution.hh>
#endif

#ifdef PETSC_HAVE_TRIANGLE
#include <triangle.h>
#endif
#ifdef PETSC_HAVE_TETGEN
#include <tetgen.h>
#endif

namespace ALE {
#ifdef PETSC_HAVE_TRIANGLE
  namespace Triangle {
    template<typename Mesh>
    class Generator {
      class SegmentVisitor {
      protected:
        const int dim;
        int *segmentlist;
        typename Mesh::numbering_type& vNumbering;
        int idx, v;
      public:
        SegmentVisitor(const int dim, int segmentlist[], typename Mesh::numbering_type& vNumbering) : dim(dim), segmentlist(segmentlist), vNumbering(vNumbering), idx(0), v(0) {};
        ~SegmentVisitor() {};
      public:
        template<typename Point>
        void visitPoint(const Point& point) {
          this->segmentlist[this->idx*dim + (this->v++)] = this->vNumbering.getIndex(point);
        }
        template<typename Arrow>
        void visitArrow(const Arrow& arrow) {}
        void setIndex(const int idx) {this->idx = idx; this->v = 0;};
      };
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
        inputCtx->trianglelist = NULL;
        inputCtx->numberofholes = 0;
        inputCtx->holelist = NULL;
        inputCtx->numberofregions = 0;
        inputCtx->regionlist = NULL;
      };
      static void initOutput(struct triangulateio *outputCtx) {
        outputCtx->numberofpoints = 0;
        outputCtx->pointlist = NULL;
        outputCtx->pointattributelist = NULL;
        outputCtx->pointmarkerlist = NULL;
        outputCtx->numberoftriangles = 0;
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
      #undef __FUNCT__
      #define __FUNCT__ "generateMesh_Triangle"
      static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false) {
        int                          dim   = 2;
        Obj<Mesh>                    mesh  = new Mesh(boundary->comm(), dim, boundary->debug());
        const Obj<typename Mesh::sieve_type>& sieve = boundary->getSieve();
        const bool                   createConvexHull = false;
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode ierr;

        initInput(&in);
        initOutput(&out);
        const Obj<typename Mesh::label_sequence>&    vertices    = boundary->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = boundary->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = boundary->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = boundary->getFactory()->getLocalNumbering(boundary, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                  idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = boundary->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& edges      = boundary->depthStratum(1);
        const Obj<typename Mesh::numbering_type>& eNumbering = boundary->getFactory()->getLocalNumbering(boundary, 1);

        in.numberofsegments = edges->size();
        if (in.numberofsegments > 0) {
          const typename Mesh::label_sequence::iterator eEnd = edges->end();

          ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != eEnd; ++e_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>&     cone = sieve->cone(*e_iter);
            const typename Mesh::sieve_type::traits::coneSequence::iterator cEnd = cone->end();
            const int                                                       idx  = eNumbering->getIndex(*e_iter);
            int                                                             v    = 0;

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
              in.segmentlist[idx*dim + (v++)] = vNumbering->getIndex(*c_iter);
            }
            in.segmentmarkerlist[idx] = boundary->getValue(markers, *e_iter);
          }
        }
        const typename Mesh::holes_type& holes = boundary->getHoles();

        in.numberofholes = holes.size();
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
          for(int h = 0; h < in.numberofholes; ++h) {
            for(int d = 0; d < dim; ++d) {
              in.holelist[h*dim+d] = holes[h][d];
            }
          }
        }
        if (mesh->commRank() == 0) {
          std::string args("pqezQ");

          if (createConvexHull) {
            args += "c";
          }
          if (constrained) {
            args = "zepDQ";
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);
        }

        if (in.pointlist)         {ierr = PetscFree(in.pointlist);CHKERRXX(ierr);}
        if (in.pointmarkerlist)   {ierr = PetscFree(in.pointmarkerlist);CHKERRXX(ierr);}
        if (in.segmentlist)       {ierr = PetscFree(in.segmentlist);CHKERRXX(ierr);}
        if (in.segmentmarkerlist) {ierr = PetscFree(in.segmentmarkerlist);CHKERRXX(ierr);}
        if (in.holelist)          {ierr = PetscFree(in.holelist);CHKERRXX(ierr);}
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;

        ALE::SieveBuilder<Mesh>::buildTopology(newSieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1, mesh->getArrowSection("orientation"));
        mesh->setSieve(newSieve);
        mesh->stratify();
        ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coords);
        const Obj<typename Mesh::label_type>& newMarkers = mesh->createLabel("marker");

        if (mesh->commRank() == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              mesh->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<typename Mesh::sieve_type::supportSet> edge = newSieve->nJoin(vertexA, vertexB, 1);

                mesh->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
              }
            }
          }
        }
        mesh->copyHoles(boundary);
        finiOutput(&out);
        return mesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "generateMeshV_Triangle"
      static Obj<Mesh> generateMeshV(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false, const bool renumber = false) {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef typename Mesh::real_section_type::value_type real;
        int                                   dim   = 2;
        const Obj<Mesh>                       mesh  = new Mesh(boundary->comm(), dim, boundary->debug());
        const Obj<typename Mesh::sieve_type>& sieve = boundary->getSieve();
        const bool                            createConvexHull = false;
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode       ierr;

        initInput(&in);
        initOutput(&out);
        const Obj<typename Mesh::label_sequence>&    vertices    = boundary->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = boundary->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = boundary->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = boundary->getFactory()->getLocalNumbering(boundary, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                           idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; ++d) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = boundary->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& edges      = boundary->depthStratum(1);
        const Obj<typename Mesh::numbering_type>& eNumbering = boundary->getFactory()->getLocalNumbering(boundary, 1);

        in.numberofsegments = edges->size();
        if (in.numberofsegments > 0) {
          const typename Mesh::label_sequence::iterator eEnd = edges->end();

          ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
          SegmentVisitor sV(dim, in.segmentlist, *vNumbering);
          for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != eEnd; ++e_iter) {
            const int idx = eNumbering->getIndex(*e_iter);

            sV.setIndex(idx);
            sieve->cone(*e_iter, sV);
            in.segmentmarkerlist[idx] = boundary->getValue(markers, *e_iter);
          }
        }
        const typename Mesh::holes_type& holes = boundary->getHoles();

        in.numberofholes = holes.size();
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
          for(int h = 0; h < in.numberofholes; ++h) {
            for(int d = 0; d < dim; ++d) {
              in.holelist[h*dim+d] = holes[h][d];
            }
          }
        }
        if (mesh->commRank() == 0) {
          std::string args("pqezQ");

          if (createConvexHull) {
            args += "c";
          }
          if (constrained) {
            args = "zepDQ";
          }
          triangulate((char *) args.c_str(), &in, &out, NULL);
        }
        if (in.pointlist)         {ierr = PetscFree(in.pointlist);CHKERRXX(ierr);}
        if (in.pointmarkerlist)   {ierr = PetscFree(in.pointmarkerlist);CHKERRXX(ierr);}
        if (in.segmentlist)       {ierr = PetscFree(in.segmentlist);CHKERRXX(ierr);}
        if (in.segmentmarkerlist) {ierr = PetscFree(in.segmentmarkerlist);CHKERRXX(ierr);}
        if (in.holelist)          {ierr = PetscFree(in.holelist);CHKERRXX(ierr);}
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        const Obj<FlexMesh>                  m        = new FlexMesh(boundary->comm(), dim, boundary->debug());
        const Obj<FlexMesh::sieve_type>      newS     = new FlexMesh::sieve_type(m->comm(), m->debug());
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;
        real   *coordsR;

        ALE::SieveBuilder<FlexMesh>::buildTopology(newS, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
        m->setSieve(newS);
        m->stratify();
        mesh->setSieve(newSieve);
        std::map<typename Mesh::point_type,typename Mesh::point_type> renumbering;
        ALE::ISieveConverter::convertSieve(*newS, *newSieve, renumbering, renumber);
        mesh->stratify();
        ALE::ISieveConverter::convertOrientation(*newS, *newSieve, renumbering,
        m->getArrowSection("orientation").ptr());
        {
          if (sizeof(double) == sizeof(real)) {
            coordsR = (real *) coords;
          } else {
            coordsR = new real[numVertices*dim];
            for(int i = 0; i < numVertices*dim; ++i) coordsR[i] = coords[i];
          }
        }
        ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coordsR);
        {
          if (sizeof(double) != sizeof(real)) {
            delete [] coordsR;
          }
        }
        const Obj<typename Mesh::label_type>& newMarkers = mesh->createLabel("marker");

        if (mesh->commRank() == 0) {
#ifdef IMESH_NEW_LABELS
          int size = 0;

          newMarkers->setChart(mesh->getSieve()->getChart());
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              newMarkers->setConeSize(v+out.numberoftriangles, 1);
              size++;
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<typename Mesh::sieve_type::supportSet> edge = newS->nJoin(vertexA, vertexB, 1);

                newMarkers->setConeSize(*edge->begin(), 1);
                size++;
              }
            }
          }
          newMarkers->setSupportSize(0, size);
          newMarkers->allocate();
#endif
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              if (renumber) {
                mesh->setValue(newMarkers, renumbering[v+out.numberoftriangles], out.pointmarkerlist[v]);
              } else {
                mesh->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
              }
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<typename Mesh::sieve_type::supportSet> edge = newS->nJoin(vertexA, vertexB, 1);

                mesh->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
              }
            }
          }
#ifdef IMESH_NEW_LABELS
          newMarkers->recalculateLabel();
#endif
        }
        mesh->copyHoles(boundary);
        finiOutput(&out);
        return mesh;
      };
    };
    template<typename Mesh>
    class Refiner {
    public:
      static Obj<Mesh> refineMesh(const Obj<Mesh>& serialMesh, const double maxVolumes[], const bool interpolate = false, const bool forceSerial = false) {
        const int                    dim         = serialMesh->getDimension();
        const Obj<Mesh>              refMesh     = new Mesh(serialMesh->comm(), dim, serialMesh->debug());
        const Obj<typename Mesh::sieve_type>& serialSieve = serialMesh->getSieve();
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode       ierr;

        Generator<Mesh>::initInput(&in);
        Generator<Mesh>::initOutput(&out);
        const Obj<typename Mesh::label_sequence>&    vertices    = serialMesh->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = serialMesh->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = serialMesh->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = serialMesh->getFactory()->getLocalNumbering(serialMesh, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                  idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = serialMesh->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& faces      = serialMesh->heightStratum(0);
        const Obj<typename Mesh::numbering_type>& fNumbering = serialMesh->getFactory()->getLocalNumbering(serialMesh, serialMesh->depth());

        in.numberofcorners   = 3;
        in.numberoftriangles = faces->size();
        in.trianglearealist  = (double *) maxVolumes;
        if (in.numberoftriangles > 0) {
          const typename Mesh::label_sequence::iterator fEnd = faces->end();

          ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRXX(ierr);
          if (serialMesh->depth() == 1) {
            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
              const Obj<typename Mesh::sieve_type::traits::coneSequence>&     cone = serialSieve->cone(*f_iter);
              const typename Mesh::sieve_type::traits::coneSequence::iterator cEnd = cone->end();
              const int                                                       idx  = fNumbering->getIndex(*f_iter);
              int                                                             v    = 0;

              for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
                in.trianglelist[idx*in.numberofcorners + v++] = vNumbering->getIndex(*c_iter);
              }
            }
          } else if (serialMesh->depth() == 2) {
            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
              typedef ALE::SieveAlg<Mesh> sieve_alg_type;
              const Obj<typename sieve_alg_type::coneArray>&       cone = sieve_alg_type::nCone(serialMesh, *f_iter, 2);
              const typename Mesh::sieve_type::coneArray::iterator cEnd = cone->end();
              const int                                            idx  = fNumbering->getIndex(*f_iter);
              int                                                  v    = 0;

              for(typename Mesh::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
                in.trianglelist[idx*in.numberofcorners + v++] = vNumbering->getIndex(*c_iter);
              }
            }
          } else {
            throw ALE::Exception("Invalid sieve: Cannot gives sieves of arbitrary depth to Triangle");
          }
        }
        if (serialMesh->depth() == 2) {
          const Obj<typename Mesh::label_sequence>&           edges    = serialMesh->depthStratum(1);
#define NEW_LABEL
#ifdef NEW_LABEL
          for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
            if (serialMesh->getValue(markers, *e_iter)) {
              in.numberofsegments++;
            }
          }
          //std::cout << "Number of segments: " << in.numberofsegments << std::endl;
          if (in.numberofsegments > 0) {
            int s = 0;

            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
            for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
              const int edgeMarker = serialMesh->getValue(markers, *e_iter);

              if (edgeMarker) {
                const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = serialSieve->cone(*e_iter);
                int                                                p    = 0;

                for(typename Mesh::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
                  in.segmentlist[s*2 + (p++)] = vNumbering->getIndex(*v_iter);
                }
                in.segmentmarkerlist[s++] = edgeMarker;
              }
            }
          }
#else
          const Obj<typename Mesh::label_type::baseSequence>& boundary = markers->base();

          in.numberofsegments = 0;
          for(typename Mesh::label_type::baseSequence::iterator b_iter = boundary->begin(); b_iter != boundary->end(); ++b_iter) {
            for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
              if (*b_iter == *e_iter) {
                in.numberofsegments++;
              }
            }
          }
          if (in.numberofsegments > 0) {
            int s = 0;

            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
            for(typename Mesh::label_type::baseSequence::iterator b_iter = boundary->begin(); b_iter != boundary->end(); ++b_iter) {
              for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
                if (*b_iter == *e_iter) {
                  const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = serialSieve->cone(*e_iter);
                  int                                                p    = 0;

                  for(typename Mesh::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
                    in.segmentlist[s*2 + (p++)] = vNumbering->getIndex(*v_iter);
                  }
                  in.segmentmarkerlist[s++] = serialMesh->getValue(markers, *e_iter);
                }
              }
            }
          }
#endif
        }

        in.numberofholes = 0;
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);CHKERRXX(ierr);
        }
        if (serialMesh->commRank() == 0) {
          std::string args("pqezQra");

          triangulate((char *) args.c_str(), &in, &out, NULL);
        }
        if (in.pointlist)         {ierr = PetscFree(in.pointlist);CHKERRXX(ierr);}
        if (in.pointmarkerlist)   {ierr = PetscFree(in.pointmarkerlist);CHKERRXX(ierr);}
        if (in.segmentlist)       {ierr = PetscFree(in.segmentlist);CHKERRXX(ierr);}
        if (in.segmentmarkerlist) {ierr = PetscFree(in.segmentmarkerlist);CHKERRXX(ierr);}
        if (in.trianglelist)      {ierr = PetscFree(in.trianglelist);CHKERRXX(ierr);}
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(serialMesh->comm(), serialMesh->debug());
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;

        ALE::SieveBuilder<Mesh>::buildTopology(newSieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1, refMesh->getArrowSection("orientation"));
        refMesh->setSieve(newSieve);
        refMesh->stratify();
        ALE::SieveBuilder<Mesh>::buildCoordinates(refMesh, dim, coords);
        const Obj<typename Mesh::label_type>& newMarkers = refMesh->createLabel("marker");

        if (refMesh->commRank() == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              refMesh->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<typename Mesh::sieve_type::supportSet> edge = newSieve->nJoin(vertexA, vertexB, 1);

                refMesh->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
              }
            }
          }
        }

        Generator<Mesh>::finiOutput(&out);
        if ((refMesh->commSize() > 1) && (!forceSerial)) {
          return ALE::Distribution<Mesh>::distributeMesh(refMesh);
        }
        return refMesh;
      };
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false) {
        Obj<Mesh>                          serialMesh       = ALE::Distribution<Mesh>::unifyMesh(mesh);
        const Obj<typename Mesh::real_section_type> serialMaxVolumes = ALE::Distribution<Mesh>::distributeSection(maxVolumes, serialMesh, serialMesh->getDistSendOverlap(), serialMesh->getDistRecvOverlap());

        return refineMesh(serialMesh, serialMaxVolumes->restrictSpace(), interpolate);
      };
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false, const bool forceSerial = false) {
        Obj<Mesh> serialMesh;
        if ((mesh->commSize() > 1) && (!forceSerial)) {
          serialMesh = ALE::Distribution<Mesh>::unifyMesh(mesh);
        } else {
          serialMesh = mesh;
        }
        const int numFaces         = serialMesh->heightStratum(0)->size();
        double   *serialMaxVolumes = new double[numFaces];

        for(int f = 0; f < numFaces; f++) {
          serialMaxVolumes[f] = maxVolume;
        }
        const Obj<Mesh> refMesh = refineMesh(serialMesh, serialMaxVolumes, interpolate, forceSerial);
        delete [] serialMaxVolumes;
        return refMesh;
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const double maxVolumes[], const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef typename Mesh::real_section_type::value_type real;
        typedef typename Mesh::point_type point_type;
        const int                             dim     = mesh->getDimension();
        const Obj<Mesh>                       refMesh = new Mesh(mesh->comm(), dim, mesh->debug());
        const Obj<typename Mesh::sieve_type>& sieve   = mesh->getSieve();
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode       ierr;

        Generator<Mesh>::initInput(&in);
        Generator<Mesh>::initOutput(&out);
        const Obj<typename Mesh::label_sequence>&    vertices    = mesh->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = mesh->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                           idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = mesh->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& faces      = mesh->heightStratum(0);
        const Obj<typename Mesh::numbering_type>& fNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());

        in.numberofcorners   = 3;
        in.numberoftriangles = faces->size();
        in.trianglearealist  = (double *) maxVolumes;
        if (in.numberoftriangles > 0) {
          ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRXX(ierr);
          if (mesh->depth() == 1) {
            ALE::ISieveVisitor::PointRetriever<typename Mesh::sieve_type> pV(3);
            const typename Mesh::label_sequence::iterator fEnd = faces->end();

            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
              sieve->cone(*f_iter, pV);
              const int         idx  = fNumbering->getIndex(*f_iter);
              const size_t      n    = pV.getSize();
              const point_type *cone = pV.getPoints();

              assert(n == 3);
              for(int v = 0; v < 3; ++v) {
                in.trianglelist[idx*in.numberofcorners + v] = vNumbering->getIndex(cone[v]);
              }
              pV.clear();
            }
          } else if (mesh->depth() == 2) {
            // Need extra space due to early error checking
            ALE::ISieveVisitor::NConeRetriever<typename Mesh::sieve_type> ncV(*sieve, 4);
            const typename Mesh::label_sequence::iterator fEnd = faces->end();

            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
              ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*sieve, *f_iter, ncV);
              const int         idx  = fNumbering->getIndex(*f_iter);
              const size_t      n    = ncV.getSize();
              const point_type *cone = ncV.getPoints();

              assert(n == 3);
              for(int v = 0; v < 3; ++v) {
                in.trianglelist[idx*in.numberofcorners + v] = vNumbering->getIndex(cone[v]);
              }
              ncV.clear();
            }
          } else {
            throw ALE::Exception("Invalid sieve: Cannot gives sieves of arbitrary depth to Triangle");
          }
        }
        if (mesh->depth() == 2) {
          const Obj<typename Mesh::label_sequence>& edges = mesh->depthStratum(1);
          for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
            if (mesh->getValue(markers, *e_iter)) {
              in.numberofsegments++;
            }
          }
          //std::cout << "Number of segments: " << in.numberofsegments << std::endl;
          if (in.numberofsegments > 0) {
            ALE::ISieveVisitor::PointRetriever<typename Mesh::sieve_type> pV(2);
            int s = 0;

            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
            for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
              const int edgeMarker = mesh->getValue(markers, *e_iter);

              if (edgeMarker) {
                sieve->cone(*e_iter, pV);
                const size_t      n    = pV.getSize();
                const point_type *cone = pV.getPoints();

                assert(n == 2);
                for(int v = 0; v < 2; ++v) {
                  in.segmentlist[s*2 + v] = vNumbering->getIndex(cone[v]);
                }
                in.segmentmarkerlist[s++] = edgeMarker;
                pV.clear();
              }
            }
          }
        }
        const typename Mesh::holes_type& holes = mesh->getHoles();

        in.numberofholes = holes.size();
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
          for(int h = 0; h < in.numberofholes; ++h) {
            for(int d = 0; d < dim; ++d) {
              in.holelist[h*dim+d] = holes[h][d];
            }
          }
        }
        if (mesh->commRank() == 0) {
          std::string args("pqezQra");

          triangulate((char *) args.c_str(), &in, &out, NULL);
        }
        if (in.pointlist)         {ierr = PetscFree(in.pointlist);CHKERRXX(ierr);}
        if (in.pointmarkerlist)   {ierr = PetscFree(in.pointmarkerlist);CHKERRXX(ierr);}
        if (in.segmentlist)       {ierr = PetscFree(in.segmentlist);CHKERRXX(ierr);}
        if (in.segmentmarkerlist) {ierr = PetscFree(in.segmentmarkerlist);CHKERRXX(ierr);}
        if (in.trianglelist)      {ierr = PetscFree(in.trianglelist);CHKERRXX(ierr);}
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        const Obj<FlexMesh>                  m        = new FlexMesh(mesh->comm(), dim, mesh->debug());
        const Obj<FlexMesh::sieve_type>      newS     = new FlexMesh::sieve_type(m->comm(), m->debug());
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;
        real   *coordsR;

        ALE::SieveBuilder<FlexMesh>::buildTopology(newS, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
        m->setSieve(newS);
        m->stratify();
        refMesh->setSieve(newSieve);
        std::map<typename Mesh::point_type,typename Mesh::point_type> renumbering;
        ALE::ISieveConverter::convertSieve(*newS, *newSieve, renumbering, renumber);
        refMesh->stratify();
        ALE::ISieveConverter::convertOrientation(*newS, *newSieve, renumbering, m->getArrowSection("orientation").ptr());
        {
          if (sizeof(double) == sizeof(real)) {
            coordsR = (real *) coords;
          } else {
            coordsR = new real[numVertices*dim];
            for(int i = 0; i < numVertices*dim; ++i) coordsR[i] = coords[i];
          }
        }
        ALE::SieveBuilder<Mesh>::buildCoordinates(refMesh, dim, coordsR);
        {
          if (sizeof(double) != sizeof(real)) {
            delete [] coordsR;
          }
        }
        const Obj<typename Mesh::label_type>& newMarkers = refMesh->createLabel("marker");

        if (refMesh->commRank() == 0) {
          for(int v = 0; v < out.numberofpoints; v++) {
            if (out.pointmarkerlist[v]) {
              if (renumber) {
                refMesh->setValue(newMarkers, renumbering[v+out.numberoftriangles], out.pointmarkerlist[v]);
              } else {
                refMesh->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
              }
            }
          }
          if (interpolate) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
                const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
                const Obj<typename Mesh::sieve_type::supportSet> edge = newS->nJoin(vertexA, vertexB, 1);

                refMesh->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
              }
            }
          }
        }
        refMesh->copyHoles(mesh);
        Generator<Mesh>::finiOutput(&out);
#if 0
        if ((refMesh->commSize() > 1) && (!forceSerial)) {
          return ALE::Distribution<Mesh>::distributeMesh(refMesh);
        }
#endif
        return refMesh;
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false, const bool renumber = false) {
        throw ALE::Exception("Not yet implemented");
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
#if 0
        Obj<Mesh> serialMesh;
        if (mesh->commSize() > 1) {
          serialMesh = ALE::Distribution<Mesh>::unifyMesh(mesh);
        } else {
          serialMesh = mesh;
        }
#endif
        const int numCells         = mesh->heightStratum(0)->size();
        double   *serialMaxVolumes = new double[numCells];

        for(int c = 0; c < numCells; c++) {
          serialMaxVolumes[c] = maxVolume;
        }
        const Obj<Mesh> refMesh = refineMeshV(mesh, serialMaxVolumes, interpolate, forceSerial, renumber);
        delete [] serialMaxVolumes;
        return refMesh;
      };
      static Obj<Mesh> refineMeshLocal(const Obj<Mesh>& mesh, const double maxVolumes[], const bool interpolate = false) {
        const int                    dim     = mesh->getDimension();
        const Obj<Mesh>              refMesh = new Mesh(mesh->comm(), dim, mesh->debug());
        const Obj<typename Mesh::sieve_type>& sieve   = mesh->getSieve();
        struct triangulateio in;
        struct triangulateio out;
        PetscErrorCode       ierr;

        Generator<Mesh>::initInput(&in);
        Generator<Mesh>::initOutput(&out);
        const Obj<typename Mesh::label_sequence>&    vertices    = mesh->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = mesh->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRXX(ierr);
          ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRXX(ierr);
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                  idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = mesh->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& faces      = mesh->heightStratum(0);
        const Obj<typename Mesh::numbering_type>& fNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());

        in.numberofcorners   = 3;
        in.numberoftriangles = faces->size();
        in.trianglearealist  = (double *) maxVolumes;
        if (in.numberoftriangles > 0) {
          ierr = PetscMalloc(in.numberoftriangles*in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRXX(ierr);
          if (mesh->depth() == 1) {
            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
              const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*f_iter);
              const int                                          idx  = fNumbering->getIndex(*f_iter);
              int                                                v    = 0;

              for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                in.trianglelist[idx*in.numberofcorners + v++] = vNumbering->getIndex(*c_iter);
              }
            }
          } else if (mesh->depth() == 2) {
            for(typename Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
              typedef ALE::SieveAlg<Mesh> sieve_alg_type;
              const Obj<typename sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(mesh, *f_iter, 2);
              const int                             idx  = fNumbering->getIndex(*f_iter);
              int                                   v    = 0;

              for(typename Mesh::sieve_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                in.trianglelist[idx*in.numberofcorners + v++] = vNumbering->getIndex(*c_iter);
              }
            }
          } else {
            throw ALE::Exception("Invalid sieve: Cannot gives sieves of arbitrary depth to Triangle");
          }
        }
        if (mesh->depth() == 2) {
          const Obj<typename Mesh::label_sequence>& edges = mesh->depthStratum(1);
          for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
            if (mesh->getValue(markers, *e_iter)) {
              in.numberofsegments++;
            }
          }
          //std::cout << "Number of segments: " << in.numberofsegments << std::endl;
          if (in.numberofsegments > 0) {
            int s = 0;

            ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRXX(ierr);
            ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRXX(ierr);
            for(typename Mesh::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
              const int edgeMarker = mesh->getValue(markers, *e_iter);

              if (edgeMarker) {
                const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*e_iter);
                int                                                p    = 0;

                for(typename Mesh::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
                  in.segmentlist[s*2 + (p++)] = vNumbering->getIndex(*v_iter);
                }
                in.segmentmarkerlist[s++] = edgeMarker;
              }
            }
          }
        }

        in.numberofholes = 0;
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);CHKERRXX(ierr);
        }
        std::string args("pqezQra");

        triangulate((char *) args.c_str(), &in, &out, NULL);
        if (in.pointlist)         {ierr = PetscFree(in.pointlist);CHKERRXX(ierr);}
        if (in.pointmarkerlist)   {ierr = PetscFree(in.pointmarkerlist);CHKERRXX(ierr);}
        if (in.segmentlist)       {ierr = PetscFree(in.segmentlist);CHKERRXX(ierr);}
        if (in.segmentmarkerlist) {ierr = PetscFree(in.segmentmarkerlist);CHKERRXX(ierr);}
        if (in.trianglelist)      {ierr = PetscFree(in.trianglelist);CHKERRXX(ierr);}
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        int     numCorners  = 3;
        int     numCells    = out.numberoftriangles;
        int    *cells       = out.trianglelist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;

        ALE::SieveBuilder<Mesh>::buildTopologyMultiple(newSieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1, refMesh->getArrowSection("orientation"));
        refMesh->setSieve(newSieve);
        refMesh->stratify();
        ALE::SieveBuilder<Mesh>::buildCoordinatesMultiple(refMesh, dim, coords);
        const Obj<typename Mesh::label_type>& newMarkers = refMesh->createLabel("marker");

        for(int v = 0; v < out.numberofpoints; v++) {
          if (out.pointmarkerlist[v]) {
            refMesh->setValue(newMarkers, v+out.numberoftriangles, out.pointmarkerlist[v]);
          }
        }
        if (interpolate) {
          for(int e = 0; e < out.numberofedges; e++) {
            if (out.edgemarkerlist[e]) {
              const typename Mesh::point_type vertexA(out.edgelist[e*2+0]+out.numberoftriangles);
              const typename Mesh::point_type vertexB(out.edgelist[e*2+1]+out.numberoftriangles);
              const Obj<typename Mesh::sieve_type::supportSet> edge = newSieve->nJoin(vertexA, vertexB, 1);

              refMesh->setValue(newMarkers, *(edge->begin()), out.edgemarkerlist[e]);
            }
          }
        }
        Generator<Mesh>::finiOutput(&out);
        return refMesh;
      };
      static Obj<Mesh> refineMeshLocal(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
        const int numLocalFaces   = mesh->heightStratum(0)->size();
        double   *localMaxVolumes = new double[numLocalFaces];

        for(int f = 0; f < numLocalFaces; f++) {
          localMaxVolumes[f] = maxVolume;
        }
        const Obj<Mesh> refMesh = refineMeshLocal(mesh, localMaxVolumes, interpolate);
        const Obj<typename Mesh::sieve_type> refSieve = refMesh->getSieve();
        delete [] localMaxVolumes;
#if 0
	typedef typename ALE::New::Completion<Mesh, typename Mesh::sieve_type::point_type> sieveCompletion;
	// This is where we enforce consistency over the overlap
	//   We need somehow to update the overlap to account for the new stuff
	//
	//   1) Since we are refining only, the vertices are invariant
	//   2) We need to make another label for the interprocess boundaries so
	//      that Triangle will respect them
	//   3) We then throw all that label into the new overlap
	//
	// Alternative: Figure out explicitly which segments were refined, and then
	//   communicated the refinement over the old overlap. Use this info to locally
	//   construct the new overlap and flip to get a decent mesh
	sieveCompletion::scatterCones(refSieve, refSieve, reMesh->getDistSendOverlap(), refMesh->getDistRecvOverlap(), refMesh);
#endif
        return refMesh;
      };
    };
  };
#endif
#ifdef PETSC_HAVE_TETGEN
  namespace TetGen {
    template<typename Mesh>
    class Generator {
    public:
      static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false) {
        typedef ALE::SieveAlg<Mesh> sieve_alg_type;
        const int         dim   = 3;
        Obj<Mesh>         mesh  = new Mesh(boundary->comm(), dim, boundary->debug());
        const PetscMPIInt rank  = mesh->commRank();
        bool              createConvexHull = false;
        ::tetgenio        in;
        ::tetgenio        out;

        const Obj<typename Mesh::label_sequence>&    vertices    = boundary->depthStratum(0);
        const Obj<typename Mesh::numbering_type>&    vNumbering  = boundary->getFactory()->getLocalNumbering(boundary, 0);
        const Obj<typename Mesh::real_section_type>& coordinates = boundary->getRealSection("coordinates");
        const Obj<typename Mesh::label_type>&        markers     = boundary->getLabel("marker");


        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                  idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = boundary->getValue(markers, *v_iter);
          }
        }

	if (boundary->depth() != 0) {  //our boundary mesh COULD be just a pointset; in which case depth = height = 0;
          const Obj<typename Mesh::label_sequence>& facets     = boundary->depthStratum(boundary->depth());
          //PetscPrintf(boundary->comm(), "%d facets on the boundary\n", facets->size());
          const Obj<typename Mesh::numbering_type>& fNumbering = boundary->getFactory()->getLocalNumbering(boundary, boundary->depth());

          in.numberoffacets = facets->size();
          if (in.numberoffacets > 0) {
            in.facetlist       = new tetgenio::facet[in.numberoffacets];
            in.facetmarkerlist = new int[in.numberoffacets];
            for(typename Mesh::label_sequence::iterator f_iter = facets->begin(); f_iter != facets->end(); ++f_iter) {
              const Obj<typename sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(boundary, *f_iter, boundary->depth());
              const int                             idx  = fNumbering->getIndex(*f_iter);

              in.facetlist[idx].numberofpolygons = 1;
              in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
              in.facetlist[idx].numberofholes    = 0;
              in.facetlist[idx].holelist         = NULL;

              tetgenio::polygon *poly = in.facetlist[idx].polygonlist;
              int                c    = 0;

              poly->numberofvertices = cone->size();
              poly->vertexlist       = new int[poly->numberofvertices];
              for(typename sieve_alg_type::coneArray::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                const int vIdx = vNumbering->getIndex(*c_iter);

                poly->vertexlist[c++] = vIdx;
              }
              in.facetmarkerlist[idx] = boundary->getValue(markers, *f_iter);
            }
          }
        }else {
          createConvexHull = true;
        }

        in.numberofholes = 0;
        if (rank == 0) {
          // Normal operation
          std::string args("pqezQ");
          //constrained operation
          if (constrained) {
            args = "pezQ";
            if (createConvexHull) {
              args = "ezQ";
              //PetscPrintf(boundary->comm(), "createConvexHull\n");
            }
          }
          // Just make tetrahedrons
//           std::string args("efzV");
          // Adds a center point
//           std::string args("pqezQi");
//           in.numberofaddpoints = 1;
//           in.addpointlist      = new double[in.numberofaddpoints*dim];
//           in.addpointlist[0]   = 0.5;
//           in.addpointlist[1]   = 0.5;
//           in.addpointlist[2]   = 0.5;

          //if (createConvexHull) args += "c";  NOT SURE, but this was basically unused before, and the convex hull should be filled in if "p" isn't included.
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        int     numCorners  = 4;
        int     numCells    = out.numberoftetrahedra;
        int    *cells       = out.tetrahedronlist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;

        if (!interpolate) {
          for(int c = 0; c < numCells; ++c) {
            int tmp = cells[c*4+0];
            cells[c*4+0] = cells[c*4+1];
            cells[c*4+1] = tmp;
          }
        }
        ALE::SieveBuilder<Mesh>::buildTopology(newSieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1, mesh->getArrowSection("orientation"));
        mesh->setSieve(newSieve);
        mesh->stratify();
        ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coords);
        const Obj<typename Mesh::label_type>& newMarkers = mesh->createLabel("marker");

        for(int v = 0; v < out.numberofpoints; v++) {
          if (out.pointmarkerlist[v]) {
            mesh->setValue(newMarkers, v+out.numberoftetrahedra, out.pointmarkerlist[v]);
          }
        }
        if (interpolate) {
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                typename Mesh::point_type endpointA(out.edgelist[e*2+0]+out.numberoftetrahedra);
                typename Mesh::point_type endpointB(out.edgelist[e*2+1]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> edge = newSieve->nJoin(endpointA, endpointB, 1);

                mesh->setValue(newMarkers, *edge->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            // Work around TetGen bug for raw tetrahedralization
            //   The boundary faces are 0,1,4,5,8,9,11,12,13,15,16,17
//             for(int f = 0; f < out.numberoftrifaces; f++) {
//               if (out.trifacemarkerlist[f]) {
//                 out.trifacemarkerlist[f] = 0;
//               } else {
//                 out.trifacemarkerlist[f] = 1;
//               }
//             }
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                typename Mesh::point_type cornerA(out.trifacelist[f*3+0]+out.numberoftetrahedra);
                typename Mesh::point_type cornerB(out.trifacelist[f*3+1]+out.numberoftetrahedra);
                typename Mesh::point_type cornerC(out.trifacelist[f*3+2]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> corners = typename Mesh::sieve_type::supportSet();
                Obj<typename Mesh::sieve_type::supportSet> edges   = typename Mesh::sieve_type::supportSet();
                corners->clear();corners->insert(cornerA);corners->insert(cornerB);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerB);corners->insert(cornerC);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerC);corners->insert(cornerA);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                const typename Mesh::point_type          face       = *newSieve->nJoin1(edges)->begin();
                const int                       faceMarker = out.trifacemarkerlist[f];
                const Obj<typename Mesh::coneArray>      closure    = sieve_alg_type::closure(mesh, face);
                const typename Mesh::coneArray::iterator end        = closure->end();

                for(typename Mesh::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
                  mesh->setValue(newMarkers, *cl_iter, faceMarker);
                }
              }
            }
          }
        }
        return mesh;
      };
      #undef __FUNCT__
      #define __FUNCT__ "generateMeshV_TetGen"
      static Obj<Mesh> generateMeshV(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false, const bool renumber = false) {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef typename Mesh::real_section_type::value_type real;
        const int                             dim   = 3;
        Obj<Mesh>                             mesh  = new Mesh(boundary->comm(), dim, boundary->debug());
        const Obj<typename Mesh::sieve_type>& sieve = boundary->getSieve();
        const PetscMPIInt                     rank  = mesh->commRank();
        bool                                  createConvexHull = false;
        PetscErrorCode                        ierr;
        ::tetgenio in;
        ::tetgenio out;

        const Obj<typename Mesh::label_sequence>&    vertices    = boundary->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = boundary->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = boundary->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = boundary->getFactory()->getLocalNumbering(boundary, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                           idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; ++d) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = boundary->getValue(markers, *v_iter);
          }
        }

        // Our boundary mesh COULD be just a pointset; in which case depth = height = 0;
        if (boundary->depth() != 0) {
          const Obj<typename Mesh::label_sequence>& facets     = boundary->depthStratum(boundary->depth());
          const Obj<typename Mesh::numbering_type>& fNumbering = boundary->getFactory()->getLocalNumbering(boundary, boundary->depth());

          in.numberoffacets = facets->size();
          if (in.numberoffacets > 0) {
            ALE::ISieveVisitor::NConeRetriever<typename Mesh::sieve_type> ncV(*sieve, 5);
            const typename Mesh::label_sequence::iterator fEnd = facets->end();

            in.facetlist       = new tetgenio::facet[in.numberoffacets];
            in.facetmarkerlist = new int[in.numberoffacets];
            for(typename Mesh::label_sequence::iterator f_iter = facets->begin(); f_iter != fEnd; ++f_iter) {
              ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*sieve, *f_iter, ncV);
              const int idx  = fNumbering->getIndex(*f_iter);

              in.facetlist[idx].numberofpolygons = 1;
              in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
              in.facetlist[idx].numberofholes    = 0;
              in.facetlist[idx].holelist         = NULL;

              tetgenio::polygon               *poly = in.facetlist[idx].polygonlist;
              const size_t                     n    = ncV.getSize();
              const typename Mesh::point_type *cone = ncV.getPoints();

              poly->numberofvertices = n;
              poly->vertexlist       = new int[poly->numberofvertices];
              for(size_t c = 0; c < n; ++c) {
                poly->vertexlist[c] = vNumbering->getIndex(cone[c]);
              }
              in.facetmarkerlist[idx] = boundary->getValue(markers, *f_iter);
              ncV.clear();
            }
          }
        } else {
          createConvexHull = true;
        }
        const typename Mesh::holes_type& holes = mesh->getHoles();

        in.numberofholes = holes.size();
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
          for(int h = 0; h < in.numberofholes; ++h) {
            for(int d = 0; d < dim; ++d) {
              in.holelist[h*dim+d] = holes[h][d];
            }
          }
        }
        if (rank == 0) {
          // Normal operation
          std::string args("pqezQ");
          // Constrained operation
          if (constrained) {
            args = "pezQ";
            if (createConvexHull) {
              args = "ezQ";
            }
          }
          // Just make tetrahedrons
//           std::string args("efzV");
          // Adds a center point
//           std::string args("pqezQi");
//           in.numberofaddpoints = 1;
//           in.addpointlist      = new double[in.numberofaddpoints*dim];
//           in.addpointlist[0]   = 0.5;
//           in.addpointlist[1]   = 0.5;
//           in.addpointlist[2]   = 0.5;
          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
        const Obj<FlexMesh>                  m        = new FlexMesh(mesh->comm(), dim, mesh->debug());
        const Obj<FlexMesh::sieve_type>      newS     = new FlexMesh::sieve_type(m->comm(), m->debug());
        int     numCorners  = 4;
        int     numCells    = out.numberoftetrahedra;
        int    *cells       = out.tetrahedronlist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;
        real   *coordsR;

        if (!interpolate) {
          // TetGen reports tetrahedra with the opposite orientation from what we expect
          for(int c = 0; c < numCells; ++c) {
            int tmp = cells[c*4+0];
            cells[c*4+0] = cells[c*4+1];
            cells[c*4+1] = tmp;
          }
        }
        ALE::SieveBuilder<FlexMesh>::buildTopology(newS, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
        m->setSieve(newS);
        m->stratify();
        mesh->setSieve(newSieve);
        std::map<typename Mesh::point_type,typename Mesh::point_type> renumbering;
        ALE::ISieveConverter::convertSieve(*newS, *newSieve, renumbering, renumber);
        mesh->stratify();
        ALE::ISieveConverter::convertOrientation(*newS, *newSieve, renumbering, m->getArrowSection("orientation").ptr());
        {
          if (sizeof(double) == sizeof(real)) {
            coordsR = (real *) coords;
          } else {
            coordsR = new real[numVertices*dim];
            for(int i = 0; i < numVertices*dim; ++i) coordsR[i] = coords[i];
          }
        }
        ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, dim, coordsR);
        {
          if (sizeof(double) != sizeof(real)) {
            delete [] coordsR;
          }
        }
        const Obj<typename Mesh::label_type>& newMarkers = mesh->createLabel("marker");

        for(int v = 0; v < out.numberofpoints; v++) {
          if (out.pointmarkerlist[v]) {
            if (renumber) {
              mesh->setValue(newMarkers, renumbering[v+out.numberoftetrahedra], out.pointmarkerlist[v]);
            } else {
              mesh->setValue(newMarkers, v+out.numberoftetrahedra, out.pointmarkerlist[v]);
            }
          }
        }
        if (interpolate) {
          // This does not work anymore (edgemarkerlist is always empty). I tried -ee and it gave bogus results
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                typename Mesh::point_type endpointA(out.edgelist[e*2+0]+out.numberoftetrahedra);
                typename Mesh::point_type endpointB(out.edgelist[e*2+1]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> edge = newS->nJoin(endpointA, endpointB, 1);

                if (renumber) {
                  mesh->setValue(newMarkers, renumbering[*edge->begin()], out.edgemarkerlist[e]);
                } else {
                  mesh->setValue(newMarkers, *edge->begin(), out.edgemarkerlist[e]);
                }
              }
            }
          }
          if (out.trifacemarkerlist) {
            // Work around TetGen bug for raw tetrahedralization
            //   The boundary faces are 0,1,4,5,8,9,11,12,13,15,16,17
//             for(int f = 0; f < out.numberoftrifaces; f++) {
//               if (out.trifacemarkerlist[f]) {
//                 out.trifacemarkerlist[f] = 0;
//               } else {
//                 out.trifacemarkerlist[f] = 1;
//               }
//             }
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                typename Mesh::point_type cornerA = out.trifacelist[f*3+0]+out.numberoftetrahedra;
                typename Mesh::point_type cornerB = out.trifacelist[f*3+1]+out.numberoftetrahedra;
                typename Mesh::point_type cornerC = out.trifacelist[f*3+2]+out.numberoftetrahedra;
                Obj<typename Mesh::sieve_type::supportSet> corners = typename Mesh::sieve_type::supportSet();
                Obj<typename Mesh::sieve_type::supportSet> edges   = typename Mesh::sieve_type::supportSet();
                corners->clear();corners->insert(cornerA);corners->insert(cornerB);
                typename Mesh::point_type edgeA = *newS->nJoin1(corners)->begin();
                corners->clear();corners->insert(cornerB);corners->insert(cornerC);
                typename Mesh::point_type edgeB = *newS->nJoin1(corners)->begin();
                corners->clear();corners->insert(cornerC);corners->insert(cornerA);
                typename Mesh::point_type edgeC = *newS->nJoin1(corners)->begin();
                edges->insert(edgeA); edges->insert(edgeB); edges->insert(edgeC);
                ALE::ISieveVisitor::PointRetriever<typename Mesh::sieve_type> pV(30);
                typename Mesh::point_type face       = *newS->nJoin1(edges)->begin();
                const int                 faceMarker = out.trifacemarkerlist[f];

                if (renumber) {face = renumbering[face];}
                ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*newSieve, face, pV);
                const size_t                     n    = pV.getSize();
                const typename Mesh::point_type *cone = pV.getPoints();

                for(size_t c = 0; c < n; ++c) {
                  mesh->setValue(newMarkers, cone[c], faceMarker);
                }
                pV.clear();
              }
            }
          }
        }
        mesh->copyHoles(boundary);
        return mesh;
      };
    };
    template<typename Mesh>
    class Refiner {
    public:
      static Obj<Mesh> refineMesh(const Obj<Mesh>& serialMesh, const double maxVolumes[], const bool interpolate = false) {
        typedef ALE::SieveAlg<Mesh> sieve_alg_type;
        const int       dim     = serialMesh->getDimension();
        const int       depth   = serialMesh->depth();
        const Obj<Mesh> refMesh = new Mesh(serialMesh->comm(), dim, serialMesh->debug());
        ::tetgenio      in;
        ::tetgenio      out;

        const Obj<typename Mesh::label_sequence>&    vertices    = serialMesh->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = serialMesh->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = serialMesh->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = serialMesh->getFactory()->getLocalNumbering(serialMesh, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                  idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = serialMesh->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& cells      = serialMesh->heightStratum(0);
        const Obj<typename Mesh::numbering_type>& cNumbering = serialMesh->getFactory()->getLocalNumbering(serialMesh, depth);

        in.numberofcorners       = 4;
        in.numberoftetrahedra    = cells->size();
        in.tetrahedronvolumelist = (double *) maxVolumes;
        if (in.numberoftetrahedra > 0) {
          in.tetrahedronlist     = new int[in.numberoftetrahedra*in.numberofcorners];
          for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            typedef ALE::SieveAlg<Mesh> sieve_alg_type;
            const Obj<typename sieve_alg_type::coneArray>& cone = sieve_alg_type::nCone(serialMesh, *c_iter, depth);
            const int                             idx  = cNumbering->getIndex(*c_iter);
            int                                   v    = 0;

            for(typename Mesh::sieve_type::coneArray::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
              in.tetrahedronlist[idx*in.numberofcorners + v++] = vNumbering->getIndex(*v_iter);
            }
          }
        }
        if (serialMesh->depth() == 3) {
          const Obj<typename Mesh::label_sequence>& boundary = serialMesh->getLabelStratum("marker", 1);

          in.numberoftrifaces = 0;
          for(typename Mesh::label_sequence::iterator b_iter = boundary->begin(); b_iter != boundary->end(); ++b_iter) {
            if (serialMesh->height(*b_iter) == 1) {
              in.numberoftrifaces++;
            }
          }
          if (in.numberoftrifaces > 0) {
            int f = 0;

            in.trifacelist       = new int[in.numberoftrifaces*3];
            in.trifacemarkerlist = new int[in.numberoftrifaces];
            for(typename Mesh::label_sequence::iterator b_iter = boundary->begin(); b_iter != boundary->end(); ++b_iter) {
              if (serialMesh->height(*b_iter) == 1) {
                const Obj<typename Mesh::coneArray>& cone = sieve_alg_type::nCone(serialMesh, *b_iter, 2);
                int                         p    = 0;

                for(typename Mesh::coneArray::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
                  in.trifacelist[f*3 + (p++)] = vNumbering->getIndex(*v_iter);
                }
                in.trifacemarkerlist[f++] = serialMesh->getValue(markers, *b_iter);
              }
            }
          }
        }

        in.numberofholes = 0;
        if (serialMesh->commRank() == 0) {
          std::string args("qezQra");

          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        in.tetrahedronvolumelist = NULL;
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(refMesh->comm(), refMesh->debug());
        int     numCorners  = 4;
        int     numCells    = out.numberoftetrahedra;
        int    *newCells       = out.tetrahedronlist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;

        if (!interpolate) {
          for(int c = 0; c < numCells; ++c) {
            int tmp = newCells[c*4+0];
            newCells[c*4+0] = newCells[c*4+1];
            newCells[c*4+1] = tmp;
          }
        }
        ALE::SieveBuilder<Mesh>::buildTopology(newSieve, dim, numCells, newCells, numVertices, interpolate, numCorners, -1, refMesh->getArrowSection("orientation"));
        refMesh->setSieve(newSieve);
        refMesh->stratify();
        ALE::SieveBuilder<Mesh>::buildCoordinates(refMesh, dim, coords);
        const Obj<typename Mesh::label_type>& newMarkers = refMesh->createLabel("marker");


        for(int v = 0; v < out.numberofpoints; v++) {
          if (out.pointmarkerlist[v]) {
            refMesh->setValue(newMarkers, v+out.numberoftetrahedra, out.pointmarkerlist[v]);
          }
        }
        if (interpolate) {
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                typename Mesh::point_type endpointA(out.edgelist[e*2+0]+out.numberoftetrahedra);
                typename Mesh::point_type endpointB(out.edgelist[e*2+1]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> edge = newSieve->nJoin(endpointA, endpointB, 1);

                refMesh->setValue(newMarkers, *edge->begin(), out.edgemarkerlist[e]);
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                typename Mesh::point_type cornerA(out.trifacelist[f*3+0]+out.numberoftetrahedra);
                typename Mesh::point_type cornerB(out.trifacelist[f*3+1]+out.numberoftetrahedra);
                typename Mesh::point_type cornerC(out.trifacelist[f*3+2]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> corners = typename Mesh::sieve_type::supportSet();
                Obj<typename Mesh::sieve_type::supportSet> edges   = typename Mesh::sieve_type::supportSet();
                corners->clear();corners->insert(cornerA);corners->insert(cornerB);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerB);corners->insert(cornerC);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerC);corners->insert(cornerA);
                edges->insert(*newSieve->nJoin1(corners)->begin());
                const typename Mesh::point_type          face       = *newSieve->nJoin1(edges)->begin();
                const int                       faceMarker = out.trifacemarkerlist[f];
                const Obj<typename Mesh::coneArray>      closure    = sieve_alg_type::closure(refMesh, face);
                const typename Mesh::coneArray::iterator end        = closure->end();

                for(typename Mesh::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
                  refMesh->setValue(newMarkers, *cl_iter, faceMarker);
                }
              }
            }
          }
        }
        if (refMesh->commSize() > 1) {
          return ALE::Distribution<Mesh>::distributeMesh(refMesh);
        }
        return refMesh;
      };
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false) {
        Obj<Mesh>                          serialMesh       = ALE::Distribution<Mesh>::unifyMesh(mesh);
        const Obj<typename Mesh::real_section_type> serialMaxVolumes = ALE::Distribution<Mesh>::distributeSection(maxVolumes, serialMesh, serialMesh->getDistSendOverlap(), serialMesh->getDistRecvOverlap());

        return refineMesh(serialMesh, serialMaxVolumes->restrictSpace(), interpolate);
      };
      static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
        Obj<Mesh> serialMesh;
        if (mesh->commSize() > 1) {
          serialMesh = ALE::Distribution<Mesh>::unifyMesh(mesh);
        } else {
          serialMesh = mesh;
        }
        const int numCells         = serialMesh->heightStratum(0)->size();
        double   *serialMaxVolumes = new double[numCells];

        for(int c = 0; c < numCells; c++) {
          serialMaxVolumes[c] = maxVolume;
        }
        const Obj<Mesh> refMesh = refineMesh(serialMesh, serialMaxVolumes, interpolate);
        delete [] serialMaxVolumes;
        return refMesh;
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const double maxVolumes[], const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef typename Mesh::real_section_type::value_type real;
        typedef typename Mesh::point_type point_type;
        const int                             dim     = mesh->getDimension();
        const int                             depth   = mesh->depth();
        const Obj<Mesh>                       refMesh = new Mesh(mesh->comm(), dim, mesh->debug());
        const Obj<typename Mesh::sieve_type>& sieve   = mesh->getSieve();
        PetscErrorCode                        ierr;
        ::tetgenio in;
        ::tetgenio out;

        const Obj<typename Mesh::label_sequence>&    vertices    = mesh->depthStratum(0);
        const Obj<typename Mesh::label_type>&        markers     = mesh->getLabel("marker");
        const Obj<typename Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
        const Obj<typename Mesh::numbering_type>&    vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);

        in.numberofpoints = vertices->size();
        if (in.numberofpoints > 0) {
          const typename Mesh::label_sequence::iterator vEnd = vertices->end();

          in.pointlist       = new double[in.numberofpoints*dim];
          in.pointmarkerlist = new int[in.numberofpoints];
          for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const typename Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
            const int                                           idx   = vNumbering->getIndex(*v_iter);

            for(int d = 0; d < dim; d++) {
              in.pointlist[idx*dim + d] = array[d];
            }
            in.pointmarkerlist[idx] = mesh->getValue(markers, *v_iter);
          }
        }
        const Obj<typename Mesh::label_sequence>& cells      = mesh->heightStratum(0);
        const Obj<typename Mesh::numbering_type>& cNumbering = mesh->getFactory()->getLocalNumbering(mesh, depth);

        in.numberofcorners       = 4;
        in.numberoftetrahedra    = cells->size();
        in.tetrahedronvolumelist = (double *) maxVolumes;
        if (in.numberoftetrahedra > 0) {
          in.tetrahedronlist     = new int[in.numberoftetrahedra*in.numberofcorners];
          if (mesh->depth() == 1) {
            ALE::ISieveVisitor::PointRetriever<typename Mesh::sieve_type> pV(4);
            const typename Mesh::label_sequence::iterator cEnd = cells->end();

            for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
              sieve->cone(*c_iter, pV);
              const int         idx  = cNumbering->getIndex(*c_iter);
              const size_t      n    = pV.getSize();
              const point_type *cone = pV.getPoints();

              assert(n == 4);
              for(int v = 0; v < 4; ++v) {
                in.tetrahedronlist[idx*in.numberofcorners + v] = vNumbering->getIndex(cone[v]);
              }
              pV.clear();
            }
          } else if (mesh->depth() == 3) {
            // Need extra space due to early error checking
            ALE::ISieveVisitor::NConeRetriever<typename Mesh::sieve_type> ncV(*sieve, 5);
            const typename Mesh::label_sequence::iterator cEnd = cells->end();

            for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
              ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*sieve, *c_iter, ncV);
              const int         idx  = cNumbering->getIndex(*c_iter);
              const size_t      n    = ncV.getSize();
              const point_type *cone = ncV.getPoints();

              assert(n == 4);
              for(int v = 0; v < 4; ++v) {
                in.tetrahedronlist[idx*in.numberofcorners + v] = vNumbering->getIndex(cone[v]);
              }
              ncV.clear();
            }
          } else {
            throw ALE::Exception("Invalid sieve: Cannot gives sieves of arbitrary depth to TetGen");
          }
        }
        if (depth == 3) {
          const Obj<typename Mesh::label_sequence>&     boundary = mesh->getLabelStratum("marker", 1);
          const typename Mesh::label_sequence::iterator bEnd     = boundary->end();

          in.numberoftrifaces = 0;
          for(typename Mesh::label_sequence::iterator b_iter = boundary->begin(); b_iter != bEnd; ++b_iter) {
            if (mesh->height(*b_iter) == 1) {
              in.numberoftrifaces++;
            }
          }
          if (in.numberoftrifaces > 0) {
            ALE::ISieveVisitor::NConeRetriever<typename Mesh::sieve_type> ncV(*sieve, 5);
            int f = 0;

            in.trifacelist       = new int[in.numberoftrifaces*3];
            in.trifacemarkerlist = new int[in.numberoftrifaces];
            for(typename Mesh::label_sequence::iterator b_iter = boundary->begin(); b_iter != bEnd; ++b_iter) {
              if (mesh->height(*b_iter) == 1) {
                ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*sieve, *b_iter, ncV);
                const size_t      n    = ncV.getSize();
                const point_type *cone = ncV.getPoints();

                for(size_t p = 0; p < n; ++p) {
                  in.trifacelist[f*3 + p] = vNumbering->getIndex(cone[p]);
                }
                in.trifacemarkerlist[f++] = mesh->getValue(markers, *b_iter);
                ncV.clear();
              }
            }
          }
        }
        const typename Mesh::holes_type& holes = mesh->getHoles();

        in.numberofholes = holes.size();
        if (in.numberofholes > 0) {
          ierr = PetscMalloc(in.numberofholes*dim * sizeof(double), &in.holelist);CHKERRXX(ierr);
          for(int h = 0; h < in.numberofholes; ++h) {
            for(int d = 0; d < dim; ++d) {
              in.holelist[h*dim+d] = holes[h][d];
            }
          }
        }
        if (mesh->commRank() == 0) {
          std::string args("qezQra");

          ::tetrahedralize((char *) args.c_str(), &in, &out);
        }
        in.tetrahedronvolumelist = NULL;
        const Obj<typename Mesh::sieve_type> newSieve = new typename Mesh::sieve_type(refMesh->comm(), refMesh->debug());
        const Obj<FlexMesh>                  m        = new FlexMesh(mesh->comm(), dim, mesh->debug());
        const Obj<FlexMesh::sieve_type>      newS     = new FlexMesh::sieve_type(m->comm(), m->debug());
        int     numCorners  = 4;
        int     numCells    = out.numberoftetrahedra;
        int    *newCells    = out.tetrahedronlist;
        int     numVertices = out.numberofpoints;
        double *coords      = out.pointlist;
        real   *coordsR;

        if (!interpolate) {
          for(int c = 0; c < numCells; ++c) {
            int tmp = newCells[c*4+0];
            newCells[c*4+0] = newCells[c*4+1];
            newCells[c*4+1] = tmp;
          }
        }
        ALE::SieveBuilder<FlexMesh>::buildTopology(newS, dim, numCells, newCells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
        m->setSieve(newS);
        m->stratify();
        refMesh->setSieve(newSieve);
        std::map<typename Mesh::point_type,typename Mesh::point_type> renumbering;
        ALE::ISieveConverter::convertSieve(*newS, *newSieve, renumbering, renumber);
        refMesh->stratify();
        ALE::ISieveConverter::convertOrientation(*newS, *newSieve, renumbering, m->getArrowSection("orientation").ptr());
        {
          if (sizeof(double) == sizeof(real)) {
            coordsR = (real *) coords;
          } else {
            coordsR = new real[numVertices*dim];
            for(int i = 0; i < numVertices*dim; ++i) coordsR[i] = coords[i];
          }
        }
        ALE::SieveBuilder<Mesh>::buildCoordinates(refMesh, dim, coordsR);
        {
          if (sizeof(double) != sizeof(real)) {
            delete [] coordsR;
          }
        }
        const Obj<typename Mesh::label_type>& newMarkers = refMesh->createLabel("marker");

        for(int v = 0; v < out.numberofpoints; v++) {
          if (out.pointmarkerlist[v]) {
            if (renumber) {
              refMesh->setValue(newMarkers, renumbering[v+out.numberoftetrahedra], out.pointmarkerlist[v]);
            } else {
              refMesh->setValue(newMarkers, v+out.numberoftetrahedra, out.pointmarkerlist[v]);
            }
          }
        }
        if (interpolate) {
          // This does not work anymore (edgemarkerlist is always empty). I tried -ee and it gave bogus results
          if (out.edgemarkerlist) {
            for(int e = 0; e < out.numberofedges; e++) {
              if (out.edgemarkerlist[e]) {
                typename Mesh::point_type endpointA(out.edgelist[e*2+0]+out.numberoftetrahedra);
                typename Mesh::point_type endpointB(out.edgelist[e*2+1]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> edge = newS->nJoin(endpointA, endpointB, 1);

                if (renumber) {
                  refMesh->setValue(newMarkers, renumbering[*edge->begin()], out.edgemarkerlist[e]);
                } else {
                  refMesh->setValue(newMarkers, *edge->begin(), out.edgemarkerlist[e]);
                }
              }
            }
          }
          if (out.trifacemarkerlist) {
            for(int f = 0; f < out.numberoftrifaces; f++) {
              if (out.trifacemarkerlist[f]) {
                typename Mesh::point_type cornerA(out.trifacelist[f*3+0]+out.numberoftetrahedra);
                typename Mesh::point_type cornerB(out.trifacelist[f*3+1]+out.numberoftetrahedra);
                typename Mesh::point_type cornerC(out.trifacelist[f*3+2]+out.numberoftetrahedra);
                Obj<typename Mesh::sieve_type::supportSet> corners = typename Mesh::sieve_type::supportSet();
                Obj<typename Mesh::sieve_type::supportSet> edges   = typename Mesh::sieve_type::supportSet();
                corners->clear();corners->insert(cornerA);corners->insert(cornerB);
                edges->insert(*newS->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerB);corners->insert(cornerC);
                edges->insert(*newS->nJoin1(corners)->begin());
                corners->clear();corners->insert(cornerC);corners->insert(cornerA);
                edges->insert(*newS->nJoin1(corners)->begin());
                ALE::ISieveVisitor::PointRetriever<typename Mesh::sieve_type> pV(30);
                typename Mesh::point_type face       = *newS->nJoin1(edges)->begin();
                const int                 faceMarker = out.trifacemarkerlist[f];

                if (renumber) {face = renumbering[face];}
                ALE::ISieveTraversal<typename Mesh::sieve_type>::orientedClosure(*newSieve, face, pV);
                const size_t                     n    = pV.getSize();
                const typename Mesh::point_type *cone = pV.getPoints();

                for(size_t c = 0; c < n; ++c) {
                  refMesh->setValue(newMarkers, cone[c], faceMarker);
                }
                pV.clear();
              }
            }
          }
        }
        return refMesh;
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
        throw ALE::Exception("Not yet implemented");
      };
      static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
        const int numCells         = mesh->heightStratum(0)->size();
        double   *serialMaxVolumes = new double[numCells];

        for(int c = 0; c < numCells; c++) {
          serialMaxVolumes[c] = maxVolume;
        }
        const Obj<Mesh> refMesh = refineMeshV(mesh, serialMaxVolumes, interpolate, forceSerial, renumber);
        delete [] serialMaxVolumes;
        return refMesh;
      };
    };
  };
#endif
  template<typename Mesh>
  class Generator {
  public:
    static Obj<Mesh> generateMesh(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false) {
      int dim = boundary->getDimension();

      if (dim == 1) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Generator<Mesh>::generateMesh(boundary, interpolate, constrained);
#else
        throw ALE::Exception("Mesh generation currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 2) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Generator<Mesh>::generateMesh(boundary, interpolate, constrained);
#else
        throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> generateMeshV(const Obj<Mesh>& boundary, const bool interpolate = false, const bool constrained = false, const bool renumber = false) {
      int dim = boundary->getDimension();

      if (dim == 1) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Generator<Mesh>::generateMeshV(boundary, interpolate, constrained, renumber);
#else
        throw ALE::Exception("Mesh generation currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 2) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Generator<Mesh>::generateMeshV(boundary, interpolate, constrained, renumber);
#else
        throw ALE::Exception("Mesh generation currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner<Mesh>::refineMesh(mesh, maxVolumes, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner<Mesh>::refineMesh(mesh, maxVolumes, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMesh(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false, const bool forceSerial = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner<Mesh>::refineMesh(mesh, maxVolume, interpolate, forceSerial);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner<Mesh>::refineMesh(mesh, maxVolume, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const Obj<typename Mesh::real_section_type>& maxVolumes, const bool interpolate = false, const bool renumber = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner<Mesh>::refineMeshV(mesh, maxVolumes, interpolate, renumber);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner<Mesh>::refineMeshV(mesh, maxVolumes, interpolate, renumber);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMeshV(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false, const bool forceSerial = false, const bool renumber = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner<Mesh>::refineMeshV(mesh, maxVolume, interpolate, forceSerial, renumber);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner<Mesh>::refineMeshV(mesh, maxVolume, interpolate, forceSerial, renumber);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
    static Obj<Mesh> refineMeshLocal(const Obj<Mesh>& mesh, const double maxVolume, const bool interpolate = false) {
      int dim = mesh->getDimension();

      if (dim == 2) {
#ifdef PETSC_HAVE_TRIANGLE
        return ALE::Triangle::Refiner<Mesh>::refineMeshLocal(mesh, maxVolume, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires Triangle to be installed. Use --download-triangle during configure.");
#endif
      } else if (dim == 3) {
#ifdef PETSC_HAVE_TETGEN
        return ALE::TetGen::Refiner<Mesh>::refineMesh(mesh, maxVolume, interpolate);
#else
        throw ALE::Exception("Mesh refinement currently requires TetGen to be installed. Use --download-tetgen during configure.");
#endif
      }
      return NULL;
    };
  };
}

#endif
