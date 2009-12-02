#ifndef included_ALE_DMBuilder_hh
#define included_ALE_DMBuilder_hh

#ifndef  included_ALE_Mesh_hh
#include <Mesh.hh>
#endif

#include <petscmesh.hh>

namespace ALE {

  class DMBuilder {
  public:
    #undef __FUNCT__
    #define __FUNCT__ "CreateBasketMesh"
    static PetscErrorCode createBasketMesh(MPI_Comm comm, const int dim, const bool structured, const bool interpolate, const int debug, DM *dm) {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (structured) {
        SETERRQ(PETSC_ERR_SUP, "Structured grids cannot handle boundary meshes");
      } else {
        typedef PETSC_MESH_TYPE::point_type point_type;
        ::Mesh boundary;

        ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
        Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
        Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
        std::map<point_type,point_type>  renumbering;
        Obj<ALE::Mesh>                   mB;

        meshBd->setSieve(sieve);
        if (dim == 2) {
          double lower[2] = {0.0, 0.0};
          double upper[2] = {1.0, 1.0};
          int    edges    = 2;

          mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(comm, lower, upper, edges, debug);
        } else if (dim == 3) {
          double lower[3] = {0.0, 0.0, 0.0};
          double upper[3] = {1.0, 1.0, 1.0};
          int    faces[3] = {3, 3, 3};
                
          mB = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(comm, lower, upper, faces, debug);
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
        ierr = MeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
        *dm = (DM) boundary;
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "CreateBoxMesh"
    static PetscErrorCode createBoxMesh(MPI_Comm comm, const int dim, const bool structured, const bool interpolate, const int debug, DM *dm) {
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (structured) {
        DA             da;
        const PetscInt dof = 1;
        const PetscInt pd  = PETSC_DECIDE;

        if (dim == 2) {
          ierr = DACreate2d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
        } else if (dim == 3) {
          ierr = DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, -3, pd, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
        *dm = (DM) da;
      } else {
        typedef PETSC_MESH_TYPE::point_type point_type;
        ::Mesh mesh;
        ::Mesh boundary;

        ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
        Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
        Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
        std::map<point_type,point_type>  renumbering;
        Obj<ALE::Mesh>                   mB;

        meshBd->setSieve(sieve);
        if (dim == 2) {
          double lower[2] = {0.0, 0.0};
          double upper[2] = {1.0, 1.0};
          int    edges[2] = {2, 2};

          mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(comm, lower, upper, edges, debug);
        } else if (dim == 3) {
          double lower[3] = {0.0, 0.0, 0.0};
          double upper[3] = {1.0, 1.0, 1.0};
          int    faces[3] = {3, 3, 3};
                
          mB = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(comm, lower, upper, faces, debug);
        } else {
          SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
        ierr = MeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
        ierr = MeshGenerate(boundary, (PetscTruth) interpolate, &mesh);CHKERRQ(ierr);
        ierr = MeshDestroy(boundary);CHKERRQ(ierr);
        *dm = (DM) mesh;
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "CreateReentrantBoxMesh"
    static PetscErrorCode createReentrantBoxMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef PETSC_MESH_TYPE::point_type point_type;
      ::Mesh         mesh;
      ::Mesh         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<ALE::Mesh>                   mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        double lower[2]  = {-1.0, -1.0};
        double upper[2]  = {1.0, 1.0};
        double offset[2] = {0.5, 0.5};

        mB = ALE::MeshBuilder<ALE::Mesh>::createReentrantBoundary(comm, lower, upper, offset, debug);
      } else if (dim == 3) {
        double lower[3]  = {-1.0, -1.0, -1.0};
        double upper[3]  = { 1.0,  1.0,  1.0};
        double offset[3] = { 0.5,  0.5,  0.5};

        mB = ALE::MeshBuilder<ALE::Mesh>::createFicheraCornerBoundary(comm, lower, upper, offset, debug);
      } else {
        SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = MeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = MeshGenerate(boundary, (PetscTruth) interpolate, &mesh);CHKERRQ(ierr);
      ierr = MeshDestroy(boundary);CHKERRQ(ierr);
      *dm = (DM) mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "CreateSphericalMesh"
    static PetscErrorCode createSphericalMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef PETSC_MESH_TYPE::point_type point_type;
      ::Mesh         mesh;
      ::Mesh         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<ALE::Mesh>                   mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        mB = ALE::MeshBuilder<ALE::Mesh>::createCircularReentrantBoundary(comm, 100, 1.0, 1.0, debug);
      } else {
        SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = MeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = MeshGenerate(boundary, (PetscTruth) interpolate, &mesh);CHKERRQ(ierr);
      ierr = MeshDestroy(boundary);CHKERRQ(ierr);
      *dm = (DM) mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "CreateReentrantSphericalMesh"
    static PetscErrorCode createReentrantSphericalMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef PETSC_MESH_TYPE::point_type point_type;
      ::Mesh         mesh;
      ::Mesh         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<ALE::Mesh>                   mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        mB = ALE::MeshBuilder<ALE::Mesh>::createCircularReentrantBoundary(comm, 100, 1.0, 0.9, debug);
      } else {
        SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = MeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = MeshGenerate(boundary, (PetscTruth) interpolate, &mesh);CHKERRQ(ierr);
      ierr = MeshDestroy(boundary);CHKERRQ(ierr);
      *dm = (DM) mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "MeshRefineSingularity"
    static PetscErrorCode MeshRefineSingularity(::Mesh mesh, double * singularity, double factor, ::Mesh *refinedMesh) {
      ALE::Obj<PETSC_MESH_TYPE> oldMesh;
      double              oldLimit;
      PetscErrorCode      ierr;

      PetscFunctionBegin;
      ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
      ierr = MeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
      int dim = oldMesh->getDimension();
      oldLimit = oldMesh->getMaxVolume();
      //double oldLimInv = 1./oldLimit;
      double curLimit, tmpLimit;
      double minLimit = oldLimit/16384.;             //arbitrary;
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = oldMesh->getRealSection("coordinates");
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& volume_limits = oldMesh->getRealSection("volume_limits");
      volume_limits->setFiberDimension(oldMesh->heightStratum(0), 1);
      oldMesh->allocate(volume_limits);
      const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& cells = oldMesh->heightStratum(0);
      PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
      PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
      double centerCoords[dim];
      while (c_iter != c_iter_end) {
        const double * coords = oldMesh->restrictClosure(coordinates, *c_iter);
        for (int i = 0; i < dim; i++) {
          centerCoords[i] = 0;
          for (int j = 0; j < dim+1; j++) {
            centerCoords[i] += coords[j*dim+i];
          }
          centerCoords[i] = centerCoords[i]/(dim+1);
        }
        double dist = 0.;
        for (int i = 0; i < dim; i++) {
          dist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
        }
        if (dist > 0.) {
          dist = sqrt(dist);
          double mu = pow(dist, factor);
          //PetscPrintf(oldMesh->comm(), "%f\n", mu);
          tmpLimit = oldLimit*pow(mu, dim);
          if (tmpLimit > minLimit) {
            curLimit = tmpLimit;
          } else curLimit = minLimit;
        } else curLimit = minLimit;
        //PetscPrintf(oldMesh->comm(), "%f, %f\n", dist, tmpLimit);
        volume_limits->updatePoint(*c_iter, &curLimit);
        c_iter++;
      }
#ifdef PETSC_OPT_SIEVE
      ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, volume_limits, true);
#else
      ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMesh(oldMesh, volume_limits, true);
#endif
      ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = newMesh->getRealSection("default");
      const Obj<std::set<std::string> >& discs = oldMesh->getDiscretizations();

      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
        newMesh->setDiscretization(*f_iter, oldMesh->getDiscretization(*f_iter));
      }
      newMesh->setupField(s);
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "MeshRefineSingularity_Fichera"
    static PetscErrorCode MeshRefineSingularity_Fichera(::Mesh mesh, double * singularity, double factor, ::Mesh *refinedMesh) {
      ALE::Obj<PETSC_MESH_TYPE> oldMesh;
      double              oldLimit;
      PetscErrorCode      ierr;

      PetscFunctionBegin;
      ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
      ierr = MeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
      int dim = oldMesh->getDimension();
      oldLimit = oldMesh->getMaxVolume();
      //double oldLimInv = 1./oldLimit;
      double curLimit, tmpLimit;
      double minLimit = oldLimit/16384.;             //arbitrary;
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = oldMesh->getRealSection("coordinates");
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& volume_limits = oldMesh->getRealSection("volume_limits");
      volume_limits->setFiberDimension(oldMesh->heightStratum(0), 1);
      oldMesh->allocate(volume_limits);
      const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& cells = oldMesh->heightStratum(0);
      PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
      PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
      double centerCoords[dim];
      while (c_iter != c_iter_end) {
        const double * coords = oldMesh->restrictClosure(coordinates, *c_iter);
        for (int i = 0; i < dim; i++) {
          centerCoords[i] = 0;
          for (int j = 0; j < dim+1; j++) {
            centerCoords[i] += coords[j*dim+i];
          }
          centerCoords[i] = centerCoords[i]/(dim+1);
          //PetscPrintf(oldMesh->comm(), "%f, ", centerCoords[i]);
        }
        //PetscPrintf(oldMesh->comm(), "\n");
        double dist = 0.;
        double cornerdist = 0.;
        //HERE'S THE DIFFERENCE: if centercoords is less than the singularity coordinate for each direction, include that direction in the distance
        /*
          for (int i = 0; i < dim; i++) {
          if (centerCoords[i] <= singularity[i]) {
          dist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
          }
          }
        */
        //determine: the per-dimension distance: cases
        for (int i = 0; i < dim; i++) {
          cornerdist = 0.;
          if (centerCoords[i] > singularity[i]) {
            for (int j = 0; j < dim; j++) {
              if (j != i) cornerdist += (centerCoords[j] - singularity[j])*(centerCoords[j] - singularity[j]);
            }
            if (cornerdist < dist || dist == 0.) dist = cornerdist; 
          }
        }
        //patch up AROUND the corner by minimizing between the distance from the relevant axis and the singular vertex
        double singdist = 0.;
        for (int i = 0; i < dim; i++) {
          singdist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
        }
        if (singdist < dist || dist == 0.) dist = singdist;
        if (dist > 0.) {
          dist = sqrt(dist);
          double mu = pow(dist, factor);
          //PetscPrintf(oldMesh->comm(), "%f, %f\n", mu, dist);
          tmpLimit = oldLimit*pow(mu, dim);
          if (tmpLimit > minLimit) {
            curLimit = tmpLimit;
          } else curLimit = minLimit;
        } else curLimit = minLimit;
        //PetscPrintf(oldMesh->comm(), "%f, %f\n", dist, tmpLimit);
        volume_limits->updatePoint(*c_iter, &curLimit);
        c_iter++;
      }
#ifdef PETSC_OPT_SIEVE
      ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, volume_limits, true);
#else
      ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMesh(oldMesh, volume_limits, true);
#endif
      ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = newMesh->getRealSection("default");
      const Obj<std::set<std::string> >& discs = oldMesh->getDiscretizations();

      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
        newMesh->setDiscretization(*f_iter, oldMesh->getDiscretization(*f_iter));
      }
      newMesh->setupField(s);
      //  PetscPrintf(newMesh->comm(), "refined\n");
      PetscFunctionReturn(0);
    };
  };
}

#endif
