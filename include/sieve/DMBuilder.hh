#ifndef included_ALE_DMBuilder_hh
#define included_ALE_DMBuilder_hh

#ifndef  included_ALE_Mesh_hh
#include <sieve/Mesh.hh>
#endif

#include <petscdmmesh.hh>

namespace ALE {

  class DMBuilder {
  public:
    #undef __FUNCT__
    #define __FUNCT__ "createBasketMesh"
    static PetscErrorCode createBasketMesh(MPI_Comm comm, const int dim, const bool structured, const bool interpolate, const int debug, DM *dm) {
      typedef PETSC_MESH_TYPE::real_section_type::value_type real;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (structured) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Structured grids cannot handle boundary meshes");
      } else {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef PETSC_MESH_TYPE::point_type point_type;
        DM boundary;

        ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
        ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
        Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
        Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
        std::map<point_type,point_type>  renumbering;
        Obj<FlexMesh>                    mB;

        meshBd->setSieve(sieve);
        if (dim == 2) {
          real lower[2] = {0.0, 0.0};
          real upper[2] = {1.0, 1.0};
          int  edges    = 2;

          mB = ALE::MeshBuilder<FlexMesh>::createSquareBoundary(comm, lower, upper, edges, debug);
        } else if (dim == 3) {
          real lower[3] = {0.0, 0.0, 0.0};
          real upper[3] = {1.0, 1.0, 1.0};
          int  faces[3] = {3, 3, 3};

          mB = ALE::MeshBuilder<FlexMesh>::createCubeBoundary(comm, lower, upper, faces, debug);
        } else {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
        ierr = DMMeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
        *dm = boundary;
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "createBoxMesh"
    static PetscErrorCode createBoxMesh(MPI_Comm comm, const int dim, const bool structured, const bool interpolate, const int debug, DM *dm) {
      typedef PETSC_MESH_TYPE::real_section_type::value_type real;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (structured) {
        DM             da;
        const PetscInt dof = 1;
        const PetscInt pd  = PETSC_DECIDE;

        if (dim == 2) {
          ierr = DMDACreate2d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
        } else if (dim == 3) {
          ierr = DMDACreate3d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, -3, pd, pd, pd, dof, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
        } else {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
        *dm = da;
      } else {
        typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
        typedef PETSC_MESH_TYPE::point_type point_type;
        DM mesh;
        DM boundary;

        ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
        ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
        Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
        Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
        std::map<point_type,point_type>  renumbering;
        Obj<FlexMesh>                    mB;

        meshBd->setSieve(sieve);
        if (dim == 2) {
          real lower[2] = {0.0, 0.0};
          real upper[2] = {1.0, 1.0};
          int  edges[2] = {2, 2};

          mB = ALE::MeshBuilder<FlexMesh>::createSquareBoundary(comm, lower, upper, edges, debug);
        } else if (dim == 3) {
          real lower[3] = {0.0, 0.0, 0.0};
          real upper[3] = {1.0, 1.0, 1.0};
          int  faces[3] = {3, 3, 3};

          mB = ALE::MeshBuilder<FlexMesh>::createCubeBoundary(comm, lower, upper, faces, debug);
        } else {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
        }
        ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
        ierr = DMMeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
        ierr = DMMeshGenerate(boundary, (PetscBool) interpolate, &mesh);CHKERRQ(ierr);
        ierr = DMDestroy(&boundary);CHKERRQ(ierr);
        *dm = mesh;
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "createReentrantBoxMesh"
    static PetscErrorCode createReentrantBoxMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      typedef PETSC_MESH_TYPE::point_type point_type;
      typedef PETSC_MESH_TYPE::real_section_type::value_type real;
      DM         mesh;
      DM         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
      ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<FlexMesh>                    mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        real lower[2]  = {-1.0, -1.0};
        real upper[2]  = {1.0, 1.0};
        real offset[2] = {0.5, 0.5};

        mB = ALE::MeshBuilder<FlexMesh>::createReentrantBoundary(comm, lower, upper, offset, debug);
      } else if (dim == 3) {
        real lower[3]  = {-1.0, -1.0, -1.0};
        real upper[3]  = { 1.0,  1.0,  1.0};
        real offset[3] = { 0.5,  0.5,  0.5};

        mB = ALE::MeshBuilder<FlexMesh>::createFicheraCornerBoundary(comm, lower, upper, offset, debug);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = DMMeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = DMMeshGenerate(boundary, (PetscBool) interpolate, &mesh);CHKERRQ(ierr);
      ierr = DMDestroy(&boundary);CHKERRQ(ierr);
      *dm = mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "createSphericalMesh"
    static PetscErrorCode createSphericalMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      typedef PETSC_MESH_TYPE::point_type point_type;
      DM         mesh;
      DM         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
      ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<FlexMesh>                    mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        mB = ALE::MeshBuilder<FlexMesh>::createCircularReentrantBoundary(comm, 100, 1.0, 1.0, debug);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = DMMeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = DMMeshGenerate(boundary, (PetscBool) interpolate, &mesh);CHKERRQ(ierr);
      ierr = DMDestroy(&boundary);CHKERRQ(ierr);
      *dm = mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "createReentrantSphericalMesh"
    static PetscErrorCode createReentrantSphericalMesh(MPI_Comm comm, const int dim, const bool interpolate, const int debug, DM *dm) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      typedef PETSC_MESH_TYPE::point_type point_type;
      DM         mesh;
      DM         boundary;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
      ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
      Obj<PETSC_MESH_TYPE>             meshBd = new PETSC_MESH_TYPE(comm, dim-1, debug);
      Obj<PETSC_MESH_TYPE::sieve_type> sieve  = new PETSC_MESH_TYPE::sieve_type(comm, debug);
      std::map<point_type,point_type>  renumbering;
      Obj<FlexMesh>                    mB;

      meshBd->setSieve(sieve);
      if (dim == 2) {
        mB = ALE::MeshBuilder<FlexMesh>::createCircularReentrantBoundary(comm, 100, 1.0, 0.9, debug);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ALE::ISieveConverter::convertMesh(*mB, *meshBd, renumbering, false);
      ierr = DMMeshSetMesh(boundary, meshBd);CHKERRQ(ierr);
      ierr = DMMeshGenerate(boundary, (PetscBool) interpolate, &mesh);CHKERRQ(ierr);
      ierr = DMDestroy(&boundary);CHKERRQ(ierr);
      *dm = mesh;
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__
    #define __FUNCT__ "MeshRefineSingularity"
    static PetscErrorCode MeshRefineSingularity(DM mesh, double * singularity, double factor, DM *refinedMesh) {
      typedef PETSC_MESH_TYPE::real_section_type::value_type real;
      ALE::Obj<PETSC_MESH_TYPE> oldMesh;
      double              oldLimit;
      PetscErrorCode      ierr;

      PetscFunctionBegin;
      ierr = DMMeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
      ierr = DMCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
      ierr = DMSetType(*refinedMesh, DMMESH);CHKERRQ(ierr);
      int dim = oldMesh->getDimension();
      oldLimit = oldMesh->getMaxVolume();
      //double oldLimInv = 1./oldLimit;
      real curLimit, tmpLimit;
      real minLimit = oldLimit/16384.;             //arbitrary;
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = oldMesh->getRealSection("coordinates");
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& volume_limits = oldMesh->getRealSection("volume_limits");
      volume_limits->setFiberDimension(oldMesh->heightStratum(0), 1);
      oldMesh->allocate(volume_limits);
      const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& cells = oldMesh->heightStratum(0);
      PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
      PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
      real centerCoords[dim];
      while (c_iter != c_iter_end) {
        const real * coords = oldMesh->restrictClosure(coordinates, *c_iter);
        for (int i = 0; i < dim; i++) {
          centerCoords[i] = 0;
          for (int j = 0; j < dim+1; j++) {
            centerCoords[i] += coords[j*dim+i];
          }
          centerCoords[i] = centerCoords[i]/(dim+1);
        }
        real dist = 0.;
        for (int i = 0; i < dim; i++) {
          dist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
        }
        if (dist > 0.) {
          dist = sqrt(dist);
          real mu = pow(dist, factor);
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
      ierr = DMMeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
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
    static PetscErrorCode MeshRefineSingularity_Fichera(DM mesh, double * singularity, double factor, DM *refinedMesh) {
      typedef PETSC_MESH_TYPE::real_section_type::value_type real;
      ALE::Obj<PETSC_MESH_TYPE> oldMesh;
      real                      oldLimit;
      PetscErrorCode            ierr;

      PetscFunctionBegin;
      ierr = DMMeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
      ierr = DMCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
      ierr = DMSetType(*refinedMesh, DMMESH);CHKERRQ(ierr);
      int dim = oldMesh->getDimension();
      oldLimit = oldMesh->getMaxVolume();
      //double oldLimInv = 1./oldLimit;
      real curLimit, tmpLimit;
      real minLimit = oldLimit/16384.;             //arbitrary;
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = oldMesh->getRealSection("coordinates");
      const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& volume_limits = oldMesh->getRealSection("volume_limits");
      volume_limits->setFiberDimension(oldMesh->heightStratum(0), 1);
      oldMesh->allocate(volume_limits);
      const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& cells = oldMesh->heightStratum(0);
      PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin();
      PETSC_MESH_TYPE::label_sequence::iterator c_iter_end = cells->end();
      real centerCoords[dim];
      while (c_iter != c_iter_end) {
        const real *coords = oldMesh->restrictClosure(coordinates, *c_iter);
        for (int i = 0; i < dim; i++) {
          centerCoords[i] = 0;
          for (int j = 0; j < dim+1; j++) {
            centerCoords[i] += coords[j*dim+i];
          }
          centerCoords[i] = centerCoords[i]/(dim+1);
          //PetscPrintf(oldMesh->comm(), "%f, ", centerCoords[i]);
        }
        //PetscPrintf(oldMesh->comm(), "\n");
        real dist = 0.;
        real cornerdist = 0.;
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
        real singdist = 0.;
        for (int i = 0; i < dim; i++) {
          singdist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
        }
        if (singdist < dist || dist == 0.) dist = singdist;
        if (dist > 0.) {
          dist = sqrt(dist);
          real mu = pow(dist, factor);
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
      ierr = DMMeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
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
