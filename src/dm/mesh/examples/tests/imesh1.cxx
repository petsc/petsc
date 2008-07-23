#include <petsc.h>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class FunctionTestIMesh : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestIMesh);

  CPPUNIT_TEST(testStratify);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh            mesh_type;
  typedef mesh_type::point_type point_type;
protected:
  ALE::Obj<mesh_type> _mesh;
  int                 _debug; // The debugging level
  PetscInt            _iters; // The number of test repetitions
  PetscInt            _size;  // The interval size
  ALE::Obj<ALE::Mesh>             _m;
  std::map<point_type,point_type> _renumbering;
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;
    this->_size  = 10;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "isection.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    double                    lower[3] = {0.0, 0.0, 0.0};
    double                    upper[3] = {1.0, 1.0, 1.0};
    int                       faces[3] = {3, 3, 3};
    const ALE::Obj<ALE::Mesh> mB       = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
    this->_m    = ALE::Generator<ALE::Mesh>::generateMesh(mB, true);
    this->_mesh = new mesh_type(PETSC_COMM_WORLD, 1, this->_debug);
    ALE::Obj<mesh_type::sieve_type> sieve = new mesh_type::sieve_type(PETSC_COMM_WORLD, 0, 119, this->_debug);

    this->_mesh->setSieve(sieve);
    ALE::ISieveConverter::convertSieve(*this->_m->getSieve(), *this->_mesh->getSieve(), this->_renumbering);
  };

  /// Tear down data.
  void tearDown(void) {};

  void testStratify() {
    this->_mesh->stratify();
    const ALE::Obj<ALE::Mesh::label_type>& h1 = this->_m->getLabel("height");
    const ALE::Obj<mesh_type::label_type>& h2 = this->_mesh->getLabel("height");

    for(int h = 0; h < 4; ++h) {
      CPPUNIT_ASSERT_EQUAL(h1->support(h)->size(), h2->support(h)->size());
      const ALE::Obj<ALE::Mesh::label_sequence>& points1 = h1->support(h);
      const ALE::Obj<mesh_type::label_sequence>& points2 = h2->support(h);
      ALE::Mesh::label_sequence::iterator        p_iter1 = points1->begin();
      mesh_type::label_sequence::iterator        p_iter2 = points2->begin();
      ALE::Mesh::label_sequence::iterator        end1    = points1->end();

      while(p_iter1 != end1) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[*p_iter1], *p_iter2);
        ++p_iter1;
        ++p_iter2;
      }
    }
    const ALE::Obj<ALE::Mesh::label_type>& d1 = this->_m->getLabel("depth");
    const ALE::Obj<mesh_type::label_type>& d2 = this->_mesh->getLabel("depth");

    for(int d = 0; d < 4; ++d) {
      CPPUNIT_ASSERT_EQUAL(d1->support(d)->size(), d2->support(d)->size());
      const ALE::Obj<ALE::Mesh::label_sequence>& points1 = d1->support(d);
      const ALE::Obj<mesh_type::label_sequence>& points2 = d2->support(d);
      ALE::Mesh::label_sequence::iterator        p_iter1 = points1->begin();
      mesh_type::label_sequence::iterator        p_iter2 = points2->begin();
      ALE::Mesh::label_sequence::iterator        end1    = points1->end();

      while(p_iter1 != end1) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[*p_iter1], *p_iter2);
        ++p_iter1;
        ++p_iter2;
      }
    }
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIMeshFunctionSuite"
PetscErrorCode RegisterIMeshFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestIMesh);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
