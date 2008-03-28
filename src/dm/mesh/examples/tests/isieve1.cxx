#include <petsc.h>
#include <ISieve.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class FunctionTestISieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestISieve);

  CPPUNIT_TEST(testBase);
  CPPUNIT_TEST(testConversion);
  CPPUNIT_TEST(testOrientedClosure);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IFSieve<int> sieve_type;
protected:
  ALE::Obj<sieve_type> _sieve;
  int                  _debug; // The debugging level
  PetscInt             _iters; // The number of test repetitions
  PetscInt             _size;  // The interval size
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
    this->_sieve = new sieve_type(PETSC_COMM_WORLD, 0, this->_size*3+1, this->_debug);
    try {
      ALE::Test::SifterBuilder::createHatISifter<sieve_type>(PETSC_COMM_WORLD, *this->_sieve, this->_size, this->_debug);
    } catch (ALE::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
  };

  /// Tear down data.
  void tearDown(void) {};

  void testBase(void) {
  };

  void testConversion(void) {
    typedef ALE::Mesh::sieve_type Sieve;
    typedef sieve_type            ISieve;
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};
    const ALE::Obj<ALE::Mesh> m = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_sieve, renumbering);
    //m->getSieve()->view("Square Mesh");
    //this->_sieve->view("Square Sieve");
    const ALE::Obj<Sieve::baseSequence>& base = m->getSieve()->base();

    for(Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<Sieve::coneSequence>& cone = m->getSieve()->cone(*b_iter);
      ALE::ISieveVisitor::PointRetriever<ISieve, 2> retriever;

      this->_sieve->cone(renumbering[*b_iter], retriever);
      const ISieve::point_type *icone = retriever.getPoints();
      int i = 0;

      CPPUNIT_ASSERT_EQUAL(cone->size(), retriever.getSize());
      for(Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
        CPPUNIT_ASSERT_EQUAL(renumbering[*c_iter], icone[i]);
      }
    }
    const ALE::Obj<Sieve::capSequence>& cap = m->getSieve()->cap();

    for(Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
      const ALE::Obj<Sieve::supportSequence>& support = m->getSieve()->support(*c_iter);
      ALE::ISieveVisitor::PointRetriever<ISieve, 4> retriever;

      this->_sieve->support(renumbering[*c_iter], retriever);
      const ISieve::point_type *isupport = retriever.getPoints();
      int i = 0;

      CPPUNIT_ASSERT_EQUAL(support->size(), retriever.getSize());
      for(Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
        CPPUNIT_ASSERT_EQUAL(renumbering[*s_iter], isupport[i]);
      }
    }
  };

  void testOrientedClosure() {
    typedef ALE::Mesh::sieve_type                         Sieve;
    typedef ALE::SieveAlg<ALE::Mesh>                      sieve_alg_type;
    typedef sieve_alg_type::orientedConeArray             oConeArray;
    typedef sieve_type                                    ISieve;
    typedef ALE::ISieveVisitor::PointSetRetriever<ISieve> Visitor;
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};
    const ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    const ALE::Obj<ALE::Mesh> m  = ALE::Generator::generateMesh(mB, true);
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_sieve, renumbering);
    const ALE::Obj<Sieve::baseSequence>& base = m->getSieve()->base();

    for(Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<oConeArray>& closure = sieve_alg_type::orientedClosure(m, *b_iter);
      Visitor retriever;

      ALE::ISieveTraversal<ISieve>::orientedClosure(*this->_sieve, renumbering[*b_iter], retriever);
      const Visitor::oriented_points_type&          icone   = retriever.getOrientedPoints();
      Visitor::oriented_points_type::const_iterator ic_iter = icone.begin();

      CPPUNIT_ASSERT_EQUAL(closure->size(), retriever.getOrientedSize());
      std::cout << "Closure of " << *b_iter <<":"<< renumbering[*b_iter] << std::endl;
      for(oConeArray::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter, ++ic_iter) {
        std::cout << "  point " << ic_iter->first << "  " << c_iter->first<<":"<<renumbering[c_iter->first] << std::endl;
        CPPUNIT_ASSERT_EQUAL(renumbering[c_iter->first], ic_iter->first);
        std::cout << "  order " << ic_iter->second << "  " << c_iter->second << std::endl;
        //CPPUNIT_ASSERT_EQUAL(c_iter->second, ic_iter->second);
      }
    }
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterISieveFunctionSuite"
PetscErrorCode RegisterISieveFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestISieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
