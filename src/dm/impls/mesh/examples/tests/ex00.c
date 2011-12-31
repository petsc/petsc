static char help[] = "Tests basic ALE memory management and logging.\n\n";

#include <Sifter.hh>
#include <iostream>

#include <petscmesh.h>

#undef __FUNCT__
#define __FUNCT__ "MemTest"
PetscErrorCode MemTest()
{
  ALE::Obj<ALE::Point_set> pointset;
  ALE::Obj<ALE::PreSieve>  presieve;
  ALE::Obj<ALE::Sieve>     sieve;
  ALE::Obj<ALE::Point>     point;
  ALE::Obj<ALE::Point>     point2(ALE::Point(0, 1));

  PetscFunctionBegin;
  point.create(ALE::Point(0, 0));
  PetscFunctionReturn(0);
}

struct source{};
struct target{};
struct color{};
#undef __FUNCT__
#define __FUNCT__ "ArrowTest"
PetscErrorCode ArrowTest()
{
  typedef ALE::def::Point Point;
  typedef ALE::def::Arrow<int> SieveArrow;
  typedef ::boost::multi_index::multi_index_container<
    SieveArrow,
    ::boost::multi_index::indexed_by<
      ::boost::multi_index::ordered_non_unique<
        ::boost::multi_index::tag<source>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,source)>,
      ::boost::multi_index::ordered_non_unique<
        ::boost::multi_index::tag<target>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,target)>,
      ::boost::multi_index::ordered_non_unique<
        ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,int,color)>
      >
    > ArrowSet;
  ArrowSet arrows;

  PetscFunctionBegin;
  arrows.insert(SieveArrow(Point(0,1), Point(0,0), 0));
  arrows.insert(SieveArrow(Point(0,2), Point(0,0), 0));
  arrows.insert(SieveArrow(Point(0,3), Point(0,0), 0));
  arrows.insert(SieveArrow(Point(1,1), Point(1,0), 0));
  arrows.insert(SieveArrow(Point(1,2), Point(1,0), 0));
  arrows.insert(SieveArrow(Point(1,3), Point(1,0), 0));
  ArrowSet::index<target>::type& i = arrows.get<target>();
  std::copy(i.begin(), i.end(), std::ostream_iterator<SieveArrow>(std::cout));
  std::copy(i.lower_bound(Point(1,0)), i.upper_bound(Point(1,0)), std::ostream_iterator<SieveArrow>(std::cout));

  Point base(0,0);
  for(int i = 4; i < 10000; ++i) {
    arrows.insert(SieveArrow(Point(0, i), base, 0));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConeTest"
PetscErrorCode ConeTest()
{
  ALE::Obj<ALE::def::Sieve<ALE::def::Point,int> > sieve = ALE::def::Sieve<ALE::def::Point,int>();

  PetscFunctionBegin;
  // Test cone of a point
  for(int j = 0; j < 1; ++j) {
    ALE::def::Point base(0, j);
    ALE::Obj<std::set<ALE::def::Point> > inCone = std::set<ALE::def::Point>();

    for(int i = 1; i < 5000; ++i) {
      inCone->insert(ALE::def::Point(0, i));
    }
    sieve->addCone(inCone, base);

    ALE::Obj<ALE::def::Sieve<ALE::def::Point,int>::coneSequence> outCone = sieve->cone(base);
    for(ALE::def::Sieve<ALE::def::Point,int>::coneSequence::iterator iter = outCone->begin(); iter != outCone->end(); ++iter) {
      std::cout << *iter << std::endl;
    }
  }
  sieve->clear();
  // Test colored cone of a point
  for(int j = 0; j < 1; ++j) {
    ALE::def::Point base(0, j);
    ALE::Obj<std::set<ALE::def::Point> > inCone = std::set<ALE::def::Point>();

    for(int c = 1; c < 3; ++c) {
      for(int i = 1; i < 4; ++i) {
        inCone->insert(ALE::def::Point(0, i));
      }
      sieve->addCone(inCone, base, c);
      inCone->clear();
    }

    for(int c = 1; c < 3; ++c) {
      ALE::Obj<ALE::def::Sieve<ALE::def::Point,int>::coneSequence> outCone = sieve->cone(base, c);
      for(ALE::def::Sieve<ALE::def::Point,int>::coneSequence::iterator iter = outCone->begin(); iter != outCone->end(); ++iter) {
        std::cout << "Color " << c << ": " << *iter << std::endl;
      }
    }
  }
  sieve->clear();
  // Test cone of a point set
  for(int j = 1; j < 6; ++j) {
    ALE::def::Point base(0, j);
    ALE::Obj<std::set<ALE::def::Point> > inCone = std::set<ALE::def::Point>();

    for(int i = 1; i < 4; ++i) {
      inCone->insert(ALE::def::Point(j, i));
    }
    sieve->addCone(inCone, base);
  }

  ALE::Obj<ALE::def::PointSet> outCone = sieve->cone(sieve->base());
  for(ALE::def::PointSet::iterator iter = outCone->begin(); iter != outCone->end(); ++iter) {
    std::cout << *iter << std::endl;
  }
  sieve->clear();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscBool      memTest, arrowTest, coneTest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  
  MPI_Comm  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for ALE memory management and logging testing", "Mesh");
    memTest = PETSC_TRUE;
    ierr = PetscOptionsBool("-mem_test", "Perform the mem test", "ex0.c", PETSC_TRUE, &memTest, PETSC_NULL);CHKERRQ(ierr);
    arrowTest = PETSC_TRUE;
    ierr = PetscOptionsBool("-arrow_test", "Perform the arrow test", "ex0.c", PETSC_TRUE, &arrowTest, PETSC_NULL);CHKERRQ(ierr);
    coneTest = PETSC_TRUE;
    ierr = PetscOptionsBool("-cone_test", "Perform the cone test", "ex0.c", PETSC_TRUE, &coneTest, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  try {
    if (memTest) {
      ierr = MemTest();CHKERRQ(ierr);
    }
    if (arrowTest) {
      ierr = ArrowTest();CHKERRQ(ierr);
    }
    if (coneTest) {
      ierr = ConeTest();CHKERRQ(ierr);
    }
    ierr = PetscFinalize();
  } catch(ALE::Exception e) {
    std::cout << e << std::endl;
  }
  PetscFunctionReturn(0);
}
