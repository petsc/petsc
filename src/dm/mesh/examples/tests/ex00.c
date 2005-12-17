static char help[] = "Tests basic ALE memory management and logging.\n\n";

#include <Sifter.hh>
#include <iostream>

#include "petscmesh.h"

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
  ALE::Obj<ALE::def::Sieve<ALE::def::Point,int> > sieve;

  PetscFunctionBegin;
  ALE::def::PointSequence ps(ALE::def::Point(0,-1));
  for(ALE::def::PointSequence::iterator iter = ps.begin(); iter != ps.end(); ++iter) {
    std::cout << *iter << std::endl;
  }

  for(int j = 0; j < 1; ++j) {
    ALE::def::Point base(0, j);

    for(int i = 1; i < 4; ++i) {
      ALE::def::Point point(0, i);

      sieve->addCone(ALE::Obj<ALE::def::const_sequence<ALE::def::Point> >(ALE::def::PointSequence(point)), base);
    }

    ALE::Obj<ALE::def::const_sequence<ALE::def::Point> > cone = sieve->cone(ALE::def::PointSequence(base));
    for(ALE::def::PointSequence::iterator iter = cone->begin(); iter != cone->end(); ++iter) {
      std::cout << *iter << std::endl;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscTruth     memTest, arrowTest, coneTest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  
  MPI_Comm  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for ALE memory management and logging testing", "Mesh");
    memTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-mem_test", "Perform the mem test", "ex0.c", PETSC_TRUE, &memTest, PETSC_NULL);CHKERRQ(ierr);
    arrowTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-arrow_test", "Perform the arrow test", "ex0.c", PETSC_TRUE, &arrowTest, PETSC_NULL);CHKERRQ(ierr);
    coneTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-cone_test", "Perform the cone test", "ex0.c", PETSC_TRUE, &coneTest, PETSC_NULL);CHKERRQ(ierr);
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
    ierr = PetscFinalize();CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << e << std::endl;
  }
  PetscFunctionReturn(0);
}
