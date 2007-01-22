static char help[] = "Sifter Basic Ordering Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE::Test::XSifter::arrow_type       arrow_type;
typedef arrow_type::source_type              source_type;
typedef arrow_type::target_type              target_type;
typedef arrow_type::color_type               color_type;

typedef ALE::Test::XSifter::xsifter_type     xsifter_type;

typedef xsifter_type::arrow_rec_type         arrow_rec_type;
typedef xsifter_type::predicate_type         predicate_type;


#undef __FUNCT__
#define __FUNCT__ "BasicTest"
PetscErrorCode BasicTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Test::XSifter::Options options, const char* xsifterName = NULL)
{

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Basic Test");
  ALE::LogStagePush(stage);
  xsifter->view(std::cout, xsifterName);
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
  {
    ALE::Test::XSifter::Options        options;
    ALE::Obj<xsifter_type> xsifterFork = ALE::Test::XSifter::createForkXSifter(PETSC_COMM_SELF, options);
    ierr = BasicTest(xsifterFork, options, "Fork"); CHKERRQ(ierr);
    ALE::Obj<xsifter_type> xsifterHat = ALE::Test::XSifter::createHatXSifter(PETSC_COMM_SELF, options);
    ierr = BasicTest(xsifterHat, options, "Hat"); CHKERRQ(ierr);
//     ////
//     std::cout << std::endl << "Testing upward_order:" << std::endl;
//     std::less<predicate_type> p_less;
//     upward_order_type         less;
//     //
//     source_type    s0(0);
//     target_type    t0(0);
//     color_type     c0('X');
//     predicate_type p0(0);
//     arrow_rec_type       r0(arrow_type(s0,t0,c0),p0);
//     //
//     source_type    s1(1);
//     target_type    t1(1);
//     color_type     c1('Y');
//     predicate_type p1(1);
//     arrow_rec_type       r1(arrow_type(s1,t1,c1),p1);
//     //
//     //
//     if(p_less(p0,p0)) {
//       std::cout << p0 << "  < " << p0 << std::endl;
//     }
//     else {
//       std::cout << p0 << " !< " << p0 << std::endl;
//     }
//     //
//     //
//     if(less(r0,r0)) {
//       std::cout << r0 << "  < " << r0 << std::endl;
//     }
//     else {
//       std::cout << r0 << " !< " << r0 << std::endl;
//     }
//     //
//     if(less(r0,ALE::singleton<predicate_type>(p0))) {
//       std::cout << r0 << "  < " << p0 << std::endl;
//     }
//     else {
//       std::cout << r0 << " !< " << p0 << std::endl;
//     }
//     if(less(ALE::singleton<predicate_type>(p0),r0)) {
//       std::cout << p0 << "  < " << r0 << std::endl;
//     }
//     else {
//       std::cout << p0 << " !< " << r0 << std::endl;
//     }
//     //
//     if(less(r0,ALE::pair<predicate_type,target_type>(p0,t0))) {
//       std::cout << r0 << "  < " << ALE::pair<predicate_type,target_type>(p0,t0) << std::endl;
//     }
//     else {
//       std::cout << r0 << " !< " << ALE::pair<predicate_type,target_type>(p0,t0) << std::endl;
//     }
//     if(less(ALE::pair<predicate_type,target_type>(p0,t0) ,r0)) {
//       std::cout << ALE::pair<predicate_type,target_type>(p0,t0)  << "  < " << r0 << std::endl;
//     }
//     else {
//       std::cout << ALE::pair<predicate_type,target_type>(p0,t0)  << " !< " << r0 << std::endl;
//     }
//     //
//     //
//     if(less(r0,r1)) {
//       std::cout << r0 << "  < " << r1 << std::endl;
//     }
//     else {
//       std::cout << r0 << " !< " << r1 << std::endl;
//     }
//     if(less(r1,r0)) {
//       std::cout << r1 << "  < " << r0 << std::endl;
//     }
//     else {
//       std::cout << r1 << " !< "  << r0 << std::endl;
//     }
//    
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
