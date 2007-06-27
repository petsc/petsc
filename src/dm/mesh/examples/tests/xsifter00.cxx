#include <petsc.h>
#include <XSifter.hh>




typedef ALE::XSifterDef::Arrow<int,char,int>       arrow_type;
typedef ALE::XSifter<arrow_type>                   xsifter_type;
typedef arrow_type::source_type                    source_type;
typedef arrow_type::target_type                    target_type;
typedef arrow_type::color_type                     color_type;


typedef xsifter_type::rec_type               rec_type;
typedef xsifter_type::predicate_type         predicate_type;


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, NULL); CHKERRQ(ierr);
  ALE::Obj<xsifter_type> xsifter = new xsifter_type(PETSC_COMM_WORLD, 0);
  // Insert arrows
  int N = 3;
  for(int i = 0; i <= 3*N; i++) {
    xsifter->addArrow(arrow_type(i,'A',i));
  }
  for(int i = 0; i < 10; i++) {
    xsifter->addArrow(arrow_type(i,'B', i));
  }
  for(int i = 0; i < 10; i++) {
    xsifter->addArrow(arrow_type(2*i,'C', i));
  }
  for(int i = 0; i < 10; i++) {
    xsifter->addArrow(arrow_type(2*i,'D', i));
  }
  // View the sifter
  xsifter->view(std::cout, "Raw");
  
  // Retrieve the raw cone index; prepare key extractors
  typedef xsifter_type::cone_index_type        index_type;
  index_type& ind = xsifter->_cone_index;
  
  // View the index tree 
  ind.view_tree(std::cout);
  
  typedef xsifter_type::cone_order_type        order_type;
  typedef order_type::pre_extractor_type       preex_type;
  typedef order_type::pos_extractor_type       posex_type;
  
  {//test1
    target_type low, high;
    source_type s;
    
    
    s = 1; low = 'A'; high = 'B';
    PetscTruth flag;
    char highstr[2], lowstr[2];
    ierr = PetscOptionsGetInt(PETSC_NULL, "-source", &s, &flag); CHKERROR(ierr, "Error in PetscOptionsGetInt");
    ierr = PetscOptionsGetString(PETSC_NULL, "-low", lowstr, 1, &flag); CHKERROR(ierr, "Error in PetscOptionsGetChar");
    if(flag) {low = lowstr[0]; }
    ierr = PetscOptionsGetString(PETSC_NULL, "-high", highstr, 1, &flag); CHKERROR(ierr, "Error in PetscOptionsGetChar");
    if(flag) {high = highstr[0];}
    
    index_type::const_iterator iter;
    
    std::cout << "\n";
    iter = ind.relative_lower_bound(low, high, s);
    std::cout << "relative_lower_bound(poskey = " << s << ", prekey low = " << low << ", prekey high = " << high << ") = ";
    if(iter != ind.end()) {
      std::cout << *iter;
    }
    else {
      std::cout << "end()";
    }
    std::cout << "\n";
    iter = ind.relative_upper_bound(low, high, s);
    std::cout << "relative_upper_bound(poskey = " << s << ", prekey low = " << low << ", prekey high = " << high << ") = ";
    if(iter != ind.end()) {
      std::cout << *iter;
    }
    else {
      std::cout << "end()";
    }
    std::cout << "\n";
  }//test1
  {//test2
    source_type s;
    
    s = 1;
    rec_type low = rec_type(arrow_type(2,'B',2));
    rec_type high = rec_type(arrow_type(8,'D',4));
    PetscTruth flag;
    ierr = PetscOptionsGetInt(PETSC_NULL, "-source", &s, &flag); CHKERROR(ierr, "Error in PetscOptionsGetInt");
    
    index_type::const_iterator iter;
    
    std::cout << "\n";
    iter = ind.relative_lower_bound(low, high, s);
    std::cout << "relative_lower_bound(poskey = " << s << ", prekey low = " << low << ", prekey high = " << high << ") = ";
    if(iter != ind.end()) {
      std::cout << *iter;
    }
    else {
      std::cout << "end()";
    }
    std::cout << "\n";
    iter = ind.relative_upper_bound(low, high, s);
    std::cout << "relative_upper_bound(poskey = " << s << ", prekey low = " << low << ", prekey high = " << high << ") = ";
    if(iter != ind.end()) {
      std::cout << *iter;
    }
    else {
      std::cout << "end()";
    }
    std::cout << "\n";
  }//test2
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// main()
