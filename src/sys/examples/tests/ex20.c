static const char help[] = "Tests PetscOptionsPrefix{Push,Pop}\n\n";

#include <petscsys.h>

#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscInt opts[6] = {0};
  PetscBool  hascl = PETSC_FALSE,hasstr = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-zero","0");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPush("a_");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-one","1");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPush("bb_");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-two","2");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-three","3");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPush("cc_");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPush("ddd_");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-four","4");CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-five","5");CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(0,"-zero",&opts[0],0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-a_one",&opts[1],0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-a_bb_two",&opts[2],0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-a_three",&opts[3],0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-a_cc_ddd_four",&opts[4],0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-five",&opts[5],0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"opts = {%D %D %D %D %D %D}\n",opts[0],opts[1],opts[2],opts[3],opts[4],opts[5]);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(0,"-cl",&hascl,0);CHKERRQ(ierr);
  if (hascl) {
    ierr = PetscMemzero(opts,sizeof opts);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_zero",&opts[0],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_a_one",&opts[1],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_a_bb_two",&opts[2],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_a_three",&opts[3],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_a_cc_ddd_four",&opts[4],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-cl_five",&opts[5],0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cl_opts = {%D %D %D %D %D %D}\n",opts[0],opts[1],opts[2],opts[3],opts[4],opts[5]);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetBool(0,"-str",&hasstr,0);CHKERRQ(ierr);
  if (hasstr) {
    ierr = PetscOptionsInsertString("-prefix_push str_ -zero 100 -prefix_push a_ -one 101 -prefix_push bb_ -two 102 -prefix_pop -three 103 -prefix_push cc_ -prefix_push ddd_ -four 104 -prefix_pop -prefix_pop -prefix_pop -five 105 -prefix_pop");CHKERRQ(ierr);
    ierr = PetscMemzero(opts,sizeof opts);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_zero",&opts[0],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_a_one",&opts[1],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_a_bb_two",&opts[2],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_a_three",&opts[3],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_a_cc_ddd_four",&opts[4],0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(0,"-str_five",&opts[5],0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"str_opts = {%D %D %D %D %D %D}\n",opts[0],opts[1],opts[2],opts[3],opts[4],opts[5]);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
