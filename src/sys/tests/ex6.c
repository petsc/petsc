static char help[] = "Tests options database";

#include <petscsys.h>

#define PetscAssert(expr) do {            \
if (PetscUnlikely(!(expr)))               \
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, \
           "Assertion: `%s' failed.",     \
           PetscStringize(expr));         \
} while(0)

int main(int argc,char **argv)
{
  const char     *val;
  PetscBool      has;
  PetscErrorCode ierr;

  ierr = PetscOptionsSetValue(NULL,"-skip_petscrc", NULL);if (ierr) return ierr;
  ierr = PetscOptionsSetValue(NULL,"-use_gpu_aware_mpi", "0");if (ierr) return ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscOptionsHasName(NULL,NULL,"-use_gpu_aware_mpi",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsHasName(NULL,NULL,"-abc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsHasName(NULL,"","-abc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsHasName(NULL,"a","-bc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsHasName(NULL,"ab","-c",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsHasName(NULL,"abc","-",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);

  ierr = PetscOptionsSetValue(NULL,"-abc",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-abc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsHasName(NULL,"","-abc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsHasName(NULL,"a","-bc",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsHasName(NULL,"ab","-c",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsHasName(NULL,"abc","-",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsClearValue(NULL,"-abc");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-ABC");CHKERRQ(ierr);

  ierr = PetscOptionsPrefixPush(NULL,"a");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-x",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-ax",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsPrefixPush(NULL,"b");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-xy",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-abxy",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsPrefixPop(NULL);CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPush(NULL,"c");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-xz",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-acxz",&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE);
  ierr = PetscOptionsPrefixPop(NULL);CHKERRQ(ierr);
  ierr = PetscOptionsPrefixPop(NULL);CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-ax");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-abxy");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-acxz");CHKERRQ(ierr);

  ierr = PetscOptionsSetValue(NULL,"-FOO",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO","BAR");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-FOO");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO","BAR");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO","BAR");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-FOO");CHKERRQ(ierr);

  {
    char name[] = "-*_42", c;
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
      ierr = PetscOptionsSetValue(NULL,name,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_TRUE);
    }
    for (c = 'A'; c <= 'Z'; c++) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_TRUE);
      ierr = PetscOptionsClearValue(NULL,name);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
    }
    for (c = 'Z'; c >= 'A'; c--) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
      ierr = PetscOptionsSetValue(NULL,name,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_TRUE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_TRUE);
      ierr = PetscOptionsClearValue(NULL,name);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      ierr = PetscOptionsHasName(NULL,NULL,name,&has);CHKERRQ(ierr);
      PetscAssert(has == PETSC_FALSE);
    }
  }

  ierr = PetscOptionsSetValue(NULL,"-abc_xyz","123");CHKERRQ(ierr);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_1_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_789_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_TRUE && !strcmp(val,"123"));
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_xyz_42",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc42xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc42_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc0_42_xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsFindPair(NULL,NULL,"-abc_42_0xyz",&val,&has);CHKERRQ(ierr);
  PetscAssert(has == PETSC_FALSE);
  ierr = PetscOptionsClearValue(NULL,"-abc_xyz");CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
