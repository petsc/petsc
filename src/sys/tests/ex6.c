static char help[] = "Tests options database";

#include <petscsys.h>

#define PetscTestCheck(expr) PetscCheck(expr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Assertion: `%s' failed.",PetscStringize(expr))

int main(int argc,char **argv)
{
  const char     *val;
  PetscBool      has;
  PetscErrorCode ierr;

  ierr = PetscOptionsSetValue(NULL,"-skip_petscrc", NULL);if (ierr) return ierr;
  ierr = PetscOptionsSetValue(NULL,"-use_gpu_aware_mpi", "0");if (ierr) return ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-use_gpu_aware_mpi",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-abc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsHasName(NULL,"","-abc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsHasName(NULL,"a","-bc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsHasName(NULL,"ab","-c",&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsHasName(NULL,"abc","-",&has));
  PetscTestCheck(has == PETSC_FALSE);

  CHKERRQ(PetscOptionsSetValue(NULL,"-abc",NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-abc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsHasName(NULL,"","-abc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsHasName(NULL,"a","-bc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsHasName(NULL,"ab","-c",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsHasName(NULL,"abc","-",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsClearValue(NULL,"-abc"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-ABC"));

  CHKERRQ(PetscOptionsPrefixPush(NULL,"a"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-x",NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-ax",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsPrefixPush(NULL,"b"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-xy",NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-abxy",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsPrefixPop(NULL));
  CHKERRQ(PetscOptionsPrefixPush(NULL,"c"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-xz",NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-acxz",&has));
  PetscTestCheck(has == PETSC_TRUE);
  CHKERRQ(PetscOptionsPrefixPop(NULL));
  CHKERRQ(PetscOptionsPrefixPop(NULL));
  CHKERRQ(PetscOptionsClearValue(NULL,"-ax"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-abxy"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-acxz"));

  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO",NULL));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO",NULL));
  CHKERRQ(PetscOptionsClearValue(NULL,"-FOO"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO",NULL));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-FOO"));

  {
    char name[] = "-*_42", c;
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
      CHKERRQ(PetscOptionsSetValue(NULL,name,NULL));
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
    }
    for (c = 'A'; c <= 'Z'; c++) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
      CHKERRQ(PetscOptionsClearValue(NULL,name));
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'Z'; c >= 'A'; c--) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
      CHKERRQ(PetscOptionsSetValue(NULL,name,NULL));
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
      CHKERRQ(PetscOptionsClearValue(NULL,name));
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      CHKERRQ(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
  }

  CHKERRQ(PetscOptionsSetValue(NULL,"-abc_xyz","123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_789_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_xyz_42",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc42xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc0_42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsFindPair(NULL,NULL,"-abc_42_0xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  CHKERRQ(PetscOptionsClearValue(NULL,"-abc_xyz"));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
