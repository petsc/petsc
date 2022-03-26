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
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-use_gpu_aware_mpi",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsHasName(NULL,NULL,"-abc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsHasName(NULL,"","-abc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsHasName(NULL,"a","-bc",&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsHasName(NULL,"ab","-c",&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsHasName(NULL,"abc","-",&has));
  PetscTestCheck(has == PETSC_FALSE);

  PetscCall(PetscOptionsSetValue(NULL,"-abc",NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-abc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsHasName(NULL,"","-abc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsHasName(NULL,"a","-bc",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsHasName(NULL,"ab","-c",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsHasName(NULL,"abc","-",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsClearValue(NULL,"-abc"));
  PetscCall(PetscOptionsClearValue(NULL,"-ABC"));

  PetscCall(PetscOptionsPrefixPush(NULL,"a"));
  PetscCall(PetscOptionsSetValue(NULL,"-x",NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-ax",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsPrefixPush(NULL,"b"));
  PetscCall(PetscOptionsSetValue(NULL,"-xy",NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-abxy",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsPrefixPush(NULL,"c"));
  PetscCall(PetscOptionsSetValue(NULL,"-xz",NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-acxz",&has));
  PetscTestCheck(has == PETSC_TRUE);
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsClearValue(NULL,"-ax"));
  PetscCall(PetscOptionsClearValue(NULL,"-abxy"));
  PetscCall(PetscOptionsClearValue(NULL,"-acxz"));

  PetscCall(PetscOptionsSetValue(NULL,"-FOO",NULL));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO",NULL));
  PetscCall(PetscOptionsClearValue(NULL,"-FOO"));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO",NULL));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO","BAR"));
  PetscCall(PetscOptionsClearValue(NULL,"-FOO"));

  {
    char name[] = "-*_42", c;
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
      PetscCall(PetscOptionsSetValue(NULL,name,NULL));
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
    }
    for (c = 'A'; c <= 'Z'; c++) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
      PetscCall(PetscOptionsClearValue(NULL,name));
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'Z'; c >= 'A'; c--) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
      PetscCall(PetscOptionsSetValue(NULL,name,NULL));
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_TRUE);
      PetscCall(PetscOptionsClearValue(NULL,name));
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
    for (c = 'a'; c <= 'z'; c++) {
      name[1] = c;
      PetscCall(PetscOptionsHasName(NULL,NULL,name,&has));
      PetscTestCheck(has == PETSC_FALSE);
    }
  }

  PetscCall(PetscOptionsSetValue(NULL,"-abc_xyz","123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_1_23_456_789_xyz",&val,&has));
  PetscTestCheck(has == PETSC_TRUE && !strcmp(val,"123"));
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_xyz_42",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc42xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc0_42_xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsFindPair(NULL,NULL,"-abc_42_0xyz",&val,&has));
  PetscTestCheck(has == PETSC_FALSE);
  PetscCall(PetscOptionsClearValue(NULL,"-abc_xyz"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
