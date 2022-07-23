
static char help[] = "Tests PetscMergeIntArray\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  const PetscInt first[] = {0,2,3,5,8}, second[] = {1,3,4,8,10,11};
  PetscInt       *result,n;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscMergeIntArray(5,first,6,second,&n,&result));
  PetscCall(PetscIntView(n,result,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscFinalize());
  return 0;
}
