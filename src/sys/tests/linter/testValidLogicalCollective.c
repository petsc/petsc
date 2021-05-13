#include <petscsys.h>

void testValidLogicalCollective(PetscInt a, PetscMPIInt b, PetscInt64 c, PetscBool d, PetscScalar e, PetscReal f)
{
  PetscViewer v; /* dummy variable to satisfy the PetscObject for the following */

  /* incorrect */
  PetscValidLogicalCollectiveInt(v,d,2);
  PetscValidLogicalCollectiveEnum(v,e,3);
  PetscValidLogicalCollectiveMPIInt(v,f,4);
  PetscValidLogicalCollectiveScalar(v,a,5);
  PetscValidLogicalCollectiveReal(v,b,6);
  PetscValidLogicalCollectiveEnum(v,c,7);

  /* correct */
  PetscValidLogicalCollectiveInt(v,a,1);
  PetscValidLogicalCollectiveMPIInt(v,b,2);
  PetscValidLogicalCollectiveInt(v,c,3);
  PetscValidLogicalCollectiveBool(v,d,4);
  PetscValidLogicalCollectiveScalar(v,e,5);
  PetscValidLogicalCollectiveReal(v,f,6);
  return;
}
