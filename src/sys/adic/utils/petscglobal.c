
#include <petscsys.h>

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalMax"
/*@C
      PetscGlobalMax - Computes the maximum value over several processors. Only for use with ADIC!

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the maximum

   Output Parameter:
.   result - the maximum value
  
   Level: intermediate

   Notes:
     These functions are to be used ONLY inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMin(), PetscGlobalSum()
@*/
PetscErrorCode  PetscGlobalMax(MPI_Comm comm,const PetscReal* local,PetscReal* result)
{
  return MPI_Allreduce((void*)local,result,1,MPIU_REAL,MPIU_MAX,comm);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalMin"
/*@C
      PetscGlobalMin - Computes the minimum value over several processors. Only for use with ADIC!

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the minimum

   Output Parameter:
.   result - the minimum value
  
   Level: intermediate

   Notes:
     These functions are to be used ONLY inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMax(), PetscGlobalSum()
@*/
PetscErrorCode  PetscGlobalMin(MPI_Comm comm,const PetscReal* local,PetscReal* result)
{
  return MPI_Allreduce((void*)local,result,1,MPIU_REAL,MPIU_MIN,comm);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalSum"
/*@C
      PetscGlobalSum - Computes the sum over several processors. Only for use with ADIC!

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the sum

   Output Parameter:
.   result - the sum
  
   Level: intermediate

   Notes:
     These functions are to be used ONLY inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMin(), PetscGlobalMax()
@*/
PetscErrorCode  PetscGlobalSum(MPI_Comm comm, const PetscScalar* local,PetscScalar* result)
{
  return MPI_Allreduce((void*)local,result,1,MPIU_SCALAR,MPIU_SUM,comm);
}


