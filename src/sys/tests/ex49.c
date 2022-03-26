
static char help[] = "Demonstrates PetscDataTypeFromString().\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/

#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscDataType  dtype;
  PetscBool      found;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscDataTypeFromString("Scalar",&dtype,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Did not find scalar datatype");
  PetscCheckFalse(dtype != PETSC_SCALAR,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Found wrong datatype for scalar");

  PetscCall(PetscDataTypeFromString("INT",&dtype,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Did not find int datatype");
  PetscCheckFalse(dtype != PETSC_INT,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Found wrong datatype for int");

  PetscCall(PetscDataTypeFromString("real",&dtype,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Did not find real datatype");
  PetscCheckFalse(dtype != PETSC_REAL,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Found wrong datatype for real");

  PetscCall(PetscDataTypeFromString("abogusdatatype",&dtype,&found));
  PetscCheck(!found,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Found a bogus datatype");

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
