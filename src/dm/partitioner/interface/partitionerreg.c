#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/

PetscClassId PETSCPARTITIONER_CLASSID = 0;

PetscFunctionList PetscPartitionerList              = NULL;
PetscBool         PetscPartitionerRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscPartitionerRegister - Adds a new PetscPartitioner implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscPartitionerRegister() may be called multiple times to add several user-defined PetscPartitioners

  Sample usage:
.vb
    PetscPartitionerRegister("my_part", MyPetscPartitionerCreate);
.ve

  Then, your PetscPartitioner type can be chosen with the procedural interface via
.vb
    PetscPartitionerCreate(MPI_Comm, PetscPartitioner *);
    PetscPartitionerSetType(PetscPartitioner, "my_part");
.ve
   or at runtime via the option
.vb
    -petscpartitioner_type my_part
.ve

  Level: advanced

.seealso: `PetscPartitionerRegisterAll()`

@*/
PetscErrorCode PetscPartitionerRegister(const char sname[], PetscErrorCode (*function)(PetscPartitioner))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&PetscPartitionerList, sname, function));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_ParMetis(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_PTScotch(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Chaco(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Shell(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Simple(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Gather(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_MatPartitioning(PetscPartitioner);

/*@C
  PetscPartitionerRegisterAll - Registers all of the PetscPartitioner components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: `PetscPartitionerRegister()`, `PetscPartitionerRegisterDestroy()`
@*/
PetscErrorCode PetscPartitionerRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscPartitionerRegisterAllCalled) PetscFunctionReturn(0);
  PetscPartitionerRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERPARMETIS, PetscPartitionerCreate_ParMetis));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERPTSCOTCH, PetscPartitionerCreate_PTScotch));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERCHACO, PetscPartitionerCreate_Chaco));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERSIMPLE, PetscPartitionerCreate_Simple));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERSHELL, PetscPartitionerCreate_Shell));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERGATHER, PetscPartitionerCreate_Gather));
  PetscCall(PetscPartitionerRegister(PETSCPARTITIONERMATPARTITIONING, PetscPartitionerCreate_MatPartitioning));
  PetscFunctionReturn(0);
}

static PetscBool PetscPartitionerPackageInitialized = PETSC_FALSE;

/*@C
  PetscPartitionerFinalizePackage - This function finalizes everything in the PetscPartitioner package.
  It is called from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscPartitionerFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscPartitionerList));
  PetscPartitionerPackageInitialized = PETSC_FALSE;
  PetscPartitionerRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerInitializePackage - This function initializes everything in the PetscPartitioner package.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscPartitionerInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscPartitionerPackageInitialized) PetscFunctionReturn(0);
  PetscPartitionerPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  PetscCall(PetscClassIdRegister("GraphPartitioner", &PETSCPARTITIONER_CLASSID));
  /* Register Constructors */
  PetscCall(PetscPartitionerRegisterAll());
  /* Register Events */
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSCPARTITIONER_CLASSID;
    PetscCall(PetscInfoProcessClass("partitioner", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("partitioner", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCPARTITIONER_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscPartitionerFinalizePackage));
  PetscFunctionReturn(0);
}
