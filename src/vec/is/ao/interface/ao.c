
/*
   Defines the abstract operations on AO (application orderings)
*/
#include <../src/vec/is/ao/aoimpl.h> /*I "petscao.h" I*/

/* Logging support */
PetscClassId  AO_CLASSID;
PetscLogEvent AO_PetscToApplication, AO_ApplicationToPetsc;

/*@C
   AOView - Displays an application ordering.

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
-  viewer - viewer used for display

   Level: intermediate

    Options Database Key:
.   -ao_view - calls `AOView()` at end of `AOCreate()`

   Notes:
   The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization context with
   `PetscViewerASCIIOpen()` - output to a specified file.

.seealso: [](sec_ao), `AO`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode AOView(AO ao, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ao), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ao, viewer));
  PetscUseTypeMethod(ao, view, viewer);
  PetscFunctionReturn(0);
}

/*@C
   AOViewFromOptions - View an `AO` based on values in the options database

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
.  obj - Optional object
-  name - command line option

   Level: intermediate

.seealso: [](sec_ao), `AO`, `AOView`, `PetscObjectViewFromOptions()`, `AOCreate()`
@*/
PetscErrorCode AOViewFromOptions(AO ao, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)ao, obj, name));
  PetscFunctionReturn(0);
}

/*@
   AODestroy - Destroys an application ordering.

   Collective on ao

   Input Parameters:
.  ao - the application ordering context

   Level: beginner

.seealso: [](sec_ao), `AO`, `AOCreate()`
@*/
PetscErrorCode AODestroy(AO *ao)
{
  PetscFunctionBegin;
  if (!*ao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ao), AO_CLASSID, 1);
  if (--((PetscObject)(*ao))->refct > 0) {
    *ao = NULL;
    PetscFunctionReturn(0);
  }
  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*ao));
  PetscCall(ISDestroy(&(*ao)->isapp));
  PetscCall(ISDestroy(&(*ao)->ispetsc));
  /* destroy the internal part */
  if ((*ao)->ops->destroy) PetscCall((*(*ao)->ops->destroy)(*ao));
  PetscCall(PetscHeaderDestroy(ao));
  PetscFunctionReturn(0);
}

#include <../src/vec/is/is/impls/general/general.h>
/* ---------------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode ISSetUp_General(IS);

/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to
   the application-defined ordering.

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
-  is - the index set; this is replaced with its mapped values

   Output Parameter:
.  is - the mapped index set

   Level: intermediate

   Notes:
   The index set cannot be of type stride or block

   Any integers in is that are negative are left unchanged. This
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions etc.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOApplicationToPetsc()`,
          `AOApplicationToPetscIS()`, `AOPetscToApplication()`
@*/
PetscErrorCode AOPetscToApplicationIS(AO ao, IS is)
{
  PetscInt  n;
  PetscInt *ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCall(ISToGeneral(is));
  /* we cheat because we know the is is general and that we can change the indices */
  PetscCall(ISGetIndices(is, (const PetscInt **)&ia));
  PetscCall(ISGetLocalSize(is, &n));
  PetscUseTypeMethod(ao, petsctoapplication, n, ia);
  PetscCall(ISRestoreIndices(is, (const PetscInt **)&ia));
  /* updated cached values (sorted, min, max, etc.)*/
  PetscCall(ISSetUp_General(is));
  PetscFunctionReturn(0);
}

/*@
   AOApplicationToPetscIS - Maps an index set in the application-defined
   ordering to the PETSc ordering.

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
-  is - the index set; this is replaced with its mapped values

   Output Parameter:
.  is - the mapped index set

   Level: beginner

   Notes:
   The index set cannot be of type stride or block

   Any integers in is that are negative are left unchanged. This
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOPetscToApplication()`,
          `AOPetscToApplicationIS()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOApplicationToPetscIS(AO ao, IS is)
{
  PetscInt n, *ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCall(ISToGeneral(is));
  /* we cheat because we know the is is general and that we can change the indices */
  PetscCall(ISGetIndices(is, (const PetscInt **)&ia));
  PetscCall(ISGetLocalSize(is, &n));
  PetscUseTypeMethod(ao, applicationtopetsc, n, ia);
  PetscCall(ISRestoreIndices(is, (const PetscInt **)&ia));
  /* updated cached values (sorted, min, max, etc.)*/
  PetscCall(ISSetUp_General(is));
  PetscFunctionReturn(0);
}

/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to
   the application-defined ordering.

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers; these are replaced with their mapped value

   Output Parameter:
.   ia - the mapped integers

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

   Integers that are out of range are mapped to -1

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOApplicationToPetsc()`,
          `AOPetscToApplicationIS()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOPetscToApplication(AO ao, PetscInt n, PetscInt ia[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  if (n) PetscValidIntPointer(ia, 3);
  PetscUseTypeMethod(ao, petsctoapplication, n, ia);
  PetscFunctionReturn(0);
}

/*@
   AOApplicationToPetsc - Maps a set of integers in the application-defined
   ordering to the PETSc ordering.

   Collective on ao

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers; these are replaced with their mapped value

   Output Parameter:
.   ia - the mapped integers

   Level: beginner

   Notes:
   Any integers in ia[] that are negative are left unchanged. This
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

   Integers that are out of range are mapped to -1

.seealso: [](sec_ao), `AOCreateBasic()`, `AOView()`, `AOPetscToApplication()`,
          `AOPetscToApplicationIS()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOApplicationToPetsc(AO ao, PetscInt n, PetscInt ia[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  if (n) PetscValidIntPointer(ia, 3);
  PetscUseTypeMethod(ao, applicationtopetsc, n, ia);
  PetscFunctionReturn(0);
}

/*@
  AOPetscToApplicationPermuteInt - Permutes an array of blocks of integers
  in the PETSc ordering to the application-defined ordering.

  Collective on ao

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Level: beginner

  Notes:
  The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOApplicationToPetsc()`, `AOPetscToApplicationIS()`
@*/
PetscErrorCode AOPetscToApplicationPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidIntPointer(array, 3);
  PetscUseTypeMethod(ao, petsctoapplicationpermuteint, block, array);
  PetscFunctionReturn(0);
}

/*@
  AOApplicationToPetscPermuteInt - Permutes an array of blocks of integers
  in the application-defined ordering to the PETSc ordering.

  Collective on ao

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Level: beginner

  Notes:
  The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOPetscToApplicationIS()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOApplicationToPetscPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidIntPointer(array, 3);
  PetscUseTypeMethod(ao, applicationtopetscpermuteint, block, array);
  PetscFunctionReturn(0);
}

/*@
  AOPetscToApplicationPermuteReal - Permutes an array of blocks of reals
  in the PETSc ordering to the application-defined ordering.

  Collective on ao

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Level: beginner

  Notes:
  The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOApplicationToPetsc()`, `AOPetscToApplicationIS()`
@*/
PetscErrorCode AOPetscToApplicationPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidRealPointer(array, 3);
  PetscUseTypeMethod(ao, petsctoapplicationpermutereal, block, array);
  PetscFunctionReturn(0);
}

/*@
  AOApplicationToPetscPermuteReal - Permutes an array of blocks of reals
  in the application-defined ordering to the PETSc ordering.

  Collective on ao

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Level: beginner

  Notes:
  The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

.seealso: [](sec_ao), `AO`, `AOCreateBasic()`, `AOView()`, `AOApplicationToPetsc()`, `AOPetscToApplicationIS()`
@*/
PetscErrorCode AOApplicationToPetscPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidRealPointer(array, 3);
  PetscUseTypeMethod(ao, applicationtopetscpermutereal, block, array);
  PetscFunctionReturn(0);
}

/*@
    AOSetFromOptions - Sets `AO` options from the options database.

   Collective on ao

   Input Parameter:
.  ao - the application ordering

   Level: beginner

.seealso: [](sec_ao), `AO`, `AOCreate()`, `AOSetType()`, `AODestroy()`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOSetFromOptions(AO ao)
{
  char        type[256];
  const char *def = AOBASIC;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)ao);
  PetscCall(PetscOptionsFList("-ao_type", "AO type", "AOSetType", AOList, def, type, 256, &flg));
  if (flg) {
    PetscCall(AOSetType(ao, type));
  } else if (!((PetscObject)ao)->type_name) {
    PetscCall(AOSetType(ao, def));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
   AOSetIS - Sets the `IS` associated with the application ordering.

   Collective

   Input Parameters:
+  ao - the application ordering
.  isapp -  index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Level: beginner

   Notes:
   The index sets isapp and ispetsc are used only for creation of ao.

   This routine increases the reference count of isapp and ispetsc so you may/should destroy these arguments after this call if you no longer need them

.seealso: [](sec_ao), [](sec_scatter), `AO`, `AOCreate()`, `AODestroy()`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOSetIS(AO ao, IS isapp, IS ispetsc)
{
  PetscFunctionBegin;
  if (ispetsc) {
    PetscInt napp, npetsc;
    PetscCall(ISGetLocalSize(isapp, &napp));
    PetscCall(ISGetLocalSize(ispetsc, &npetsc));
    PetscCheck(napp == npetsc, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "napp %" PetscInt_FMT " != npetsc %" PetscInt_FMT ". Local IS lengths must match", napp, npetsc);
  }
  if (isapp) PetscCall(PetscObjectReference((PetscObject)isapp));
  if (ispetsc) PetscCall(PetscObjectReference((PetscObject)ispetsc));
  PetscCall(ISDestroy(&ao->isapp));
  PetscCall(ISDestroy(&ao->ispetsc));
  ao->isapp   = isapp;
  ao->ispetsc = ispetsc;
  PetscFunctionReturn(0);
}

/*@
   AOCreate - Creates an application ordering. That is an object that maps from an application ordering to a PETSc ordering and vice versa

   Collective

   Input Parameters:
.  comm - MPI communicator that is to share the `AO`

   Output Parameter:
.  ao - the new application ordering

   Options Database Key:
+   -ao_type <aotype> - create ao with particular format
-   -ao_view - call AOView() at the conclusion of AOCreate()

   Level: beginner

.seealso: [](sec_ao), `AO`, `AOSetIS()`, `AODestroy()`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOCreate(MPI_Comm comm, AO *ao)
{
  AO aonew;

  PetscFunctionBegin;
  PetscValidPointer(ao, 2);
  *ao = NULL;
  PetscCall(AOInitializePackage());

  PetscCall(PetscHeaderCreate(aonew, AO_CLASSID, "AO", "Application Ordering", "AO", comm, AODestroy, AOView));
  *ao = aonew;
  PetscFunctionReturn(0);
}
