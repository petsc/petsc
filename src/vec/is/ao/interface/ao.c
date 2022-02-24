
/*
   Defines the abstract operations on AO (application orderings)
*/
#include <../src/vec/is/ao/aoimpl.h>      /*I "petscao.h" I*/

/* Logging support */
PetscClassId  AO_CLASSID;
PetscLogEvent AO_PetscToApplication, AO_ApplicationToPetsc;

/*@C
   AOView - Displays an application ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
-  viewer - viewer used for display

   Level: intermediate

    Options Database Key:
.   -ao_view - calls AOView() at end of AOCreate()

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  AOView(AO ao,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ao),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)ao,viewer));
  CHKERRQ((*ao->ops->view)(ao,viewer));
  PetscFunctionReturn(0);
}

/*@C
   AOViewFromOptions - View from Options

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  AO, AOView, PetscObjectViewFromOptions(), AOCreate()
@*/
PetscErrorCode  AOViewFromOptions(AO ao,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)ao,obj,name));
  PetscFunctionReturn(0);
}

/*@
   AODestroy - Destroys an application ordering.

   Collective on AO

   Input Parameters:
.  ao - the application ordering context

   Level: beginner

.seealso: AOCreate()
@*/
PetscErrorCode  AODestroy(AO *ao)
{
  PetscFunctionBegin;
  if (!*ao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ao),AO_CLASSID,1);
  if (--((PetscObject)(*ao))->refct > 0) {*ao = NULL; PetscFunctionReturn(0);}
  /* if memory was published with SAWs then destroy it */
  CHKERRQ(PetscObjectSAWsViewOff((PetscObject)*ao));
  CHKERRQ(ISDestroy(&(*ao)->isapp));
  CHKERRQ(ISDestroy(&(*ao)->ispetsc));
  /* destroy the internal part */
  if ((*ao)->ops->destroy) {
    CHKERRQ((*(*ao)->ops->destroy)(*ao));
  }
  CHKERRQ(PetscHeaderDestroy(ao));
  PetscFunctionReturn(0);
}

#include <../src/vec/is/is/impls/general/general.h>
/* ---------------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode ISSetUp_General(IS);

/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to
   the application-defined ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
-  is - the index set; this is replaced with its mapped values

   Output Parameter:
.  is - the mapped index set

   Level: intermediate

   Notes:
   The index set cannot be of type stride or block

   Any integers in ia[] that are negative are left unchanged. This
         allows one to convert, for example, neighbor lists that use negative
         entries to indicate nonexistent neighbors due to boundary conditions
         etc.

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOApplicationToPetscIS(),AOPetscToApplication()
@*/
PetscErrorCode  AOPetscToApplicationIS(AO ao,IS is)
{
  PetscInt       n;
  PetscInt       *ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  CHKERRQ(ISToGeneral(is));
  /* we cheat because we know the is is general and that we can change the indices */
  CHKERRQ(ISGetIndices(is,(const PetscInt**)&ia));
  CHKERRQ(ISGetLocalSize(is,&n));
  CHKERRQ((*ao->ops->petsctoapplication)(ao,n,ia));
  CHKERRQ(ISRestoreIndices(is,(const PetscInt**)&ia));
  /* updated cached values (sorted, min, max, etc.)*/
  CHKERRQ(ISSetUp_General(is));
  PetscFunctionReturn(0);
}

/*@
   AOApplicationToPetscIS - Maps an index set in the application-defined
   ordering to the PETSc ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
-  is - the index set; this is replaced with its mapped values

   Output Parameter:
.  is - the mapped index set

   Level: beginner

   Note:
   The index set cannot be of type stride or block

   Any integers in ia[] that are negative are left unchanged. This
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetscIS(AO ao,IS is)
{
  PetscInt       n,*ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  CHKERRQ(ISToGeneral(is));
  /* we cheat because we know the is is general and that we can change the indices */
  CHKERRQ(ISGetIndices(is,(const PetscInt**)&ia));
  CHKERRQ(ISGetLocalSize(is,&n));
  CHKERRQ((*ao->ops->applicationtopetsc)(ao,n,ia));
  CHKERRQ(ISRestoreIndices(is,(const PetscInt**)&ia));
  /* updated cached values (sorted, min, max, etc.)*/
  CHKERRQ(ISSetUp_General(is));
  PetscFunctionReturn(0);
}

/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to
   the application-defined ordering.

   Collective on AO

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

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOPetscToApplication(AO ao,PetscInt n,PetscInt ia[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (n) PetscValidIntPointer(ia,3);
  CHKERRQ((*ao->ops->petsctoapplication)(ao,n,ia));
  PetscFunctionReturn(0);
}

/*@
   AOApplicationToPetsc - Maps a set of integers in the application-defined
   ordering to the PETSc ordering.

   Collective on AO

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

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetsc(AO ao,PetscInt n,PetscInt ia[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (n) PetscValidIntPointer(ia,3);
  CHKERRQ((*ao->ops->applicationtopetsc)(ao,n,ia));
  PetscFunctionReturn(0);
}

/*@
  AOPetscToApplicationPermuteInt - Permutes an array of blocks of integers
  in the PETSc ordering to the application-defined ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOPetscToApplicationPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  CHKERRQ((*ao->ops->petsctoapplicationpermuteint)(ao, block, array));
  PetscFunctionReturn(0);
}

/*@
  AOApplicationToPetscPermuteInt - Permutes an array of blocks of integers
  in the application-defined ordering to the PETSc ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.seealso: AOCreateBasic(), AOView(), AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetscPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  CHKERRQ((*ao->ops->applicationtopetscpermuteint)(ao, block, array));
  PetscFunctionReturn(0);
}

/*@
  AOPetscToApplicationPermuteReal - Permutes an array of blocks of reals
  in the PETSc ordering to the application-defined ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOPetscToApplicationPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidRealPointer(array,3);
  CHKERRQ((*ao->ops->petsctoapplicationpermutereal)(ao, block, array));
  PetscFunctionReturn(0);
}

/*@
  AOApplicationToPetscPermuteReal - Permutes an array of blocks of reals
  in the application-defined ordering to the PETSc ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Output Parameter:
. array - The permuted array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOApplicationToPetscPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidRealPointer(array,3);
  CHKERRQ((*ao->ops->applicationtopetscpermutereal)(ao, block, array));
  PetscFunctionReturn(0);
}

/*@
    AOSetFromOptions - Sets AO options from the options database.

   Collective on AO

   Input Parameter:
.  ao - the application ordering

   Level: beginner

.seealso: AOCreate(), AOSetType(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode AOSetFromOptions(AO ao)
{
  PetscErrorCode ierr;
  char           type[256];
  const char     *def=AOBASIC;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)ao);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-ao_type","AO type","AOSetType",AOList,def,type,256,&flg));
  if (flg) {
    CHKERRQ(AOSetType(ao,type));
  } else if (!((PetscObject)ao)->type_name) {
    CHKERRQ(AOSetType(ao,def));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   AOSetIS - Sets the IS associated with the application ordering.

   Collective

   Input Parameters:
+  ao - the application ordering
.  isapp -  index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Notes:
   The index sets isapp and ispetsc are used only for creation of ao.

   This routine increases the reference count of isapp and ispetsc so you may/should destroy these arguments after this call if you no longer need them

   Level: beginner

.seealso: AOCreate(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode AOSetIS(AO ao,IS isapp,IS ispetsc)
{
  PetscFunctionBegin;
  if (ispetsc) {
    PetscInt napp,npetsc;
    CHKERRQ(ISGetLocalSize(isapp,&napp));
    CHKERRQ(ISGetLocalSize(ispetsc,&npetsc));
    PetscCheckFalse(napp != npetsc,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"napp %" PetscInt_FMT " != npetsc %" PetscInt_FMT ". Local IS lengths must match",napp,npetsc);
  }
  if (isapp) CHKERRQ(PetscObjectReference((PetscObject)isapp));
  if (ispetsc) CHKERRQ(PetscObjectReference((PetscObject)ispetsc));
  CHKERRQ(ISDestroy(&ao->isapp));
  CHKERRQ(ISDestroy(&ao->ispetsc));
  ao->isapp   = isapp;
  ao->ispetsc = ispetsc;
  PetscFunctionReturn(0);
}

/*@
   AOCreate - Creates an application ordering.

   Collective

   Input Parameters:
.  comm - MPI communicator that is to share AO

   Output Parameter:
.  ao - the new application ordering

   Options Database Key:
+   -ao_type <aotype> - create ao with particular format
-   -ao_view - call AOView() at the conclusion of AOCreate()

   Level: beginner

.seealso: AOSetIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOCreate(MPI_Comm comm,AO *ao)
{
  AO             aonew;

  PetscFunctionBegin;
  PetscValidPointer(ao,2);
  *ao = NULL;
  CHKERRQ(AOInitializePackage());

  CHKERRQ(PetscHeaderCreate(aonew,AO_CLASSID,"AO","Application Ordering","AO",comm,AODestroy,AOView));
  *ao  = aonew;
  PetscFunctionReturn(0);
}
