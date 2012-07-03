
/*  
   Defines the abstract operations on AO (application orderings) 
*/
#include <../src/dm/ao/aoimpl.h>      /*I "petscao.h" I*/

/* Logging support */
PetscClassId  AO_CLASSID;
PetscLogEvent  AO_PetscToApplication, AO_ApplicationToPetsc;

#undef __FUNCT__  
#define __FUNCT__ "AOView" 
/*@C
   AOView - Displays an application ordering.

   Collective on AO and PetscViewer

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

.keywords: application ordering

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  AOView(AO ao,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)ao)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = (*ao->ops->view)(ao,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy" 
/*@C
   AODestroy - Destroys an application ordering.

   Collective on AO

   Input Parameters:
.  ao - the application ordering context

   Level: beginner

.keywords: destroy, application ordering

.seealso: AOCreate()
@*/
PetscErrorCode  AODestroy(AO *ao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ao),AO_CLASSID,1);
  if (--((PetscObject)(*ao))->refct > 0) {*ao = 0; PetscFunctionReturn(0);}
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish((*ao));CHKERRQ(ierr);
  /* destroy the internal part */
  if ((*ao)->ops->destroy) {
    ierr = (*(*ao)->ops->destroy)(*ao);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#include <../src/vec/is/impls/general/general.h>
/* ---------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplicationIS" 
/*@
   AOPetscToApplicationIS - Maps an index set in the PETSc ordering to 
   the application-defined ordering.

   Collective on AO and IS

   Input Parameters:
+  ao - the application ordering context
-  is - the index set

   Level: intermediate

   Notes:
   The index set cannot be of type stride or block
   
   Any integers in ia[] that are negative are left unchanged. This 
         allows one to convert, for example, neighbor lists that use negative
         entries to indicate nonexistent neighbors due to boundary conditions
         etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOApplicationToPetscIS(),AOPetscToApplication()
@*/
PetscErrorCode  AOPetscToApplicationIS(AO ao,IS is)
{
  PetscErrorCode ierr;
  PetscInt       n;
  PetscInt       *ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  ierr = ISToGeneral(is);CHKERRQ(ierr);
  /* we cheat because we know the is is general and that we can change the indices */
  ierr = ISGetIndices(is,(const PetscInt**)&ia);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = (*ao->ops->petsctoapplication)(ao,n,ia);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,(const PetscInt**)&ia);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetscIS" 
/*@
   AOApplicationToPetscIS - Maps an index set in the application-defined
   ordering to the PETSc ordering.

   Collective on AO and IS

   Input Parameters:
+  ao - the application ordering context
-  is - the index set

   Level: beginner

   Note:
   The index set cannot be of type stride or block
   
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetscIS(AO ao,IS is)
{
  PetscErrorCode ierr;
  PetscInt       n,*ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  ierr = ISToGeneral(is);CHKERRQ(ierr);
  /* we cheat because we know the is is general and that we can change the indices */
  ierr = ISGetIndices(is,(const PetscInt**)&ia);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = (*ao->ops->applicationtopetsc)(ao,n,ia);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,(const PetscInt**)&ia);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplication" 
/*@
   AOPetscToApplication - Maps a set of integers in the PETSc ordering to 
   the application-defined ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOPetscToApplication(AO ao,PetscInt n,PetscInt ia[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (n) {PetscValidIntPointer(ia,3);}
  ierr = (*ao->ops->petsctoapplication)(ao,n,ia);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetsc" 
/*@
   AOApplicationToPetsc - Maps a set of integers in the application-defined
   ordering to the PETSc ordering.

   Collective on AO

   Input Parameters:
+  ao - the application ordering context
.  n - the number of integers
-  ia - the integers; these are replaced with their mapped value

   Output Parameter:
.   ia - the mapped interges

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetsc(AO ao,PetscInt n,PetscInt ia[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_CLASSID,1);
  if (n) {PetscValidIntPointer(ia,3);}
  ierr = (*ao->ops->applicationtopetsc)(ao,n,ia);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplicationPermuteInt"
/*@
  AOPetscToApplicationPermuteInt - Permutes an array of blocks of integers
  in the PETSc ordering to the application-defined ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.keywords: application ordering, mapping
.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOPetscToApplicationPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  ierr = (*ao->ops->petsctoapplicationpermuteint)(ao, block, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetscPermuteInt"
/*@
  AOApplicationToPetscPermuteInt - Permutes an array of blocks of integers
  in the application-defined ordering to the PETSc ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOApplicationToPetscPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  ierr = (*ao->ops->applicationtopetscpermuteint)(ao, block, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplicationPermuteReal"
/*@
  AOPetscToApplicationPermuteReal - Permutes an array of blocks of reals
  in the PETSc ordering to the application-defined ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_pet] --> array[i_app], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOPetscToApplicationPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  ierr = (*ao->ops->petsctoapplicationpermutereal)(ao, block, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetscPermuteReal"
/*@
  AOApplicationToPetscPermuteReal - Permutes an array of blocks of reals
  in the application-defined ordering to the PETSc ordering.

  Collective on AO

  Input Parameters:
+ ao    - The application ordering context
. block - The block size
- array - The integer array

  Note: The length of the array should be block*N, where N is length
  provided to the AOCreate*() method that created the AO.

  The permutation takes array[i_app] --> array[i_pet], where i_app is
  the index of 'i' in the application ordering and i_pet is the index
  of 'i' in the petsc ordering.

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode  AOApplicationToPetscPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidIntPointer(array,3);
  ierr = (*ao->ops->applicationtopetscpermutereal)(ao, block, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOSetFromOptions" 
/*@C
    AOSetFromOptions - Sets AO options from the options database.

   Collective on AO

   Input Parameter:
.  ao - the application ordering

   Level: beginner

.keywords: AO, options, database

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
    ierr = PetscOptionsList("-ao_type","AO type","AOSetType",AOList,def,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = AOSetType(ao,type);CHKERRQ(ierr);
    } else if (!((PetscObject)ao)->type_name){
      ierr = AOSetType(ao,def);CHKERRQ(ierr);
    } 

    /* not used here, but called so will go into help messaage */
    ierr = PetscOptionsName("-ao_view","Print detailed information on AO used","AOView",0);CHKERRQ(ierr);
 
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOSetIS" 
/*@C
   AOSetIS - Sets the IS associated with the application ordering.

   Collective on MPI_Comm

   Input Parameters:
+  ao - the application ordering
.  isapp -  index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be PETSC_NULL to use the
             natural ordering)

   Notes: 
   The index sets isapp and ispetsc are used only for creation of ao.

   Level: beginner

.keywords: AO, create

.seealso: AOCreate(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode AOSetIS(AO ao,IS isapp,IS ispetsc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ispetsc){
    PetscInt       napp,npetsc;
    ierr = ISGetLocalSize(isapp,&napp);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ispetsc,&npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"napp %D != npetsc %d. Local IS lengths must match",napp,npetsc);
  }
  ao->isapp   = isapp;
  ao->ispetsc = ispetsc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreate" 
/*@C
   AOCreate - Creates an application ordering.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator that is to share AO

   Output Parameter:
.  ao - the new application ordering

   Options Database Key:
+   -ao_type <aotype> - create ao with particular format
-   -ao_view - call AOView() at the conclusion of AOCreate()

   Level: beginner

.keywords: AO, create

.seealso: AOSetIS(), AODestroy(), AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode  AOCreate(MPI_Comm comm,AO *ao)
{
  PetscErrorCode ierr;
  AO             aonew;
  PetscBool      opt;

  PetscFunctionBegin;
  PetscValidPointer(ao,2);
  *ao = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = AOInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(aonew,_p_AO,struct _AOOps,AO_CLASSID,-1,"AO","Application Ordering","AO",comm,AODestroy,AOView);CHKERRQ(ierr);
  ierr = PetscMemzero(aonew->ops, sizeof(struct _AOOps));CHKERRQ(ierr);
  *ao = aonew;

  opt = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-ao_view", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {
    ierr = AOView(aonew, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
