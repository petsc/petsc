/*  
   Defines the abstract operations on AO (application orderings) 
*/
#include "src/dm/ao/aoimpl.h"      /*I "petscao.h" I*/

/* Logging support */
PetscCookie AO_COOKIE = 0, AODATA_COOKIE = 0;
PetscEvent  AO_PetscToApplication = 0, AO_ApplicationToPetsc = 0;

#undef __FUNCT__  
#define __FUNCT__ "AOView" 
/*@C
   AOView - Displays an application ordering.

   Collective on AO and PetscViewer

   Input Parameters:
+  ao - the application ordering context
-  viewer - viewer used for display

   Level: intermediate

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
PetscErrorCode AOView(AO ao,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(ao->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = (*ao->ops->view)(ao,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODestroy" 
/*@
   AODestroy - Destroys an application ordering set.

   Collective on AO

   Input Parameters:
.  ao - the application ordering context

   Level: beginner

.keywords: destroy, application ordering

.seealso: AOCreateBasic()
@*/
PetscErrorCode AODestroy(AO ao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  if (--ao->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(ao);CHKERRQ(ierr);

  ierr = (*ao->ops->destroy)(ao);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


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
PetscErrorCode AOPetscToApplicationIS(AO ao,IS is)
{
  PetscErrorCode ierr;
  PetscInt       n,*ia;
  PetscTruth     flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  ierr = ISBlock(is,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(PETSC_ERR_SUP,"Cannot translate block index sets");
  ierr = ISStride(is,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISStrideToGeneral(is);CHKERRQ(ierr);
  }

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia);CHKERRQ(ierr);
  ierr = (*ao->ops->petsctoapplication)(ao,n,ia);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia);CHKERRQ(ierr);
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
PetscErrorCode AOApplicationToPetscIS(AO ao,IS is)
{
  PetscErrorCode ierr;
  PetscInt       n,*ia;
  PetscTruth     flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  ierr = ISBlock(is,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ(PETSC_ERR_SUP,"Cannot translate block index sets");
  ierr = ISStride(is,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = ISStrideToGeneral(is);CHKERRQ(ierr);
  }

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ia);CHKERRQ(ierr);
  ierr = (*ao->ops->applicationtopetsc)(ao,n,ia);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ia);CHKERRQ(ierr);
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
PetscErrorCode AOPetscToApplication(AO ao,PetscInt n,PetscInt ia[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  PetscValidIntPointer(ia,3);
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
-  ia - the integers

   Level: beginner

   Note:
   Any integers in ia[] that are negative are left unchanged. This 
   allows one to convert, for example, neighbor lists that use negative
   entries to indicate nonexistent neighbors due to boundary conditions, etc.

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplication(),
          AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode AOApplicationToPetsc(AO ao,PetscInt n,PetscInt ia[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AO_COOKIE,1);
  PetscValidIntPointer(ia,3);
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

  Level: beginner

.keywords: application ordering, mapping
.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode AOPetscToApplicationPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
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

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOPetscToApplicationIS(), AOApplicationToPetsc()
@*/
PetscErrorCode AOApplicationToPetscPermuteInt(AO ao, PetscInt block, PetscInt array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
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

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(), AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode AOPetscToApplicationPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
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

  Level: beginner

.keywords: application ordering, mapping

.seealso: AOCreateBasic(), AOView(),AOApplicationToPetsc(), AOPetscToApplicationIS()
@*/
PetscErrorCode AOApplicationToPetscPermuteReal(AO ao, PetscInt block, PetscReal array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
  PetscValidIntPointer(array,3);
  ierr = (*ao->ops->applicationtopetscpermutereal)(ao, block, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
