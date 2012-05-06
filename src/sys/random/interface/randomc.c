
/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We are still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include <../src/sys/random/randomimpl.h>                              /*I "petscsys.h" I*/
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

/* Logging support */
PetscClassId  PETSC_RANDOM_CLASSID;

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomDestroy" 
/*@
   PetscRandomDestroy - Destroys a context that has been formed by 
   PetscRandomCreate().

   Collective on PetscRandom

   Intput Parameter:
.  r  - the random number generator context

   Level: intermediate

.seealso: PetscRandomGetValue(), PetscRandomCreate(), VecSetRandom()
@*/
PetscErrorCode  PetscRandomDestroy(PetscRandom *r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!*r) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*r,PETSC_RANDOM_CLASSID,1);
  if (--((PetscObject)(*r))->refct > 0) {*r = 0; PetscFunctionReturn(0);}
  ierr = PetscHeaderDestroy(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetSeed"
/*@
   PetscRandomGetSeed - Gets the random seed.

   Not collective

   Input Parameters:
.  r - The random number generator context

   Output Parameter:
.  seed - The random seed

   Level: intermediate

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomSetSeed(), PetscRandomSeed()
@*/
PetscErrorCode  PetscRandomGetSeed(PetscRandom r,unsigned long *seed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  if (seed) {
    PetscValidPointer(seed,2);
    *seed = r->seed;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetSeed"
/*@
   PetscRandomSetSeed - Sets the random seed. You MUST call PetscRandomSeed() after this call to have the new seed used.

   Not collective

   Input Parameters:
+  r  - The random number generator context
-  seed - The random seed

   Level: intermediate

   Usage:
      PetscRandomSetSeed(r,a positive integer);
      PetscRandomSeed(r);  PetscRandomGetValue() will now start with the new seed.

      PetscRandomSeed(r) without a call to PetscRandomSetSeed() re-initializes
        the seed. The random numbers generated will be the same as before.

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomGetSeed(), PetscRandomSeed()
@*/
PetscErrorCode  PetscRandomSetSeed(PetscRandom r,unsigned long seed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  r->seed = seed;
  ierr = PetscInfo1(PETSC_NULL,"Setting seed to %d\n",(int)seed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetTypeFromOptions_Private"
/*
  PetscRandomSetTypeFromOptions_Private - Sets the type of random generator from user options. Defaults to type PETSCRAND48 or PETSCRAND.

  Collective on PetscRandom

  Input Parameter:
. rnd - The random number generator context

  Level: intermediate

.keywords: PetscRandom, set, options, database, type
.seealso: PetscRandomSetFromOptions(), PetscRandomSetType()
*/
static PetscErrorCode PetscRandomSetTypeFromOptions_Private(PetscRandom rnd)
{
  PetscBool      opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)rnd)->type_name) {
    defaultType = ((PetscObject)rnd)->type_name;
  } else {
#if defined(PETSC_HAVE_DRAND48)    
    defaultType = PETSCRAND48;
#elif defined(PETSC_HAVE_RAND)
    defaultType = PETSCRAND;
#endif
  }

  if (!PetscRandomRegisterAllCalled) {ierr = PetscRandomRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsList("-random_type","PetscRandom type","PetscRandomSetType",PetscRandomList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscRandomSetType(rnd, typeName);CHKERRQ(ierr);
  } else {
    ierr = PetscRandomSetType(rnd, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetFromOptions"
/*@
  PetscRandomSetFromOptions - Configures the random number generator from the options database.

  Collective on PetscRandom

  Input Parameter:
. rnd - The random number generator context

  Options Database:
.  -random_seed <integer> - provide a seed to the random number generater

  Notes:  To see all options, run your program with the -help option.
          Must be called after PetscRandomCreate() but before the rnd is used.

  Level: beginner

.keywords: PetscRandom, set, options, database
.seealso: PetscRandomCreate(), PetscRandomSetType()
@*/
PetscErrorCode  PetscRandomSetFromOptions(PetscRandom rnd)
{
  PetscErrorCode ierr;
  PetscBool      set;
  PetscInt       seed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd,PETSC_RANDOM_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)rnd);CHKERRQ(ierr);

    /* Handle PetscRandom type options */
    ierr = PetscRandomSetTypeFromOptions_Private(rnd);CHKERRQ(ierr);

    /* Handle specific random generator's options */
    if (rnd->ops->setfromoptions) {
      ierr = (*rnd->ops->setfromoptions)(rnd);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-random_seed","Seed to use to generate random numbers","PetscRandomSetSeed",0,&seed,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PetscRandomSetSeed(rnd,(unsigned long int)seed);CHKERRQ(ierr);
      ierr = PetscRandomSeed(rnd);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscRandomViewFromOptions(rnd, ((PetscObject)rnd)->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomView"
/*@C
   PetscRandomView - Views a random number generator object. 

   Collective on PetscRandom

   Input Parameters:
+  rnd - The random number generator context
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the vector is printed using the 
   option PetscViewerSetFormat().

   Level: beginner

.seealso:  PetscRealView(), PetscScalarView(), PetscIntView()
@*/
PetscErrorCode  PetscRandomView(PetscRandom rnd,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd,PETSC_RANDOM_CLASSID,1);
  PetscValidType(rnd,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)rnd)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(rnd,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(((PetscObject)rnd)->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%D] Random type %s, seed %D\n",rank,((PetscObject)rnd)->type_name,rnd->seed);CHKERRQ(ierr); 
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    const char *tname;
    ierr = PetscObjectGetName((PetscObject)viewer,&tname);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this object",tname);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscRandomViewFromOptions"
/*@
  PetscRandomViewFromOptions - This function visualizes the type and the seed of the generated random numbers based upon user options.

  Collective on PetscRandom

  Input Parameters:
. rnd   - The random number generator context
. title - The title

  Level: intermediate

.keywords: PetscRandom, view, options, database
.seealso: PetscRandomSetFromOptions()
@*/
PetscErrorCode  PetscRandomViewFromOptions(PetscRandom rnd, char *title)
{
  PetscBool      opt = PETSC_FALSE;
  PetscViewer    viewer;
  char           typeName[1024];
  char           fileName[PETSC_MAX_PATH_LEN];
  size_t         len;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject)rnd)->prefix, "-random_view", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {   
    ierr = PetscOptionsGetString(((PetscObject)rnd)->prefix, "-random_view", typeName, 1024, &opt);CHKERRQ(ierr);
    ierr = PetscStrlen(typeName, &len);CHKERRQ(ierr);
    if (len > 0) {
      ierr = PetscViewerCreate(((PetscObject)rnd)->comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, typeName);CHKERRQ(ierr);
      ierr = PetscOptionsGetString(((PetscObject)rnd)->prefix, "-random_view_file", fileName, 1024, &opt);CHKERRQ(ierr);
      if (opt) {
        ierr = PetscViewerFileSetName(viewer, fileName);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerFileSetName(viewer, ((PetscObject)rnd)->name);CHKERRQ(ierr);
      }
      ierr = PetscRandomView(rnd, viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {    
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)rnd)->comm,&viewer);CHKERRQ(ierr);
      ierr = PetscRandomView(rnd, viewer);CHKERRQ(ierr);
    } 
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomPublish_Petsc"
static PetscErrorCode PetscRandomPublish_Petsc(PetscObject obj)
{
  PetscRandom    rand = (PetscRandom) obj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AMS_Memory_add_field(obj->amem,"Low",&rand->low,1,AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate" 
/*@
   PetscRandomCreate - Creates a context for generating random numbers,
   and initializes the random-number generator.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator

   Output Parameter:
.  r  - the random number generator context

   Level: intermediate

   Notes:
   The random type has to be set by PetscRandomSetType().

   This is only a primative "parallel" random number generator, it should NOT
   be used for sophisticated parallel Monte Carlo methods since it will very likely
   not have the correct statistics across processors. You can provide your own
   parallel generator using PetscRandomRegister();

   If you create a PetscRandom() using PETSC_COMM_SELF on several processors then
   the SAME random numbers will be generated on all those processors. Use PETSC_COMM_WORLD
   or the appropriate parallel communicator to eliminate this issue.

   Use VecSetRandom() to set the elements of a vector to random numbers.

   Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_SELF,&r);
      PetscRandomSetType(r,PETSCRAND48);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValueReal(r,&value2);
      PetscRandomDestroy(&r);
.ve

   Concepts: random numbers^creating

.seealso: PetscRandomSetType(), PetscRandomGetValue(), PetscRandomGetValueReal(), PetscRandomSetInterval(), 
          PetscRandomDestroy(), VecSetRandom(), PetscRandomType
@*/

PetscErrorCode  PetscRandomCreate(MPI_Comm comm,PetscRandom *r)
{
  PetscRandom    rr;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidPointer(r,3);
  *r = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscRandomInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(rr,_p_PetscRandom,struct _PetscRandomOps,PETSC_RANDOM_CLASSID,-1,"PetscRandom","Random number generator","Sys",comm,PetscRandomDestroy,0);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  rr->data  = PETSC_NULL;
  rr->low   = 0.0;
  rr->width = 1.0;
  rr->iset  = PETSC_FALSE;
  rr->seed  = 0x12345678 + 76543*rank;
#if defined(PETSC_HAVE_AMS)
  ((PetscObject)rr)->bops->publish = PetscRandomPublish_Petsc;
#endif
  *r = rr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSeed"
/*@
   PetscRandomSeed - Seed the generator.

   Not collective

   Input Parameters:
.  r - The random number generator context

   Level: intermediate

   Usage:
      PetscRandomSetSeed(r,a positive integer);
      PetscRandomSeed(r);  PetscRandomGetValue() will now start with the new seed.

      PetscRandomSeed(r) without a call to PetscRandomSetSeed() re-initializes
        the seed. The random numbers generated will be the same as before.

   Concepts: random numbers^seed

.seealso: PetscRandomCreate(), PetscRandomGetSeed(), PetscRandomSetSeed()
@*/
PetscErrorCode  PetscRandomSeed(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_CLASSID,1);
  PetscValidType(r,1);

  ierr = (*r->ops->seed)(r);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

