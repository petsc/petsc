#define PETSC_DLL
/*
   These routines simplify the use of command line, file options, etc.,
   and are used to manipulate the options database.

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

typedef struct {
  char            *prefix,*mprefix;  
  char            *title;
  MPI_Comm        comm;
  PetscTruth      printhelp;
  PetscTruth      changedmethod;
} PetscOptionsPublishObject;
static PetscOptionsPublishObject amspub;
PetscInt                         PetscOptionsPublishCount = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsBegin_Private"
PetscErrorCode PetscOptionsBegin_Private(MPI_Comm comm,const char prefix[],const char title[],const char mansec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(prefix,&amspub.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&amspub.title);CHKERRQ(ierr);
  amspub.comm   = comm;
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&amspub.printhelp);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount) {
    ierr = (*PetscHelpPrintf)(comm,"%s -------------------------------------------------\n",title);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEnd_Private"
PetscErrorCode PetscOptionsEnd_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(amspub.title);CHKERRQ(ierr); amspub.title  = 0;
  ierr = PetscStrfree(amspub.prefix);CHKERRQ(ierr); amspub.prefix = 0;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInt"
/*@C
   PetscOptionsInt - Gets the integer value for a particular option in the database.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  defaultv - the default (current) value

   Output Parameter:
+  value - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsInt(const char opt[],const char text[],const char man[],PetscInt defaultv,PetscInt *value,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(amspub.prefix,opt,value,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%d>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsString"
/*@C
   PetscOptionsString - Gets the string value for a particular option in the database.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  defaultv - the default (current) value

   Output Parameter:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsString(const char opt[],const char text[],const char man[],const char defaultv[],char value[],size_t len,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(amspub.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%s>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     Publishes an AMS double field (with the default value in it) and with a name
   given by the text string
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsReal"
/*@C
   PetscOptionsReal - Gets the PetscReal value for a particular option in the database.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  defaultv - the default (current) value

   Output Parameter:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsReal(const char opt[],const char text[],const char man[],PetscReal defaultv,PetscReal *value,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(amspub.prefix,opt,value,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%g>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsScalar"
/*@C
   PetscOptionsScalar - Gets the scalar value for a particular option in the database.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  defaultv - the default (current) value

   Output Parameter:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsScalar(const char opt[],const char text[],const char man[],PetscScalar defaultv,PetscScalar *value,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsReal(opt,text,man,defaultv,value,set);CHKERRQ(ierr);
#else
  ierr = PetscOptionsGetScalar(amspub.prefix,opt,value,set);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*
     Publishes an AMS logical field (with the default value in it) and with a name
   given by the text string
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsName"
/*@C
   PetscOptionsName - Determines if a particular option is in the database

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsName(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsList"
/*@C
     PetscOptionsList - Puts a list of option values that a single one may be selected from

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices
-  defaultv - the default (current) value

   Output Parameter:
+  value - the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate
   
   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsEList() for when the choices are given in a string array

   To get a listing of all currently specified options,
    see PetscOptionsPrint() or PetscOptionsGetAll()

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsList(const char opt[],const char ltext[],const char man[],PetscFList list,const char defaultv[],char value[],PetscInt len,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(amspub.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = PetscFListPrintTypes(amspub.comm,stdout,amspub.prefix,opt,ltext,man,list);CHKERRQ(ierr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEList"
/*@C
     PetscOptionsEList - Puts a list of option values that a single one may be selected from

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  ltext - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices
.  ntext - number of choices
-  defaultv - the default (current) value

   Output Parameter:
+  value - the index of the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE
   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsList() for when the choices are given in a PetscFList()

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsEList(const char opt[],const char ltext[],const char man[],const char *list[],PetscInt ntext,const char defaultv[],PetscInt *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  size_t         alen,len = 0;
  char           *svalue;
  PetscTruth     aset,flg;
  PetscInt       i;

  PetscFunctionBegin;
  for ( i=0; i<ntext; i++) {
    ierr = PetscStrlen(list[i],&alen);CHKERRQ(ierr);
    if (alen > len) len = alen;
  }
  len += 5; /* a little extra space for user mistypes */
  ierr = PetscMalloc(len*sizeof(char),&svalue);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(amspub.prefix,opt,svalue,len,&aset);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%s> (one of)",amspub.prefix?amspub.prefix:"",opt+1,defaultv);CHKERRQ(ierr);
    for (i=0; i<ntext; i++){
      ierr = (*PetscHelpPrintf)(amspub.comm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(amspub.comm,"\n");CHKERRQ(ierr);
  }
  if (aset) {
    if (set) *set = PETSC_TRUE;
    for (i=0; i<ntext; i++) {
      ierr = PetscStrcmp(svalue,list[i],&flg);CHKERRQ(ierr);
      if (flg) {
        *value = i;
        ierr = PetscFree(svalue);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscFree(svalue);CHKERRQ(ierr);
    SETERRQ3(PETSC_ERR_USER,"Unknown option %s for -%s%s",svalue,amspub.prefix?amspub.prefix:"",opt+1);
  } else if (set) {
    ierr = PetscFree(svalue);CHKERRQ(ierr);
    *set = PETSC_FALSE;
  } else {
    ierr = PetscFree(svalue);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLogicalGroupBegin"
/*@C
     PetscOptionsLogicalGroupBegin - First in a series of logical queries on the options database for
       which only a single value can be true.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - whether that option was set or not
   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must be followed by 0 or more PetscOptionsLogicalGroup()s and PetscOptionsLogicalGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroupBegin(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  Pick at most one of -------------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(amspub.comm,"    -%s%s: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLogicalGroup"
/*@C
     PetscOptionsLogicalGroup - One in a series of logical queries on the options database for
       which only a single value can be true.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE
   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsLogicalGroupBegin() and preceded a PetscOptionsLogicalGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroup(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"    -%s%s: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLogicalGroupEnd"
/*@C
     PetscOptionsLogicalGroupEnd - Last in a series of logical queries on the options database for
       which only a single value can be true.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE
   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsLogicalGroupBegin()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogicalGroupEnd(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"    -%s%s: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLogical"
/*@C
   PetscOptionsLogical - Determines if a particular option is in the database with a true or false

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE or PETSC_FALSE
.  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^logical

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsLogical(const char opt[],const char text[],const char man[],PetscTruth deflt,PetscTruth *flg,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscTruth     iset;

  PetscFunctionBegin;
  ierr = PetscOptionsGetLogical(amspub.prefix,opt,flg,&iset);CHKERRQ(ierr);
  if (!iset) {
    if (flg) *flg = deflt;
  }
  if (set) *set = iset;
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    const char *v = PetscTruths[deflt];
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s: <%s> %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,v,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsRealArray"
/*@C
   PetscOptionsRealArray - Gets an array of double values for a particular
   option in the database. The values must be separated with commas with 
   no intervening spaces. 

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  nmax - maximum number of values

   Output Parameter:
+  value - location to copy values
.  nmax - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes: 
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Concepts: options database^array of strings

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsRealArray(const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsGetRealArray(amspub.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%g",amspub.prefix?amspub.prefix:"",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(amspub.comm,",%g",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(amspub.comm,">: %s (%s)\n",text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsIntArray"
/*@C
   PetscOptionsIntArray - Gets an array of integers for a particular
   option in the database. The values must be separated with commas with 
   no intervening spaces. 

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  nmax - maximum number of values

   Output Parameter:
+  value - location to copy values
.  nmax - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes: 
   The user should pass in an array of integers

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Concepts: options database^array of strings

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsRealArray()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsIntArray(const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsGetIntArray(amspub.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%d",amspub.prefix?amspub.prefix:"",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(amspub.comm,",%d",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(amspub.comm,">: %s (%s)\n",text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsStringArray"
/*@C
   PetscOptionsStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with 
   no intervening spaces. 

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  nmax - maximum number of strings

   Output Parameter:
+  value - location to copy strings
.  nmax - actual number of strings found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes: 
   The user should pass in an array of pointers to char, to hold all the
   strings returned by this function.

   The user is responsible for deallocating the strings that are
   returned. The Fortran interface for this routine is not supported.

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Concepts: options database^array of strings

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsStringArray(const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetStringArray(amspub.prefix,opt,value,nmax,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <string1,string2,...>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsHead"
/*@C
     PetscOptionsHead - Puts a heading before list any more published options. Used, for example,
            in KSPSetFromOptions_GMRES().

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameter:
.   head - the heading text

   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          Must be followed by a call to PetscOptionsTail() in the same function.

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsHead(const char head[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  %s\n",head);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






