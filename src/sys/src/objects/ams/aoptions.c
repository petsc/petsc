/*$Id: aoptions.c,v 1.34 2001/08/31 16:19:18 bsmith Exp $*/
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

#if defined(PETSC_HAVE_AMS)
/*
    We keep a linked list of options that have been posted and we are waiting for 
   user selection

    Eventually we'll attach this beast to a MPI_Comm
*/
typedef enum {OPTION_INT,OPTION_LOGICAL,OPTION_REAL,OPTION_LIST,OPTION_STRING,OPTION_REAL_ARRAY,OPTION_HEAD} OptionType;
typedef struct _p_OptionsAMS* PetscOptionsAMS;
struct _p_OptionsAMS {
  char            *option;
  char            *text;
  void            *data;
  void            *edata;
  int             arraylength;
  PetscTruth      set;
  OptionType      type;
  PetscOptionsAMS next;
  char            *man;
};
#endif

typedef struct {
#if defined(PETSC_HAVE_AMS)
  AMS_Memory      amem;
  PetscOptionsAMS next;
#endif
  char            *prefix,*mprefix;  /* publish mprefix, not prefix cause the AMS will change it BUT we need to free it*/
  char            *title;
  MPI_Comm        comm;
  PetscTruth      printhelp;
  PetscTruth      changedmethod;
} PetscOptionsPublishObject;
static PetscOptionsPublishObject amspub;
int PetscOptionsPublishCount;

/*MC
    PetscOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

   Synopsis: int PetscOptionsBegin(MPI_Comm comm,char *prefix,char *title,char *mansec)

    Collective on MPI_Comm

  Input Parameters:
+   comm - communicator that shares GUI
.   prefix - options prefix for all options displayed on window
.   title - short descriptive text, for example "Krylov Solver Options"
-   mansec - section of manual pages for options, for example KSP

  Level: intermediate

  Notes: Needs to be ended by a call the PetscOptionsEnd()

         Can add subheadings with PetscOptionsHead()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsBegin_Private"
int PetscOptionsBegin_Private(MPI_Comm comm,char *prefix,char *title,char *mansec)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(prefix,&amspub.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&amspub.title);CHKERRQ(ierr);
  amspub.comm   = comm;
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&amspub.printhelp);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount) {
    ierr = (*PetscHelpPrintf)(comm,"%s -------------------------------------------------\n",title);CHKERRQ(ierr);
  }
 
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    AMS_Comm   acomm;
    static int count = 0;
    char       options[16];
    /* the next line is a bug, this will only work if all processors are here, the comm passed in is ignored!!! */
    ierr = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(PETSC_COMM_WORLD),&acomm);CHKERRQ(ierr);
    sprintf(options,"Options_%d",count++);
    ierr = AMS_Memory_create(acomm,options,&amspub.amem);CHKERRQ(ierr);
    ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr); 
    amspub.mprefix = amspub.prefix;
    ierr = AMS_Memory_add_field(amspub.amem,title,&amspub.mprefix,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,mansec,&amspub.mprefix,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    amspub.changedmethod = PETSC_FALSE;
    ierr = AMS_Memory_add_field(amspub.amem,"ChangedMethod",&amspub.changedmethod,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Collective on the MPI_Comm used in PetscOptionsBegin()

   Synopsis: int PetscOptionsEnd(void)

  Level: intermediate

  Notes: Needs to be preceded by a call to PetscOptionsBegin()

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

M*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEnd_Private"
int PetscOptionsEnd_Private(void)
{
  int ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS last;
    char       option[256],value[1024],tmp[32];
    int        j;

    if (amspub.amem < 0) SETERRQ(1,"Called without a call to PetscOptionsBegin()");
    ierr = AMS_Memory_publish(amspub.amem);CHKERRQ(ierr);
    ierr = AMS_Memory_grant_access(amspub.amem);CHKERRQ(ierr);
    /* wait until accessor has unlocked the memory */
    ierr = AMS_Memory_lock(amspub.amem,0);CHKERRQ(ierr);
    ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr);

    /* reset counter to -2; this updates the screen with the new options for the selected method */
    if (amspub.changedmethod) PetscOptionsPublishCount = -2; 

    /*
        Free all the PetscOptions in the linked list and add any changed ones to the database
    */
    while (amspub.next) {
      if (amspub.next->set) {
        if (amspub.prefix) {
          ierr = PetscStrcpy(option,"-");CHKERRQ(ierr);
          ierr = PetscStrcat(option,amspub.prefix);CHKERRQ(ierr);
          ierr = PetscStrcat(option,amspub.next->option+1);CHKERRQ(ierr);
        } else {
          ierr = PetscStrcpy(option,amspub.next->option);CHKERRQ(ierr);
        }

        switch (amspub.next->type) {
          case OPTION_HEAD:
            break;
          case OPTION_INT: 
            sprintf(value,"%d",*(int*)amspub.next->data);
            break;
          case OPTION_REAL:
            sprintf(value,"%g",*(double*)amspub.next->data);
            break;
          case OPTION_REAL_ARRAY:
            sprintf(value,"%g",((PetscReal*)amspub.next->data)[0]);
            for (j=1; j<amspub.next->arraylength; j++) {
              sprintf(tmp,"%g",((PetscReal*)amspub.next->data)[j]);
              ierr = PetscStrcat(value,",");CHKERRQ(ierr);
              ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
            }
            break;
          case OPTION_LOGICAL:
            sprintf(value,"%d",*(int*)amspub.next->data);
            break;
          case OPTION_LIST:
            ierr = PetscStrcpy(value,*(char**)amspub.next->data);CHKERRQ(ierr);
            break;
          case OPTION_STRING: /* also handles string arrays */
            ierr = PetscStrcpy(value,*(char**)amspub.next->data);CHKERRQ(ierr);
            break;
        }
        ierr = PetscOptionsSetValue(option,value);CHKERRQ(ierr);
      }
      ierr   = PetscStrfree(amspub.next->text);CHKERRQ(ierr);
      ierr   = PetscStrfree(amspub.next->option);CHKERRQ(ierr);
      ierr   = PetscFree(amspub.next->man);CHKERRQ(ierr);
      if (amspub.next->data)  {ierr = PetscFree(amspub.next->data);CHKERRQ(ierr);}
      if (amspub.next->edata) {ierr = PetscFree(amspub.next->edata);CHKERRQ(ierr);}
      last        = amspub.next;
      amspub.next = amspub.next->next;
      ierr        = PetscFree(last);CHKERRQ(ierr);
    }
    ierr = AMS_Memory_grant_access(amspub.amem);CHKERRQ(ierr);
    ierr = AMS_Memory_destroy(amspub.amem);CHKERRQ(ierr);
  }
#endif
  ierr = PetscStrfree(amspub.title);CHKERRQ(ierr); amspub.title  = 0;
  ierr = PetscStrfree(amspub.prefix);CHKERRQ(ierr); amspub.prefix = 0;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_AMS)
/*
     Publishes the "lock" for an option; with a name that is the command line
   option name. This is the first item that is always published for an option
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCreate_Private"
static int PetscOptionsCreate_Private(char *opt,char *text,char *man,PetscOptionsAMS *amsopt)
{
  int             ierr;
  static int      mancount = 0;
  PetscOptionsAMS next;
  char            manname[16];

  PetscFunctionBegin;
  ierr             = PetscNew(struct _p_OptionsAMS,amsopt);CHKERRQ(ierr);
  (*amsopt)->next  = 0;
  (*amsopt)->set   = PETSC_FALSE;
  (*amsopt)->data  = 0;
  (*amsopt)->edata = 0;
  ierr             = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr             = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amspub.amem,opt,&(*amsopt)->set,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  sprintf(manname,"man_%d",mancount++);
  ierr                     = PetscMalloc(sizeof(char*),&(*amsopt)->man);CHKERRQ(ierr);
  *(char **)(*amsopt)->man = man;
  ierr = AMS_Memory_add_field(amspub.amem,manname,(*amsopt)->man,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  if (!amspub.next) {
    amspub.next = *amsopt;
  } else {
    next = amspub.next;
    while (next->next) next = next->next;
    next->next = *amsopt;
  }
  PetscFunctionReturn(0);
}
#endif

/* -------------------------------------------------------------------------------------------------------------*/
/*
     Publishes an AMS int field (with the default value in it) and with a name
   given by the text string
*/
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
int PetscOptionsInt(char *opt,char *text,char *man,int defaultv,int *value,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_INT;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsString(char *opt,char *text,char *man,char *defaultv,char *value,int len,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type          = OPTION_STRING;
    ierr = PetscMalloc(sizeof(char*),&amsopt->data);CHKERRQ(ierr);
    *(char**)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsReal(char *opt,char *text,char *man,PetscReal defaultv,PetscReal *value,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type           = OPTION_REAL;
    ierr = PetscMalloc(sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);
    *(PetscReal*)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsScalar(char *opt,char *text,char *man,PetscScalar defaultv,PetscScalar *value,PetscTruth *set)
{
  int ierr;

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
int PetscOptionsName(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_LOGICAL;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (flg) *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
int PetscOptionsList(char *opt,char *ltext,char *man,PetscFList list,char *defaultv,char *value,int len,PetscTruth *set)
{
  int        ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    int        ntext;
    char       ldefault[128];

    ierr = PetscOptionsCreate_Private(opt,ltext,man,&amsopt);CHKERRQ(ierr);
    amsopt->type             = OPTION_LIST;
    ierr = PetscMalloc(sizeof(char*),&amsopt->data);CHKERRQ(ierr);
    *(char **)(amsopt->data) = defaultv;
    ierr = PetscStrcpy(ldefault,"DEFAULT:");CHKERRQ(ierr);
    ierr = PetscStrcat(ldefault,ltext);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,ldefault,amsopt->data,1,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

    ierr = PetscFListGet(list,(char***)&amsopt->edata,&ntext);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,ltext,amsopt->edata,ntext,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices
.  ntext - number of choices
.  defaultv - the default (current) value
-  len - the size of the output value array

   Output Parameter:
+  value - the value to return
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
int PetscOptionsEList(char *opt,char *ltext,char *man,char **list,int ntext,char *defaultv,char *value,int len,PetscTruth *set)
{
  int i,ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    char       ldefault[128];

    ierr = PetscOptionsCreate_Private(opt,ltext,man,&amsopt);CHKERRQ(ierr);
    amsopt->type             = OPTION_LIST;
    ierr = PetscMalloc(sizeof(char*),&amsopt->data);CHKERRQ(ierr);
    *(char **)(amsopt->data) = defaultv;
    ierr = PetscStrcpy(ldefault,"DEFAULT:");CHKERRQ(ierr);
    ierr = PetscStrcat(ldefault,ltext);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,ldefault,amsopt->data,1,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

    ierr = PetscMalloc((ntext+1)*sizeof(char**),&amsopt->edata);CHKERRQ(ierr);
    ierr = PetscMemcpy(amsopt->edata,list,ntext*sizeof(char*));CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,ltext,amsopt->edata,ntext,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = PetscOptionsGetString(amspub.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%s> (one of)",amspub.prefix?amspub.prefix:"",opt+1,defaultv);CHKERRQ(ierr);
    for (i=0; i<ntext; i++){
      ierr = (*PetscHelpPrintf)(amspub.comm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(amspub.comm,"\n");CHKERRQ(ierr);
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
int PetscOptionsLogicalGroupBegin(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_LOGICAL;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = 1; /* the first one listed is always the default */
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (flg) *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsLogicalGroup(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_LOGICAL;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (flg) *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsLogicalGroupEnd(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_LOGICAL;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (flg) *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsLogical(char *opt,char *text,char *man,PetscTruth deflt,PetscTruth *flg,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = OPTION_LOGICAL;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = (int)deflt;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (flg) *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = PetscOptionsGetLogical(amspub.prefix,opt,flg,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    const char *v = (deflt ? "true" : "false");
    ierr = (*PetscHelpPrintf)(amspub.comm,"    -%s%s: <%s> %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,v,text,man);CHKERRQ(ierr);
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
int PetscOptionsRealArray(char *opt,char *text,char *man,PetscReal *value,int *n,PetscTruth *set)
{
  int             ierr,i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type           = OPTION_REAL_ARRAY;
    amsopt->arraylength    = *n;
    ierr = PetscMalloc((*n)*sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);
    ierr                   = PetscMemcpy(amsopt->data,value,(*n)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,*n,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
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
int PetscOptionsStringArray(char *opt,char *text,char *man,char **value,int *nmax,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type          = OPTION_STRING;
    ierr = PetscMalloc((*nmax)*sizeof(char*),&amsopt->data);CHKERRQ(ierr);
    ierr                  = PetscMemzero(amsopt->data,(*nmax)*sizeof(char*));CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,*nmax,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = PetscOptionsGetStringArray(amspub.prefix,opt,value,nmax,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <string1,string2,...>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsTail - Ends a section of options begun with PetscOptionsHead()
            See, for example, KSPSetFromOptions_GMRES().

   Collective on the communicator passed in PetscOptionsBegin()

   Synopsis: int PetscOptionsTail(void)

  Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          Must be preceded by a call to PetscOptionsHead() in the same function.

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
M*/

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
int PetscOptionsHead(char *head)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private("-amshead",head,"None",&amsopt);CHKERRQ(ierr);
    amsopt->type = OPTION_HEAD;
    ierr = PetscMalloc(sizeof(int),&amsopt->data);CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,head,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  }
#endif
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  %s\n",head);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}






