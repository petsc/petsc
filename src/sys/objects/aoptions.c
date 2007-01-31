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

/*
    Keep a linked list of options that have been posted and we are waiting for 
   user selection

    Eventually we'll attach this beast to a MPI_Comm
*/
typedef enum {OPTION_INT,OPTION_LOGICAL,OPTION_REAL,OPTION_LIST,OPTION_STRING,OPTION_REAL_ARRAY,OPTION_HEAD} OptionType;
typedef struct _p_Options* PetscOptions;
struct _p_Options {
  char         *option;
  char         *text;
  void         *data;
  void         *edata;
  char         *man;
  int          arraylength;
  PetscTruth   set;
  OptionType   type;
  PetscOptions next;
};

typedef struct _p_OptionsHelp* OptionsHelp;
struct _p_OptionsHelp {
  char        *prefix;
  char        *title;
  char        *mansec;
  OptionsHelp next;
};

static struct {
  PetscOptions    next;
  char            *prefix,*mprefix;  
  char            *title;
  MPI_Comm        comm;
  PetscTruth      printhelp,changedmethod,alreadyprinted;
  OptionsHelp     help;
}                                   PetscOptionsObject;
PetscInt                            PetscOptionsPublishCount = 0;


#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHelpAddList"
PetscErrorCode PetscOptionsHelpAddList(const char prefix[],const char title[],const char mansec[])
{
  int          ierr;
  OptionsHelp  newhelp;
  PetscFunctionBegin;
  ierr = PetscNew(struct _p_OptionsHelp,&newhelp);CHKERRQ(ierr);
  ierr = PetscStrallocpy(prefix,&newhelp->prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&newhelp->title);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mansec,&newhelp->mansec);CHKERRQ(ierr);
  newhelp->next = 0;

  if (!PetscOptionsObject.help) {
    PetscOptionsObject.help = newhelp;
  } else {
    newhelp->next = PetscOptionsObject.help;
    PetscOptionsObject.help = newhelp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHelpDestroyList"
PetscErrorCode PetscOptionsHelpDestroyList(void)
{
  PetscErrorCode ierr;
  OptionsHelp    help = PetscOptionsObject.help, next;

  PetscFunctionBegin;
  while (help) {
    next = help->next;
    ierr = PetscStrfree(help->prefix);CHKERRQ(ierr);
    ierr = PetscStrfree(help->title);CHKERRQ(ierr);
    ierr = PetscStrfree(help->mansec);CHKERRQ(ierr);
    ierr = PetscFree(help);CHKERRQ(ierr);
    help = next;
  }
  PetscFunctionReturn(0);
}
  

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHelpFindList"
PetscErrorCode PetscOptionsHelpFindList(const char prefix[],const char title[],const char mansec[],PetscTruth *flg)
{
  PetscErrorCode ierr;
  PetscTruth     flg1,flg2,flg3;
  OptionsHelp help = PetscOptionsObject.help;
  PetscFunctionBegin;
  while (help) {
    ierr = PetscStrcmp(help->prefix,prefix,&flg1);CHKERRQ(ierr);
    ierr = PetscStrcmp(help->title,title,&flg2);CHKERRQ(ierr);
    ierr = PetscStrcmp(help->mansec,mansec,&flg3);CHKERRQ(ierr);
    if (flg1 && flg2 && flg3) {
      *flg = PETSC_TRUE;
      break;
    }
    help = help->next;
  }
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHelpCheckAddList"
PetscErrorCode PetscOptionsHelpCheckAddList(const char prefix[],const char title[],const char mansec[],PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscOptionsHelpFindList(prefix,title,mansec,flg);
  if (!(*flg)) PetscOptionsHelpAddList(prefix,title,mansec);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsBegin_Private"
/*
    Handles setting up the data structure in a call to PetscOptionsBegin()
*/
PetscErrorCode PetscOptionsBegin_Private(MPI_Comm comm,const char prefix[],const char title[],const char mansec[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscOptionsObject.next          = 0;
  PetscOptionsObject.comm          = comm;
  PetscOptionsObject.changedmethod = PETSC_FALSE;
  ierr = PetscStrallocpy(prefix,&PetscOptionsObject.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&PetscOptionsObject.title);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&PetscOptionsObject.printhelp);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1) {
    ierr = PetscOptionsHelpCheckAddList(prefix,title,mansec,&PetscOptionsObject.alreadyprinted);
    if (!PetscOptionsObject.alreadyprinted) {
      ierr = (*PetscHelpPrintf)(comm,"%s -------------------------------------------------\n",title);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
     Handles adding another option to the list of options within this particular PetscOptionsBegin() PetscOptionsEnd()
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCreate_Private"
static int PetscOptionsCreate_Private(const char opt[],const char text[],const char man[],OptionType t,PetscOptions *amsopt)
{
  int          ierr;
  PetscOptions next;

  PetscFunctionBegin;
  ierr             = PetscNew(struct _p_Options,amsopt);CHKERRQ(ierr);
  (*amsopt)->next  = 0;
  (*amsopt)->set   = PETSC_FALSE;
  (*amsopt)->type  = t;
  (*amsopt)->data  = 0;
  (*amsopt)->edata = 0;

  ierr             = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr             = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr             = PetscStrallocpy(man,&(*amsopt)->man);CHKERRQ(ierr);

  if (!PetscOptionsObject.next) {
    PetscOptionsObject.next = *amsopt;
  } else {
    next = PetscOptionsObject.next;
    while (next->next) next = next->next;
    next->next = *amsopt;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetFromGui"
PetscErrorCode PetscOptionsGetFromGUI()
{
  PetscErrorCode ierr;
  PetscOptions   next = PetscOptionsObject.next;
  char           str[512];

  ierr = (*PetscPrintf)(PetscOptionsObject.comm,"%s -------------------------------------------------\n",PetscOptionsObject.title);CHKERRQ(ierr);
  while (next) {
    switch (next->type) {
      case OPTION_HEAD:
        break;
      case OPTION_INT: 
        ierr = PetscPrintf(PetscOptionsObject.comm,"-%s%s <%d>: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",next->option,*(int*)next->data,next->text,next->man);CHKERRQ(ierr);
        scanf("%s\n",str);
        if (str[0] != '\n') {
          printf("changing value\n");
        }
        break;
    default:
      break;
    }
    next = next->next;
  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEnd_Private"
PetscErrorCode PetscOptionsEnd_Private(void)
{
  PetscErrorCode ierr;
  PetscOptions   last;
  char           option[256],value[1024],tmp[32];
  PetscInt       j;

  PetscFunctionBegin;

  /*  if (PetscOptionsObject.next) { 
    ierr = PetscOptionsGetFromGUI();
    }*/

  ierr = PetscStrfree(PetscOptionsObject.title);CHKERRQ(ierr); PetscOptionsObject.title  = 0;
  ierr = PetscStrfree(PetscOptionsObject.prefix);CHKERRQ(ierr); PetscOptionsObject.prefix = 0;

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (PetscOptionsObject.changedmethod) PetscOptionsPublishCount = -2; 
  /* reset alreadyprinted flag */
  PetscOptionsObject.alreadyprinted = PETSC_FALSE;

  while (PetscOptionsObject.next) { 
    if (PetscOptionsObject.next->set) {
      if (PetscOptionsObject.prefix) {
        ierr = PetscStrcpy(option,"-");CHKERRQ(ierr);
        ierr = PetscStrcat(option,PetscOptionsObject.prefix);CHKERRQ(ierr);
        ierr = PetscStrcat(option,PetscOptionsObject.next->option+1);CHKERRQ(ierr);
      } else {
        ierr = PetscStrcpy(option,PetscOptionsObject.next->option);CHKERRQ(ierr);
      }

      switch (PetscOptionsObject.next->type) {
        case OPTION_HEAD:
          break;
        case OPTION_INT: 
          sprintf(value,"%d",*(PetscInt*)PetscOptionsObject.next->data);
          break;
        case OPTION_REAL:
          sprintf(value,"%g",*(PetscReal*)PetscOptionsObject.next->data);
          break;
        case OPTION_REAL_ARRAY:
          sprintf(value,"%g",((PetscReal*)PetscOptionsObject.next->data)[0]);
          for (j=1; j<PetscOptionsObject.next->arraylength; j++) {
            sprintf(tmp,"%g",((PetscReal*)PetscOptionsObject.next->data)[j]);
            ierr = PetscStrcat(value,",");CHKERRQ(ierr);
            ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
          }
          break;
        case OPTION_LOGICAL:
          sprintf(value,"%d",*(PetscInt*)PetscOptionsObject.next->data);
          break;
        case OPTION_LIST:
          ierr = PetscStrcpy(value,*(char**)PetscOptionsObject.next->data);CHKERRQ(ierr);
          break;
        case OPTION_STRING: /* also handles string arrays */
          ierr = PetscStrcpy(value,*(char**)PetscOptionsObject.next->data);CHKERRQ(ierr);
          break;
      }
      ierr = PetscOptionsSetValue(option,value);CHKERRQ(ierr);
    }
    ierr   = PetscStrfree(PetscOptionsObject.next->text);CHKERRQ(ierr);
    ierr   = PetscStrfree(PetscOptionsObject.next->option);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->man);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->data);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->edata);CHKERRQ(ierr);
    last                    = PetscOptionsObject.next;
    PetscOptionsObject.next = PetscOptionsObject.next->next;
    ierr                    = PetscFree(last);CHKERRQ(ierr);
  }
  PetscOptionsObject.next = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEnum"
/*@C
   PetscOptionsEnum - Gets the enum value for a particular option in the database.

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  defaultv - the default (current) value

   Output Parameter:
+  value - the  value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          list is usually something like PCASMTypes or some other predefined list of enum names

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsEnum(const char opt[],const char text[],const char man[],const char **list,PetscEnum defaultv,PetscEnum *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       ntext = 0;

  PetscFunctionBegin;
  while (list[ntext++]) {
    if (ntext > 50) SETERRQ(PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  if (ntext < 3) SETERRQ(PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  ierr = PetscOptionsEList(opt,text,man,list,ntext,list[defaultv],(PetscInt*)value,set);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsInt(const char opt[],const char text[],const char man[],PetscInt defaultv,PetscInt *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (PetscOptionsPublishCount == 1) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_INT,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt),&amsopt->data);CHKERRQ(ierr);
    *(PetscInt*)amsopt->data = defaultv;
  }
  ierr = PetscOptionsGetInt(PetscOptionsObject.prefix,opt,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%d>: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsString(const char opt[],const char text[],const char man[],const char defaultv[],char value[],size_t len,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(PetscOptionsObject.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s>: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsReal(const char opt[],const char text[],const char man[],PetscReal defaultv,PetscReal *value,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(PetscOptionsObject.prefix,opt,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%G>: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsScalar(const char opt[],const char text[],const char man[],PetscScalar defaultv,PetscScalar *value,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsReal(opt,text,man,defaultv,value,set);CHKERRQ(ierr);
#else
  ierr = PetscOptionsGetScalar(PetscOptionsObject.prefix,opt,value,set);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsName(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PetscOptionsObject.prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,text,man);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsList(const char opt[],const char ltext[],const char man[],PetscFList list,const char defaultv[],char value[],PetscInt len,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(PetscOptionsObject.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = PetscFListPrintTypes(PetscOptionsObject.comm,stdout,PetscOptionsObject.prefix,opt,ltext,man,list);CHKERRQ(ierr);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsEList(const char opt[],const char ltext[],const char man[],const char **list,PetscInt ntext,const char defaultv[],PetscInt *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsGetEList(PetscOptionsObject.prefix,opt,list,ntext,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s> (choose one of)",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,defaultv);CHKERRQ(ierr);
    for (i=0; i<ntext; i++){
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsTruthGroupBegin"
/*@C
     PetscOptionsTruthGroupBegin - First in a series of logical queries on the options database for
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

   Must be followed by 0 or more PetscOptionsTruthGroup()s and PetscOptionsTruthGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroupBegin(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PetscOptionsObject.prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  Pick at most one of -------------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsTruthGroup"
/*@C
     PetscOptionsTruthGroup - One in a series of logical queries on the options database for
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

   Must follow a PetscOptionsTruthGroupBegin() and preceded a PetscOptionsTruthGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroup(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PetscOptionsObject.prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsTruthGroupEnd"
/*@C
     PetscOptionsTruthGroupEnd - Last in a series of logical queries on the options database for
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

   Must follow a PetscOptionsTruthGroupBegin()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruthGroupEnd(const char opt[],const char text[],const char man[],PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PetscOptionsObject.prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsTruth"
/*@C
   PetscOptionsTruth - Determines if a particular option is in the database with a true or false

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsTruth(const char opt[],const char text[],const char man[],PetscTruth deflt,PetscTruth *flg,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscTruth     iset;

  PetscFunctionBegin;
  ierr = PetscOptionsGetTruth(PetscOptionsObject.prefix,opt,flg,&iset);CHKERRQ(ierr);
  if (!iset) {
    if (flg) *flg = deflt;
  }
  if (set) *set = iset;
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    const char *v = PetscTruths[deflt];
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s: <%s> %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,v,text,man);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsRealArray(const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsGetRealArray(PetscOptionsObject.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%G",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%G",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (%s)\n",text,man);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsRealArray()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsIntArray(const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsGetIntArray(PetscOptionsObject.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%d",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%d",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (%s)\n",text,man);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsStringArray(const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscTruth *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetStringArray(PetscOptionsObject.prefix,opt,value,nmax,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <string1,string2,...>: %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsHead"
/*@C
     PetscOptionsHead - Puts a heading before listing any more published options. Used, for example,
            in KSPSetFromOptions_GMRES().

   Collective on the communicator passed in PetscOptionsBegin()

   Input Parameter:
.   head - the heading text

   
   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          Can be followed by a call to PetscOptionsTail() in the same function.

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsHead(const char head[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  %s\n",head);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






