/*$Id: aoptions.c,v 1.26 2001/03/09 04:13:10 bsmith Exp balay $*/
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
typedef enum {OPTION_INT,OPTION_LOGICAL,OPTION_DOUBLE,OPTION_LIST,OPTION_STRING,OPTION_DOUBLE_ARRAY,OPTION_HEAD} OptionType;
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
  AMS_Memory amem;
  PetscOptionsAMS next;
#endif
  char       *prefix,*mprefix;  /* publish mprefix, not prefix cause the AMS will change it BUT we need to free it*/
  char       *title;
  MPI_Comm   comm;
  PetscTruth printhelp;
  PetscTruth changedmethod;
} PetscOptionsPublishObject;
static PetscOptionsPublishObject amspub;
int PetscOptionsPublishCount;

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
          case OPTION_DOUBLE:
            sprintf(value,"%g",*(double*)amspub.next->data);
            break;
          case OPTION_DOUBLE_ARRAY:
            sprintf(value,"%g",((double*)amspub.next->data)[0]);
            for (j=1; j<amspub.next->arraylength; j++) {
              sprintf(tmp,"%g",((double*)amspub.next->data)[j]);
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
#define __FUNCT__ "PetscOptionsDouble"
int PetscOptionsDouble(char *opt,char *text,char *man,double defaultv,double *value,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type           = OPTION_DOUBLE;
    ierr = PetscMalloc(sizeof(double),&amsopt->data);CHKERRQ(ierr);
    *(double*)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = PetscOptionsGetDouble(amspub.prefix,opt,value,set);CHKERRQ(ierr);
  if (amspub.printhelp && PetscOptionsPublishCount == 1) {
    ierr = (*PetscHelpPrintf)(amspub.comm,"  -%s%s <%g>: %s (%s)\n",amspub.prefix?amspub.prefix:"",opt+1,defaultv,text,man);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsScalar"
int PetscOptionsScalar(char *opt,char *text,char *man,Scalar defaultv,Scalar *value,PetscTruth *set)
{
  int ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsDouble(opt,text,man,defaultv,value,set);CHKERRQ(ierr);
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

/*
     Publishes a single string (the default) with a name given by the DEFAULT: + text
  and an AMS array of strings which are to be ed from with a name given by the text

*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsList"
int PetscOptionsList(char *opt,char *ltext,char *man,PetscFList list,char *defaultv,char *value,int len,PetscTruth *set)
{
  int   ierr;

#if defined(PETSC_HAVE_AMS)
  PetscFList next = list;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    int        ntext,i;
    char       ldefault[128],**vtext;

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

/*
     Publishes a single string (the default) with a name given by the DEFAULT: + text
  and an AMS array of strings which are to be ed from with a name given by the text

*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsList"
int PetscOptionsEList(char *opt,char *ltext,char *man,char **list,int ntext,char *defaultv,char *value,int len,PetscTruth *set)
{
  int i,ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    char       ldefault[128],**vtext;

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

/*
     Publishes an AMS logical field, only one in a group can be on
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLogicalGroup"
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
#define __FUNCT__ "PetscOptionsLogicalGroup"
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

/*
     Publishes an AMS double field (with the default value in it) and with a name
   given by the text string
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsDoubleArray"
int PetscOptionsDoubleArray(char *opt,char *text,char *man,double *value,int *n,PetscTruth *set)
{
  int             ierr,i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = PetscOptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type           = OPTION_DOUBLE_ARRAY;
    amsopt->arraylength    = *n;
    ierr = PetscMalloc((*n)*sizeof(double),&amsopt->data);CHKERRQ(ierr);
    ierr                   = PetscMemcpy(amsopt->data,value,(*n)*sizeof(double));CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,*n,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = PetscOptionsGetDoubleArray(amspub.prefix,opt,value,n,set);CHKERRQ(ierr);
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

/*
    Put a subheading into the GUI list of PetscOptions
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsHead"
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

