/*$Id: aoptions.c,v 1.14 2000/08/23 17:11:02 bsmith Exp bsmith $*/
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
typedef enum {PETSC_OPTION_INT,PETSC_OPTION_LOGICAL,PETSC_OPTION_DOUBLE,PETSC_OPTION_LIST,
              PETSC_OPTION_STRING} PetscOptionType;
typedef struct _p_PetscOptionsAMS* PetscOptionsAMS;
struct _p_PetscOptionsAMS {
  char            *option;
  char            *text;
  void            *data;
  void            *edata;
  PetscTruth      set;
  PetscOptionType type;
  PetscOptionsAMS next;
  char            manname[16],man[128];
};
#endif

typedef struct {
#if defined(PETSC_HAVE_AMS)
  AMS_Memory      amem;
  PetscOptionsAMS next;
#endif
  char            *prefix;
  char            *title;
  MPI_Comm        comm;
} PetscOptionsPublish;
static PetscOptionsPublish amspub;
int PetscOptionsPublishCount;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsBegin_Private"
int OptionsBegin_Private(MPI_Comm comm,char *prefix,char *title)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(prefix,&amspub.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(prefix,&amspub.title);CHKERRQ(ierr);
  amspub.comm = comm;

#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    AMS_Comm   acomm;
    static int count = 0;
    char       options[16];
    ierr = ViewerAMSGetAMSComm(VIEWER_AMS_(comm),&acomm);CHKERRQ(ierr);
    sprintf(options,"Options_%d",count++);
    ierr = AMS_Memory_create(acomm,options,&amspub.amem);CHKERRQ(ierr);
    ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr); 
    ierr = AMS_Memory_add_field(amspub.amem,title,&amspub.title,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsEnd_Private"
int OptionsEnd_Private(void)
{
  int ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS last;
    char            option[256],value[256];
    if (amspub.amem < 0) SETERRQ(1,1,"Called without a call to OptionsBegin()");
    ierr = AMS_Memory_publish(amspub.amem);CHKERRQ(ierr);
    ierr = AMS_Memory_grant_access(amspub.amem);CHKERRQ(ierr);
    /* wait until accessor has unlocked the memory */
    ierr = AMS_Memory_lock(amspub.amem,0);CHKERRQ(ierr);
    ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr);

    /*
        Free all the options in the linked list and add any changed ones to the database
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
          case PETSC_OPTION_INT: 
            sprintf(value,"%d",*(int*)amspub.next->data);
            break;
          case PETSC_OPTION_DOUBLE:
            sprintf(value,"%g",*(double*)amspub.next->data);
            break;
          case PETSC_OPTION_LOGICAL:
            sprintf(value,"%d",*(int*)amspub.next->data);
            break;
          case PETSC_OPTION_LIST:
            ierr = PetscStrcpy(value,*(char**)amspub.next->data);CHKERRQ(ierr);
            break;
          case PETSC_OPTION_STRING:
            ierr = PetscStrcpy(value,*(char**)amspub.next->data);CHKERRQ(ierr);
            break;
        }
        ierr = OptionsSetValue(option,value);CHKERRQ(ierr);
      }
      ierr   = PetscStrfree(amspub.next->text);CHKERRQ(ierr);
      ierr   = PetscStrfree(amspub.next->option);CHKERRQ(ierr);
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
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsCreate_Private"
static int OptionsCreate_Private(char *opt,char *text,char *man,PetscOptionsAMS *amsopt)
{
  int             ierr;
  static int      mancount = 0;
  PetscOptionsAMS next;

  PetscFunctionBegin;
  *amsopt          = PetscNew(struct _p_PetscOptionsAMS);CHKPTRQ(amsopt);
  (*amsopt)->next  = 0;
  (*amsopt)->set   = PETSC_FALSE;
  (*amsopt)->data  = 0;
  (*amsopt)->edata = 0;
  ierr             = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr             = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amspub.amem,opt,&(*amsopt)->set,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  sprintf((*amsopt)->manname,"man_%d",mancount++);
  ierr = PetscStrcpy((*amsopt)->man,man);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amspub.amem,(*amsopt)->manname,&(*amsopt)->man,1,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

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
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsInt"
int OptionsInt(char *opt,char *text,char *man,int defaultv,int *value,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = PETSC_OPTION_INT;
    amsopt->data        = (void *)PetscMalloc(sizeof(int));CHKERRQ(ierr);
    *(int*)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsGetInt(amspub.prefix,opt,value,set);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Publishes an AMS double field (with the default value in it) and with a name
   given by the text string
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsDouble"
int OptionsDouble(char *opt,char *text,char *man,double defaultv,double *value,PetscTruth *set)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type           = PETSC_OPTION_DOUBLE;
    amsopt->data           = (void *)PetscMalloc(sizeof(double));CHKERRQ(ierr);
    *(double*)amsopt->data = defaultv;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsGetDouble(amspub.prefix,opt,value,set);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     Publishes an AMS logical field (with the default value in it) and with a name
   given by the text string
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsName"
int OptionsName(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = PETSC_OPTION_LOGICAL;
    amsopt->data        = (void *)PetscMalloc(sizeof(int));CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     Publishes a single string (the default) with a name given by the DEFAULT: + text
  and an AMS array of strings which are to be ed from with a name given by the text

*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsList"
int OptionsList(char *opt,char *ltext,char *man,char **text,int ntext,char *defaultv,char *value,int len,PetscTruth *set)
{
  int ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    int             mlen,tlen,i;
    char            ldefault[128],*mtext,**vtext;

    ierr = OptionsCreate_Private(opt,ltext,man,&amsopt);CHKERRQ(ierr);
    amsopt->type             = PETSC_OPTION_LIST;
    amsopt->data             = (void *)PetscMalloc(sizeof(char*));CHKERRQ(ierr);
    *(char **)(amsopt->data) = defaultv;
    ierr = PetscStrcpy(ldefault,"DEFAULT:");CHKERRQ(ierr);
    ierr = PetscStrcat(ldefault,ltext);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(amspub.amem,ldefault,amsopt->data,1,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

    /* need to copy text array so they are not freed before AMS uses them */
    tlen = 0;
    for (i=0; i<ntext; i++) {
      ierr = PetscStrlen(text[i],&mlen);CHKERRQ(ierr);
      tlen += mlen + 1;
    }
    amsopt->edata = (void *)PetscMalloc(tlen*sizeof(char)+ntext*sizeof(char*));CHKPTRQ(amsopt->edata);
    vtext         = (char**)amsopt->edata;
    mtext         = (char*)(vtext+ntext);
    for (i=0; i<ntext; i++) {
      vtext[i]    = mtext;
      ierr        = PetscStrlen(text[i],&mlen);CHKERRQ(ierr);
      ierr        = PetscStrcpy(mtext,text[i]);CHKERRQ(ierr);
      mtext      += mlen+1;
    }
    ierr = AMS_Memory_add_field(amspub.amem,ltext,vtext,ntext,AMS_STRING,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsGetString(amspub.prefix,opt,value,len,set);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     Publishes an AMS logical field, only one in a group can be on
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsLogicalGroup"
int OptionsLogicalGroupBegin(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = PETSC_OPTION_LOGICAL;
    amsopt->data        = (void *)PetscMalloc(sizeof(int));CHKERRQ(ierr);
    *(int*)amsopt->data = 1; /* the first one listed is always the default */
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsLogicalGroup"
int OptionsLogicalGroup(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = PETSC_OPTION_LOGICAL;
    amsopt->data        = (void *)PetscMalloc(sizeof(int));CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsLogicalGroup"
int OptionsLogicalGroupEnd(char *opt,char *text,char *man,PetscTruth *flg)
{
  int             ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_AMS)
  if (!PetscOptionsPublishCount) {
    PetscOptionsAMS amsopt;
    ierr = OptionsCreate_Private(opt,text,man,&amsopt);CHKERRQ(ierr);
    amsopt->type        = PETSC_OPTION_LOGICAL;
    amsopt->data        = (void *)PetscMalloc(sizeof(int));CHKERRQ(ierr);
    *(int*)amsopt->data = 0;
    ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#endif
  ierr = OptionsHasName(amspub.prefix,opt,flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}





