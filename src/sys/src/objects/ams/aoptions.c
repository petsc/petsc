/*$Id: aoptions.c,v 1.1 2000/01/06 20:50:44 bsmith Exp bsmith $*/
/*
   These routines simplify the use of command line, file options, etc.,
   and are used to manipulate the options database.

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

/*
    We keep a linked list of options that have been posted and we are waiting for 
   user selection

    Eventually we'll attach this beast to a MPI_Comm
*/
typedef enum {PETSC_OPTION_INT, PETSC_OPTION_NAME, PETSC_OPTION_DOUBLE} PetscOptionType;
typedef struct _p_PetscOptionsAMS* PetscOptionsAMS;
struct _p_PetscOptionsAMS {
  char            *option;
  char            *text;
  void            *data;
  PetscTruth      set;
  PetscOptionType type;
  PetscOptionsAMS next;
};

typedef struct {
  AMS_Memory      amem;
  PetscOptionsAMS next;
  int             lock;
  char            *prefix;
  char            *title;
} PetscOptionsPublish;
static PetscOptionsPublish amspub;

#undef __FUNC__  
#define __FUNC__ "OptionsSelectBegin"
int OptionsSelectBegin(MPI_Comm comm,char *prefix,char *title)
{
  AMS_Comm   acomm;
  int        ierr;

  PetscFunctionBegin;
  ierr = ViewerAMSGetAMSComm(VIEWER_AMS_(comm),&acomm);CHKERRQ(ierr);
  
  amspub.lock = 1;
  ierr = PetscStrallocpy(prefix,&amspub.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(prefix,&amspub.title);CHKERRQ(ierr);

  ierr = AMS_Memory_create(acomm,"Options",&amspub.amem);CHKERRQ(ierr);
  ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr); 

  ierr = AMS_Memory_add_field(amspub.amem,title,&amspub.lock,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amspub.amem,"lock",&amspub.lock,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSelectEnd"
int OptionsSelectEnd(MPI_Comm comm)
{
  int             ierr;
  PetscOptionsAMS last;
  char             option[256],value[256];

  PetscFunctionBegin;
  if (amspub.amem < 0) SETERRQ(1,1,"Called without a call to OptionsSelectBegin()");
  ierr = AMS_Memory_publish(amspub.amem);CHKERRQ(ierr);

  while (amspub.lock) {
    ierr = AMS_Memory_grant_access(amspub.amem);CHKERRQ(ierr);
    ierr = PetscSleep(10);CHKERRQ(ierr);
    ierr = AMS_Memory_take_access(amspub.amem);CHKERRQ(ierr);
  }

  /*
        Free all the options in the linked list
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
          sprintf(value,"%g",*(int*)amspub.next->data);
          break;
        case PETSC_OPTION_NAME:
          break;
      }
      ierr = OptionsSetValue(option,value);CHKERRQ(ierr);
    }
    ierr   = PetscStrfree(amspub.next->text);CHKERRQ(ierr);
    ierr   = PetscStrfree(amspub.next->option);CHKERRQ(ierr);
    if (amspub.next->data) {ierr = PetscFree(amspub.next->data);CHKERRQ(ierr);}
    last   = amspub.next;
    amspub.next = amspub.next->next;
    ierr   = PetscFree(last);CHKERRQ(ierr);
  }
  ierr = PetscStrfree(amspub.prefix);CHKERRQ(ierr); amspub.prefix = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSelectCreate"
static int OptionsSelectCreate(char *opt,char *text,PetscOptionsAMS *amsopt)
{
  int             ierr;
  PetscOptionsAMS next;

  PetscFunctionBegin;
  *amsopt         = PetscNew(struct _p_PetscOptionsAMS);CHKPTRQ(amsopt);
  (*amsopt)->next = 0;
  (*amsopt)->set  = PETSC_FALSE;
  ierr            = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr            = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amspub.amem,opt,&(*amsopt)->set,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  if (!amspub.next) {
    amspub.next = *amsopt;
  } else {
    next = amspub.next;
    while (next->next) next = next->next;
    next->next = *amsopt;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSelectInt"
int OptionsSelectInt(MPI_Comm comm,char *opt,char *text,int defaultv)
{
  int             ierr;
  PetscOptionsAMS amsopt;

  PetscFunctionBegin;
  ierr = OptionsSelectCreate(opt,text,&amsopt);CHKERRQ(ierr);
  amsopt->type        = PETSC_OPTION_INT;
  amsopt->data        = (void *) PetscMalloc(sizeof(int));CHKERRQ(ierr);
  *(int*)amsopt->data = defaultv;

  ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSelectDouble"
int OptionsSelectDouble(MPI_Comm comm,char *opt,char *text,double defaultv)
{
  int             ierr;
  PetscOptionsAMS amsopt;

  PetscFunctionBegin;
  ierr = OptionsSelectCreate(opt,text,&amsopt);CHKERRQ(ierr);
  amsopt->type           = PETSC_OPTION_DOUBLE;
  amsopt->data           = (void *) PetscMalloc(sizeof(double));CHKERRQ(ierr);
  *(double*)amsopt->data = defaultv;

  ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_DOUBLE,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSelectName"
int OptionsSelectName(MPI_Comm comm,char *opt,char *text)
{
  int             ierr;
  PetscOptionsAMS amsopt;

  PetscFunctionBegin;
  ierr = OptionsSelectCreate(opt,text,&amsopt);CHKERRQ(ierr);
  amsopt->type        = PETSC_OPTION_NAME;
  amsopt->data        = (void *) PetscMalloc(sizeof(int));CHKERRQ(ierr);
  *(int*)amsopt->data = 0;

  ierr = AMS_Memory_add_field(amspub.amem,text,amsopt->data,1,AMS_INT,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


