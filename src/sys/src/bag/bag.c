#define PETSC_DLL

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"
#include "petscbag.h"     /*I  "petscbag.h"   I*/

/*
   Ugly variable to indicate if we are inside a PetscBagLoad() and should not call PetscOptions....
*/
static PetscTruth PetscBagInLoad = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterEnum"
/*@C
   PetscBagRegisterEnum - add an enum value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of enum in struct
.  mdefault - the initial value
.  name - name of the enum
-  help - longer string with more information about the value

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterEnum(PetscBag* bag,void *addr,PetscEnum mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%d>: %s \n",nname,mdefault,help);CHKERRQ(ierr);
    }
    ierr     = PetscOptionsGetInt(PETSC_NULL,nname,&mdefault,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_ENUM;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = 1;
  *(PetscEnum*)addr = mdefault;
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterInt"
/*@C
   PetscBagRegisterInt - add an integer value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of integer in struct
.  mdefault - the initial value
.  name - name of the integer
-  help - longer string with more information about the value

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterInt(PetscBag* bag,void *addr,PetscInt mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%d>: %s \n",nname,mdefault,help);CHKERRQ(ierr);
    }
    ierr     = PetscOptionsGetInt(PETSC_NULL,nname,&mdefault,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = 1;
  *(PetscInt*)addr = mdefault;
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterString"
/*@C
   PetscBagRegisterString - add a string value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of start of string in struct
.  msize - length of the string space in the struct
.  mdefault - the initial value
.  name - name of the string
-  help - longer string with more information about the value

   Level: beginner

   Note: The struct should have the field char mystring[msize]; not char *mystring

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterString(PetscBag* bag,void *addr,size_t msize,const char* mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%s>: %s \n",nname,mdefault,help);CHKERRQ(ierr);
    }
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_CHAR;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = msize;
  ierr = PetscStrncpy((char*)addr,mdefault,msize-1);CHKERRQ(ierr);
  if (!PetscBagInLoad) {
    ierr = PetscOptionsGetString(PETSC_NULL,nname,(char*)addr,msize,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterReal"
/*@C
   PetscBagRegisterReal - add a real value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of double in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterReal(PetscBag* bag,void *addr,PetscReal mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%g>: %s \n",nname,mdefault,help);CHKERRQ(ierr);
    }
    ierr     = PetscOptionsGetReal(PETSC_NULL,nname,&mdefault,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_REAL;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = 1;
  *(PetscReal*)addr = mdefault;
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterScalar"
/*@C
   PetscBagRegisterScalar - add a real value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of scalar in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value


   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterScalar(PetscBag* bag,void *addr,PetscScalar mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%g + %gi>: %s \n",nname,PetscRealPart(mdefault),PetscImaginaryPart(mdefault),help);CHKERRQ(ierr);
    }
    ierr     = PetscOptionsGetScalar(PETSC_NULL,nname,&mdefault,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_SCALAR;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = 1;
  *(PetscScalar*)addr = mdefault;
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagRegisterTruth"
/*@C
   PetscBagRegisterTruth - add a logical value to the bag

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
.  addr - location of logical in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value


   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterTruth(PetscBag* bag,void *addr,PetscTruth mdefault, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscTruth     printhelp;

  PetscFunctionBegin;
  if (!PetscBagInLoad) {
    nname[0] = '-';
    nname[1] = 0;
    ierr     = PetscStrncat(nname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
    ierr     = PetscOptionsHasName(PETSC_NULL,"-help",&printhelp);CHKERRQ(ierr);
    if (printhelp) {
      ierr     = (*PetscHelpPrintf)(bag->bagcomm,"  %s <%s>: %s \n",nname,PetscTruths[mdefault],help);CHKERRQ(ierr);
    }
    ierr     = PetscOptionsGetLogical(PETSC_NULL,nname,&mdefault,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNew(struct _p_PetscBagItem,&item);CHKERRQ(ierr);
  item->dtype  = PETSC_LOGICAL;
  item->offset = ((char*)addr) - ((char*)bag);
  item->next   = 0;
  item->msize  = 1;
  *(PetscTruth*)addr = mdefault;
  ierr = PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
  if (!bag->bagitems) bag->bagitems = item;
  else {
    PetscBagItem nitem = bag->bagitems;
    while (nitem->next) {
      nitem = nitem->next;
    }
    nitem->next = item;
  }
  bag->count++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagDestroy"
/*@C
   PetscBagDestroy - Destroys a bag values

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscBagDestroy(PetscBag *bag)
{
  PetscErrorCode ierr;
  PetscBagItem   nitem = bag->bagitems,item;

  PetscFunctionBegin;
  while (nitem) {
    item  = nitem->next;
    ierr  = PetscFree(nitem);CHKERRQ(ierr);
    nitem = item;
  }
  ierr = PetscFree(bag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagSetFromOptions"
/*@C
   PetscBagSetFromOptions - Allows setting options from a bag

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName(), PetscBagView()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscBagSetFromOptions(PetscBag *bag)
{
  PetscErrorCode ierr;
  PetscBagItem   nitem = bag->bagitems;
  char           name[PETSC_BAG_NAME_LENGTH+1],helpname[PETSC_BAG_NAME_LENGTH+PETSC_BAG_HELP_LENGTH+3];
  
  PetscFunctionBegin;

  ierr = PetscStrcpy(helpname,bag->bagname);CHKERRQ(ierr);
  ierr = PetscStrcat(helpname," ");CHKERRQ(ierr);
  ierr = PetscStrcat(helpname,bag->baghelp);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(bag->bagcomm,PETSC_NULL,helpname,0);
    while (nitem) {
      name[0] = '-';
      name[1] = 0;
      ierr    = PetscStrcat(name,nitem->name);CHKERRQ(ierr);
      if (nitem->dtype == PETSC_CHAR) {
        char *value = (char*)(((char*)bag) + nitem->offset);
        ierr = PetscOptionsString(name,nitem->help,"",value,value,nitem->msize,PETSC_NULL);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_REAL) {
        PetscReal *value = (PetscReal*)(((char*)bag) + nitem->offset);
        ierr = PetscOptionsReal(name,nitem->help,"",*value,value,PETSC_NULL);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_SCALAR) {
        PetscScalar *value = (PetscScalar*)(((char*)bag) + nitem->offset);
        ierr = PetscOptionsScalar(name,nitem->help,"",*value,value,PETSC_NULL);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_INT) {
        PetscInt *value = (PetscInt*)(((char*)bag) + nitem->offset);
        ierr = PetscOptionsInt(name,nitem->help,"",*value,value,PETSC_NULL);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_ENUM) {
        PetscInt *value = (PetscInt*)(((char*)bag) + nitem->offset);                     
        ierr = PetscOptionsInt(name,nitem->help,"",*value,value,PETSC_NULL);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_LOGICAL) {
        PetscTruth *value = (PetscTruth*)(((char*)bag) + nitem->offset);
        ierr = PetscOptionsLogical(name,nitem->help,"",*value,value,PETSC_NULL);CHKERRQ(ierr);
      }
      nitem = nitem->next;
    }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagView"
/*@C
   PetscBagView - Views a bag of values as either ASCII text or a binary file

   Collective on PetscBag

   Input Parameter:
+  bag - the bag of values
-  viewer - location to view the values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscBagView(PetscBag *bag,PetscViewer view)
{
  PetscTruth     isascii,isbinary;
  PetscErrorCode ierr;
  PetscBagItem   nitem = bag->bagitems;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)view,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)view,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(view,"PetscBag Object:  %s %s\n",bag->bagname,bag->baghelp);CHKERRQ(ierr);
    while (nitem) {
      if (nitem->dtype == PETSC_CHAR) {
        char* value = (char*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = %s; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_REAL) {
        PetscReal value = *(PetscReal*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_SCALAR) {
        PetscScalar value = *(PetscScalar*)(((char*)bag) + nitem->offset);
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(view,"  %s = %g + %gi; %s\n",nitem->name,PetscRealPart(value),PetscImaginaryPart(value),nitem->help);CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
#endif
      } else if (nitem->dtype == PETSC_INT) {
        PetscInt value = *(PetscInt*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = %D; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_ENUM) {
        PetscInt value = *(PetscInt*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = %D; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_LOGICAL) {
        PetscTruth value = *(PetscTruth*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = %s; %s\n",nitem->name,PetscTruths[value],nitem->help);CHKERRQ(ierr);
      }
      nitem = nitem->next;
    }
  } else if (isbinary) {
    PetscInt cookie = PETSC_BAG_FILE_COOKIE, bagsize = (PetscInt) bag->bagsize, dtype;
    ierr = PetscViewerBinaryWrite(view,&cookie,1,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,&bagsize,1,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,&bag->count,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,bag->bagname,PETSC_BAG_NAME_LENGTH,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,bag->baghelp,PETSC_BAG_HELP_LENGTH,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
    while (nitem) {
      ierr  = PetscViewerBinaryWrite(view,&nitem->offset,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
      dtype = (PetscInt)nitem->dtype;
      ierr  = PetscViewerBinaryWrite(view,&dtype,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,nitem->name,PETSC_BAG_NAME_LENGTH,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,nitem->help,PETSC_BAG_HELP_LENGTH,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,&nitem->msize,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,(((char*)bag) + nitem->offset),nitem->msize,nitem->dtype,PETSC_FALSE);CHKERRQ(ierr);
      nitem = nitem->next;
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for this viewer type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBagLoad"
/*@C
   PetscBagLoad - Loads a bag of values from a binary file

   Collective on PetscViewer

   Input Parameter:
.  viewer - file to load values from

   Output Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagView()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterTruth(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagRegisterVec(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscBagLoad(PetscViewer view,PetscBag **bag)
{
  PetscErrorCode ierr;
  PetscTruth     isbinary;
  PetscInt       cookie,bagsizecount[2],i,offsetdtype[2],msize;
  char           name[PETSC_BAG_NAME_LENGTH],help[PETSC_BAG_HELP_LENGTH];

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)view,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_ERR_SUP,"No support for this viewer type");

  ierr = PetscViewerBinaryRead(view,&cookie,1,PETSC_INT);CHKERRQ(ierr);
  if (cookie != PETSC_BAG_FILE_COOKIE) SETERRQ(PETSC_ERR_ARG_WRONG,"Not PetscBag next in binary file");
  ierr = PetscViewerBinaryRead(view,bagsizecount,2,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscMalloc((size_t)bagsizecount[0],bag);CHKERRQ(ierr);
  ierr = PetscMemzero(*bag,bagsizecount[0]);CHKERRQ(ierr);
  (*bag)->bagsize = bagsizecount[0];

  ierr = PetscViewerBinaryRead(view,(*bag)->bagname,PETSC_BAG_NAME_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(view,(*bag)->baghelp,PETSC_BAG_HELP_LENGTH,PETSC_CHAR);CHKERRQ(ierr);

  PetscBagInLoad = PETSC_TRUE;
  for (i=0; i<bagsizecount[1]; i++) {
    ierr = PetscViewerBinaryRead(view,offsetdtype,2,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,name,PETSC_BAG_NAME_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,help,PETSC_BAG_HELP_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,&msize,1,PETSC_INT);CHKERRQ(ierr);

    if (offsetdtype[1] == (PetscInt) PETSC_CHAR) {
      ierr = PetscViewerBinaryRead(view,((char*)(*bag))+offsetdtype[0],msize,PETSC_CHAR);CHKERRQ(ierr);
      ierr = PetscBagRegisterString(*bag,((char*)(*bag))+offsetdtype[0],msize,((char*)(*bag))+offsetdtype[0],name,help);CHKERRQ(ierr);
    } else if (offsetdtype[1] == (PetscInt) PETSC_REAL) {
      PetscReal mdefault;
      ierr = PetscViewerBinaryRead(view,&mdefault,1,PETSC_REAL);CHKERRQ(ierr);
      ierr = PetscBagRegisterReal(*bag,((char*)(*bag))+offsetdtype[0],mdefault,name,help);CHKERRQ(ierr);
    } else if (offsetdtype[1] == (PetscInt) PETSC_SCALAR) {
      PetscScalar mdefault;
      ierr = PetscViewerBinaryRead(view,&mdefault,1,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = PetscBagRegisterScalar(*bag,((char*)(*bag))+offsetdtype[0],mdefault,name,help);CHKERRQ(ierr);
    } else if (offsetdtype[1] == (PetscInt) PETSC_INT) {
      PetscInt mdefault;
      ierr = PetscViewerBinaryRead(view,&mdefault,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscBagRegisterInt(*bag,((char*)(*bag))+offsetdtype[0],mdefault,name,help);CHKERRQ(ierr);
    } else if (offsetdtype[1] == (PetscInt) PETSC_LOGICAL) {
      PetscTruth mdefault;
      ierr = PetscViewerBinaryRead(view,&mdefault,1,PETSC_LOGICAL);CHKERRQ(ierr);
      ierr = PetscBagRegisterTruth(*bag,((char*)(*bag))+offsetdtype[0],mdefault,name,help);CHKERRQ(ierr);
    }
  }
  PetscBagInLoad = PETSC_FALSE;
  PetscFunctionReturn(0);
}
