#include <petsc/private/petscimpl.h>
#include <petsc/private/bagimpl.h>     /*I  "petscbag.h"   I*/
#include <petscviewer.h>

/*
      Adds item to the linked list in a bag
*/
static PetscErrorCode PetscBagRegister_Private(PetscBag bag,PetscBagItem item,const char *name,const char *help)
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(item->name,name,PETSC_BAG_NAME_LENGTH-1));
  PetscCall(PetscStrncpy(item->help,help,PETSC_BAG_HELP_LENGTH-1));
  if (bag->bagitems) {
    PetscBagItem nitem = bag->bagitems;

    while (nitem->next) nitem = nitem->next;
    nitem->next = item;
  } else bag->bagitems = item;
  bag->count++;
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterEnum - add an enum value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of enum in struct
.  mdefault - the initial value
.  list - array of strings containing names of enum values followed by enum name followed by enum prefix
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`

@*/
PetscErrorCode PetscBagRegisterEnum(PetscBag bag,void *addr,const char *const *list,PetscEnum mdefault, const char *name, const char *help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i = 0;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidPointer(list,3);
  PetscValidCharPointer(name,5);
  PetscValidCharPointer(help,6);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    while (list[i++]) ;
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: (%s) %s (choose one of) ",bag->bagprefix ? bag->bagprefix : "",name,list[mdefault],list[i-3],help));
    for (i=0; list[i+2]; i++) PetscCall((*PetscHelpPrintf)(bag->bagcomm," %s",list[i]));
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"\n"));
  }
  PetscCall(PetscOptionsGetEnum(NULL,bag->bagprefix,nname,list,&mdefault,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_ENUM;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = 1;
  PetscCall(PetscStrArrayallocpy(list,(char***)&item->list));
  *(PetscEnum*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterIntArray - add an integer value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of integer in struct
.  msize - number of entries in array
.  name - name of the integer array
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterIntArray(PetscBag bag,void *addr,PetscInt msize, const char *name, const char *help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix ? bag->bagprefix : "",name));
    for (i=0; i<msize; i++) {
      PetscCall((*PetscHelpPrintf)(bag->bagcomm,"%" PetscInt_FMT " ",*((PetscInt*)addr)+i));
    }
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help));
  }
  PetscCall(PetscOptionsGetIntArray(NULL,bag->bagprefix,nname,(PetscInt*)addr,&tmp,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterRealArray - add an real array to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of real array in struct
.  msize - number of entries in array
.  name - name of the integer array
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterRealArray(PetscBag bag,void *addr,PetscInt msize, const char *name, const char *help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix ? bag->bagprefix : "",name));
    for (i=0; i<msize; i++) {
      PetscCall((*PetscHelpPrintf)(bag->bagcomm,"%g ",(double)*((PetscReal*)addr)+i));
    }
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help));
  }
  PetscCall(PetscOptionsGetRealArray(NULL,bag->bagprefix,nname,(PetscReal*)addr,&tmp,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_REAL;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterInt - add an integer value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of integer in struct
.  mdefault - the initial value
.  name - name of the integer
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt64()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterInt(PetscBag bag,void *addr,PetscInt mdefault,const char *name,const char *help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%" PetscInt_FMT ">: %s \n",bag->bagprefix ? bag->bagprefix : "",name,mdefault,help));
  }
  PetscCall(PetscOptionsGetInt(NULL,bag->bagprefix,nname,&mdefault,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next       = NULL;
  item->msize      = 1;
  *(PetscInt*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterInt64 - add an integer value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of integer in struct
.  mdefault - the initial value
.  name - name of the integer
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterInt64(PetscBag bag,void *addr,PetscInt64 mdefault,const char *name,const char *help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       odefault = (PetscInt)mdefault;
  PetscBool      flg;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%" PetscInt_FMT ">: %s \n",bag->bagprefix ? bag->bagprefix : "",name,odefault,help));
  }
  PetscCall(PetscOptionsGetInt(NULL,bag->bagprefix,nname,&odefault,&flg));
  if (flg) mdefault = (PetscInt64)odefault;

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next       = NULL;
  item->msize      = 1;
  *(PetscInt64*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterBoolArray - add a n logical values to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of boolean array in struct
.  msize - number of entries in array
.  name - name of the boolean array
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterBoolArray(PetscBag bag,void *addr,PetscInt msize, const char* name, const char* help)
{
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix?bag->bagprefix:"",name));
    for (i=0; i<msize; i++) {
      PetscCall((*PetscHelpPrintf)(bag->bagcomm,"%" PetscInt_FMT " ",*((PetscInt*)addr)+i));
    }
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help));
  }
  PetscCall(PetscOptionsGetBoolArray(NULL,bag->bagprefix,nname,(PetscBool*)addr,&tmp,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_BOOL;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next   = NULL;
  item->msize  = msize;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterString - add a string value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of start of string in struct
.  msize - length of the string space in the struct
.  mdefault - the initial value
.  name - name of the string
-  help - longer string with more information about the value

   Level: beginner

   Note: The struct should have the field char mystring[msize]; not char *mystring

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions(),PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterString(PetscBag bag,void *addr,PetscInt msize,const char* mdefault,const char* name,const char* help)
{
  PetscBagItem item;
  char         nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool    printhelp;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(mdefault,4);
  PetscValidCharPointer(name,5);
  PetscValidCharPointer(help,6);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,mdefault,help));
  }

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_CHAR;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  if (mdefault != (char*)addr) {
    PetscCall(PetscStrncpy((char*)addr,mdefault,msize-1));
  }
  PetscCall(PetscOptionsGetString(NULL,bag->bagprefix,nname,(char*)addr,msize,NULL));
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterReal - add a real value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of double in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterReal(PetscBag bag,void *addr,PetscReal mdefault, const char *name, const char *help)
{
  PetscBagItem item;
  char         nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool    printhelp;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%g>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,(double)mdefault,help));
  }
  PetscCall(PetscOptionsGetReal(NULL,bag->bagprefix,nname,&mdefault,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_REAL;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next        = NULL;
  item->msize       = 1;
  *(PetscReal*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterScalar - add a real or complex number value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of scalar in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterScalar(PetscBag bag,void *addr,PetscScalar mdefault,const char *name,const char *help)
{
  PetscBagItem item;
  char         nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool    printhelp;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%g + %gi>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,(double)PetscRealPart(mdefault),(double)PetscImaginaryPart(mdefault),help));
  }
  PetscCall(PetscOptionsGetScalar(NULL,bag->bagprefix,nname,&mdefault,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_SCALAR;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next          = NULL;
  item->msize         = 1;
  *(PetscScalar*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagRegisterBool - add a logical value to the bag

   Logically Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
.  addr - location of logical in struct
.  mdefault - the initial value
.  name - name of the variable
-  help - longer string with more information about the value

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode PetscBagRegisterBool(PetscBag bag,void *addr,PetscBool mdefault,const char *name,const char *help)
{
  PetscBagItem item;
  char         nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool    printhelp;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(addr,2);
  PetscValidCharPointer(name,4);
  PetscValidCharPointer(help,5);
  /* the checks here with != PETSC_FALSE and PETSC_TRUE is a special case; here we truly demand that the value be 0 or 1 */
  PetscCheck(mdefault == PETSC_FALSE || mdefault == PETSC_TRUE,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Boolean %s %s must be boolean; integer value %d",name,help,(int)mdefault);
  nname[0] = '-';
  nname[1] = 0;
  PetscCall(PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH));
  PetscCall(PetscOptionsHasHelp(NULL,&printhelp));
  if (printhelp) {
    PetscCall((*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,PetscBools[mdefault],help));
  }
  PetscCall(PetscOptionsGetBool(NULL,bag->bagprefix,nname,&mdefault,NULL));

  PetscCall(PetscNew(&item));
  item->dtype  = PETSC_BOOL;
  item->offset = ((char*)addr) - ((char*)bag);
  PetscCheck(item->offset <= bag->bagsize,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next        = NULL;
  item->msize       = 1;
  *(PetscBool*)addr = mdefault;
  PetscCall(PetscBagRegister_Private(bag,item,name,help));
  PetscFunctionReturn(0);
}

/*@C
   PetscBagDestroy - Destroys a bag values

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode  PetscBagDestroy(PetscBag *bag)
{
  PetscBagItem nitem;

  PetscFunctionBegin;
  if (!*bag) PetscFunctionReturn(0);
  PetscValidPointer(*bag,1);
  nitem = (*bag)->bagitems;
  while (nitem) {
    PetscBagItem item = nitem->next;

    if (nitem->list) PetscCall(PetscStrArrayDestroy(&nitem->list));
    PetscCall(PetscFree(nitem));
    nitem = item;
  }
  if ((*bag)->bagprefix) PetscCall(PetscFree((*bag)->bagprefix));
  PetscCall(PetscFree(*bag));
  PetscFunctionReturn(0);
}

/*@
   PetscBagSetFromOptions - Allows setting options from a bag

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagDestroy()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagView()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode  PetscBagSetFromOptions(PetscBag bag)
{
  PetscBagItem   nitem = bag->bagitems;
  char           name[PETSC_BAG_NAME_LENGTH+1],helpname[PETSC_BAG_NAME_LENGTH+PETSC_BAG_HELP_LENGTH+3];
  PetscInt       n;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscCall(PetscStrncpy(helpname,bag->bagname,sizeof(helpname)));
  PetscCall(PetscStrlcat(helpname," ",sizeof(helpname)));
  PetscCall(PetscStrlcat(helpname,bag->baghelp,sizeof(helpname)));
  PetscOptionsBegin(bag->bagcomm,bag->bagprefix,helpname,NULL);
  while (nitem) {
    name[0] = '-';
    name[1] = 0;
    PetscCall(PetscStrlcat(name,nitem->name,sizeof(name)));
    if (nitem->dtype == PETSC_CHAR) {   /* special handling for fortran required? [due to space padding vs null termination] */
      char *value = (char*)(((char*)bag) + nitem->offset);
      PetscCall(PetscOptionsString(name,nitem->help,"",value,value,nitem->msize,NULL));
    } else if (nitem->dtype == PETSC_REAL) {
      PetscReal *value = (PetscReal*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        PetscCall(PetscOptionsReal(name,nitem->help,"",*value,value,NULL));
      } else {
        n    = nitem->msize;
        PetscCall(PetscOptionsRealArray(name,nitem->help,"",value,&n,NULL));
      }
    } else if (nitem->dtype == PETSC_SCALAR) {
      PetscScalar *value = (PetscScalar*)(((char*)bag) + nitem->offset);
      PetscCall(PetscOptionsScalar(name,nitem->help,"",*value,value,NULL));
    } else if (nitem->dtype == PETSC_INT) {
      PetscInt *value = (PetscInt*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        PetscCall(PetscOptionsInt(name,nitem->help,"",*value,value,NULL));
      } else {
        n    = nitem->msize;
        PetscCall(PetscOptionsIntArray(name,nitem->help,"",value,&n,NULL));
      }
    } else if (nitem->dtype == PETSC_ENUM) {
      PetscEnum *value = (PetscEnum*)(((char*)bag) + nitem->offset);
      PetscInt  i      = 0;
      while (nitem->list[i++]) ;
      PetscCall(PetscOptionsEnum(name,nitem->help,nitem->list[i-3],(const char*const*)nitem->list,*value,value,NULL));
    } else if (nitem->dtype == PETSC_BOOL) {
      PetscBool *value = (PetscBool*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        PetscCall(PetscOptionsBool(name,nitem->help,"",*value,value,NULL));
      } else {
        n = nitem->msize;
        PetscCall(PetscOptionsBoolArray(name,nitem->help,"",value,&n,NULL));
      }
    }
    nitem = nitem->next;
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@C
   PetscBagView - Views a bag of values as either ASCII text or a binary file

   Collective on PetscBag

   Input Parameters:
+  bag - the bag of values
-  viewer - location to view the values

   Level: beginner

   Warning: Currently PETSc bags saved in a binary file can only be read back
     in on a machine of the same architecture. Let us know when this is a problem
     and we'll fix it.

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagDestroy()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`, `PetscBagRegisterEnum()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`

@*/
PetscErrorCode  PetscBagView(PetscBag bag,PetscViewer view)
{
  PetscBool    isascii,isbinary;
  PetscBagItem nitem = bag->bagitems;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidHeaderSpecific(view,PETSC_VIEWER_CLASSID,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERBINARY,&isbinary));
  if (isascii) {
    if (bag->bagprefix) {
      PetscCall(PetscViewerASCIIPrintf(view,"PetscBag Object:  %s (%s) %s\n",bag->bagname,bag->bagprefix,bag->baghelp));
    } else {
      PetscCall(PetscViewerASCIIPrintf(view,"PetscBag Object:  %s %s\n",bag->bagname,bag->baghelp));
    }
    while (nitem) {
      if (nitem->dtype == PETSC_CHAR) {
        char *value = (char*)(((char*)bag) + nitem->offset);
        char tmp    = value[nitem->msize-1]; /* special handling for fortran chars wihout null terminator */
        value[nitem->msize-1] =0;
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = %s; %s\n",nitem->name,value,nitem->help));
        value[nitem->msize-1] = tmp;
      } else if (nitem->dtype == PETSC_REAL) {
        PetscReal *value = (PetscReal*)(((char*)bag) + nitem->offset);
        PetscInt  i;
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = ",nitem->name));
        for (i=0; i<nitem->msize; i++) {
          PetscCall(PetscViewerASCIIPrintf(view,"%g ",(double)value[i]));
        }
        PetscCall(PetscViewerASCIIPrintf(view,"; %s\n",nitem->help));
      } else if (nitem->dtype == PETSC_SCALAR) {
        PetscScalar value = *(PetscScalar*)(((char*)bag) + nitem->offset);
#if defined(PETSC_USE_COMPLEX)
        if ((double)PetscImaginaryPart(value)) {
          PetscCall(PetscViewerASCIIPrintf(view,"  %s = %g + %gi; %s\n",nitem->name,(double)PetscRealPart(value),(double)PetscImaginaryPart(value),nitem->help));
        } else {
          PetscCall(PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,(double)PetscRealPart(value),nitem->help));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,(double)value,nitem->help));
#endif
      } else if (nitem->dtype == PETSC_INT) {
        PetscInt i,*value = (PetscInt*)(((char*)bag) + nitem->offset);
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = ",nitem->name));
        for (i=0; i<nitem->msize; i++) {
          PetscCall(PetscViewerASCIIPrintf(view,"%" PetscInt_FMT " ",value[i]));
        }
        PetscCall(PetscViewerASCIIPrintf(view,"; %s\n",nitem->help));
      } else if (nitem->dtype == PETSC_BOOL) {
        PetscBool  *value = (PetscBool*)(((char*)bag) + nitem->offset);
        PetscInt  i;
         /* some Fortran compilers use -1 as boolean */
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = ",nitem->name));
        for (i=0; i<nitem->msize; i++) {
          if (((int) value[i]) == -1) value[i] = PETSC_TRUE;
          /* the checks here with != PETSC_FALSE and PETSC_TRUE is a special case; here we truly demand that the value be 0 or 1 */
          PetscCheck(value[i] == PETSC_FALSE || value[i] == PETSC_TRUE,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Boolean value for %s %s is corrupt; integer value %" PetscInt_FMT,nitem->name,nitem->help,(PetscInt)(value[i]));
          PetscCall(PetscViewerASCIIPrintf(view," %s",PetscBools[value[i]]));
        }
        PetscCall(PetscViewerASCIIPrintf(view,"; %s\n",nitem->help));
      } else if (nitem->dtype == PETSC_ENUM) {
        PetscEnum value = *(PetscEnum*)(((char*)bag) + nitem->offset);
        PetscInt  i     = 0;
        while (nitem->list[i++]) ;
        PetscCall(PetscViewerASCIIPrintf(view,"  %s = %s; (%s) %s\n",nitem->name,nitem->list[value],nitem->list[i-3],nitem->help));
      }
      nitem = nitem->next;
    }
  } else if (isbinary) {
    PetscInt          classid           = PETSC_BAG_FILE_CLASSID, dtype;
    PetscInt          deprecatedbagsize = 0;
    PetscViewerFormat format;
    PetscCall(PetscViewerBinaryWrite(view,&classid,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(view,&deprecatedbagsize,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(view,&bag->count,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(view,bag->bagname,PETSC_BAG_NAME_LENGTH,PETSC_CHAR));
    PetscCall(PetscViewerBinaryWrite(view,bag->baghelp,PETSC_BAG_HELP_LENGTH,PETSC_CHAR));
    while (nitem) {
      PetscCall(PetscViewerBinaryWrite(view,&nitem->offset,1,PETSC_INT));
      dtype = (PetscInt)nitem->dtype;
      PetscCall(PetscViewerBinaryWrite(view,&dtype,1,PETSC_INT));
      PetscCall(PetscViewerBinaryWrite(view,nitem->name,PETSC_BAG_NAME_LENGTH,PETSC_CHAR));
      PetscCall(PetscViewerBinaryWrite(view,nitem->help,PETSC_BAG_HELP_LENGTH,PETSC_CHAR));
      PetscCall(PetscViewerBinaryWrite(view,&nitem->msize,1,PETSC_INT));
      /* some Fortran compilers use -1 as boolean */
      if (dtype == PETSC_BOOL && ((*(int*) (((char*)bag) + nitem->offset) == -1))) *(int*) (((char*)bag) + nitem->offset) = PETSC_TRUE;

      PetscCall(PetscViewerBinaryWrite(view,(((char*)bag) + nitem->offset),nitem->msize,nitem->dtype));
      if (dtype == PETSC_ENUM) {
        PetscCall(PetscViewerBinaryWriteStringArray(view,(const char* const*)nitem->list));
      }
      nitem = nitem->next;
    }
    PetscCall(PetscViewerGetFormat(view,&format));
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      MPI_Comm comm;
      FILE     *info;
      PetscCall(PetscObjectGetComm((PetscObject)view,&comm));
      PetscCall(PetscViewerBinaryGetInfoPointer(view,&info));
      PetscCall(PetscFPrintf(comm,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
      PetscCall(PetscFPrintf(comm,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",bag->bagname));
      PetscCall(PetscFPrintf(comm,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscBagViewFromOptions - Processes command line options to determine if/how a PetscBag is to be viewed.

  Collective on PetscBag

  Input Parameters:
+ obj   - the object
. bobj  - optional other object that provides prefix (if NULL then the prefix in obj is used)
- optionname - option to activate viewing

  Level: intermediate

.seealso: `PetscBagCreate()`, `PetscBag`, `PetscViewer`
@*/
PetscErrorCode PetscBagViewFromOptions(PetscBag bag, PetscObject bobj, const char optionname[])
{
  static PetscBool  incall = PETSC_FALSE;
  PetscViewer       viewer;
  PetscViewerFormat format;
  const char       *prefix, *bprefix = NULL;
  PetscBool         flg;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscValidPointer(bag,1);
  if (bobj) PetscCall(PetscObjectGetOptionsPrefix(bobj, &bprefix));
  prefix = bobj ? bprefix : bag->bagprefix;
  PetscCall(PetscOptionsGetViewer(bag->bagcomm, NULL, prefix, optionname, &viewer, &format, &flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(PetscBagView(bag, viewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscBagLoad - Loads a bag of values from a binary file

   Collective on PetscViewer

   Input Parameters:
+  viewer - file to load values from
-  bag - the bag of values

   Notes:
    You must have created and registered all the fields in the bag before loading into it.

   Notes:
   Level: beginner

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagDestroy()`, `PetscBagView()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagGetName()`, `PetscBagRegisterEnum()`

@*/
PetscErrorCode  PetscBagLoad(PetscViewer view,PetscBag bag)
{
  PetscBool    isbinary;
  PetscInt     classid,bagcount,dtype,msize,offset,deprecatedbagsize;
  char         name[PETSC_BAG_NAME_LENGTH],help[PETSC_BAG_HELP_LENGTH],**list;
  PetscBagItem nitem;
  MPI_Comm     comm;
  PetscMPIInt  flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(view,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(bag,2);
  PetscCall(PetscObjectGetComm((PetscObject)view,&comm));
  PetscCallMPI(MPI_Comm_compare(comm,bag->bagcomm,&flag));
  PetscCheck(flag == MPI_CONGRUENT || flag == MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the viewer and bag");
  PetscCall(PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERBINARY,&isbinary));
  PetscCheck(isbinary,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this viewer type");

  PetscCall(PetscViewerBinaryRead(view,&classid,1,NULL,PETSC_INT));
  PetscCheck(classid == PETSC_BAG_FILE_CLASSID,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not PetscBag next in binary file");
  PetscCall(PetscViewerBinaryRead(view,&deprecatedbagsize,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(view,&bagcount,1,NULL,PETSC_INT));
  PetscCheck(bagcount == bag->count,comm,PETSC_ERR_ARG_INCOMP,"Bag in file has different number of entries %d then passed in bag %d",(int)bagcount,(int)bag->count);
  PetscCall(PetscViewerBinaryRead(view,bag->bagname,PETSC_BAG_NAME_LENGTH,NULL,PETSC_CHAR));
  PetscCall(PetscViewerBinaryRead(view,bag->baghelp,PETSC_BAG_HELP_LENGTH,NULL,PETSC_CHAR));

  nitem = bag->bagitems;
  for (PetscInt i=0; i<bagcount; i++) {
    PetscCall(PetscViewerBinaryRead(view,&offset,1,NULL,PETSC_INT));
    /* ignore the offset in the file */
    PetscCall(PetscViewerBinaryRead(view,&dtype,1,NULL,PETSC_INT));
    PetscCall(PetscViewerBinaryRead(view,name,PETSC_BAG_NAME_LENGTH,NULL,PETSC_CHAR));
    PetscCall(PetscViewerBinaryRead(view,help,PETSC_BAG_HELP_LENGTH,NULL,PETSC_CHAR));
    PetscCall(PetscViewerBinaryRead(view,&msize,1,NULL,PETSC_INT));

    if (dtype == (PetscInt) PETSC_CHAR) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_CHAR));
    } else if (dtype == (PetscInt) PETSC_REAL) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_REAL));
    } else if (dtype == (PetscInt) PETSC_SCALAR) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,1,NULL,PETSC_SCALAR));
    } else if (dtype == (PetscInt) PETSC_INT) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_INT));
    } else if (dtype == (PetscInt) PETSC_BOOL) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_BOOL));
    } else if (dtype == (PetscInt) PETSC_ENUM) {
      PetscCall(PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,1,NULL,PETSC_ENUM));
      PetscCall(PetscViewerBinaryReadStringArray(view,&list));
      /* don't need to save list because it is already registered in the bag */
      PetscCall(PetscFree(list));
    }
    nitem = nitem->next;
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscBagCreate - Create a bag of values

  Collective

  Level: Intermediate

  Input Parameters:
+  comm - communicator to share bag
-  bagsize - size of the C structure holding the values

  Output Parameter:
.   bag - the bag of values

   Notes:
      The size of the A struct must be small enough to fit in a PetscInt; by default
      PetscInt is 4 bytes; this means a bag cannot be larger than 2 gigabytes in length.
      The warning about casting to a shorter length can be ignored below unless your A struct is too large

.seealso: `PetscBag`, `PetscBagGetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagDestroy()`, `PetscBagRegisterEnum()`
@*/
PetscErrorCode PetscBagCreate(MPI_Comm comm, size_t bagsize, PetscBag *bag)
{
  const size_t totalsize = bagsize+sizeof(struct _n_PetscBag)+sizeof(PetscScalar);

  PetscFunctionBegin;
  PetscValidPointer(bag,3);
  PetscCall(PetscInfo(NULL,"Creating Bag with total size %d\n",(int)totalsize));
  PetscCall(PetscCalloc(totalsize,bag));

  (*bag)->bagsize        = totalsize;
  (*bag)->bagcomm        = comm;
  (*bag)->bagprefix      = NULL;
  (*bag)->structlocation = (void*)(((char*)(*bag)) + sizeof(PetscScalar)*(sizeof(struct _n_PetscBag)/sizeof(PetscScalar)) + sizeof(PetscScalar));
  PetscFunctionReturn(0);
}

/*@C
    PetscBagSetName - Sets the name of a bag of values

  Not Collective

  Level: Intermediate

  Input Parameters:
+   bag - the bag of values
.   name - the name assigned to the bag
-   help - help message for bag

.seealso: `PetscBag`, `PetscBagGetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagDestroy()`, `PetscBagRegisterEnum()`
@*/

PetscErrorCode PetscBagSetName(PetscBag bag, const char *name, const char *help)
{
  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidCharPointer(name,2);
  PetscValidCharPointer(help,3);
  PetscCall(PetscStrncpy(bag->bagname,name,PETSC_BAG_NAME_LENGTH-1));
  PetscCall(PetscStrncpy(bag->baghelp,help,PETSC_BAG_HELP_LENGTH-1));
  PetscFunctionReturn(0);
}

/*@C
    PetscBagGetName - Gets the name of a bag of values

  Not Collective

  Level: Intermediate

  Input Parameter:
.   bag - the bag of values

  Output Parameter:
.   name - the name assigned to the bag

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagDestroy()`, `PetscBagRegisterEnum()`
@*/
PetscErrorCode PetscBagGetName(PetscBag bag, char **name)
{
  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(name,2);
  *name = bag->bagname;
  PetscFunctionReturn(0);
}

/*@C
    PetscBagGetData - Gives back the user - access to memory that
    should be used for storing user-data-structure

  Not Collective

  Level: Intermediate

  Input Parameter:
.   bag - the bag of values

  Output Parameter:
.   data - pointer to memory that will have user-data-structure

.seealso: `PetscBag`, `PetscBagSetName()`, `PetscBagView()`, `PetscBagLoad()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagDestroy()`, `PetscBagRegisterEnum()`
@*/
PetscErrorCode PetscBagGetData(PetscBag bag, void **data)
{
  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(data,2);
  *data = bag->structlocation;
  PetscFunctionReturn(0);
}

/*@C
  PetscBagSetOptionsPrefix - Sets the prefix used for searching for all
  PetscBag items in the options database.

  Logically collective on Bag.

  Level: Intermediate

  Input Parameters:
+   bag - the bag of values
-   prefix - the prefix to prepend all Bag item names with.

  NOTES: Must be called prior to registering any of the bag items.

.seealso: `PetscBag`, `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`
          `PetscBagSetFromOptions()`, `PetscBagCreate()`, `PetscBagDestroy()`, `PetscBagRegisterEnum()`
@*/

PetscErrorCode PetscBagSetOptionsPrefix(PetscBag bag, const char pre[])
{
  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  if (pre) {
    PetscValidCharPointer(pre,2);
    PetscCheck(pre[0] != '-',PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hyphen");
    PetscCall(PetscFree(bag->bagprefix));
    PetscCall(PetscStrallocpy(pre,&(bag->bagprefix)));
  } else PetscCall(PetscFree(bag->bagprefix));
  PetscFunctionReturn(0);
}

/*@C
  PetscBagGetNames - Get the names of all entries in the bag

  Not collective

  Input Parameters:
+ bag   - the bag of values
- names - array of the correct size to hold names

  Output Parameter:
. names - array of char pointers for names

  Level: intermediate

.seealso: `PetscBag`, `PetscBagGetName()`, `PetscBagSetName()`, `PetscBagCreate()`, `PetscBagGetData()`
          `PetscBagRegisterReal()`, `PetscBagRegisterInt()`, `PetscBagRegisterBool()`, `PetscBagRegisterScalar()`, `PetscBagRegisterEnum()`
@*/
PetscErrorCode PetscBagGetNames(PetscBag bag, const char *names[])
{
  PetscBagItem nitem = bag->bagitems;

  PetscFunctionBegin;
  PetscValidPointer(bag,1);
  PetscValidPointer(names,2);
  for (PetscInt n = 0; nitem; ++n, nitem = nitem->next) names[n] = nitem->name;
  PetscFunctionReturn(0);
}
