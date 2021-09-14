
#include <petsc/private/bagimpl.h>     /*I  "petscbag.h"   I*/
#include <petscviewer.h>

/*
      Adds item to the linked list in a bag
*/
static PetscErrorCode PetscBagRegister_Private(PetscBag bag,PetscBagItem item,const char *name,const char *help)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode PetscBagRegisterEnum(PetscBag bag,void *addr,const char *const *list,PetscEnum mdefault, const char *name, const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i = 0;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    while (list[i++]) ;
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: (%s) %s (choose one of) ",bag->bagprefix ? bag->bagprefix : "",name,list[mdefault],list[i-3],help);CHKERRQ(ierr);
    for (i=0; list[i+2]; i++) {
      ierr = (*PetscHelpPrintf)(bag->bagcomm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"\n");CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetEnum(NULL,bag->bagprefix,nname,list,&mdefault,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_ENUM;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next        = NULL;
  item->msize       = 1;
  ierr              = PetscStrArrayallocpy(list,(char***)&item->list);CHKERRQ(ierr);
  *(PetscEnum*)addr = mdefault;
  ierr              = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterIntArray(PetscBag bag,void *addr,PetscInt msize, const char *name, const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix ? bag->bagprefix : "",name);CHKERRQ(ierr);
    for (i=0; i<msize; i++) {
      ierr = (*PetscHelpPrintf)(bag->bagcomm,"%D ",*((PetscInt*)addr)+i);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetIntArray(NULL,bag->bagprefix,nname,(PetscInt*)addr,&tmp,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  ierr        = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterRealArray(PetscBag bag,void *addr,PetscInt msize, const char *name, const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix ? bag->bagprefix : "",name);CHKERRQ(ierr);
    for (i=0; i<msize; i++) {
      ierr = (*PetscHelpPrintf)(bag->bagcomm,"%g ",(double)*((PetscReal*)addr)+i);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetRealArray(NULL,bag->bagprefix,nname,(PetscReal*)addr,&tmp,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_REAL;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  ierr        = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt64(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterInt(PetscBag bag,void *addr,PetscInt mdefault,const char *name,const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%d>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,mdefault,help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetInt(NULL,bag->bagprefix,nname,&mdefault,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next       = NULL;
  item->msize      = 1;
  *(PetscInt*)addr = mdefault;
  ierr             = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterInt64(PetscBag bag,void *addr,PetscInt64 mdefault,const char *name,const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       odefault = (PetscInt)mdefault;
  PetscBool      flg;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%d>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,odefault,help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetInt(NULL,bag->bagprefix,nname,&odefault,&flg);CHKERRQ(ierr);
  if (flg) mdefault = (PetscInt64)odefault;

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_INT;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next       = NULL;
  item->msize      = 1;
  *(PetscInt64*)addr = mdefault;
  ierr             = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterBoolArray(PetscBag bag,void *addr,PetscInt msize, const char* name, const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;
  PetscInt       i,tmp = msize;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <",bag->bagprefix?bag->bagprefix:"",name);CHKERRQ(ierr);
    for (i=0; i<msize; i++) {
      ierr = (*PetscHelpPrintf)(bag->bagcomm,"%D ",*((PetscInt*)addr)+i);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(bag->bagcomm,">: %s \n",help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBoolArray(NULL,bag->bagprefix,nname,(PetscBool*)addr,&tmp,NULL);CHKERRQ(ierr);

  ierr = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_BOOL;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next   = NULL;
  item->msize  = msize;
  ierr = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(),PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterString(PetscBag bag,void *addr,PetscInt msize,const char* mdefault,const char* name,const char* help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,mdefault,help);CHKERRQ(ierr);
  }

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_CHAR;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next  = NULL;
  item->msize = msize;
  if (mdefault != (char*)addr) {
    ierr = PetscStrncpy((char*)addr,mdefault,msize-1);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,bag->bagprefix,nname,(char*)addr,msize,NULL);CHKERRQ(ierr);
  ierr = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterReal(PetscBag bag,void *addr,PetscReal mdefault, const char *name, const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%g>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,(double)mdefault,help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetReal(NULL,bag->bagprefix,nname,&mdefault,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_REAL;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next        = NULL;
  item->msize       = 1;
  *(PetscReal*)addr = mdefault;
  ierr              = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterScalar(PetscBag bag,void *addr,PetscScalar mdefault,const char *name,const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%g + %gi>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,(double)PetscRealPart(mdefault),(double)PetscImaginaryPart(mdefault),help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetScalar(NULL,bag->bagprefix,nname,&mdefault,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_SCALAR;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next          = NULL;
  item->msize         = 1;
  *(PetscScalar*)addr = mdefault;
  ierr                = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode PetscBagRegisterBool(PetscBag bag,void *addr,PetscBool mdefault,const char *name,const char *help)
{
  PetscErrorCode ierr;
  PetscBagItem   item;
  char           nname[PETSC_BAG_NAME_LENGTH+1];
  PetscBool      printhelp;

  PetscFunctionBegin;
  /* the checks here with != PETSC_FALSE and PETSC_TRUE is a special case; here we truly demand that the value be 0 or 1 */
  if (mdefault != PETSC_FALSE && mdefault != PETSC_TRUE) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Boolean %s %s must be boolean; integer value %d",name,help,(int)mdefault);
  nname[0] = '-';
  nname[1] = 0;
  ierr     = PetscStrlcat(nname,name,PETSC_BAG_NAME_LENGTH);CHKERRQ(ierr);
  ierr     = PetscOptionsHasHelp(NULL,&printhelp);CHKERRQ(ierr);
  if (printhelp) {
    ierr = (*PetscHelpPrintf)(bag->bagcomm,"  -%s%s <%s>: %s \n",bag->bagprefix ? bag->bagprefix : "",name,PetscBools[mdefault],help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL,bag->bagprefix,nname,&mdefault,NULL);CHKERRQ(ierr);

  ierr         = PetscNew(&item);CHKERRQ(ierr);
  item->dtype  = PETSC_BOOL;
  item->offset = ((char*)addr) - ((char*)bag);
  if (item->offset > bag->bagsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Registered item %s %s is not in bag memory space",name,help);
  item->next        = NULL;
  item->msize       = 1;
  *(PetscBool*)addr = mdefault;
  ierr              = PetscBagRegister_Private(bag,item,name,help);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscBagDestroy - Destroys a bag values

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode  PetscBagDestroy(PetscBag *bag)
{
  PetscErrorCode ierr;
  PetscBagItem   nitem = (*bag)->bagitems,item;

  PetscFunctionBegin;
  while (nitem) {
    item = nitem->next;
    if (nitem->list) {
      ierr = PetscStrArrayDestroy(&nitem->list);CHKERRQ(ierr);
    }
    ierr  = PetscFree(nitem);CHKERRQ(ierr);
    nitem = item;
  }
  if ((*bag)->bagprefix) { ierr = PetscFree((*bag)->bagprefix);CHKERRQ(ierr); }
  ierr = PetscFree(*bag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscBagSetFromOptions - Allows setting options from a bag

   Collective on PetscBag

   Input Parameter:
.  bag - the bag of values

   Level: beginner

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagView(), PetscBagRegisterEnum()

@*/
PetscErrorCode  PetscBagSetFromOptions(PetscBag bag)
{
  PetscErrorCode ierr;
  PetscBagItem   nitem = bag->bagitems;
  char           name[PETSC_BAG_NAME_LENGTH+1],helpname[PETSC_BAG_NAME_LENGTH+PETSC_BAG_HELP_LENGTH+3];
  PetscInt       n;

  PetscFunctionBegin;
  ierr = PetscStrncpy(helpname,bag->bagname,sizeof(helpname));CHKERRQ(ierr);
  ierr = PetscStrlcat(helpname," ",sizeof(helpname));CHKERRQ(ierr);
  ierr = PetscStrlcat(helpname,bag->baghelp,sizeof(helpname));CHKERRQ(ierr);
  ierr = PetscOptionsBegin(bag->bagcomm,bag->bagprefix,helpname,NULL);CHKERRQ(ierr);
  while (nitem) {
    name[0] = '-';
    name[1] = 0;
    ierr    = PetscStrlcat(name,nitem->name,sizeof(name));CHKERRQ(ierr);
    if (nitem->dtype == PETSC_CHAR) {   /* special handling for fortran required? [due to space padding vs null termination] */
      char *value = (char*)(((char*)bag) + nitem->offset);
      ierr = PetscOptionsString(name,nitem->help,"",value,value,nitem->msize,NULL);CHKERRQ(ierr);
    } else if (nitem->dtype == PETSC_REAL) {
      PetscReal *value = (PetscReal*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        ierr = PetscOptionsReal(name,nitem->help,"",*value,value,NULL);CHKERRQ(ierr);
      } else {
        n    = nitem->msize;
        ierr = PetscOptionsRealArray(name,nitem->help,"",value,&n,NULL);CHKERRQ(ierr);
      }
    } else if (nitem->dtype == PETSC_SCALAR) {
      PetscScalar *value = (PetscScalar*)(((char*)bag) + nitem->offset);
      ierr = PetscOptionsScalar(name,nitem->help,"",*value,value,NULL);CHKERRQ(ierr);
    } else if (nitem->dtype == PETSC_INT) {
      PetscInt *value = (PetscInt*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        ierr = PetscOptionsInt(name,nitem->help,"",*value,value,NULL);CHKERRQ(ierr);
      } else {
        n    = nitem->msize;
        ierr = PetscOptionsIntArray(name,nitem->help,"",value,&n,NULL);CHKERRQ(ierr);
      }
    } else if (nitem->dtype == PETSC_ENUM) {
      PetscEnum *value = (PetscEnum*)(((char*)bag) + nitem->offset);
      PetscInt  i      = 0;
      while (nitem->list[i++]) ;
      ierr = PetscOptionsEnum(name,nitem->help,nitem->list[i-3],(const char*const*)nitem->list,*value,value,NULL);CHKERRQ(ierr);
    } else if (nitem->dtype == PETSC_BOOL) {
      PetscBool *value = (PetscBool*)(((char*)bag) + nitem->offset);
      if (nitem->msize == 1) {
        ierr = PetscOptionsBool(name,nitem->help,"",*value,value,NULL);CHKERRQ(ierr);
      } else {
        n = nitem->msize;
        ierr = PetscOptionsBoolArray(name,nitem->help,"",value,&n,NULL);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar(), PetscBagRegisterEnum()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName()

@*/
PetscErrorCode  PetscBagView(PetscBag bag,PetscViewer view)
{
  PetscBool      isascii,isbinary;
  PetscErrorCode ierr;
  PetscBagItem   nitem = bag->bagitems;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isascii) {
    if (bag->bagprefix) {
      ierr = PetscViewerASCIIPrintf(view,"PetscBag Object:  %s (%s) %s\n",bag->bagname,bag->bagprefix,bag->baghelp);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(view,"PetscBag Object:  %s %s\n",bag->bagname,bag->baghelp);CHKERRQ(ierr);
    }
    while (nitem) {
      if (nitem->dtype == PETSC_CHAR) {
        char *value = (char*)(((char*)bag) + nitem->offset);
        char tmp    = value[nitem->msize-1]; /* special handling for fortran chars wihout null terminator */
        value[nitem->msize-1] =0;
        ierr = PetscViewerASCIIPrintf(view,"  %s = %s; %s\n",nitem->name,value,nitem->help);CHKERRQ(ierr);
        value[nitem->msize-1] = tmp;
      } else if (nitem->dtype == PETSC_REAL) {
        PetscReal *value = (PetscReal*)(((char*)bag) + nitem->offset);
        PetscInt  i;
        ierr = PetscViewerASCIIPrintf(view,"  %s = ",nitem->name);CHKERRQ(ierr);
        for (i=0; i<nitem->msize; i++) {
          ierr = PetscViewerASCIIPrintf(view,"%g ",(double)value[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(view,"; %s\n",nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_SCALAR) {
        PetscScalar value = *(PetscScalar*)(((char*)bag) + nitem->offset);
#if defined(PETSC_USE_COMPLEX)
        if ((double)PetscImaginaryPart(value)) {
          ierr = PetscViewerASCIIPrintf(view,"  %s = %g + %gi; %s\n",nitem->name,(double)PetscRealPart(value),(double)PetscImaginaryPart(value),nitem->help);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,(double)PetscRealPart(value),nitem->help);CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(view,"  %s = %g; %s\n",nitem->name,(double)value,nitem->help);CHKERRQ(ierr);
#endif
      } else if (nitem->dtype == PETSC_INT) {
        PetscInt i,*value = (PetscInt*)(((char*)bag) + nitem->offset);
        ierr = PetscViewerASCIIPrintf(view,"  %s = ",nitem->name);CHKERRQ(ierr);
        for (i=0; i<nitem->msize; i++) {
          ierr = PetscViewerASCIIPrintf(view,"%D ",value[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(view,"; %s\n",nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_BOOL) {
        PetscBool  *value = (PetscBool*)(((char*)bag) + nitem->offset);
        PetscInt  i;
         /* some Fortran compilers use -1 as boolean */
        ierr = PetscViewerASCIIPrintf(view,"  %s = ",nitem->name);CHKERRQ(ierr);
        for (i=0; i<nitem->msize; i++) {
          if (((int) value[i]) == -1) value[i] = PETSC_TRUE;
          /* the checks here with != PETSC_FALSE and PETSC_TRUE is a special case; here we truly demand that the value be 0 or 1 */
          if (value[i] != PETSC_FALSE && value[i] != PETSC_TRUE) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Boolean value for %s %s is corrupt; integer value %d",nitem->name,nitem->help,value);
          ierr = PetscViewerASCIIPrintf(view," %s",PetscBools[value[i]]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(view,"; %s\n",nitem->help);CHKERRQ(ierr);
      } else if (nitem->dtype == PETSC_ENUM) {
        PetscEnum value = *(PetscEnum*)(((char*)bag) + nitem->offset);
        PetscInt  i     = 0;
        while (nitem->list[i++]) ;
        ierr = PetscViewerASCIIPrintf(view,"  %s = %s; (%s) %s\n",nitem->name,nitem->list[value],nitem->list[i-3],nitem->help);CHKERRQ(ierr);
      }
      nitem = nitem->next;
    }
  } else if (isbinary) {
    PetscInt          classid           = PETSC_BAG_FILE_CLASSID, dtype;
    PetscInt          deprecatedbagsize = 0;
    PetscViewerFormat format;
    ierr = PetscViewerBinaryWrite(view,&classid,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,&deprecatedbagsize,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,&bag->count,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,bag->bagname,PETSC_BAG_NAME_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(view,bag->baghelp,PETSC_BAG_HELP_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
    while (nitem) {
      ierr  = PetscViewerBinaryWrite(view,&nitem->offset,1,PETSC_INT);CHKERRQ(ierr);
      dtype = (PetscInt)nitem->dtype;
      ierr  = PetscViewerBinaryWrite(view,&dtype,1,PETSC_INT);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,nitem->name,PETSC_BAG_NAME_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,nitem->help,PETSC_BAG_HELP_LENGTH,PETSC_CHAR);CHKERRQ(ierr);
      ierr  = PetscViewerBinaryWrite(view,&nitem->msize,1,PETSC_INT);CHKERRQ(ierr);
      /* some Fortran compilers use -1 as boolean */
      if (dtype == PETSC_BOOL && ((*(int*) (((char*)bag) + nitem->offset) == -1))) *(int*) (((char*)bag) + nitem->offset) = PETSC_TRUE;

      ierr = PetscViewerBinaryWrite(view,(((char*)bag) + nitem->offset),nitem->msize,nitem->dtype);CHKERRQ(ierr);
      if (dtype == PETSC_ENUM) {
        ierr = PetscViewerBinaryWriteStringArray(view,(const char* const*)nitem->list);CHKERRQ(ierr);
      }
      nitem = nitem->next;
    }
    ierr = PetscViewerGetFormat(view,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_BINARY_MATLAB) {
      MPI_Comm comm;
      FILE     *info;
      ierr = PetscObjectGetComm((PetscObject)view,&comm);CHKERRQ(ierr);
      ierr = PetscViewerBinaryGetInfoPointer(view,&info);CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#$$ Set.%s = PetscBinaryRead(fd);\n",bag->bagname);CHKERRQ(ierr);
      ierr = PetscFPrintf(comm,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
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

.seealso: PetscBagCreate(), PetscBag, PetscViewer
@*/
PetscErrorCode PetscBagViewFromOptions(PetscBag bag, PetscObject bobj, const char optionname[])
{
  static PetscBool  incall = PETSC_FALSE;
  PetscViewer       viewer;
  PetscViewerFormat format;
  const char       *prefix, *bprefix = NULL;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  if (bobj) {ierr = PetscObjectGetOptionsPrefix(bobj, &bprefix);CHKERRQ(ierr);}
  prefix = bobj ? bprefix : bag->bagprefix;
  ierr   = PetscOptionsGetViewer(bag->bagcomm, NULL, prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscBagView(bag, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagDestroy(), PetscBagView(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagGetName(), PetscBagRegisterEnum()

@*/
PetscErrorCode  PetscBagLoad(PetscViewer view,PetscBag bag)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  PetscInt       classid,bagcount,i,dtype,msize,offset,deprecatedbagsize;
  char           name[PETSC_BAG_NAME_LENGTH],help[PETSC_BAG_HELP_LENGTH],**list;
  PetscBagItem   nitem;
  MPI_Comm       comm;
  PetscMPIInt    flag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)view,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,bag->bagcomm,&flag);CHKERRMPI(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the viewer and bag"); \
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this viewer type");

  ierr = PetscViewerBinaryRead(view,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (classid != PETSC_BAG_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not PetscBag next in binary file");
  ierr = PetscViewerBinaryRead(view,&deprecatedbagsize,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(view,&bagcount,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (bagcount != bag->count) SETERRQ2(comm,PETSC_ERR_ARG_INCOMP,"Bag in file has different number of entries %d then passed in bag %d\n",(int)bagcount,(int)bag->count);
  ierr = PetscViewerBinaryRead(view,bag->bagname,PETSC_BAG_NAME_LENGTH,NULL,PETSC_CHAR);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(view,bag->baghelp,PETSC_BAG_HELP_LENGTH,NULL,PETSC_CHAR);CHKERRQ(ierr);

  nitem = bag->bagitems;
  for (i=0; i<bagcount; i++) {
    ierr = PetscViewerBinaryRead(view,&offset,1,NULL,PETSC_INT);CHKERRQ(ierr);
    /* ignore the offset in the file */
    ierr = PetscViewerBinaryRead(view,&dtype,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,name,PETSC_BAG_NAME_LENGTH,NULL,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,help,PETSC_BAG_HELP_LENGTH,NULL,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(view,&msize,1,NULL,PETSC_INT);CHKERRQ(ierr);

    if (dtype == (PetscInt) PETSC_CHAR) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_CHAR);CHKERRQ(ierr);
    } else if (dtype == (PetscInt) PETSC_REAL) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_REAL);CHKERRQ(ierr);
    } else if (dtype == (PetscInt) PETSC_SCALAR) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,1,NULL,PETSC_SCALAR);CHKERRQ(ierr);
    } else if (dtype == (PetscInt) PETSC_INT) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_INT);CHKERRQ(ierr);
    } else if (dtype == (PetscInt) PETSC_BOOL) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,msize,NULL,PETSC_BOOL);CHKERRQ(ierr);
    } else if (dtype == (PetscInt) PETSC_ENUM) {
      ierr = PetscViewerBinaryRead(view,((char*)bag)+nitem->offset,1,NULL,PETSC_ENUM);CHKERRQ(ierr);
      ierr = PetscViewerBinaryReadStringArray(view,&list);CHKERRQ(ierr);
      /* don't need to save list because it is already registered in the bag */
      ierr = PetscFree(list);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagGetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
@*/
PetscErrorCode PetscBagCreate(MPI_Comm comm, size_t bagsize, PetscBag *bag)
{
  PetscErrorCode ierr;
  size_t         totalsize = bagsize+sizeof(struct _n_PetscBag)+sizeof(PetscScalar);

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Creating Bag with total size %d\n",(int)totalsize);CHKERRQ(ierr);
  ierr = PetscMalloc(totalsize,bag);CHKERRQ(ierr);
  ierr = PetscMemzero(*bag,bagsize+sizeof(struct _n_PetscBag)+sizeof(PetscScalar));CHKERRQ(ierr);

  (*bag)->bagsize        = bagsize+sizeof(struct _n_PetscBag)+sizeof(PetscScalar);
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

.seealso: PetscBag, PetscBagGetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
@*/

PetscErrorCode PetscBagSetName(PetscBag bag, const char *name, const char *help)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(bag->bagname,name,PETSC_BAG_NAME_LENGTH-1);CHKERRQ(ierr);
  ierr = PetscStrncpy(bag->baghelp,help,PETSC_BAG_HELP_LENGTH-1);CHKERRQ(ierr);
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad(), PetscBagGetData()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
@*/
PetscErrorCode PetscBagGetName(PetscBag bag, char **name)
{
  PetscFunctionBegin;
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

.seealso: PetscBag, PetscBagSetName(), PetscBagView(), PetscBagLoad()
           PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
@*/
PetscErrorCode PetscBagGetData(PetscBag bag, void **data)
{
  PetscFunctionBegin;
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

.seealso: PetscBag, PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar()
           PetscBagSetFromOptions(), PetscBagCreate(), PetscBagDestroy(), PetscBagRegisterEnum()
@*/

PetscErrorCode PetscBagSetOptionsPrefix(PetscBag bag, const char pre[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pre) {
    ierr = PetscFree(bag->bagprefix);CHKERRQ(ierr);
  } else {
    if (pre[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hyphen");
    ierr = PetscFree(bag->bagprefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(pre,&(bag->bagprefix));CHKERRQ(ierr);
  }
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

.seealso: PetscBag, PetscBagGetName(), PetscBagSetName(), PetscBagCreate(), PetscBagGetData()
          PetscBagRegisterReal(), PetscBagRegisterInt(), PetscBagRegisterBool(), PetscBagRegisterScalar(), PetscBagRegisterEnum()
@*/
PetscErrorCode PetscBagGetNames(PetscBag bag, const char *names[])
{
  PetscBagItem nitem = bag->bagitems;
  PetscInt     n;

  PetscFunctionBegin;
  for (n = 0; nitem; ++n, nitem = nitem->next) {names[n] = nitem->name;}
  PetscFunctionReturn(0);
}
