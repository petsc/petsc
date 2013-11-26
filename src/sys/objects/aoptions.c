
/*
   Implements the higher-level options database querying methods. These are self-documenting and can attach at runtime to
   GUI code to display the options and get values from the users.

*/

#include <petsc-private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscviewer.h>

#define ManSection(str) ((str) ? (str) : "None")

/*
    Keep a linked list of options that have been posted and we are waiting for
   user selection. See the manual page for PetscOptionsBegin()

    Eventually we'll attach this beast to a MPI_Comm
*/
PetscOptionsObjectType PetscOptionsObject;
PetscInt               PetscOptionsPublishCount = 0;

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

  ierr = PetscFree(PetscOptionsObject.prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(prefix,&PetscOptionsObject.prefix);CHKERRQ(ierr);
  ierr = PetscFree(PetscOptionsObject.title);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&PetscOptionsObject.title);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,"-help",&PetscOptionsObject.printhelp);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1) {
    if (!PetscOptionsObject.alreadyprinted) {
      ierr = (*PetscHelpPrintf)(comm,"%s -------------------------------------------------\n",title);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectOptionsBegin_Private"
/*
    Handles setting up the data structure in a call to PetscObjectOptionsBegin()
*/
PetscErrorCode PetscObjectOptionsBegin_Private(PetscObject obj)
{
  PetscErrorCode ierr;
  char           title[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscOptionsObject.object         = obj;
  PetscOptionsObject.alreadyprinted = obj->optionsprinted;

  ierr = PetscStrcmp(obj->description,obj->class_name,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscSNPrintf(title,sizeof(title),"%s options",obj->class_name);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(title,sizeof(title),"%s (%s) options",obj->description,obj->class_name);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBegin_Private(obj->comm,obj->prefix,title,obj->mansec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Handles adding another option to the list of options within this particular PetscOptionsBegin() PetscOptionsEnd()
*/
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCreate_Private"
static int PetscOptionsCreate_Private(const char opt[],const char text[],const char man[],PetscOptionType t,PetscOptions *amsopt)
{
  int          ierr;
  PetscOptions next;
  PetscBool    valid;

  PetscFunctionBegin;
  ierr = PetscOptionsValidKey(opt,&valid);CHKERRQ(ierr);
  if (!valid) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"The option '%s' is not a valid key",opt);

  ierr            = PetscNew(struct _n_PetscOptions,amsopt);CHKERRQ(ierr);
  (*amsopt)->next = 0;
  (*amsopt)->set  = PETSC_FALSE;
  (*amsopt)->type = t;
  (*amsopt)->data = 0;

  ierr = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr = PetscStrallocpy(man,&(*amsopt)->man);CHKERRQ(ierr);

  if (!PetscOptionsObject.next) PetscOptionsObject.next = *amsopt;
  else {
    next = PetscOptionsObject.next;
    while (next->next) next = next->next;
    next->next = *amsopt;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscScanString"
/*
    PetscScanString -  Gets user input via stdin from process and broadcasts to all processes

    Collective on MPI_Comm

   Input Parameters:
+     commm - communicator for the broadcast, must be PETSC_COMM_WORLD
.     n - length of the string, must be the same on all processes
-     str - location to store input

    Bugs:
.   Assumes process 0 of the given communicator has access to stdin

*/
static PetscErrorCode PetscScanString(MPI_Comm comm,size_t n,char str[])
{
  size_t         i;
  char           c;
  PetscMPIInt    rank,nm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    c = (char) getchar();
    i = 0;
    while (c != '\n' && i < n-1) {
      str[i++] = c;
      c = (char)getchar();
    }
    str[i] = 0;
  }
  ierr = PetscMPIIntCast(n,&nm);CHKERRQ(ierr);
  ierr = MPI_Bcast(str,nm,MPI_CHAR,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsGetFromTextInput"
/*
    PetscOptionsGetFromTextInput - Presents all the PETSc Options processed by the program so the user may change them at runtime

    Notes: this isn't really practical, it is just to demonstrate the principle

    A carriage return indicates no change from the default; but this like -ksp_monitor <stdout>  the default is actually not stdout the default
    is to do nothing so to get it to use stdout you need to type stdout. This is kind of bug?

    Bugs:
+    All processes must traverse through the exact same set of option queries due to the call to PetscScanString()
.    Internal strings have arbitrary length and string copies are not checked that they fit into string space
-    Only works for PetscInt == int, PetscReal == double etc

    Developer Notes: Normally the GUI that presents the options the user and retrieves the values would be running in a different
     address space and communicating with the PETSc program

*/
PetscErrorCode PetscOptionsGetFromTextInput()
{
  PetscErrorCode ierr;
  PetscOptions   next = PetscOptionsObject.next;
  char           str[512];
  PetscBool      bid;
  PetscReal      ir,*valr;
  PetscInt       *vald;
  size_t         i;

  ierr = (*PetscPrintf)(PETSC_COMM_WORLD,"%s -------------------------------------------------\n",PetscOptionsObject.title);CHKERRQ(ierr);
  while (next) {
    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1);CHKERRQ(ierr);
      vald = (PetscInt*) next->data;
      for (i=0; i<next->arraylength; i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%d",vald[i]);CHKERRQ(ierr);
        if (i < next->arraylength-1) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,",");CHKERRQ(ierr);
        }
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,">: %s (%s) ",next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        PetscToken token;
        PetscInt   n=0,nmax = next->arraylength,*dvalue = (PetscInt*)next->data,start,end;
        size_t     len;
        char       *value;
        PetscBool  foundrange;

        next->set = PETSC_TRUE;
        value     = str;
        ierr      = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
        ierr      = PetscTokenFind(token,&value);CHKERRQ(ierr);
        while (n < nmax) {
          if (!value) break;

          /* look for form  d-D where d and D are integers */
          foundrange = PETSC_FALSE;
          ierr       = PetscStrlen(value,&len);CHKERRQ(ierr);
          if (value[0] == '-') i=2;
          else i=1;
          for (;i<len; i++) {
            if (value[i] == '-') {
              if (i == len-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry %s\n",n,value);
              value[i] = 0;
              ierr     = PetscOptionsStringToInt(value,&start);CHKERRQ(ierr);
              ierr     = PetscOptionsStringToInt(value+i+1,&end);CHKERRQ(ierr);
              if (end <= start) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
              if (n + end - start - 1 >= nmax) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, not enough space in left in array (%D) to contain entire range from %D to %D",n,nmax-n,start,end);
              for (; start<end; start++) {
                *dvalue = start; dvalue++;n++;
              }
              foundrange = PETSC_TRUE;
              break;
            }
          }
          if (!foundrange) {
            ierr = PetscOptionsStringToInt(value,dvalue);CHKERRQ(ierr);
            dvalue++;
            n++;
          }
          ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
        }
        ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
      }
      break;
    case OPTION_REAL_ARRAY:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1);CHKERRQ(ierr);
      valr = (PetscReal*) next->data;
      for (i=0; i<next->arraylength; i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%g",valr[i]);CHKERRQ(ierr);
        if (i < next->arraylength-1) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,",");CHKERRQ(ierr);
        }
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,">: %s (%s) ",next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        PetscToken token;
        PetscInt   n = 0,nmax = next->arraylength;
        PetscReal  *dvalue = (PetscReal*)next->data;
        char       *value;

        next->set = PETSC_TRUE;
        value     = str;
        ierr      = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
        ierr      = PetscTokenFind(token,&value);CHKERRQ(ierr);
        while (n < nmax) {
          if (!value) break;
          ierr = PetscOptionsStringToReal(value,dvalue);CHKERRQ(ierr);
          dvalue++;
          n++;
          ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
        }
        ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
      }
      break;
    case OPTION_INT:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%d>: %s (%s) ",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,*(int*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
#if defined(PETSC_SIZEOF_LONG_LONG)
        long long lid;
        sscanf(str,"%lld",&lid);
        if (lid > PETSC_MAX_INT || lid < PETSC_MIN_INT) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %lld",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,lid);
#else
        long  lid;
        sscanf(str,"%ld",&lid);
        if (lid > PETSC_MAX_INT || lid < PETSC_MIN_INT) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %ld",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,lid);
#endif

        next->set = PETSC_TRUE;
        *((PetscInt*)next->data) = (PetscInt)lid;
      }
      break;
    case OPTION_REAL:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%g>: %s (%s) ",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,*(double*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
#if defined(PETSC_USE_REAL_SINGLE)
        sscanf(str,"%e",&ir);
#elif defined(PETSC_USE_REAL_DOUBLE)
        sscanf(str,"%le",&ir);
#elif defined(PETSC_USE_REAL___FLOAT128)
        ir = strtoflt128(str,0);
#else
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unknown scalar type");
#endif
        next->set                 = PETSC_TRUE;
        *((PetscReal*)next->data) = ir;
      }
      break;
    case OPTION_BOOL:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,*(PetscBool*)next->data ? "true": "false",next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        ierr = PetscOptionsStringToBool(str,&bid);CHKERRQ(ierr);
        next->set = PETSC_TRUE;
        *((PetscBool*)next->data) = bid;
      }
      break;
    case OPTION_STRING:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",next->option+1,(char*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        next->set = PETSC_TRUE;
        ierr = PetscStrallocpy(str,(char**) &next->data);CHKERRQ(ierr);
      }
      break;
    case OPTION_FLIST:
      ierr = PetscFunctionListPrintTypes(PETSC_COMM_WORLD,stdout,PetscOptionsObject.prefix,next->option,next->text,next->man,next->flist,(char*)next->data);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        PetscOptionsObject.changedmethod = PETSC_TRUE;
        next->set = PETSC_TRUE;
        ierr = PetscStrallocpy(str,(char**) &next->data);CHKERRQ(ierr);
      }
      break;
    default:
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>

static int count = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsSAWsDestroy"
PetscErrorCode PetscOptionsSAWsDestroy(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsSAWsInput"
/*
    PetscOptionsSAWsInput - Presents all the PETSc Options processed by the program so the user may change them at runtime using the SAWs

    Bugs:
+    All processes must traverse through the exact same set of option queries do to the call to PetscScanString()
.    Internal strings have arbitrary length and string copies are not checked that they fit into string space
-    Only works for PetscInt == int, PetscReal == double etc


*/
PetscErrorCode PetscOptionsSAWsInput()
{
  PetscErrorCode ierr;
  PetscOptions   next     = PetscOptionsObject.next;
  static int     mancount = 0;
  char           options[16];
  PetscBool      changedmethod = PETSC_FALSE;
  char           manname[16],textname[16];
  char           dir[1024];

  /* the next line is a bug, this will only work if all processors are here, the comm passed in is ignored!!! */
  sprintf(options,"Options_%d",count++);

  PetscOptionsObject.pprefix = PetscOptionsObject.prefix; /* SAWs will change this, so cannot pass prefix directly */

  ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s","_title");CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&PetscOptionsObject.title,1,SAWs_READ,SAWs_STRING));
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s","prefix");CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&PetscOptionsObject.pprefix,1,SAWs_READ,SAWs_STRING));
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Options/ChangedMethod",&changedmethod,1,SAWs_WRITE,SAWs_BOOLEAN));

  while (next) {
    sprintf(manname,"_man_%d",mancount);
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",manname);CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Register,(dir,&next->man,1,SAWs_READ,SAWs_STRING));
    sprintf(textname,"_text_%d",mancount++);
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",textname);CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Register,(dir,&next->text,1,SAWs_READ,SAWs_STRING));

    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_INT));
      break;
    case OPTION_REAL_ARRAY:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_DOUBLE));
      break;
    case OPTION_INT:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_INT));
      break;
    case OPTION_REAL:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_DOUBLE));
      break;
    case OPTION_BOOL:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_BOOLEAN));
      break;
    case OPTION_BOOL_ARRAY:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_BOOLEAN));
      break;
    case OPTION_STRING:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      break;
    case OPTION_STRING_ARRAY:
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_STRING));
      break;
    case OPTION_FLIST:
      {
      PetscInt ntext;
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      ierr = PetscFunctionListGet(next->flist,(const char***)&next->edata,&ntext);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Set_Legal_Variable_Values,(dir,ntext,next->edata));
      }
      break;
    case OPTION_ELIST:
      {
      PetscInt ntext = next->nlist;
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      ierr = PetscMalloc((ntext+1)*sizeof(char*),&next->edata);CHKERRQ(ierr);
      ierr = PetscMemcpy(next->edata,next->list,ntext*sizeof(char*));CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Set_Legal_Variable_Values,(dir,ntext,next->edata));
      }
      break;
    default:
      break;
    }
    next = next->next;
  }

  /* wait until accessor has unlocked the memory */
  ierr = PetscSAWsBlock();CHKERRQ(ierr);

  /* determine if any values have been set in GUI */
  next = PetscOptionsObject.next;
  while (next) {
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Selected,(dir,&next->set));
    next = next->next;
  }

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (changedmethod) PetscOptionsPublishCount = -2;

  PetscStackCallSAWs(SAWs_Delete,("/PETSc/Options"));
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsEnd_Private"
PetscErrorCode PetscOptionsEnd_Private(void)
{
  PetscErrorCode ierr;
  PetscOptions   last;
  char           option[256],value[1024],tmp[32];
  size_t         j;

  PetscFunctionBegin;
  if (PetscOptionsObject.next) {
    if (!PetscOptionsPublishCount) {
#if defined(PETSC_HAVE_SAWS)
      ierr = PetscOptionsSAWsInput();CHKERRQ(ierr);
#else
      ierr = PetscOptionsGetFromTextInput();CHKERRQ(ierr);
#endif
    }
  }

  ierr = PetscFree(PetscOptionsObject.title);CHKERRQ(ierr);
  ierr = PetscFree(PetscOptionsObject.prefix);CHKERRQ(ierr);

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (PetscOptionsObject.changedmethod) PetscOptionsPublishCount = -2;
  /* reset alreadyprinted flag */
  PetscOptionsObject.alreadyprinted = PETSC_FALSE;
  if (PetscOptionsObject.object) PetscOptionsObject.object->optionsprinted = PETSC_TRUE;
  PetscOptionsObject.object = NULL;

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
      case OPTION_INT_ARRAY:
        sprintf(value,"%d",(int)((PetscInt*)PetscOptionsObject.next->data)[0]);
        for (j=1; j<PetscOptionsObject.next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscInt*)PetscOptionsObject.next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_INT:
        sprintf(value,"%d",(int) *(PetscInt*)PetscOptionsObject.next->data);
        break;
      case OPTION_REAL:
        sprintf(value,"%g",(double) *(PetscReal*)PetscOptionsObject.next->data);
        break;
      case OPTION_REAL_ARRAY:
        sprintf(value,"%g",(double)((PetscReal*)PetscOptionsObject.next->data)[0]);
        for (j=1; j<PetscOptionsObject.next->arraylength; j++) {
          sprintf(tmp,"%g",(double)((PetscReal*)PetscOptionsObject.next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_BOOL:
        sprintf(value,"%d",*(int*)PetscOptionsObject.next->data);
        break;
      case OPTION_BOOL_ARRAY:
        sprintf(value,"%d",(int)((PetscBool*)PetscOptionsObject.next->data)[0]);
        for (j=1; j<PetscOptionsObject.next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscBool*)PetscOptionsObject.next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_FLIST:
      case OPTION_ELIST:
        ierr = PetscStrcpy(value,(char*)PetscOptionsObject.next->data);CHKERRQ(ierr);
        break;
      case OPTION_STRING:
        ierr = PetscStrcpy(value,(char*)PetscOptionsObject.next->data);CHKERRQ(ierr);
        break;
      case OPTION_STRING_ARRAY:
        sprintf(value,"%s",((char**)PetscOptionsObject.next->data)[0]);
        for (j=1; j<PetscOptionsObject.next->arraylength; j++) {
          sprintf(tmp,"%s",((char**)PetscOptionsObject.next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      }
      ierr = PetscOptionsSetValue(option,value);CHKERRQ(ierr);
    }
    ierr   = PetscFree(PetscOptionsObject.next->text);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->option);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->man);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->edata);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject.next->data);CHKERRQ(ierr);

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

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  defaultv - the default (current) value

   Output Parameter:
+  value - the  value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          list is usually something like PCASMTypes or some other predefined list of enum names

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsEnum(const char opt[],const char text[],const char man[],const char *const *list,PetscEnum defaultv,PetscEnum *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt       ntext = 0;
  PetscInt       tval;
  PetscBool      tflg;

  PetscFunctionBegin;
  while (list[ntext++]) {
    if (ntext > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  if (ntext < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  ierr   = PetscOptionsEList(opt,text,man,list,ntext,list[defaultv],&tval,&tflg);CHKERRQ(ierr);
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (tflg) *value = (PetscEnum)tval;
  if (set)  *set   = tflg;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsInt"
/*@C
   PetscOptionsInt - Gets the integer value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsInt(const char opt[],const char text[],const char man[],PetscInt defaultv,PetscInt *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_INT,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt),&amsopt->data);CHKERRQ(ierr);

    *(PetscInt*)amsopt->data = defaultv;
  }
  ierr = PetscOptionsGetInt(PetscOptionsObject.prefix,opt,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%d>: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,defaultv,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsString"
/*@C
   PetscOptionsString - Gets the string value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  defaultv - the default (current) value
-  len - length of the result string including null terminator

   Output Parameter:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Even if the user provided no string (for example -optionname -someotheroption) the flag is set to PETSC_TRUE (and the string is fulled with nulls).

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsString(const char opt[],const char text[],const char man[],const char defaultv[],char value[],size_t len,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_STRING,&amsopt);CHKERRQ(ierr);
    ierr = PetscStrallocpy(defaultv ? defaultv : "",(char**) &amsopt->data);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(PetscOptionsObject.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s>: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,defaultv,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsReal"
/*@C
   PetscOptionsReal - Gets the PetscReal value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsReal(const char opt[],const char text[],const char man[],PetscReal defaultv,PetscReal *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_REAL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);

    *(PetscReal*)amsopt->data = defaultv;
  }
  ierr = PetscOptionsGetReal(PetscOptionsObject.prefix,opt,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%G>: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,defaultv,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsScalar"
/*@C
   PetscOptionsScalar - Gets the scalar value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsScalar(const char opt[],const char text[],const char man[],PetscScalar defaultv,PetscScalar *value,PetscBool  *set)
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

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsName"
/*@C
   PetscOptionsName - Determines if a particular option has been set in the database. This returns true whether the option is a number, string or boolean, even
                      its value is set to false.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsName(const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  ierr = PetscOptionsHasName(PetscOptionsObject.prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsFList"
/*@C
     PetscOptionsFList - Puts a list of option values that a single one may be selected from

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices
.  defaultv - the default (current) value
-  len - the length of the character array value

   Output Parameter:
+  value - the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsEList() for when the choices are given in a string array

   To get a listing of all currently specified options,
    see PetscOptionsView() or PetscOptionsGetAll()

   Developer Note: This cannot check for invalid selection because of things like MATAIJ that are not included in the list

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsEnum()
@*/
PetscErrorCode  PetscOptionsFList(const char opt[],const char ltext[],const char man[],PetscFunctionList list,const char defaultv[],char value[],size_t len,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,ltext,man,OPTION_FLIST,&amsopt);CHKERRQ(ierr);
    ierr = PetscStrallocpy(defaultv ? defaultv : "",(char**) &amsopt->data);CHKERRQ(ierr);
    amsopt->flist = list;
  }
  ierr = PetscOptionsGetString(PetscOptionsObject.prefix,opt,value,len,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = PetscFunctionListPrintTypes(PetscOptionsObject.comm,stdout,PetscOptionsObject.prefix,opt,ltext,man,list,defaultv);CHKERRQ(ierr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsEList"
/*@C
     PetscOptionsEList - Puts a list of option values that a single one may be selected from

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  ltext - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices (one of these must be selected, anything else is invalid)
.  ntext - number of choices
-  defaultv - the default (current) value

   Output Parameter:
+  value - the index of the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsFList() for when the choices are given in a PetscFunctionList()

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEnum()
@*/
PetscErrorCode  PetscOptionsEList(const char opt[],const char ltext[],const char man[],const char *const *list,PetscInt ntext,const char defaultv[],PetscInt *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,ltext,man,OPTION_ELIST,&amsopt);CHKERRQ(ierr);
    ierr = PetscStrallocpy(defaultv ? defaultv : "",(char**) &amsopt->data);CHKERRQ(ierr);
    amsopt->list  = list;
    amsopt->nlist = ntext;
  }
  ierr = PetscOptionsGetEList(PetscOptionsObject.prefix,opt,list,ntext,value,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s> (choose one of)",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,defaultv);CHKERRQ(ierr);
    for (i=0; i<ntext; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm," (%s)\n",ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsBoolGroupBegin"
/*@C
     PetscOptionsBoolGroupBegin - First in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - whether that option was set or not

   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must be followed by 0 or more PetscOptionsBoolGroup()s and PetscOptionsBoolGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsBoolGroupBegin(const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject.prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  Pick at most one of -------------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsBoolGroup"
/*@C
     PetscOptionsBoolGroup - One in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsBoolGroupBegin() and preceded a PetscOptionsBoolGroupEnd()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsBoolGroup(const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject.prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsBoolGroupEnd"
/*@C
     PetscOptionsBoolGroupEnd - Last in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsBoolGroupBegin()

    Concepts: options database^logical group

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsBoolGroupEnd(const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject.prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"    -%s%s: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsBool"
/*@C
   PetscOptionsBool - Determines if a particular option is in the database with a true or false

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsBool(const char opt[],const char text[],const char man[],PetscBool deflt,PetscBool  *flg,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscBool      iset;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = deflt;
  }
  ierr = PetscOptionsGetBool(PetscOptionsObject.prefix,opt,flg,&iset);CHKERRQ(ierr);
  if (!iset) {
    if (flg) *flg = deflt;
  }
  if (set) *set = iset;
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    const char *v = PetscBools[deflt];
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s: <%s> %s (%s)\n",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,v,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsRealArray"
/*@C
   PetscOptionsRealArray - Gets an array of double values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsRealArray(const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    PetscReal *vals;

    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_REAL_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*n)*sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);
    vals = (PetscReal*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetRealArray(PetscOptionsObject.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%G",PetscOptionsObject.prefix?PetscOptionsObject.prefix:"",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%G",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscOptionsIntArray"
/*@C
   PetscOptionsIntArray - Gets an array of integers for a particular
   option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  n - maximum number of values

   Output Parameter:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The array can be passed as
   a comma seperated list:                                 0,1,2,3,4,5,6,7
   a range (start-end+1):                                  0-8
   a range with given increment (start-end+1:inc):         0-7:2
   a combination of values and ranges seperated by commas: 0,1-8,8-15:2

   There must be no intervening spaces between the values.

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Concepts: options database^array of ints

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsRealArray()
@*/
PetscErrorCode  PetscOptionsIntArray(const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    PetscInt *vals;

    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_INT_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*n)*sizeof(PetscInt),&amsopt->data);CHKERRQ(ierr);
    vals = (PetscInt*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetIntArray(PetscOptionsObject.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%d",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%d",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsStringArray"
/*@C
   PetscOptionsStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsStringArray(const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_STRING_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*nmax)*sizeof(char*),&amsopt->data);CHKERRQ(ierr);

    amsopt->arraylength = *nmax;
  }
  ierr = PetscOptionsGetStringArray(PetscOptionsObject.prefix,opt,value,nmax,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <string1,string2,...>: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsBoolArray"
/*@C
   PetscOptionsBoolArray - Gets an array of logical values (true or false) for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsBoolArray(const char opt[],const char text[],const char man[],PetscBool value[],PetscInt *n,PetscBool *set)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    PetscBool *vals;

    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_BOOL_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*n)*sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);
    vals = (PetscBool*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetBoolArray(PetscOptionsObject.prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%d",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%d",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsViewer"
/*@C
   PetscOptionsInt - Gets a viewer appropriate for the type indicated by the user

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
+  viewer - the viewer
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()
If no value is provided ascii:stdout is used
$       ascii[:[filename][:format]]   defaults to stdout - format can be one of info, info_detailed, or matlab, for example ascii::info prints just the info
$                                     about the object to standard out
$       binary[:filename]   defaults to binaryoutput
$       draw
$       socket[:port]    defaults to the standard output port

   Use PetscRestoreViewerDestroy() after using the viewer, otherwise a memory leak will occur

.seealso: PetscOptionsGetViewer(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsViewer(const char opt[],const char text[],const char man[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscOptions   amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsPublishCount) {
    ierr = PetscOptionsCreate_Private(opt,text,man,OPTION_STRING,&amsopt);CHKERRQ(ierr);
    ierr = PetscStrallocpy("",(char**) &amsopt->data);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(PetscOptionsObject.comm,PetscOptionsObject.prefix,opt,viewer,format,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s>: %s (%s)\n",PetscOptionsObject.prefix ? PetscOptionsObject.prefix : "",opt+1,"",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHead"
/*@C
     PetscOptionsHead - Puts a heading before listing any more published options. Used, for example,
            in KSPSetFromOptions_GMRES().

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameter:
.   head - the heading text


   Level: intermediate

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          Can be followed by a call to PetscOptionsTail() in the same function.

   Concepts: options database^subheading

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsHead(const char head[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  %s\n",head);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






