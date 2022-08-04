/*
   Implements the higher-level options database querying methods. These are self-documenting and can attach at runtime to
   GUI code to display the options and get values from the users.

*/

#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscviewer.h>

#define ManSection(str) ((str) ? (str) : "None")

/*
    Keep a linked list of options that have been posted and we are waiting for
   user selection. See the manual page for PetscOptionsBegin()

    Eventually we'll attach this beast to a MPI_Comm
*/

/*
    Handles setting up the data structure in a call to PetscOptionsBegin()
*/
PetscErrorCode PetscOptionsBegin_Private(PetscOptionItems *PetscOptionsObject,MPI_Comm comm,const char prefix[],const char title[],const char mansec[])
{
  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix,3);
  PetscValidCharPointer(title,4);
  if (mansec) PetscValidCharPointer(mansec,5);
  if (!PetscOptionsObject->alreadyprinted) {
    if (!PetscOptionsHelpPrintedSingleton) PetscCall(PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton));
    PetscCall(PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton,prefix,title,&PetscOptionsObject->alreadyprinted));
  }
  PetscOptionsObject->next          = NULL;
  PetscOptionsObject->comm          = comm;
  PetscOptionsObject->changedmethod = PETSC_FALSE;

  PetscCall(PetscStrallocpy(prefix,&PetscOptionsObject->prefix));
  PetscCall(PetscStrallocpy(title,&PetscOptionsObject->title));

  PetscCall(PetscOptionsHasHelp(PetscOptionsObject->options,&PetscOptionsObject->printhelp));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1) {
    if (!PetscOptionsObject->alreadyprinted) {
      PetscCall((*PetscHelpPrintf)(comm,"----------------------------------------\n%s:\n",title));
    }
  }
  PetscFunctionReturn(0);
}

/*
    Handles setting up the data structure in a call to PetscObjectOptionsBegin()
*/
PetscErrorCode PetscObjectOptionsBegin_Private(PetscOptionItems *PetscOptionsObject,PetscObject obj)
{
  char      title[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidPointer(PetscOptionsObject,1);
  PetscValidHeader(obj,2);
  PetscOptionsObject->object         = obj;
  PetscOptionsObject->alreadyprinted = obj->optionsprinted;

  PetscCall(PetscStrcmp(obj->description,obj->class_name,&flg));
  if (flg) PetscCall(PetscSNPrintf(title,sizeof(title),"%s options",obj->class_name));
  else     PetscCall(PetscSNPrintf(title,sizeof(title),"%s (%s) options",obj->description,obj->class_name));
  PetscCall(PetscOptionsBegin_Private(PetscOptionsObject,obj->comm,obj->prefix,title,obj->mansec));
  PetscFunctionReturn(0);
}

/*
     Handles adding another option to the list of options within this particular PetscOptionsBegin() PetscOptionsEnd()
*/
static int PetscOptionItemCreate_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscOptionType t,PetscOptionItem *amsopt)
{
  PetscOptionItem next;
  PetscBool       valid;

  PetscFunctionBegin;
  PetscCall(PetscOptionsValidKey(opt,&valid));
  PetscCheck(valid,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"The option '%s' is not a valid key",opt);

  PetscCall(PetscNew(amsopt));
  (*amsopt)->next = NULL;
  (*amsopt)->set  = PETSC_FALSE;
  (*amsopt)->type = t;
  (*amsopt)->data = NULL;

  PetscCall(PetscStrallocpy(text,&(*amsopt)->text));
  PetscCall(PetscStrallocpy(opt,&(*amsopt)->option));
  PetscCall(PetscStrallocpy(man,&(*amsopt)->man));

  if (!PetscOptionsObject->next) PetscOptionsObject->next = *amsopt;
  else {
    next = PetscOptionsObject->next;
    while (next->next) next = next->next;
    next->next = *amsopt;
  }
  PetscFunctionReturn(0);
}

/*
    PetscScanString -  Gets user input via stdin from process and broadcasts to all processes

    Collective

   Input Parameters:
+     commm - communicator for the broadcast, must be PETSC_COMM_WORLD
.     n - length of the string, must be the same on all processes
-     str - location to store input

    Bugs:
.   Assumes process 0 of the given communicator has access to stdin

*/
static PetscErrorCode PetscScanString(MPI_Comm comm, size_t n, char str[])
{
  PetscMPIInt rank,nm;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    char   c = (char)getchar();
    size_t i = 0;

    while (c != '\n' && i < n-1) {
      str[i++] = c;
      c = (char)getchar();
    }
    str[i] = 0;
  }
  PetscCall(PetscMPIIntCast(n,&nm));
  PetscCallMPI(MPI_Bcast(str,nm,MPI_CHAR,0,comm));
  PetscFunctionReturn(0);
}

/*
    This is needed because certain strings may be freed by SAWs, hence we cannot use PetscStrallocpy()
*/
static PetscErrorCode PetscStrdup(const char s[], char *t[])
{
  char *tmp = NULL;

  PetscFunctionBegin;
  if (s) {
    size_t len;

    PetscCall(PetscStrlen(s,&len));
    tmp = (char*) malloc((len+1)*sizeof(*tmp));
    PetscCheck(tmp,PETSC_COMM_SELF,PETSC_ERR_MEM,"No memory to duplicate string");
    PetscCall(PetscStrcpy(tmp,s));
  }
  *t = tmp;
  PetscFunctionReturn(0);
}

/*
    PetscOptionsGetFromTextInput - Presents all the PETSc Options processed by the program so the user may change them at runtime

    Notes:
    this isn't really practical, it is just to demonstrate the principle

    A carriage return indicates no change from the default; but this like -ksp_monitor <stdout>  the default is actually not stdout the default
    is to do nothing so to get it to use stdout you need to type stdout. This is kind of bug?

    Bugs:
+    All processes must traverse through the exact same set of option queries due to the call to PetscScanString()
.    Internal strings have arbitrary length and string copies are not checked that they fit into string space
-    Only works for PetscInt == int, PetscReal == double etc

    Developer Notes:
    Normally the GUI that presents the options the user and retrieves the values would be running in a different
     address space and communicating with the PETSc program

*/
PetscErrorCode PetscOptionsGetFromTextInput(PetscOptionItems *PetscOptionsObject)
{
  PetscOptionItem next = PetscOptionsObject->next;
  char            str[512];
  PetscBool       bid;
  PetscReal       ir,*valr;
  PetscInt        *vald;
  size_t          i;

  PetscFunctionBegin;
  PetscCall((*PetscPrintf)(PETSC_COMM_WORLD,"%s --------------------\n",PetscOptionsObject->title));
  while (next) {
    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1));
      vald = (PetscInt*) next->data;
      for (i=0; i<next->arraylength; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT,vald[i]));
        if (i < next->arraylength-1) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,","));
        }
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,">: %s (%s) ",next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
        PetscToken token;
        PetscInt   n=0,nmax = next->arraylength,*dvalue = (PetscInt*)next->data,start,end;
        size_t     len;
        char       *value;
        PetscBool  foundrange;

        next->set = PETSC_TRUE;
        value     = str;
        PetscCall(PetscTokenCreate(value,',',&token));
        PetscCall(PetscTokenFind(token,&value));
        while (n < nmax) {
          if (!value) break;

          /* look for form  d-D where d and D are integers */
          foundrange = PETSC_FALSE;
          PetscCall(PetscStrlen(value,&len));
          if (value[0] == '-') i=2;
          else i=1;
          for (;i<len; i++) {
            if (value[i] == '-') {
              PetscCheck(i != len-1,PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry %s",n,value);
              value[i] = 0;
              PetscCall(PetscOptionsStringToInt(value,&start));
              PetscCall(PetscOptionsStringToInt(value+i+1,&end));
              PetscCheck(end > start,PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
              PetscCheck(n + end - start - 1 < nmax,PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry, not enough space in left in array (%" PetscInt_FMT ") to contain entire range from %" PetscInt_FMT " to %" PetscInt_FMT,n,nmax-n,start,end);
              for (; start<end; start++) {
                *dvalue = start; dvalue++;n++;
              }
              foundrange = PETSC_TRUE;
              break;
            }
          }
          if (!foundrange) {
            PetscCall(PetscOptionsStringToInt(value,dvalue));
            dvalue++;
            n++;
          }
          PetscCall(PetscTokenFind(token,&value));
        }
        PetscCall(PetscTokenDestroy(&token));
      }
      break;
    case OPTION_REAL_ARRAY:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1));
      valr = (PetscReal*) next->data;
      for (i=0; i<next->arraylength; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%g",(double)valr[i]));
        if (i < next->arraylength-1) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,","));
        }
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,">: %s (%s) ",next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
        PetscToken token;
        PetscInt   n = 0,nmax = next->arraylength;
        PetscReal  *dvalue = (PetscReal*)next->data;
        char       *value;

        next->set = PETSC_TRUE;
        value     = str;
        PetscCall(PetscTokenCreate(value,',',&token));
        PetscCall(PetscTokenFind(token,&value));
        while (n < nmax) {
          if (!value) break;
          PetscCall(PetscOptionsStringToReal(value,dvalue));
          dvalue++;
          n++;
          PetscCall(PetscTokenFind(token,&value));
        }
        PetscCall(PetscTokenDestroy(&token));
      }
      break;
    case OPTION_INT:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%d>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(int*)next->data,next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
#if defined(PETSC_SIZEOF_LONG_LONG)
        long long lid;
        sscanf(str,"%lld",&lid);
        PetscCheck(lid <= PETSC_MAX_INT && lid >= PETSC_MIN_INT,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %lld",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,lid);
#else
        long  lid;
        sscanf(str,"%ld",&lid);
        PetscCheck(lid <= PETSC_MAX_INT && lid >= PETSC_MIN_INT,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %ld",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,lid);
#endif

        next->set = PETSC_TRUE;
        *((PetscInt*)next->data) = (PetscInt)lid;
      }
      break;
    case OPTION_REAL:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%g>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(double*)next->data,next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
#if defined(PETSC_USE_REAL_SINGLE)
        sscanf(str,"%e",&ir);
#elif defined(PETSC_USE_REAL___FP16)
        float irtemp;
        sscanf(str,"%e",&irtemp);
        ir = irtemp;
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
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(PetscBool*)next->data ? "true": "false",next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
        PetscCall(PetscOptionsStringToBool(str,&bid));
        next->set = PETSC_TRUE;
        *((PetscBool*)next->data) = bid;
      }
      break;
    case OPTION_STRING:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,(char*)next->data,next->text,next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
        next->set = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        PetscCall(PetscStrdup(str,(char**)&next->data));
      }
      break;
    case OPTION_FLIST:
      PetscCall(PetscFunctionListPrintTypes(PETSC_COMM_WORLD,stdout,PetscOptionsObject->prefix,next->option,next->text,next->man,next->flist,(char*)next->data,(char*)next->data));
      PetscCall(PetscScanString(PETSC_COMM_WORLD,512,str));
      if (str[0]) {
        PetscOptionsObject->changedmethod = PETSC_TRUE;
        next->set = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        PetscCall(PetscStrdup(str,(char**)&next->data));
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

PetscErrorCode PetscOptionsSAWsDestroy(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static const char *OptionsHeader = "<head>\n"
                                   "<script type=\"text/javascript\" src=\"https://www.mcs.anl.gov/research/projects/saws/js/jquery-1.9.1.js\"></script>\n"
                                   "<script type=\"text/javascript\" src=\"https://www.mcs.anl.gov/research/projects/saws/js/SAWs.js\"></script>\n"
                                   "<script type=\"text/javascript\" src=\"js/PETSc.js\"></script>\n"
                                   "<script>\n"
                                      "jQuery(document).ready(function() {\n"
                                         "PETSc.getAndDisplayDirectory(null,\"#variablesInfo\")\n"
                                      "})\n"
                                  "</script>\n"
                                  "</head>\n";

/*  Determines the size and style of the scroll region where PETSc options selectable from users are displayed */
static const char *OptionsBodyBottom = "<div id=\"variablesInfo\" style=\"background-color:lightblue;height:auto;max-height:500px;overflow:scroll;\"></div>\n<br>\n</body>";

/*
    PetscOptionsSAWsInput - Presents all the PETSc Options processed by the program so the user may change them at runtime using the SAWs

    Bugs:
+    All processes must traverse through the exact same set of option queries do to the call to PetscScanString()
.    Internal strings have arbitrary length and string copies are not checked that they fit into string space
-    Only works for PetscInt == int, PetscReal == double etc

*/
PetscErrorCode PetscOptionsSAWsInput(PetscOptionItems *PetscOptionsObject)
{
  PetscOptionItem next     = PetscOptionsObject->next;
  static int      mancount = 0;
  char            options[16];
  PetscBool       changedmethod = PETSC_FALSE;
  PetscBool       stopasking    = PETSC_FALSE;
  char            manname[16],textname[16];
  char            dir[1024];

  PetscFunctionBegin;
  /* the next line is a bug, this will only work if all processors are here, the comm passed in is ignored!!! */
  sprintf(options,"Options_%d",count++);

  PetscOptionsObject->pprefix = PetscOptionsObject->prefix; /* SAWs will change this, so cannot pass prefix directly */

  PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s","_title"));
  PetscCallSAWs(SAWs_Register,(dir,&PetscOptionsObject->title,1,SAWs_READ,SAWs_STRING));
  PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s","prefix"));
  PetscCallSAWs(SAWs_Register,(dir,&PetscOptionsObject->pprefix,1,SAWs_READ,SAWs_STRING));
  PetscCallSAWs(SAWs_Register,("/PETSc/Options/ChangedMethod",&changedmethod,1,SAWs_WRITE,SAWs_BOOLEAN));
  PetscCallSAWs(SAWs_Register,("/PETSc/Options/StopAsking",&stopasking,1,SAWs_WRITE,SAWs_BOOLEAN));

  while (next) {
    sprintf(manname,"_man_%d",mancount);
    PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",manname));
    PetscCallSAWs(SAWs_Register,(dir,&next->man,1,SAWs_READ,SAWs_STRING));
    sprintf(textname,"_text_%d",mancount++);
    PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",textname));
    PetscCallSAWs(SAWs_Register,(dir,&next->text,1,SAWs_READ,SAWs_STRING));

    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_INT));
      break;
    case OPTION_REAL_ARRAY:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_DOUBLE));
      break;
    case OPTION_INT:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_INT));
      break;
    case OPTION_REAL:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_DOUBLE));
      break;
    case OPTION_BOOL:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,1,SAWs_WRITE,SAWs_BOOLEAN));
      break;
    case OPTION_BOOL_ARRAY:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_BOOLEAN));
      break;
    case OPTION_STRING:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      break;
    case OPTION_STRING_ARRAY:
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,next->data,next->arraylength,SAWs_WRITE,SAWs_STRING));
      break;
    case OPTION_FLIST:
      {
      PetscInt ntext;
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      PetscCall(PetscFunctionListGet(next->flist,(const char***)&next->edata,&ntext));
      PetscCallSAWs(SAWs_Set_Legal_Variable_Values,(dir,ntext,next->edata));
      }
      break;
    case OPTION_ELIST:
      {
      PetscInt ntext = next->nlist;
      PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
      PetscCallSAWs(SAWs_Register,(dir,&next->data,1,SAWs_WRITE,SAWs_STRING));
      PetscCall(PetscMalloc1((ntext+1),(char***)&next->edata));
      PetscCall(PetscMemcpy(next->edata,next->list,ntext*sizeof(char*)));
      PetscCallSAWs(SAWs_Set_Legal_Variable_Values,(dir,ntext,next->edata));
      }
      break;
    default:
      break;
    }
    next = next->next;
  }

  /* wait until accessor has unlocked the memory */
  PetscCallSAWs(SAWs_Push_Header,("index.html",OptionsHeader));
  PetscCallSAWs(SAWs_Push_Body,("index.html",2,OptionsBodyBottom));
  PetscCall(PetscSAWsBlock());
  PetscCallSAWs(SAWs_Pop_Header,("index.html"));
  PetscCallSAWs(SAWs_Pop_Body,("index.html",2));

  /* determine if any values have been set in GUI */
  next = PetscOptionsObject->next;
  while (next) {
    PetscCall(PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option));
    PetscCallSAWs(SAWs_Selected,(dir,(int*)&next->set));
    next = next->next;
  }

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (changedmethod) PetscOptionsObject->count = -2;

  if (stopasking) {
    PetscOptionsPublish      = PETSC_FALSE;
    PetscOptionsObject->count = 0;//do not ask for same thing again
  }

  PetscCallSAWs(SAWs_Delete,("/PETSc/Options"));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode PetscOptionsEnd_Private(PetscOptionItems *PetscOptionsObject)
{
  PetscOptionItem last;
  char            option[256],value[1024],tmp[32];
  size_t          j;

  PetscFunctionBegin;
  if (PetscOptionsObject->next) {
    if (!PetscOptionsObject->count) {
#if defined(PETSC_HAVE_SAWS)
      PetscCall(PetscOptionsSAWsInput(PetscOptionsObject));
#else
      PetscCall(PetscOptionsGetFromTextInput(PetscOptionsObject));
#endif
    }
  }

  PetscCall(PetscFree(PetscOptionsObject->title));

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (PetscOptionsObject->changedmethod) PetscOptionsObject->count = -2;
  /* reset alreadyprinted flag */
  PetscOptionsObject->alreadyprinted = PETSC_FALSE;
  if (PetscOptionsObject->object) PetscOptionsObject->object->optionsprinted = PETSC_TRUE;
  PetscOptionsObject->object = NULL;

  while (PetscOptionsObject->next) {
    if (PetscOptionsObject->next->set) {
      if (PetscOptionsObject->prefix) {
        PetscCall(PetscStrcpy(option,"-"));
        PetscCall(PetscStrcat(option,PetscOptionsObject->prefix));
        PetscCall(PetscStrcat(option,PetscOptionsObject->next->option+1));
      } else PetscCall(PetscStrcpy(option,PetscOptionsObject->next->option));

      switch (PetscOptionsObject->next->type) {
      case OPTION_HEAD:
        break;
      case OPTION_INT_ARRAY:
        sprintf(value,"%d",(int)((PetscInt*)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscInt*)PetscOptionsObject->next->data)[j]);
          PetscCall(PetscStrcat(value,","));
          PetscCall(PetscStrcat(value,tmp));
        }
        break;
      case OPTION_INT:
        sprintf(value,"%d",(int) *(PetscInt*)PetscOptionsObject->next->data);
        break;
      case OPTION_REAL:
        sprintf(value,"%g",(double) *(PetscReal*)PetscOptionsObject->next->data);
        break;
      case OPTION_REAL_ARRAY:
        sprintf(value,"%g",(double)((PetscReal*)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%g",(double)((PetscReal*)PetscOptionsObject->next->data)[j]);
          PetscCall(PetscStrcat(value,","));
          PetscCall(PetscStrcat(value,tmp));
        }
        break;
      case OPTION_SCALAR_ARRAY:
        sprintf(value,"%g+%gi",(double)PetscRealPart(((PetscScalar*)PetscOptionsObject->next->data)[0]),(double)PetscImaginaryPart(((PetscScalar*)PetscOptionsObject->next->data)[0]));
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%g+%gi",(double)PetscRealPart(((PetscScalar*)PetscOptionsObject->next->data)[j]),(double)PetscImaginaryPart(((PetscScalar*)PetscOptionsObject->next->data)[j]));
          PetscCall(PetscStrcat(value,","));
          PetscCall(PetscStrcat(value,tmp));
        }
        break;
      case OPTION_BOOL:
        sprintf(value,"%d",*(int*)PetscOptionsObject->next->data);
        break;
      case OPTION_BOOL_ARRAY:
        sprintf(value,"%d",(int)((PetscBool*)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscBool*)PetscOptionsObject->next->data)[j]);
          PetscCall(PetscStrcat(value,","));
          PetscCall(PetscStrcat(value,tmp));
        }
        break;
      case OPTION_FLIST:
        PetscCall(PetscStrcpy(value,(char*)PetscOptionsObject->next->data));
        break;
      case OPTION_ELIST:
        PetscCall(PetscStrcpy(value,(char*)PetscOptionsObject->next->data));
        break;
      case OPTION_STRING:
        PetscCall(PetscStrcpy(value,(char*)PetscOptionsObject->next->data));
        break;
      case OPTION_STRING_ARRAY:
        sprintf(value,"%s",((char**)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%s",((char**)PetscOptionsObject->next->data)[j]);
          PetscCall(PetscStrcat(value,","));
          PetscCall(PetscStrcat(value,tmp));
        }
        break;
      }
      PetscCall(PetscOptionsSetValue(PetscOptionsObject->options,option,value));
    }
    if (PetscOptionsObject->next->type == OPTION_ELIST) {
      PetscCall(PetscStrNArrayDestroy(PetscOptionsObject->next->nlist,(char ***)&PetscOptionsObject->next->list));
    }
    PetscCall(PetscFree(PetscOptionsObject->next->text));
    PetscCall(PetscFree(PetscOptionsObject->next->option));
    PetscCall(PetscFree(PetscOptionsObject->next->man));
    PetscCall(PetscFree(PetscOptionsObject->next->edata));

    if ((PetscOptionsObject->next->type == OPTION_STRING) || (PetscOptionsObject->next->type == OPTION_FLIST) || (PetscOptionsObject->next->type == OPTION_ELIST)) {
      free(PetscOptionsObject->next->data);
    } else {
      PetscCall(PetscFree(PetscOptionsObject->next->data));
    }

    last                     = PetscOptionsObject->next;
    PetscOptionsObject->next = PetscOptionsObject->next->next;
    PetscCall(PetscFree(last));
  }
  PetscCall(PetscFree(PetscOptionsObject->prefix));
  PetscOptionsObject->next = NULL;
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsEnum - Gets the enum value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsEnum(const char opt[],const char text[],const char man[],const char *const *list,PetscEnum currentvalue,PetscEnum *value,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsEnum(..., obj->value,&object->value,...) or
$                 value = defaultvalue
$                 PetscOptionsEnum(..., value,&value,&flg);
$                 if (flg) {

   Output Parameters:
+  value - the  value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          list is usually something like PCASMTypes or some other predefined list of enum names

          If the user does not supply the option at all value is NOT changed. Thus
          you should ALWAYS initialize value if you access it without first checking if the set flag is true.

          The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsEnum_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char *const *list,PetscEnum currentvalue,PetscEnum *value,PetscBool  *set)
{
  PetscInt       ntext = 0;
  PetscInt       tval;
  PetscBool      tflg;

  PetscFunctionBegin;
  while (list[ntext++]) {
    PetscCheck(ntext <= 50,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  PetscCheck(ntext >= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  PetscCall(PetscOptionsEList_Private(PetscOptionsObject,opt,text,man,list,ntext,list[currentvalue],&tval,&tflg));
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (tflg) *value = (PetscEnum)tval;
  if (set)  *set   = tflg;
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsEnumArray - Gets an array of enum values for a particular
   option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsEnumArray(const char opt[],const char text[],const char man[],const char *const *list,PetscEnum value[],PetscInt *n,PetscBool  *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  n - maximum number of values allowed in the value array

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The array must be passed as a comma separated list.

   There must be no intervening spaces between the values.

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsRealArray()`
M*/

PetscErrorCode  PetscOptionsEnumArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char *const *list,PetscEnum value[],PetscInt *n,PetscBool  *set)
{
  PetscInt        i,nlist = 0;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  while (list[nlist++]) PetscCheck(nlist <= 50,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  PetscCheck(nlist >= 3,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  nlist -= 3; /* drop enum name, prefix, and null termination */
  if (0 && !PetscOptionsObject->count) { /* XXX Requires additional support */
    PetscEnum *vals;
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT_ARRAY/*XXX OPTION_ENUM_ARRAY*/,&amsopt));
    PetscCall(PetscStrNArrayallocpy(nlist,list,(char***)&amsopt->list));
    amsopt->nlist = nlist;
    PetscCall(PetscMalloc1(*n,(PetscEnum**)&amsopt->data));
    amsopt->arraylength = *n;
    vals = (PetscEnum*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
  }
  PetscCall(PetscOptionsGetEnumArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,list,value,n,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%s",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,list[value[0]]));
    for (i=1; i<*n; i++) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,",%s",list[value[i]]));
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (choose from)",text));
    for (i=0; i<nlist; i++) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm," %s",list[i]));
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm," (%s)\n",ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsBoundedInt - Gets an integer value greater than or equal a given bound for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsBoundInt(const char opt[],const char text[],const char man[],PetscInt currentvalue,PetscInt *value,PetscBool *flg,PetscInt bound)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsInt(..., obj->value,&obj->value,...) or
$                 value = defaultvalue
$                 PetscOptionsInt(..., value,&value,&flg);
$                 if (flg) {
-  bound - the requested value should be greater than or equal this bound or an error is generated

   Output Parameters:
+  value - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
    If the user does not supply the option at all value is NOT changed. Thus
    you should ALWAYS initialize value if you access it without first checking if the set flag is true.

    The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

/*MC
   PetscOptionsRangeInt - Gets an integer value within a range of values for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
PetscErrorCode  PetscOptionsRangeInt(const char opt[],const char text[],const char man[],PetscInt currentvalue,PetscInt *value,PetscBool *flg,PetscInt lb,PetscInt ub)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsInt(..., obj->value,&obj->value,...) or
$                 value = defaultvalue
$                 PetscOptionsInt(..., value,&value,&flg);
$                 if (flg) {
.  lb - the lower bound, provided value must be greater than or equal to this value or an error is generated
-  ub - the upper bound, provided value must be less than or equal to this value or an error is generated

   Output Parameters:
+  value - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
    If the user does not supply the option at all value is NOT changed. Thus
    you should ALWAYS initialize value if you access it without first checking if the set flag is true.

    The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsBoundedInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

/*MC
   PetscOptionsInt - Gets the integer value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
PetscErrorCode  PetscOptionsInt(const char opt[],const char text[],const char man[],PetscInt currentvalue,PetscInt *value,PetscBool *flg))

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsInt(..., obj->value,&obj->value,...) or
$                 value = defaultvalue
$                 PetscOptionsInt(..., value,&value,&flg);
$                 if (flg) {

   Output Parameters:
+  value - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
    If the user does not supply the option at all value is NOT changed. Thus
    you should ALWAYS initialize value if you access it without first checking if the set flag is true.

    The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsBoundedInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsInt_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscInt currentvalue,PetscInt *value,PetscBool  *set,PetscInt lb,PetscInt ub)
{
  PetscOptionItem amsopt;
  PetscBool       wasset;

  PetscFunctionBegin;
  PetscCheck(currentvalue >= lb,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Current value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT,currentvalue,lb);
  PetscCheck(currentvalue <= ub,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Current value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT,currentvalue,ub);
     if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscInt),&amsopt->data));
    *(PetscInt*)amsopt->data = currentvalue;

    PetscCall(PetscOptionsGetInt(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,&currentvalue,&wasset));
    if (wasset) {
      *(PetscInt*)amsopt->data = currentvalue;
    }
  }
  PetscCall(PetscOptionsGetInt(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,&wasset));
  PetscCheck(!wasset || *value >= lb,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Newly set value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT,*value,lb);
  PetscCheck(!wasset || *value <= ub,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Newly set value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT,*value,ub);
  if (set) *set = wasset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %" PetscInt_FMT " : formerly %" PetscInt_FMT ">: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,wasset && value ? *value : currentvalue,currentvalue,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsString - Gets the string value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsString(const char opt[],const char text[],const char man[],const char currentvalue[],char value[],size_t len,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. This is not used to set value
-  len - length of the result string including null terminator

   Output Parameters:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Even if the user provided no string (for example -optionname -someotheroption) the flag is set to PETSC_TRUE (and the string is fulled with nulls).

          If the user does not supply the option at all value is NOT changed. Thus
          you should ALWAYS initialize value if you access it without first checking if the set flag is true.

          The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsString_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char currentvalue[],char value[],size_t len,PetscBool  *set)
{
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING,&amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data));
  }
  PetscCall(PetscOptionsGetString(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,len,&lset));
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %s : formerly %s>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,lset && value ? value : currentvalue,currentvalue,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsReal - Gets the PetscReal value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsReal(const char opt[],const char text[],const char man[],PetscReal currentvalue,PetscReal *value,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsReal(..., obj->value,&obj->value,...) or
$                 value = defaultvalue
$                 PetscOptionsReal(..., value,&value,&flg);
$                 if (flg) {

   Output Parameters:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
    If the user does not supply the option at all value is NOT changed. Thus
    you should ALWAYS initialize value if you access it without first checking if the set flag is true.

    The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsReal_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscReal currentvalue,PetscReal *value,PetscBool  *set)
{
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_REAL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscReal),&amsopt->data));

    *(PetscReal*)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetReal(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,&lset));
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g : %g>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,lset && value ? (double)*value : (double) currentvalue,(double)currentvalue,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsScalar - Gets the scalar value for a particular option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsScalar(const char opt[],const char text[],const char man[],PetscScalar currentvalue,PetscScalar *value,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
$                 PetscOptionsScalar(..., obj->value,&obj->value,...) or
$                 value = defaultvalue
$                 PetscOptionsScalar(..., value,&value,&flg);
$                 if (flg) {

   Output Parameters:
+  value - the value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
    If the user does not supply the option at all value is NOT changed. Thus
    you should ALWAYS initialize value if you access it without first checking if the set flag is true.

    The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsScalar_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscScalar currentvalue,PetscScalar *value,PetscBool  *set)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscOptionsReal(opt,text,man,currentvalue,value,set));
#else
  PetscCall(PetscOptionsGetScalar(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,set));
#endif
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsName - Determines if a particular option has been set in the database. This returns true whether the option is a number, string or boolean, even
                      its value is set to false.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsName(const char opt[],const char text[],const char man[],PetscBool  *flg)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsName_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool),&amsopt->data));

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  PetscCall(PetscOptionsHasName(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsFList - Puts a list of option values that a single one may be selected from

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsFList(const char opt[],const char ltext[],const char man[],PetscFunctionList list,const char currentvalue[],char value[],size_t len,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with
$                 PetscOptionsFlist(..., obj->value,value,len,&flg);
$                 if (flg) {
-  len - the length of the character array value

   Output Parameters:
+  value - the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          If the user does not supply the option at all value is NOT changed. Thus
          you should ALWAYS initialize value if you access it without first checking if the set flag is true.

          The default/currentvalue passed into this routine does not get transferred to the output value variable automatically.

   See PetscOptionsEList() for when the choices are given in a string array

   To get a listing of all currently specified options,
    see PetscOptionsView() or PetscOptionsGetAll()

   Developer Note: This cannot check for invalid selection because of things like MATAIJ that are not included in the list

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsEnum()`
M*/

PetscErrorCode  PetscOptionsFList_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char ltext[],const char man[],PetscFunctionList list,const char currentvalue[],char value[],size_t len,PetscBool  *set)
{
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,ltext,man,OPTION_FLIST,&amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data));
    amsopt->flist = list;
  }
  PetscCall(PetscOptionsGetString(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,len,&lset));
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall(PetscFunctionListPrintTypes(PetscOptionsObject->comm,stdout,PetscOptionsObject->prefix,opt,ltext,man,list,currentvalue,lset && value ? value : currentvalue));
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsEList - Puts a list of option values that a single one may be selected from

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode  PetscOptionsEList(const char opt[],const char ltext[],const char man[],const char *const *list,PetscInt ntext,const char currentvalue[],PetscInt *value,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  ltext - short string that describes the option
.  man - manual page with additional information on option
.  list - the possible choices (one of these must be selected, anything else is invalid)
.  ntext - number of choices
-  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with
$                 PetscOptionsElist(..., obj->value,&value,&flg);
$                 if (flg) {

   Output Parameters:
+  value - the index of the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

         If the user does not supply the option at all value is NOT changed. Thus
          you should ALWAYS initialize value if you access it without first checking if the set flag is true.

   See PetscOptionsFList() for when the choices are given in a PetscFunctionList()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEnum()`
M*/

PetscErrorCode  PetscOptionsEList_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char ltext[],const char man[],const char *const *list,PetscInt ntext,const char currentvalue[],PetscInt *value,PetscBool  *set)
{
  PetscInt        i;
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,ltext,man,OPTION_ELIST,&amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data));
    PetscCall(PetscStrNArrayallocpy(ntext,list,(char***)&amsopt->list));
    amsopt->nlist = ntext;
  }
  PetscCall(PetscOptionsGetEList(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,list,ntext,value,&lset));
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %s : formerly %s> %s (choose one of)",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,lset && value ? list[*value] : currentvalue,currentvalue,ltext));
    for (i=0; i<ntext; i++) {
      PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm," %s",list[i]));
    }
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm," (%s)\n",ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsBoolGroupBegin - First in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsBoolGroupBegin(const char opt[],const char text[],const char man[],PetscBool  *flg)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - whether that option was set or not

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must be followed by 0 or more PetscOptionsBoolGroup()s and PetscOptionsBoolGroupEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsBoolGroupBegin_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool),&amsopt->data));

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  Pick at most one of -------------\n"));
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsBoolGroup - One in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsBoolGroup(const char opt[],const char text[],const char man[],PetscBool  *flg)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsBoolGroupBegin() and preceded a PetscOptionsBoolGroupEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsBoolGroup_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool),&amsopt->data));

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
     PetscOptionsBoolGroupEnd - Last in a series of logical queries on the options database for
       which at most a single value can be true.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsBoolGroupEnd(const char opt[],const char text[],const char man[],PetscBool  *flg)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameter:
.  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Must follow a PetscOptionsBoolGroupBegin()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsBoolGroupEnd_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool),&amsopt->data));

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsBool - Determines if a particular option is in the database with a true or false

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsBool(const char opt[],const char text[],const char man[],PetscBool currentvalue,PetscBool  *flg,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
.  man - manual page with additional information on option
-  currentvalue - the current value

   Output Parameters:
+  flg - PETSC_TRUE or PETSC_FALSE
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

      If the option is given, but no value is provided, then flg and set are both given the value PETSC_TRUE. That is -requested_bool
     is equivalent to -requested_bool true

       If the user does not supply the option at all flg is NOT changed. Thus
     you should ALWAYS initialize the flg if you access it without first checking if the set flag is true.

    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   Level: beginner

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsBool_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool currentvalue,PetscBool  *flg,PetscBool  *set)
{
  PetscBool       iset;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool),&amsopt->data));

    *(PetscBool*)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,&iset));
  if (set) *set = iset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    const char *v = PetscBools[currentvalue], *vn = PetscBools[iset && flg ? *flg : currentvalue];
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s: <%s : %s> %s (%s)\n",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,v,vn,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsRealArray - Gets an array of double values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsRealArray(const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscBool  *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  n - maximum number of values that value has room for

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode PetscOptionsRealArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscBool  *set)
{
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscReal *vals;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_REAL_ARRAY,&amsopt));
    PetscCall(PetscMalloc((*n)*sizeof(PetscReal),&amsopt->data));
    vals = (PetscReal*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  PetscCall(PetscOptionsGetRealArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,(double)value[0]));
    for (i=1; i<*n; i++) {
      PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,",%g",(double)value[i]));
    }
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsScalarArray - Gets an array of Scalar values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsScalarArray(const char opt[],const char text[],const char man[],PetscScalar value[],PetscInt *n,PetscBool  *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  n - maximum number of values allowed in the value array

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode PetscOptionsScalarArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscScalar value[],PetscInt *n,PetscBool  *set)
{
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscScalar *vals;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_SCALAR_ARRAY,&amsopt));
    PetscCall(PetscMalloc((*n)*sizeof(PetscScalar),&amsopt->data));
    vals = (PetscScalar*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  PetscCall(PetscOptionsGetScalarArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g+%gi",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,(double)PetscRealPart(value[0]),(double)PetscImaginaryPart(value[0])));
    for (i=1; i<*n; i++) {
      PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,",%g+%gi",(double)PetscRealPart(value[i]),(double)PetscImaginaryPart(value[i])));
    }
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsIntArray - Gets an array of integers for a particular
   option in the database.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsIntArray(const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscBool  *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  n - maximum number of values

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The array can be passed as
   a comma separated list:                                 0,1,2,3,4,5,6,7
   a range (start-end+1):                                  0-8
   a range with given increment (start-end+1:inc):         0-7:2
   a combination of values and ranges separated by commas: 0,1-8,8-15:2

   There must be no intervening spaces between the values.

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsRealArray()`
M*/

PetscErrorCode  PetscOptionsIntArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscBool  *set)
{
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscInt *vals;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT_ARRAY,&amsopt));
    PetscCall(PetscMalloc1(*n,(PetscInt**)&amsopt->data));
    vals = (PetscInt*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  PetscCall(PetscOptionsGetIntArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%" PetscInt_FMT,PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,value[0]));
    for (i=1; i<*n; i++) {
      PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,",%" PetscInt_FMT,value[i]));
    }
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsStringArray(const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscBool  *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  nmax - maximum number of strings

   Output Parameters:
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

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsStringArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscBool  *set)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING_ARRAY,&amsopt));
    PetscCall(PetscMalloc1(*nmax,(char**)&amsopt->data));

    amsopt->arraylength = *nmax;
  }
  PetscCall(PetscOptionsGetStringArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,nmax,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <string1,string2,...>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsBoolArray - Gets an array of logical values (true or false) for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsBoolArray(const char opt[],const char text[],const char man[],PetscBool value[],PetscInt *n,PetscBool *set)

   Input Parameters:
+  opt - the option one is seeking
.  text - short string describing option
.  man - manual page for option
-  n - maximum number of values allowed in the value array

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsBoolArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool value[],PetscInt *n,PetscBool *set)
{
  PetscInt         i;
  PetscOptionItem  amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscBool *vals;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL_ARRAY,&amsopt));
    PetscCall(PetscMalloc1(*n,(PetscBool**)&amsopt->data));
    vals = (PetscBool*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  PetscCall(PetscOptionsGetBoolArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%d",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,value[0]));
    for (i=1; i<*n; i++) {
      PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,",%d",value[i]));
    }
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsViewer - Gets a viewer appropriate for the type indicated by the user

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsViewer(const char opt[],const char text[],const char man[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameters:
+  viewer - the viewer
.  format - the PetscViewerFormat requested by the user, pass NULL if not needed
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsGetViewer() for the format of the supplied viewer and its options

.seealso: `PetscOptionsGetViewer()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/

PetscErrorCode  PetscOptionsViewer_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool  *set)
{
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING,&amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup("",(char**)&amsopt->data));
  }
  PetscCall(PetscOptionsGetViewer(PetscOptionsObject->comm,PetscOptionsObject->options,PetscOptionsObject->prefix,opt,viewer,format,set));
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%s>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,"",text,ManSection(man)));
  }
  PetscFunctionReturn(0);
}
