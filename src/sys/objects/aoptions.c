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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix,3);
  PetscValidCharPointer(title,4);
  if (mansec) PetscValidCharPointer(mansec,5);
  if (!PetscOptionsObject->alreadyprinted) {
    if (!PetscOptionsHelpPrintedSingleton) {
      ierr = PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton);CHKERRQ(ierr);
    }
    ierr = PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton,prefix,title,&PetscOptionsObject->alreadyprinted);CHKERRQ(ierr);
  }
  PetscOptionsObject->next          = NULL;
  PetscOptionsObject->comm          = comm;
  PetscOptionsObject->changedmethod = PETSC_FALSE;

  ierr = PetscStrallocpy(prefix,&PetscOptionsObject->prefix);CHKERRQ(ierr);
  ierr = PetscStrallocpy(title,&PetscOptionsObject->title);CHKERRQ(ierr);

  ierr = PetscOptionsHasHelp(PetscOptionsObject->options,&PetscOptionsObject->printhelp);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1) {
    if (!PetscOptionsObject->alreadyprinted) {
      ierr = (*PetscHelpPrintf)(comm,"----------------------------------------\n%s:\n",title);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
    Handles setting up the data structure in a call to PetscObjectOptionsBegin()
*/
PetscErrorCode PetscObjectOptionsBegin_Private(PetscOptionItems *PetscOptionsObject,PetscObject obj)
{
  PetscErrorCode ierr;
  char           title[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeader(obj,2);
  PetscOptionsObject->object         = obj;
  PetscOptionsObject->alreadyprinted = obj->optionsprinted;

  ierr = PetscStrcmp(obj->description,obj->class_name,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscSNPrintf(title,sizeof(title),"%s options",obj->class_name);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(title,sizeof(title),"%s (%s) options",obj->description,obj->class_name);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBegin_Private(PetscOptionsObject,obj->comm,obj->prefix,title,obj->mansec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Handles adding another option to the list of options within this particular PetscOptionsBegin() PetscOptionsEnd()
*/
static int PetscOptionItemCreate_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscOptionType t,PetscOptionItem *amsopt)
{
  int             ierr;
  PetscOptionItem next;
  PetscBool       valid;

  PetscFunctionBegin;
  ierr = PetscOptionsValidKey(opt,&valid);CHKERRQ(ierr);
  if (!valid) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"The option '%s' is not a valid key",opt);

  ierr            = PetscNew(amsopt);CHKERRQ(ierr);
  (*amsopt)->next = NULL;
  (*amsopt)->set  = PETSC_FALSE;
  (*amsopt)->type = t;
  (*amsopt)->data = NULL;

  ierr = PetscStrallocpy(text,&(*amsopt)->text);CHKERRQ(ierr);
  ierr = PetscStrallocpy(opt,&(*amsopt)->option);CHKERRQ(ierr);
  ierr = PetscStrallocpy(man,&(*amsopt)->man);CHKERRQ(ierr);

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
static PetscErrorCode PetscScanString(MPI_Comm comm,size_t n,char str[])
{
  size_t         i;
  char           c;
  PetscMPIInt    rank,nm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    c = (char) getchar();
    i = 0;
    while (c != '\n' && i < n-1) {
      str[i++] = c;
      c = (char)getchar();
    }
    str[i] = 0;
  }
  ierr = PetscMPIIntCast(n,&nm);CHKERRQ(ierr);
  ierr = MPI_Bcast(str,nm,MPI_CHAR,0,comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/*
    This is needed because certain strings may be freed by SAWs, hence we cannot use PetscStrallocpy()
*/
static PetscErrorCode  PetscStrdup(const char s[],char *t[])
{
  PetscErrorCode ierr;
  size_t         len;
  char           *tmp = NULL;

  PetscFunctionBegin;
  if (s) {
    ierr = PetscStrlen(s,&len);CHKERRQ(ierr);
    tmp = (char*) malloc((len+1)*sizeof(char));
    if (!tmp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"No memory to duplicate string");
    ierr = PetscStrcpy(tmp,s);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
  PetscOptionItem next = PetscOptionsObject->next;
  char            str[512];
  PetscBool       bid;
  PetscReal       ir,*valr;
  PetscInt        *vald;
  size_t          i;

  PetscFunctionBegin;
  ierr = (*PetscPrintf)(PETSC_COMM_WORLD,"%s --------------------\n",PetscOptionsObject->title);CHKERRQ(ierr);
  while (next) {
    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1);CHKERRQ(ierr);
      vald = (PetscInt*) next->data;
      for (i=0; i<next->arraylength; i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT,vald[i]);CHKERRQ(ierr);
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
              if (i == len-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry %s",n,value);
              value[i] = 0;
              ierr     = PetscOptionsStringToInt(value,&start);CHKERRQ(ierr);
              ierr     = PetscOptionsStringToInt(value+i+1,&end);CHKERRQ(ierr);
              if (end <= start) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
              if (n + end - start - 1 >= nmax) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %" PetscInt_FMT "-th array entry, not enough space in left in array (%" PetscInt_FMT ") to contain entire range from %" PetscInt_FMT " to %" PetscInt_FMT,n,nmax-n,start,end);
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
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1);CHKERRQ(ierr);
      valr = (PetscReal*) next->data;
      for (i=0; i<next->arraylength; i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"%g",(double)valr[i]);CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%d>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(int*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
#if defined(PETSC_SIZEOF_LONG_LONG)
        long long lid;
        sscanf(str,"%lld",&lid);
        if (lid > PETSC_MAX_INT || lid < PETSC_MIN_INT) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %lld",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,lid);
#else
        long  lid;
        sscanf(str,"%ld",&lid);
        if (lid > PETSC_MAX_INT || lid < PETSC_MIN_INT) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Argument: -%s%s %ld",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,lid);
#endif

        next->set = PETSC_TRUE;
        *((PetscInt*)next->data) = (PetscInt)lid;
      }
      break;
    case OPTION_REAL:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%g>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(double*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,*(PetscBool*)next->data ? "true": "false",next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        ierr = PetscOptionsStringToBool(str,&bid);CHKERRQ(ierr);
        next->set = PETSC_TRUE;
        *((PetscBool*)next->data) = bid;
      }
      break;
    case OPTION_STRING:
      ierr = PetscPrintf(PETSC_COMM_WORLD,"-%s%s <%s>: %s (%s) ",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",next->option+1,(char*)next->data,next->text,next->man);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        next->set = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        ierr = PetscStrdup(str,(char**)&next->data);CHKERRQ(ierr);
      }
      break;
    case OPTION_FLIST:
      ierr = PetscFunctionListPrintTypes(PETSC_COMM_WORLD,stdout,PetscOptionsObject->prefix,next->option,next->text,next->man,next->flist,(char*)next->data,(char*)next->data);CHKERRQ(ierr);
      ierr = PetscScanString(PETSC_COMM_WORLD,512,str);CHKERRQ(ierr);
      if (str[0]) {
        PetscOptionsObject->changedmethod = PETSC_TRUE;
        next->set = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        ierr = PetscStrdup(str,(char**)&next->data);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
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

  ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s","_title");CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&PetscOptionsObject->title,1,SAWs_READ,SAWs_STRING));
  ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s","prefix");CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Register,(dir,&PetscOptionsObject->pprefix,1,SAWs_READ,SAWs_STRING));
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Options/ChangedMethod",&changedmethod,1,SAWs_WRITE,SAWs_BOOLEAN));
  PetscStackCallSAWs(SAWs_Register,("/PETSc/Options/StopAsking",&stopasking,1,SAWs_WRITE,SAWs_BOOLEAN));

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
      ierr = PetscMalloc1((ntext+1),(char***)&next->edata);CHKERRQ(ierr);
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
  PetscStackCallSAWs(SAWs_Push_Header,("index.html",OptionsHeader));
  PetscStackCallSAWs(SAWs_Push_Body,("index.html",2,OptionsBodyBottom));
  ierr = PetscSAWsBlock();CHKERRQ(ierr);
  PetscStackCallSAWs(SAWs_Pop_Header,("index.html"));
  PetscStackCallSAWs(SAWs_Pop_Body,("index.html",2));

  /* determine if any values have been set in GUI */
  next = PetscOptionsObject->next;
  while (next) {
    ierr = PetscSNPrintf(dir,1024,"/PETSc/Options/%s",next->option);CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Selected,(dir,(int*)&next->set));
    next = next->next;
  }

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (changedmethod) PetscOptionsObject->count = -2;

  if (stopasking) {
    PetscOptionsPublish      = PETSC_FALSE;
    PetscOptionsObject->count = 0;//do not ask for same thing again
  }

  PetscStackCallSAWs(SAWs_Delete,("/PETSc/Options"));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode PetscOptionsEnd_Private(PetscOptionItems *PetscOptionsObject)
{
  PetscErrorCode  ierr;
  PetscOptionItem last;
  char            option[256],value[1024],tmp[32];
  size_t          j;

  PetscFunctionBegin;
  if (PetscOptionsObject->next) {
    if (!PetscOptionsObject->count) {
#if defined(PETSC_HAVE_SAWS)
      ierr = PetscOptionsSAWsInput(PetscOptionsObject);CHKERRQ(ierr);
#else
      ierr = PetscOptionsGetFromTextInput(PetscOptionsObject);CHKERRQ(ierr);
#endif
    }
  }

  ierr = PetscFree(PetscOptionsObject->title);CHKERRQ(ierr);

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (PetscOptionsObject->changedmethod) PetscOptionsObject->count = -2;
  /* reset alreadyprinted flag */
  PetscOptionsObject->alreadyprinted = PETSC_FALSE;
  if (PetscOptionsObject->object) PetscOptionsObject->object->optionsprinted = PETSC_TRUE;
  PetscOptionsObject->object = NULL;

  while (PetscOptionsObject->next) {
    if (PetscOptionsObject->next->set) {
      if (PetscOptionsObject->prefix) {
        ierr = PetscStrcpy(option,"-");CHKERRQ(ierr);
        ierr = PetscStrcat(option,PetscOptionsObject->prefix);CHKERRQ(ierr);
        ierr = PetscStrcat(option,PetscOptionsObject->next->option+1);CHKERRQ(ierr);
      } else {
        ierr = PetscStrcpy(option,PetscOptionsObject->next->option);CHKERRQ(ierr);
      }

      switch (PetscOptionsObject->next->type) {
      case OPTION_HEAD:
        break;
      case OPTION_INT_ARRAY:
        sprintf(value,"%d",(int)((PetscInt*)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscInt*)PetscOptionsObject->next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
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
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_SCALAR_ARRAY:
        sprintf(value,"%g+%gi",(double)PetscRealPart(((PetscScalar*)PetscOptionsObject->next->data)[0]),(double)PetscImaginaryPart(((PetscScalar*)PetscOptionsObject->next->data)[0]));
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%g+%gi",(double)PetscRealPart(((PetscScalar*)PetscOptionsObject->next->data)[j]),(double)PetscImaginaryPart(((PetscScalar*)PetscOptionsObject->next->data)[j]));
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_BOOL:
        sprintf(value,"%d",*(int*)PetscOptionsObject->next->data);
        break;
      case OPTION_BOOL_ARRAY:
        sprintf(value,"%d",(int)((PetscBool*)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%d",(int)((PetscBool*)PetscOptionsObject->next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      case OPTION_FLIST:
        ierr = PetscStrcpy(value,(char*)PetscOptionsObject->next->data);CHKERRQ(ierr);
        break;
      case OPTION_ELIST:
        ierr = PetscStrcpy(value,(char*)PetscOptionsObject->next->data);CHKERRQ(ierr);
        break;
      case OPTION_STRING:
        ierr = PetscStrcpy(value,(char*)PetscOptionsObject->next->data);CHKERRQ(ierr);
        break;
      case OPTION_STRING_ARRAY:
        sprintf(value,"%s",((char**)PetscOptionsObject->next->data)[0]);
        for (j=1; j<PetscOptionsObject->next->arraylength; j++) {
          sprintf(tmp,"%s",((char**)PetscOptionsObject->next->data)[j]);
          ierr = PetscStrcat(value,",");CHKERRQ(ierr);
          ierr = PetscStrcat(value,tmp);CHKERRQ(ierr);
        }
        break;
      }
      ierr = PetscOptionsSetValue(PetscOptionsObject->options,option,value);CHKERRQ(ierr);
    }
    if (PetscOptionsObject->next->type == OPTION_ELIST) {
      ierr = PetscStrNArrayDestroy(PetscOptionsObject->next->nlist,(char ***)&PetscOptionsObject->next->list);CHKERRQ(ierr);
    }
    ierr   = PetscFree(PetscOptionsObject->next->text);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject->next->option);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject->next->man);CHKERRQ(ierr);
    ierr   = PetscFree(PetscOptionsObject->next->edata);CHKERRQ(ierr);

    if ((PetscOptionsObject->next->type == OPTION_STRING) || (PetscOptionsObject->next->type == OPTION_FLIST) || (PetscOptionsObject->next->type == OPTION_ELIST)) {
      free(PetscOptionsObject->next->data);
    } else {
      ierr   = PetscFree(PetscOptionsObject->next->data);CHKERRQ(ierr);
    }

    last                    = PetscOptionsObject->next;
    PetscOptionsObject->next = PetscOptionsObject->next->next;
    ierr                    = PetscFree(last);CHKERRQ(ierr);
  }
  ierr = PetscFree(PetscOptionsObject->prefix);CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsEnum_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char *const *list,PetscEnum currentvalue,PetscEnum *value,PetscBool  *set)
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
  ierr   = PetscOptionsEList_Private(PetscOptionsObject,opt,text,man,list,ntext,list[currentvalue],&tval,&tflg);CHKERRQ(ierr);
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
-  n - maximum number of values

   Output Parameters:
+  value - location to copy values
.  n - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The array must be passed as a comma separated list.

   There must be no intervening spaces between the values.

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsRealArray()
M*/

PetscErrorCode  PetscOptionsEnumArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char *const *list,PetscEnum value[],PetscInt *n,PetscBool  *set)
{
  PetscInt        i,nlist = 0;
  PetscOptionItem amsopt;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  while (list[nlist++]) if (nlist > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  if (nlist < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  nlist -= 3; /* drop enum name, prefix, and null termination */
  if (0 && !PetscOptionsObject->count) { /* XXX Requires additional support */
    PetscEnum *vals;
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT_ARRAY/*XXX OPTION_ENUM_ARRAY*/,&amsopt);CHKERRQ(ierr);
    ierr = PetscStrNArrayallocpy(nlist,list,(char***)&amsopt->list);CHKERRQ(ierr);
    amsopt->nlist = nlist;
    ierr = PetscMalloc1(*n,(PetscEnum**)&amsopt->data);CHKERRQ(ierr);
    amsopt->arraylength = *n;
    vals = (PetscEnum*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
  }
  ierr = PetscOptionsGetEnumArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,list,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%s",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,list[value[0]]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,",%s",list[value[i]]);CHKERRQ(ierr);}
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (choose from)",text);CHKERRQ(ierr);
    for (i=0; i<nlist; i++) {ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm," %s",list[i]);CHKERRQ(ierr);}
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm," (%s)\n",ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsInt(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(), PetscOptionsRangeInt()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
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

.seealso: PetscOptionsInt(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(), PetscOptionsBoundedInt()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
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

.seealso: PetscOptionsBoundedInt(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(), PetscOptionsRangeInt()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsInt_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscInt currentvalue,PetscInt *value,PetscBool  *set,PetscInt lb,PetscInt ub)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;
  PetscBool       wasset;

  PetscFunctionBegin;
  if (currentvalue < lb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Current value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT,currentvalue,lb);
  if (currentvalue > ub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Current value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT,currentvalue,ub);
     if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt),&amsopt->data);CHKERRQ(ierr);
    *(PetscInt*)amsopt->data = currentvalue;

    ierr = PetscOptionsGetInt(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,&currentvalue,&wasset);CHKERRQ(ierr);
    if (wasset) {
      *(PetscInt*)amsopt->data = currentvalue;
    }
  }
  ierr = PetscOptionsGetInt(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,&wasset);CHKERRQ(ierr);
  if (wasset && *value < lb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Newly set value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT,*value,lb);
  if (wasset && *value > ub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Newly set value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT,*value,ub);
  if (set) *set = wasset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %" PetscInt_FMT " : formerly %" PetscInt_FMT ">: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,wasset && value ? *value : currentvalue,currentvalue,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsString_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],const char currentvalue[],char value[],size_t len,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING,&amsopt);CHKERRQ(ierr);
    /* must use system malloc since SAWs may free this */
    ierr = PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,len,&lset);CHKERRQ(ierr);
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %s : formerly %s>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,lset && value ? value : currentvalue,currentvalue,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsReal_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscReal currentvalue,PetscReal *value,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_REAL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);

    *(PetscReal*)amsopt->data = currentvalue;
  }
  ierr = PetscOptionsGetReal(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,&lset);CHKERRQ(ierr);
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g : %g>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,lset && value ? (double)*value : (double) currentvalue,(double)currentvalue,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsScalar_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscScalar currentvalue,PetscScalar *value,PetscBool  *set)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsReal(opt,text,man,currentvalue,value,set);CHKERRQ(ierr);
#else
  ierr = PetscOptionsGetScalar(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,set);CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsName_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  ierr = PetscOptionsHasName(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsEnum()
M*/

PetscErrorCode  PetscOptionsFList_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char ltext[],const char man[],PetscFunctionList list,const char currentvalue[],char value[],size_t len,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,ltext,man,OPTION_FLIST,&amsopt);CHKERRQ(ierr);
    /* must use system malloc since SAWs may free this */
    ierr = PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data);CHKERRQ(ierr);
    amsopt->flist = list;
  }
  ierr = PetscOptionsGetString(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,len,&lset);CHKERRQ(ierr);
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = PetscFunctionListPrintTypes(PetscOptionsObject->comm,stdout,PetscOptionsObject->prefix,opt,ltext,man,list,currentvalue,lset && value ? value : currentvalue);CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEnum()
M*/

PetscErrorCode  PetscOptionsEList_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char ltext[],const char man[],const char *const *list,PetscInt ntext,const char currentvalue[],PetscInt *value,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscOptionItem amsopt;
  PetscBool       lset;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,ltext,man,OPTION_ELIST,&amsopt);CHKERRQ(ierr);
    /* must use system malloc since SAWs may free this */
    ierr = PetscStrdup(currentvalue ? currentvalue : "",(char**)&amsopt->data);CHKERRQ(ierr);
    ierr = PetscStrNArrayallocpy(ntext,list,(char***)&amsopt->list);CHKERRQ(ierr);
    amsopt->nlist = ntext;
  }
  ierr = PetscOptionsGetEList(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,list,ntext,value,&lset);CHKERRQ(ierr);
  if (set) *set = lset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <now %s : formerly %s> %s (choose one of)",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,lset && value ? list[*value] : currentvalue,currentvalue,ltext);CHKERRQ(ierr);
    for (i=0; i<ntext; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm," %s",list[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm," (%s)\n",ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsBoolGroupBegin_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  Pick at most one of -------------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsBoolGroup_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsBoolGroupEnd_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool  *flg)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"    -%s%s: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetBool(),
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsBool_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool currentvalue,PetscBool  *flg,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscBool       iset;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscBool),&amsopt->data);CHKERRQ(ierr);

    *(PetscBool*)amsopt->data = currentvalue;
  }
  ierr = PetscOptionsGetBool(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,flg,&iset);CHKERRQ(ierr);
  if (set) *set = iset;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    const char *v = PetscBools[currentvalue], *vn = PetscBools[iset && flg ? *flg : currentvalue];
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s: <%s : %s> %s (%s)\n",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,v,vn,text,ManSection(man));CHKERRQ(ierr);
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
-  nmax - maximum number of values

   Output Parameters:
+  value - location to copy values
.  nmax - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode PetscOptionsRealArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscReal value[],PetscInt *n,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscReal *vals;

    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_REAL_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*n)*sizeof(PetscReal),&amsopt->data);CHKERRQ(ierr);
    vals = (PetscReal*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetRealArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,(double)value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,",%g",(double)value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
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
-  nmax - maximum number of values

   Output Parameters:
+  value - location to copy values
.  nmax - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode PetscOptionsScalarArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscScalar value[],PetscInt *n,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscScalar *vals;

    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_SCALAR_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc((*n)*sizeof(PetscScalar),&amsopt->data);CHKERRQ(ierr);
    vals = (PetscScalar*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetScalarArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%g+%gi",PetscOptionsObject->prefix?PetscOptionsObject->prefix:"",opt+1,(double)PetscRealPart(value[0]),(double)PetscImaginaryPart(value[0]));CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,",%g+%gi",(double)PetscRealPart(value[i]),(double)PetscImaginaryPart(value[i]));CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsRealArray()
M*/

PetscErrorCode  PetscOptionsIntArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscInt value[],PetscInt *n,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt        i;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscInt *vals;

    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_INT_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc1(*n,(PetscInt**)&amsopt->data);CHKERRQ(ierr);
    vals = (PetscInt*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetIntArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%" PetscInt_FMT,PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,",%" PetscInt_FMT,value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsStringArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],char *value[],PetscInt *nmax,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc1(*nmax,(char**)&amsopt->data);CHKERRQ(ierr);

    amsopt->arraylength = *nmax;
  }
  ierr = PetscOptionsGetStringArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,nmax,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <string1,string2,...>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,text,ManSection(man));CHKERRQ(ierr);
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
-  nmax - maximum number of values

   Output Parameters:
+  value - location to copy values
.  nmax - actual number of values found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of doubles

   Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsBoolArray_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscBool value[],PetscInt *n,PetscBool *set)
{
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscOptionItem  amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    PetscBool *vals;

    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_BOOL_ARRAY,&amsopt);CHKERRQ(ierr);
    ierr = PetscMalloc1(*n,(PetscBool**)&amsopt->data);CHKERRQ(ierr);
    vals = (PetscBool*)amsopt->data;
    for (i=0; i<*n; i++) vals[i] = value[i];
    amsopt->arraylength = *n;
  }
  ierr = PetscOptionsGetBoolArray(PetscOptionsObject->options,PetscOptionsObject->prefix,opt,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%d",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,value[0]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {
      ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,",%d",value[i]);CHKERRQ(ierr);
    }
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,">: %s (%s)\n",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   PetscOptionsViewer - Gets a viewer appropriate for the type indicated by the user

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode PetscOptionsViewer(const char opt[],const char text[],const char man[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool  *set)

   Input Parameters:
+  opt - option name
.  text - short string that describes the option
-  man - manual page with additional information on option

   Output Parameters:
+  viewer - the viewer
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

   See PetscOptionsGetViewer() for the format of the supplied viewer and its options

.seealso: PetscOptionsGetViewer(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
M*/

PetscErrorCode  PetscOptionsViewer_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool  *set)
{
  PetscErrorCode  ierr;
  PetscOptionItem amsopt;

  PetscFunctionBegin;
  if (!PetscOptionsObject->count) {
    ierr = PetscOptionItemCreate_Private(PetscOptionsObject,opt,text,man,OPTION_STRING,&amsopt);CHKERRQ(ierr);
    /* must use system malloc since SAWs may free this */
    ierr = PetscStrdup("",(char**)&amsopt->data);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(PetscOptionsObject->comm,PetscOptionsObject->options,PetscOptionsObject->prefix,opt,viewer,format,set);CHKERRQ(ierr);
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  -%s%s <%s>: %s (%s)\n",PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "",opt+1,"",text,ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscOptionsHead - Puts a heading before listing any more published options. Used, for example,
            in KSPSetFromOptions_GMRES().

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameter:
.   head - the heading text

   Level: intermediate

   Notes:
    Must be between a PetscOptionsBegin() and a PetscOptionsEnd(), and PetscOptionsObject created in PetscOptionsBegin() should be the first argument

          Can be followed by a call to PetscOptionsTail() in the same function.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsHead(PetscOptionItems *PetscOptionsObject,const char head[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject->comm,"  %s\n",head);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

