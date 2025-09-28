/*
   Implements the higher-level options database querying methods. These are self-documenting and can attach at runtime to
   GUI code to display the options and get values from the users.

*/

#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/
#include <petscviewer.h>

static const char *ManSection(const char *str)
{
  return str ? str : "None";
}

static const char *Prefix(const char *str)
{
  return str ? str : "";
}

static int ShouldPrintHelp(const PetscOptionItems opts)
{
  return opts->printhelp && opts->count == 1 && !opts->alreadyprinted;
}

/*
    Keep a linked list of options that have been posted and we are waiting for
   user selection. See the manual page for PetscOptionsBegin()

    Eventually we'll attach this beast to a MPI_Comm
*/

/*
    Handles setting up the data structure in a call to PetscOptionsBegin()
*/
PetscErrorCode PetscOptionsBegin_Private(PetscOptionItems PetscOptionsObject, MPI_Comm comm, const char prefix[], const char title[], const char mansec[])
{
  PetscFunctionBegin;
  if (prefix) PetscAssertPointer(prefix, 3);
  PetscAssertPointer(title, 4);
  if (mansec) PetscAssertPointer(mansec, 5);
  if (!PetscOptionsObject->alreadyprinted) {
    if (!PetscOptionsHelpPrintedSingleton) PetscCall(PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton));
    PetscCall(PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton, prefix, title, &PetscOptionsObject->alreadyprinted));
  }
  PetscOptionsObject->next          = NULL;
  PetscOptionsObject->comm          = comm;
  PetscOptionsObject->changedmethod = PETSC_FALSE;

  PetscCall(PetscStrallocpy(prefix, &PetscOptionsObject->prefix));
  PetscCall(PetscStrallocpy(title, &PetscOptionsObject->title));

  PetscCall(PetscOptionsHasHelp(PetscOptionsObject->options, &PetscOptionsObject->printhelp));
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(comm, "----------------------------------------\n%s:\n", title));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Handles setting up the data structure in a call to PetscObjectOptionsBegin()
*/
PetscErrorCode PetscObjectOptionsBegin_Private(PetscObject obj, PetscOptionItems PetscOptionsObject)
{
  char      title[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscAssertPointer(PetscOptionsObject, 2);
  PetscValidHeader(obj, 1);
  PetscOptionsObject->object         = obj;
  PetscOptionsObject->alreadyprinted = obj->optionsprinted;

  PetscCall(PetscStrcmp(obj->description, obj->class_name, &flg));
  if (flg) PetscCall(PetscSNPrintf(title, sizeof(title), "%s options", obj->class_name));
  else PetscCall(PetscSNPrintf(title, sizeof(title), "%s (%s) options", obj->description, obj->class_name));
  PetscCall(PetscOptionsBegin_Private(PetscOptionsObject, obj->comm, obj->prefix, title, obj->mansec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Handles adding another option to the list of options within this particular PetscOptionsBegin() PetscOptionsEnd()
*/
static PetscErrorCode PetscOptionItemCreate_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscOptionType t, PetscOptionItem *amsopt)
{
  PetscBool valid;

  PetscFunctionBegin;
  PetscCall(PetscOptionsValidKey(opt, &valid));
  PetscCheck(valid, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "The option '%s' is not a valid key", opt);

  PetscCall(PetscNew(amsopt));
  (*amsopt)->next = NULL;
  (*amsopt)->set  = PETSC_FALSE;
  (*amsopt)->type = t;
  (*amsopt)->data = NULL;

  PetscCall(PetscStrallocpy(text, &(*amsopt)->text));
  PetscCall(PetscStrallocpy(opt, &(*amsopt)->option));
  PetscCall(PetscStrallocpy(man, &(*amsopt)->man));

  {
    PetscOptionItem cur = PetscOptionsObject->next;

    while (cur->next) cur = cur->next;
    cur->next = *amsopt;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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

    PetscCall(PetscStrlen(s, &len));
    tmp = (char *)malloc((len + 1) * sizeof(*tmp));
    PetscCheck(tmp, PETSC_COMM_SELF, PETSC_ERR_MEM, "No memory to duplicate string");
    PetscCall(PetscArraycpy(tmp, s, len + 1));
  }
  *t = tmp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>

static int count = 0;

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
+    All processes must traverse through the exact same set of option queries due to the call to PetscScanString()
.    Internal strings have arbitrary length and string copies are not checked that they fit into string space
-    Only works for PetscInt == int, PetscReal == double etc

*/
static PetscErrorCode PetscOptionsSAWsInput(PetscOptionItems PetscOptionsObject)
{
  PetscOptionItem next     = PetscOptionsObject->next;
  static int      mancount = 0;
  char            options[16];
  PetscBool       changedmethod = PETSC_FALSE;
  PetscBool       stopasking    = PETSC_FALSE;
  char            manname[16], textname[16];
  char            dir[1024];

  PetscFunctionBegin;
  /* the next line is a bug, this will only work if all processors are here, the comm passed in is ignored!!! */
  PetscCall(PetscSNPrintf(options, PETSC_STATIC_ARRAY_LENGTH(options), "Options_%d", count++));

  PetscOptionsObject->pprefix = PetscOptionsObject->prefix; /* SAWs will change this, so cannot pass prefix directly */

  PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", "_title"));
  PetscCallSAWs(SAWs_Register, (dir, &PetscOptionsObject->title, 1, SAWs_READ, SAWs_STRING));
  PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", "prefix"));
  PetscCallSAWs(SAWs_Register, (dir, &PetscOptionsObject->pprefix, 1, SAWs_READ, SAWs_STRING));
  PetscCallSAWs(SAWs_Register, ("/PETSc/Options/ChangedMethod", &changedmethod, 1, SAWs_WRITE, SAWs_BOOLEAN));
  PetscCallSAWs(SAWs_Register, ("/PETSc/Options/StopAsking", &stopasking, 1, SAWs_WRITE, SAWs_BOOLEAN));

  while (next) {
    PetscCall(PetscSNPrintf(manname, PETSC_STATIC_ARRAY_LENGTH(manname), "_man_%d", mancount));
    PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", manname));
    PetscCallSAWs(SAWs_Register, (dir, &next->man, 1, SAWs_READ, SAWs_STRING));
    PetscCall(PetscSNPrintf(textname, PETSC_STATIC_ARRAY_LENGTH(textname), "_text_%d", mancount++));
    PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", textname));
    PetscCallSAWs(SAWs_Register, (dir, &next->text, 1, SAWs_READ, SAWs_STRING));

    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, next->arraylength, SAWs_WRITE, SAWs_INT));
      break;
    case OPTION_REAL_ARRAY:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, next->arraylength, SAWs_WRITE, SAWs_DOUBLE));
      break;
    case OPTION_INT:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, 1, SAWs_WRITE, SAWs_INT));
      break;
    case OPTION_REAL:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, 1, SAWs_WRITE, SAWs_DOUBLE));
      break;
    case OPTION_BOOL:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, 1, SAWs_WRITE, SAWs_BOOLEAN));
      break;
    case OPTION_BOOL_ARRAY:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, next->arraylength, SAWs_WRITE, SAWs_BOOLEAN));
      break;
    case OPTION_STRING:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, &next->data, 1, SAWs_WRITE, SAWs_STRING));
      break;
    case OPTION_STRING_ARRAY:
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, next->data, next->arraylength, SAWs_WRITE, SAWs_STRING));
      break;
    case OPTION_FLIST: {
      PetscInt ntext;
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, &next->data, 1, SAWs_WRITE, SAWs_STRING));
      PetscCall(PetscFunctionListGet(next->flist, (const char ***)&next->edata, &ntext));
      PetscCallSAWs(SAWs_Set_Legal_Variable_Values, (dir, ntext, next->edata));
    } break;
    case OPTION_ELIST: {
      PetscInt ntext = next->nlist;
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
      PetscCallSAWs(SAWs_Register, (dir, &next->data, 1, SAWs_WRITE, SAWs_STRING));
      PetscCall(PetscMalloc1(ntext + 1, (char ***)&next->edata));
      PetscCall(PetscMemcpy(next->edata, next->list, ntext * sizeof(char *)));
      PetscCallSAWs(SAWs_Set_Legal_Variable_Values, (dir, ntext, next->edata));
    } break;
    default:
      break;
    }
    next = next->next;
  }

  /* wait until accessor has unlocked the memory */
  PetscCallSAWs(SAWs_Push_Header, ("index.html", OptionsHeader));
  PetscCallSAWs(SAWs_Push_Body, ("index.html", 2, OptionsBodyBottom));
  PetscCall(PetscSAWsBlock());
  PetscCallSAWs(SAWs_Pop_Header, ("index.html"));
  PetscCallSAWs(SAWs_Pop_Body, ("index.html", 2));

  /* determine if any values have been set in GUI */
  next = PetscOptionsObject->next;
  while (next) {
    PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Options/%s", next->option));
    PetscCallSAWs(SAWs_Selected, (dir, (int *)&next->set));
    next = next->next;
  }

  /* reset counter to -2; this updates the screen with the new options for the selected method */
  if (changedmethod) PetscOptionsObject->count = -2;

  if (stopasking) {
    PetscOptionsPublish       = PETSC_FALSE;
    PetscOptionsObject->count = 0; //do not ask for same thing again
  }

  PetscCallSAWs(SAWs_Delete, ("/PETSc/Options"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
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
  PetscMPIInt rank, nm;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    char   c = (char)getchar();
    size_t i = 0;

    while (c != '\n' && i < n - 1) {
      str[i++] = c;
      c        = (char)getchar();
    }
    str[i] = '\0';
  }
  PetscCall(PetscMPIIntCast(n, &nm));
  PetscCallMPI(MPI_Bcast(str, nm, MPI_CHAR, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
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
static PetscErrorCode PetscOptionsGetFromTextInput(PetscOptionItems PetscOptionsObject)
{
  PetscOptionItem next = PetscOptionsObject->next;
  char            str[512];
  PetscBool       bid;
  PetscReal       ir, *valr;
  PetscInt       *vald;

  PetscFunctionBegin;
  PetscCall((*PetscPrintf)(PETSC_COMM_WORLD, "%s --------------------\n", PetscOptionsObject->title));
  while (next) {
    switch (next->type) {
    case OPTION_HEAD:
      break;
    case OPTION_INT_ARRAY:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1));
      vald = (PetscInt *)next->data;
      for (PetscInt i = 0; i < next->arraylength; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT, vald[i]));
        if (i < next->arraylength - 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, ","));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">: %s (%s) ", next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
        PetscToken  token;
        PetscInt    n = 0, nmax = next->arraylength, *dvalue = (PetscInt *)next->data, start, end;
        size_t      len;
        const char *value;
        PetscBool   foundrange;

        next->set = PETSC_TRUE;
        value     = str;
        PetscCall(PetscTokenCreate(value, ',', &token));
        PetscCall(PetscTokenFind(token, &value));
        while (n < nmax) {
          char    *ivalue;
          PetscInt i;

          if (!value) break;
          PetscCall(PetscStrallocpy(value, &ivalue));

          /* look for form  d-D where d and D are integers */
          foundrange = PETSC_FALSE;
          PetscCall(PetscStrlen(ivalue, &len));
          if (ivalue[0] == '-') i = 2;
          else i = 1;
          for (; i < (PetscInt)len; i++) {
            if (ivalue[i] == '-') {
              PetscCheck(i != (PetscInt)(len - 1), PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry %s", n, ivalue);
              ivalue[i] = 0;
              PetscCall(PetscOptionsStringToInt(ivalue, &start));
              PetscCall(PetscOptionsStringToInt(ivalue + i + 1, &end));
              PetscCheck(end > start, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry, %s-%s cannot have decreasing list", n, ivalue, ivalue + i + 1);
              PetscCheck(n + end - start - 1 < nmax, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry, not enough space in left in array (%" PetscInt_FMT ") to contain entire range from %" PetscInt_FMT " to %" PetscInt_FMT, n, nmax - n, start, end);
              for (; start < end; start++) {
                *dvalue = start;
                dvalue++;
                n++;
              }
              foundrange = PETSC_TRUE;
              break;
            }
          }
          if (!foundrange) {
            PetscCall(PetscOptionsStringToInt(ivalue, dvalue));
            dvalue++;
            n++;
          }
          PetscCall(PetscFree(ivalue));
          PetscCall(PetscTokenFind(token, &value));
        }
        PetscCall(PetscTokenDestroy(&token));
      }
      break;
    case OPTION_REAL_ARRAY:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1));
      valr = (PetscReal *)next->data;
      for (PetscInt i = 0; i < next->arraylength; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g", (double)valr[i]));
        if (i < next->arraylength - 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, ","));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">: %s (%s) ", next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
        PetscToken  token;
        PetscInt    n = 0, nmax = next->arraylength;
        PetscReal  *dvalue = (PetscReal *)next->data;
        const char *value;

        next->set = PETSC_TRUE;
        value     = str;
        PetscCall(PetscTokenCreate(value, ',', &token));
        PetscCall(PetscTokenFind(token, &value));
        while (n < nmax) {
          if (!value) break;
          PetscCall(PetscOptionsStringToReal(value, dvalue));
          dvalue++;
          n++;
          PetscCall(PetscTokenFind(token, &value));
        }
        PetscCall(PetscTokenDestroy(&token));
      }
      break;
    case OPTION_INT:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <%d>: %s (%s) ", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, *(int *)next->data, next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
  #if defined(PETSC_SIZEOF_LONG_LONG)
        long long lid;
        sscanf(str, "%lld", &lid);
        PetscCheck(lid <= PETSC_INT_MAX && lid >= PETSC_INT_MIN, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Argument: -%s%s %lld", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, lid);
  #else
        long lid;
        sscanf(str, "%ld", &lid);
        PetscCheck(lid <= PETSC_INT_MAX && lid >= PETSC_INT_MIN, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Argument: -%s%s %ld", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, lid);
  #endif

        next->set                 = PETSC_TRUE;
        *((PetscInt *)next->data) = (PetscInt)lid;
      }
      break;
    case OPTION_REAL:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <%g>: %s (%s) ", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, *(double *)next->data, next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
  #if defined(PETSC_USE_REAL_SINGLE)
        sscanf(str, "%e", &ir);
  #elif defined(PETSC_USE_REAL___FP16)
        float irtemp;
        sscanf(str, "%e", &irtemp);
        ir = irtemp;
  #elif defined(PETSC_USE_REAL_DOUBLE)
        sscanf(str, "%le", &ir);
  #elif defined(PETSC_USE_REAL___FLOAT128)
        ir = strtoflt128(str, 0);
  #else
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unknown scalar type");
  #endif
        next->set                  = PETSC_TRUE;
        *((PetscReal *)next->data) = ir;
      }
      break;
    case OPTION_BOOL:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <%s>: %s (%s) ", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, *(PetscBool *)next->data ? "true" : "false", next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
        PetscCall(PetscOptionsStringToBool(str, &bid));
        next->set                  = PETSC_TRUE;
        *((PetscBool *)next->data) = bid;
      }
      break;
    case OPTION_STRING:
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%s%s: <%s>: %s (%s) ", PetscOptionsObject->prefix ? PetscOptionsObject->prefix : "", next->option + 1, (char *)next->data, next->text, next->man));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
        next->set = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        PetscCall(PetscStrdup(str, (char **)&next->data));
      }
      break;
    case OPTION_FLIST:
      PetscCall(PetscFunctionListPrintTypes(PETSC_COMM_WORLD, stdout, PetscOptionsObject->prefix, next->option, next->text, next->man, next->flist, (char *)next->data, (char *)next->data));
      PetscCall(PetscScanString(PETSC_COMM_WORLD, 512, str));
      if (str[0]) {
        PetscOptionsObject->changedmethod = PETSC_TRUE;
        next->set                         = PETSC_TRUE;
        /* must use system malloc since SAWs may free this */
        PetscCall(PetscStrdup(str, (char **)&next->data));
      }
      break;
    default:
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode PetscOptionsEnd_Private(PetscOptionItems PetscOptionsObject)
{
  PetscOptionItem next, last;

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

  while ((next = PetscOptionsObject->next)) {
    const PetscOptionType type        = next->type;
    const size_t          arraylength = next->arraylength;
    void                 *data        = next->data;

    if (next->set) {
      char option[256], value[1024], tmp[32];

      if (PetscOptionsObject->prefix) {
        PetscCall(PetscStrncpy(option, "-", sizeof(option)));
        PetscCall(PetscStrlcat(option, PetscOptionsObject->prefix, sizeof(option)));
        PetscCall(PetscStrlcat(option, next->option + 1, sizeof(option)));
      } else {
        PetscCall(PetscStrncpy(option, next->option, sizeof(option)));
      }

      switch (type) {
      case OPTION_HEAD:
        break;
      case OPTION_INT_ARRAY:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%" PetscInt_FMT, ((PetscInt *)data)[0]));
        for (size_t j = 1; j < arraylength; ++j) {
          PetscCall(PetscSNPrintf(tmp, PETSC_STATIC_ARRAY_LENGTH(tmp), "%" PetscInt_FMT, ((PetscInt *)data)[j]));
          PetscCall(PetscStrlcat(value, ",", sizeof(value)));
          PetscCall(PetscStrlcat(value, tmp, sizeof(value)));
        }
        break;
      case OPTION_INT:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%" PetscInt_FMT, *(PetscInt *)data));
        break;
      case OPTION_REAL:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%g", (double)*(PetscReal *)data));
        break;
      case OPTION_REAL_ARRAY:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%g", (double)((PetscReal *)data)[0]));
        for (size_t j = 1; j < arraylength; ++j) {
          PetscCall(PetscSNPrintf(tmp, PETSC_STATIC_ARRAY_LENGTH(tmp), "%g", (double)((PetscReal *)data)[j]));
          PetscCall(PetscStrlcat(value, ",", sizeof(value)));
          PetscCall(PetscStrlcat(value, tmp, sizeof(value)));
        }
        break;
      case OPTION_SCALAR_ARRAY:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%g+%gi", (double)PetscRealPart(((PetscScalar *)data)[0]), (double)PetscImaginaryPart(((PetscScalar *)data)[0])));
        for (size_t j = 1; j < arraylength; ++j) {
          PetscCall(PetscSNPrintf(tmp, PETSC_STATIC_ARRAY_LENGTH(tmp), "%g+%gi", (double)PetscRealPart(((PetscScalar *)data)[j]), (double)PetscImaginaryPart(((PetscScalar *)data)[j])));
          PetscCall(PetscStrlcat(value, ",", sizeof(value)));
          PetscCall(PetscStrlcat(value, tmp, sizeof(value)));
        }
        break;
      case OPTION_BOOL:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%d", *(int *)data));
        break;
      case OPTION_BOOL_ARRAY:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%d", (int)((PetscBool *)data)[0]));
        for (size_t j = 1; j < arraylength; ++j) {
          PetscCall(PetscSNPrintf(tmp, PETSC_STATIC_ARRAY_LENGTH(tmp), "%d", (int)((PetscBool *)data)[j]));
          PetscCall(PetscStrlcat(value, ",", sizeof(value)));
          PetscCall(PetscStrlcat(value, tmp, sizeof(value)));
        }
        break;
      case OPTION_FLIST: // fall-through
      case OPTION_ELIST: // fall-through
      case OPTION_STRING:
        PetscCall(PetscStrncpy(value, (char *)data, sizeof(value)));
        break;
      case OPTION_STRING_ARRAY:
        PetscCall(PetscSNPrintf(value, PETSC_STATIC_ARRAY_LENGTH(value), "%s", ((char **)data)[0]));
        for (size_t j = 1; j < arraylength; j++) {
          PetscCall(PetscSNPrintf(tmp, PETSC_STATIC_ARRAY_LENGTH(tmp), "%s", ((char **)data)[j]));
          PetscCall(PetscStrlcat(value, ",", sizeof(value)));
          PetscCall(PetscStrlcat(value, tmp, sizeof(value)));
        }
        break;
      }
      PetscCall(PetscOptionsSetValue(PetscOptionsObject->options, option, value));
    }
    if (type == OPTION_ELIST) PetscCall(PetscStrNArrayDestroy(next->nlist, (char ***)&next->list));
    PetscCall(PetscFree(next->text));
    PetscCall(PetscFree(next->option));
    PetscCall(PetscFree(next->man));
    PetscCall(PetscFree(next->edata));

    if (type == OPTION_STRING || type == OPTION_FLIST || type == OPTION_ELIST) {
      free(data);
    } else {
      // use next->data instead of data because PetscFree() sets it to NULL
      PetscCall(PetscFree(next->data));
    }

    last                     = next;
    PetscOptionsObject->next = next->next;
    PetscCall(PetscFree(last));
  }
  PetscCall(PetscFree(PetscOptionsObject->prefix));
  PetscOptionsObject->next = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetListLength(const char *const *list, PetscInt *len)
{
  PetscInt retlen = 0;

  PetscFunctionBegin;
  PetscAssertPointer(len, 2);
  while (list[retlen]) {
    PetscAssertPointer(list[retlen], 1);
    PetscCheck(++retlen < 50, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "List argument appears to be wrong or have more than 50 entries");
  }
  PetscCheck(retlen > 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "List argument must have at least 2 entries: typename and type prefix");
  /* drop item name and prefix*/
  *len = retlen - 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsEnum_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], const char *const *list, PetscEnum currentvalue, PetscEnum *value, PetscBool *set)
{
  PetscInt  ntext = 0;
  PetscInt  tval;
  PetscBool tflg;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(list, 5);
  PetscAssertPointer(value, 7);
  if (set) PetscAssertPointer(set, 8);
  PetscCall(GetListLength(list, &ntext));
  PetscCall(PetscOptionsEList_Private(PetscOptionsObject, opt, text, man, list, ntext, list[(int)currentvalue], &tval, &tflg));
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (tflg) *value = (PetscEnum)tval;
  if (set) *set = tflg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsEnumArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], const char *const *list, PetscEnum value[], PetscInt *n, PetscBool *set)
{
  PetscInt    nlist  = 0;
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(list, 5);
  PetscAssertPointer(value, 6);
  PetscAssertPointer(n, 7);
  PetscCheck(*n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") must be > 0", *n);
  if (set) PetscAssertPointer(set, 8);
  PetscCall(GetListLength(list, &nlist));
  const PetscInt nin = *n;
  PetscCall(PetscOptionsGetEnumArray(PetscOptionsObject->options, prefix, opt, list, value, n, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) {
    const MPI_Comm comm = PetscOptionsObject->comm;
    const PetscInt nv   = *n;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%s", Prefix(prefix), opt + 1, list[value[0]]));
    for (PetscInt i = 1; i < nv; ++i) PetscCall((*PetscHelpPrintf)(comm, ",%s", list[value[i]]));
    PetscCall((*PetscHelpPrintf)(comm, ">: %s (choose from)", text));
    for (PetscInt i = 0; i < nlist; ++i) PetscCall((*PetscHelpPrintf)(comm, " %s", list[i]));
    PetscCall((*PetscHelpPrintf)(comm, " (%s)\n", ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsInt_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscInt currentvalue, PetscInt *value, PetscBool *set, PetscInt lb, PetscInt ub)
{
  const char        *prefix  = PetscOptionsObject->prefix;
  const PetscOptions options = PetscOptionsObject->options;
  PetscBool          wasset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 6);
  if (set) PetscAssertPointer(set, 7);
  PetscCheck(currentvalue >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT, currentvalue, lb);
  PetscCheck(currentvalue <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT, currentvalue, ub);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_INT, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscInt), &amsopt->data));
    *(PetscInt *)amsopt->data = currentvalue;

    PetscCall(PetscOptionsGetInt(options, prefix, opt, &currentvalue, &wasset));
    if (wasset) *(PetscInt *)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetInt(options, prefix, opt, value, &wasset));
  PetscCheck(!wasset || *value >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %" PetscInt_FMT " less than allowed bound %" PetscInt_FMT, *value, lb);
  PetscCheck(!wasset || *value <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %" PetscInt_FMT " greater than allowed bound %" PetscInt_FMT, *value, ub);
  if (set) *set = wasset;
  if (ShouldPrintHelp(PetscOptionsObject)) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %" PetscInt_FMT " : formerly %" PetscInt_FMT ">: %s (%s)\n", Prefix(prefix), opt + 1, wasset ? *value : currentvalue, currentvalue, text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsMPIInt_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscMPIInt currentvalue, PetscMPIInt *value, PetscBool *set, PetscMPIInt lb, PetscMPIInt ub)
{
  const char        *prefix  = PetscOptionsObject->prefix;
  const PetscOptions options = PetscOptionsObject->options;
  PetscBool          wasset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 6);
  if (set) PetscAssertPointer(set, 7);
  PetscCheck(currentvalue >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %d less than allowed bound %d", currentvalue, lb);
  PetscCheck(currentvalue <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %d greater than allowed bound %d", currentvalue, ub);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_INT, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscInt), &amsopt->data));
    *(PetscMPIInt *)amsopt->data = currentvalue;

    PetscCall(PetscOptionsGetMPIInt(options, prefix, opt, &currentvalue, &wasset));
    if (wasset) *(PetscMPIInt *)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetMPIInt(options, prefix, opt, value, &wasset));
  PetscCheck(!wasset || *value >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %d less than allowed bound %d", *value, lb);
  PetscCheck(!wasset || *value <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %d greater than allowed bound %d", *value, ub);
  if (set) *set = wasset;
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %d : formerly %d>: %s (%s)\n", Prefix(prefix), opt + 1, wasset ? *value : currentvalue, currentvalue, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsString_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], const char currentvalue[], char value[], size_t len, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;
  PetscBool   lset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 6);
  if (set) PetscAssertPointer(set, 8);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_STRING, &amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "", (char **)&amsopt->data));
  }
  PetscCall(PetscOptionsGetString(PetscOptionsObject->options, prefix, opt, value, len, &lset));
  if (set) *set = lset;
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %s : formerly %s>: %s (%s)\n", Prefix(prefix), opt + 1, lset ? value : currentvalue, currentvalue, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsReal_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscReal currentvalue, PetscReal *value, PetscBool *set, PetscReal lb, PetscReal ub)
{
  const char        *prefix  = PetscOptionsObject->prefix;
  const PetscOptions options = PetscOptionsObject->options;
  PetscBool          wasset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 6);
  if (set) PetscAssertPointer(set, 7);
  PetscCheck(currentvalue >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %g less than allowed bound %g", (double)currentvalue, (double)lb);
  PetscCheck(currentvalue <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Current value %g greater than allowed bound %g", (double)currentvalue, (double)ub);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_REAL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscReal), &amsopt->data));
    *(PetscReal *)amsopt->data = currentvalue;

    PetscCall(PetscOptionsGetReal(options, prefix, opt, &currentvalue, &wasset));
    if (wasset) *(PetscReal *)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetReal(PetscOptionsObject->options, prefix, opt, value, &wasset));
  PetscCheck(!wasset || *value >= lb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %g less than allowed bound %g", (double)*value, (double)lb);
  PetscCheck(!wasset || *value <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Newly set value %g greater than allowed bound %g", (double)*value, (double)ub);
  if (set) *set = wasset;
  if (ShouldPrintHelp(PetscOptionsObject)) {
    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %g : formerly %g>: %s (%s)\n", Prefix(prefix), opt + 1, wasset ? (double)*value : (double)currentvalue, (double)currentvalue, text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsScalar_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscScalar currentvalue, PetscScalar *value, PetscBool *set)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscOptionsReal(opt, text, man, currentvalue, value, set));
#else
  PetscCall(PetscOptionsGetScalar(PetscOptionsObject->options, PetscOptionsObject->prefix, opt, value, set));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsName_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool *flg)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 5);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool), &amsopt->data));

    *(PetscBool *)amsopt->data = PETSC_FALSE;
  }
  PetscCall(PetscOptionsHasName(PetscOptionsObject->options, prefix, opt, flg));
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: %s (%s)\n", Prefix(prefix), opt + 1, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsFList_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char ltext[], const char man[], PetscFunctionList list, const char currentvalue[], char value[], size_t len, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;
  PetscBool   lset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 7);
  if (set) PetscAssertPointer(set, 9);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, ltext, man, OPTION_FLIST, &amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "", (char **)&amsopt->data));
    amsopt->flist = list;
  }
  PetscCall(PetscOptionsGetString(PetscOptionsObject->options, prefix, opt, value, len, &lset));
  if (set) *set = lset;
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall(PetscFunctionListPrintTypes(PetscOptionsObject->comm, stdout, Prefix(prefix), opt, ltext, man, list, currentvalue, lset ? value : currentvalue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef __cplusplus
  #include <type_traits>
#endif

PetscErrorCode PetscOptionsEList_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char ltext[], const char man[], const char *const *list, PetscInt ntext, const char currentvalue[], PetscInt *value, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;
  PetscBool   lset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(value, 8);
  if (set) PetscAssertPointer(set, 9);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, ltext, man, OPTION_ELIST, &amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup(currentvalue ? currentvalue : "", (char **)&amsopt->data));
    PetscCall(PetscStrNArrayallocpy(ntext, list, (char ***)&amsopt->list));
    PetscCheck(ntext <= CHAR_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of list entries %" PetscInt_FMT " > %d", ntext, CHAR_MAX);
#ifdef __cplusplus
    static_assert(std::is_same<typename std::decay<decltype(amsopt->nlist)>::type, char>::value, "");
#endif
    amsopt->nlist = (char)ntext;
  }
  PetscCall(PetscOptionsGetEList(PetscOptionsObject->options, prefix, opt, list, ntext, value, &lset));
  if (set) *set = lset;
  if (ShouldPrintHelp(PetscOptionsObject)) {
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <now %s : formerly %s> %s (choose one of)", Prefix(prefix), opt + 1, lset ? list[*value] : currentvalue, currentvalue, ltext));
    for (PetscInt i = 0; i < ntext; ++i) PetscCall((*PetscHelpPrintf)(comm, " %s", list[i]));
    PetscCall((*PetscHelpPrintf)(comm, " (%s)\n", ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBoolGroupBegin_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool *flg)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 5);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool), &amsopt->data));

    *(PetscBool *)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options, prefix, opt, flg, NULL));
  if (ShouldPrintHelp(PetscOptionsObject)) {
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  Pick at most one of -------------\n"));
    PetscCall((*PetscHelpPrintf)(comm, "    -%s%s: %s (%s)\n", Prefix(prefix), opt + 1, text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBoolGroup_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool *flg)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 5);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool), &amsopt->data));

    *(PetscBool *)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options, prefix, opt, flg, NULL));
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "    -%s%s: %s (%s)\n", Prefix(prefix), opt + 1, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBoolGroupEnd_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool *flg)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 5);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool), &amsopt->data));

    *(PetscBool *)amsopt->data = PETSC_FALSE;
  }
  *flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options, prefix, opt, flg, NULL));
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "    -%s%s: %s (%s)\n", Prefix(prefix), opt + 1, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBool_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool currentvalue, PetscBool *flg, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;
  PetscBool   iset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 6);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL, &amsopt));
    PetscCall(PetscMalloc(sizeof(PetscBool), &amsopt->data));

    *(PetscBool *)amsopt->data = currentvalue;
  }
  PetscCall(PetscOptionsGetBool(PetscOptionsObject->options, prefix, opt, flg, &iset));
  if (set) *set = iset;
  if (ShouldPrintHelp(PetscOptionsObject)) {
    const char *curvalue = PetscBools[currentvalue];

    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %s : formerly %s> %s (%s)\n", Prefix(prefix), opt + 1, iset ? PetscBools[*flg] : curvalue, curvalue, text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBool3_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool3 currentvalue, PetscBool3 *flg, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;
  PetscBool   iset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(flg, 6);
  if (set) PetscAssertPointer(set, 7);
  PetscCall(PetscOptionsGetBool3(PetscOptionsObject->options, prefix, opt, flg, &iset));
  if (set) *set = iset;
  if (ShouldPrintHelp(PetscOptionsObject)) {
    const char *curvalue = PetscBool3s[currentvalue];

    PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <now %s : formerly %s> %s (%s)\n", Prefix(prefix), opt + 1, iset ? PetscBools[*flg] : curvalue, curvalue, text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsRealArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscReal value[], PetscInt *n, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(n, 6);
  PetscCheck(*n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") cannot be negative", *n);
  if (*n) PetscAssertPointer(value, 5);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    const PetscInt  nv = *n;
    PetscReal      *vals;
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_REAL_ARRAY, &amsopt));
    PetscCall(PetscMalloc(nv * sizeof(*vals), &vals));
    for (PetscInt i = 0; i < nv; ++i) vals[i] = value[i];
    amsopt->arraylength = nv;
    amsopt->data        = vals;
  }
  const PetscInt nin = *n;
  PetscCall(PetscOptionsGetRealArray(PetscOptionsObject->options, prefix, opt, value, n, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) {
    const PetscInt nv   = *n;
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%g", Prefix(prefix), opt + 1, (double)value[0]));
    for (PetscInt i = 1; i < nv; ++i) PetscCall((*PetscHelpPrintf)(comm, ",%g", (double)value[i]));
    PetscCall((*PetscHelpPrintf)(comm, ">: %s (%s)\n", text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsScalarArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscScalar value[], PetscInt *n, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(n, 6);
  PetscCheck(*n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") cannot be negative", *n);
  if (*n) PetscAssertPointer(value, 5);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    const PetscInt  nv = *n;
    PetscOptionItem amsopt;
    PetscScalar    *vals;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_SCALAR_ARRAY, &amsopt));
    PetscCall(PetscMalloc(nv * sizeof(*vals), &vals));
    for (PetscInt i = 0; i < nv; ++i) vals[i] = value[i];
    amsopt->arraylength = nv;
    amsopt->data        = vals;
  }
  const PetscInt nin = *n;
  PetscCall(PetscOptionsGetScalarArray(PetscOptionsObject->options, prefix, opt, value, n, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) {
    const PetscInt nv   = *n;
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%g+%gi", Prefix(prefix), opt + 1, (double)PetscRealPart(value[0]), (double)PetscImaginaryPart(value[0])));
    for (PetscInt i = 1; i < nv; ++i) PetscCall((*PetscHelpPrintf)(comm, ",%g+%gi", (double)PetscRealPart(value[i]), (double)PetscImaginaryPart(value[i])));
    PetscCall((*PetscHelpPrintf)(comm, ">: %s (%s)\n", text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsIntArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscInt value[], PetscInt *n, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(n, 6);
  PetscCheck(*n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") cannot be negative", *n);
  if (*n) PetscAssertPointer(value, 5);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    const PetscInt  nv = *n;
    PetscInt       *vals;
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_INT_ARRAY, &amsopt));
    PetscCall(PetscMalloc1(nv, &vals));
    for (PetscInt i = 0; i < nv; ++i) vals[i] = value[i];
    amsopt->arraylength = nv;
    amsopt->data        = vals;
  }
  const PetscInt nin = *n;
  PetscCall(PetscOptionsGetIntArray(PetscOptionsObject->options, prefix, opt, value, n, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) {
    const PetscInt nv   = *n;
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%" PetscInt_FMT, Prefix(prefix), opt + 1, value[0]));
    for (PetscInt i = 1; i < nv; ++i) PetscCall((*PetscHelpPrintf)(comm, ",%" PetscInt_FMT, value[i]));
    PetscCall((*PetscHelpPrintf)(comm, ">: %s (%s)\n", text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsStringArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], char *value[], PetscInt *nmax, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(nmax, 6);
  PetscCheck(*nmax >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") cannot be negative", *nmax);
  if (*nmax) PetscAssertPointer(value, 5);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    const PetscInt  nmaxv = *nmax;
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_STRING_ARRAY, &amsopt));
    PetscCall(PetscMalloc1(nmaxv, (char **)&amsopt->data));
    amsopt->arraylength = nmaxv;
  }
  const PetscInt nin = *nmax;
  PetscCall(PetscOptionsGetStringArray(PetscOptionsObject->options, prefix, opt, value, nmax, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  -%s%s: <string1,string2,...>: %s (%s)\n", Prefix(prefix), opt + 1, text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsBoolArray_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscBool value[], PetscInt *n, PetscBool *set)
{
  const char *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(n, 6);
  PetscCheck(*n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n (%" PetscInt_FMT ") cannot be negative", *n);
  if (*n) PetscAssertPointer(value, 5);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    const PetscInt  nv = *n;
    PetscBool      *vals;
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_BOOL_ARRAY, &amsopt));
    PetscCall(PetscMalloc1(nv, &vals));
    for (PetscInt i = 0; i < nv; ++i) vals[i] = value[i];
    amsopt->arraylength = nv;
    amsopt->data        = vals;
  }
  const PetscInt nin = *n;
  PetscCall(PetscOptionsGetBoolArray(PetscOptionsObject->options, prefix, opt, value, n, set));
  if (ShouldPrintHelp(PetscOptionsObject) && nin) {
    const PetscInt nv   = *n;
    const MPI_Comm comm = PetscOptionsObject->comm;

    PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%d", Prefix(prefix), opt + 1, value[0]));
    for (PetscInt i = 1; i < nv; ++i) PetscCall((*PetscHelpPrintf)(comm, ",%d", value[i]));
    PetscCall((*PetscHelpPrintf)(comm, ">: %s (%s)\n", text, ManSection(man)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscOptionsViewer - Creates a viewer appropriate for the type indicated by the user

  Synopsis:
  #include <petscviewer.h>
  PetscErrorCode PetscOptionsViewer(const char opt[], const char text[], const char man[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - option name
. text - short string that describes the option
- man  - manual page with additional information on option

  Output Parameters:
+ viewer - the viewer
. format - the PetscViewerFormat requested by the user, pass `NULL` if not needed
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  See `PetscOptionsCreateViewer()` for the format of the supplied viewer and its options

.seealso: `PetscOptionsCreateViewer()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
PetscErrorCode PetscOptionsViewer_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set)
{
  const MPI_Comm comm   = PetscOptionsObject->comm;
  const char    *prefix = PetscOptionsObject->prefix;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 2);
  PetscAssertPointer(viewer, 5);
  if (format) PetscAssertPointer(format, 6);
  if (set) PetscAssertPointer(set, 7);
  if (!PetscOptionsObject->count) {
    PetscOptionItem amsopt;

    PetscCall(PetscOptionItemCreate_Private(PetscOptionsObject, opt, text, man, OPTION_STRING, &amsopt));
    /* must use system malloc since SAWs may free this */
    PetscCall(PetscStrdup("", (char **)&amsopt->data));
  }
  PetscCall(PetscOptionsCreateViewer(comm, PetscOptionsObject->options, prefix, opt, viewer, format, set));
  if (ShouldPrintHelp(PetscOptionsObject)) PetscCall((*PetscHelpPrintf)(comm, "  -%s%s: <%s>: %s (%s)\n", Prefix(prefix), opt + 1, "", text, ManSection(man)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
