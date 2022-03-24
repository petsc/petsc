static const char help[] = "Tests PETSc -- Mathematica connection\n";
#include <petscksp.h>
#include <mathlink.h>

typedef enum {MATHEMATICA_LINK_CREATE, MATHEMATICA_LINK_CONNECT, MATHEMATICA_LINK_LAUNCH} LinkMode;

static int setupConnection(MLENV *env, MLINK *link, const char *linkhost, LinkMode linkmode)
{
  int  argc = 5;
  char *argv[5];
  char hostname[256];
  long lerr;
  int  ierr;

  PetscFunctionBegin;
  /* Link name */
  argv[0] = "-linkname";
  argv[1] = "8001";

  /* Link host */
  argv[2] = "-linkhost";
  if (!linkhost) {
    CHKERRQ(PetscGetHostName(hostname, sizeof(hostname)));
    argv[3] = hostname;
  } else argv[3] = (char*) linkhost;

  /* Link mode */
  switch (linkmode) {
  case MATHEMATICA_LINK_CREATE:
    argv[4] = "-linkcreate";
    break;
  case MATHEMATICA_LINK_CONNECT:
    argv[4] = "-linkconnect";
    break;
  case MATHEMATICA_LINK_LAUNCH:
    argv[4] = "-linklaunch";
    break;
  }

  *env = MLInitialize(0);
  for (lerr = 0; lerr < argc; lerr++) printf("argv[%ld] = %s\n", lerr, argv[lerr]);
  *link = MLOpenInEnv(*env, argc, argv, &lerr);
  printf("lerr = %ld\n", lerr);
  PetscFunctionReturn(0);
}

static int printIndent(int indent)
{
  int i;

  PetscFunctionBegin;
  for (i = 0; i < indent; i++) printf(" ");
  PetscFunctionReturn(0);
}

static int processPacket(MLINK link, int indent)
{
  static int isHead    = 0;
  int        tokenType = MLGetNext(link);
  int        ierr;

  PetscFunctionBegin;
  CHKERRQ(printIndent(indent));
  switch (tokenType) {
  case MLTKFUNC:
  {
    long numArguments;
    int  arg;

    printf("Function:\n");
    MLGetArgCount(link, &numArguments);
    /* Process head */
    printf("  Head:\n");
    isHead = 1;
    ierr   = processPacket(link, indent+4);
    if (ierr) PetscFunctionReturn(ierr);
    isHead = 0;
    /* Process arguments */
    printf("  Arguments:\n");
    for (arg = 0; arg < numArguments; arg++) {
      CHKERRQ(processPacket(link, indent+4));
    }
  }
    break;
  case MLTKSYM:
  {
    const char *symbol;

    MLGetSymbol(link, &symbol);
    printf("Symbol: %s\n", symbol);
    if (isHead && !strcmp(symbol, "Shutdown")) {
      MLDisownSymbol(link, symbol);
      PetscFunctionReturn(2);
    }
    MLDisownSymbol(link, symbol);
  }
    break;
  case MLTKINT:
  {
    int i;

    MLGetInteger(link, &i);
    printf("Integer: %d\n", i);
  }
    break;
  case MLTKREAL:
  {
    double r;

    MLGetReal(link, &r);
    printf("Real: %g\n", r);
  }
    break;
  case MLTKSTR:
  {
    const char *string;

    MLGetString(link, &string);
    printf("String: %s\n", string);
    MLDisownString(link, string);
  }
    break;
  default:
    printf("Unknown code %d\n", tokenType);
    MLClearError(link);
    fprintf(stderr, "ERROR: %s\n", (char*) MLErrorMessage(link));
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

static int processPackets(MLINK link)
{
  int packetType;
  int loop   = 1;
  int errors = 0;
  int err;

  PetscFunctionBegin;
  while (loop) {
    while ((packetType = MLNextPacket(link)) && (packetType != RETURNPKT)) {
      switch (packetType) {
      case BEGINDLGPKT:
        printf("Begin dialog packet\n");
        break;
      case CALLPKT:
        printf("Call packet\n");
        break;
      case DISPLAYPKT:
        printf("Display packet\n");
        break;
      case DISPLAYENDPKT:
        printf("Display end packet\n");
        break;
      case ENDDLGPKT:
        printf("End dialog packet\n");
        break;
      case ENTERTEXTPKT:
        printf("Enter text packet\n");
        break;
      case ENTEREXPRPKT:
        printf("Enter expression packet\n");
        break;
      case EVALUATEPKT:
        printf("Evaluate packet\n");
        break;
      case INPUTPKT:
        printf("Input packet\n");
        break;
      case INPUTNAMEPKT:
        printf("Input name packet\n");
        break;
      case INPUTSTRPKT:
        printf("Input string packet\n");
        break;
      case MENUPKT:
        printf("Menu packet\n");
        break;
      case MESSAGEPKT:
        printf("Message packet\n");
        break;
      case OUTPUTNAMEPKT:
        printf("Output name packet\n");
        break;
      case RESUMEPKT:
        printf("Resume packet\n");
        break;
      case RETURNTEXTPKT:
        printf("Return text packet\n");
        break;
      case RETURNEXPRPKT:
        printf("Return expression packet\n");
        break;
      case SUSPENDPKT:
        printf("Suspend packet\n");
        break;
      case SYNTAXPKT:
        printf("Syntax packet\n");
        break;
      case TEXTPKT:
        printf("Text packet\n");
        break;
      }
      MLNewPacket(link);
    }

    /* Got a Return packet */
    if (!packetType) {
      MLClearError(link);
      printf("ERROR: %s\n", (char*) MLErrorMessage(link));
      errors++;
    } else if (packetType == RETURNPKT) {
      err = processPacket(link, 0);
      PetscCheckFalse(err == 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from Mathematica");
      if (err == 2) loop = 0;
    } else {
      fprintf(stderr, "Invalid packet type %d\n", packetType);
      loop = 0;
    }
    if (errors > 10) loop = 0;
  }
  PetscFunctionReturn(0);
}

static int cleanupConnection(MLENV env, MLINK link)
{
  PetscFunctionBegin;
  MLClose(link);
  MLDeinitialize(env);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MLENV env;
  MLINK link;
  int   ierr;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(setupConnection(&env, &link, "192.168.119.1", MATHEMATICA_LINK_CONNECT));
  CHKERRQ(processPackets(link));
  CHKERRQ(cleanupConnection(env, link));
  CHKERRQ(PetscFinalize());
  return(ierr);
}
