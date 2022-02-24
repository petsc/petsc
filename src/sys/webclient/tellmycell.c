
#include <petscwebclient.h>

/*@C
     PetscTellMyCell - Sends an SMS to an American/Canadian phone number

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
.  number - the 10 digit telephone number
-  message - the message

   Output Parameter:
.   flg - PETSC_TRUE if the text was sent

   Options Database:
+   -tellmycell <number[,message]> - send a message to the give number when the program ends
.   -tellmycell_user <Username> - this value is created when registering at tellmycell.com
-   -tellmycell_password <Password> - this value is created when registering at tellmycell.com

   Level: intermediate

   Notes:
    You must register for an account at tellmycell.com (you get 10 free texts with registration)

   You must provide -tellmycell_user <Username> and -tellmycell_password <Password> in the options database

   It would be nice to provide this as a free service but that would require making the PETSc TellMyCell password public.

   Developer Notes:
    Perhaps the Username and Password should be arguments to this function.

.seealso: PetscTextBelt(), PetscHTTPSRequest(), PetscHTTPSConnect(), PetscSSLInitializeContext()
@*/
PetscErrorCode PetscTellMyCell(MPI_Comm comm,const char number[],const char message[],PetscBool *flg)
{
  PetscErrorCode ierr;
  size_t         nlen,mlen,blen;
  PetscMPIInt    rank;
  char           Username[64],Password[64];

  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(number,&nlen));
  PetscCheckFalse(nlen != 10,comm,PETSC_ERR_ARG_WRONG,"Number %s is not ten digits",number);
  CHKERRQ(PetscStrlen(message,&mlen));
  PetscCheckFalse(mlen > 100,comm,PETSC_ERR_ARG_WRONG,"Message  %s is too long",message);
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    int       sock;
    char      buff[1000],*body;
    PetscInt  i;
    SSL_CTX   *ctx;
    SSL       *ssl;
    PetscBool set;

    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-tellmycell_user",Username,sizeof(Username),&set));
    PetscCheckFalse(!set,PETSC_COMM_SELF,PETSC_ERR_USER,"You must pass in a tellmycell user name with -tellmycell_user <Username>");
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-tellmycell_password",Password,sizeof(Password),&set));
    PetscCheckFalse(!set,PETSC_COMM_SELF,PETSC_ERR_USER,"You must pass in a tellmycell password with -tellmycell_password <Password>");
    CHKERRQ(PetscMalloc1(mlen+nlen+100,&body));
    CHKERRQ(PetscStrcpy(body,"User="));
    CHKERRQ(PetscStrcat(body,Username));
    CHKERRQ(PetscStrcat(body,"&Password="));
    CHKERRQ(PetscStrcat(body,Password));
    CHKERRQ(PetscStrcat(body,"&PhoneNumbers[]="));
    CHKERRQ(PetscStrcat(body,number));
    CHKERRQ(PetscStrcat(body,"&"));
    CHKERRQ(PetscStrcat(body,"Message="));
    CHKERRQ(PetscStrcat(body,message));
    CHKERRQ(PetscStrlen(body,&blen));
    for (i=0; i<(int)blen; i++) {
      if (body[i] == ' ') body[i] = '+';
    }
    CHKERRQ(PetscSSLInitializeContext(&ctx));
    CHKERRQ(PetscHTTPSConnect("app.tellmycell.com",443,ctx,&sock,&ssl));
    CHKERRQ(PetscHTTPSRequest("POST","app.tellmycell.com/sending/messages?format=json",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff)));
    CHKERRQ(PetscSSLDestroyContext(ctx));
    close(sock);
    CHKERRQ(PetscFree(body));
    if (flg) {
      char *found;
      CHKERRQ(PetscStrstr(buff,"\"success\":tr",&found));
      *flg = found ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  PetscFunctionReturn(0);
}
