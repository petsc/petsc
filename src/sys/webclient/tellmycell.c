
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
  ierr = PetscStrlen(number,&nlen);CHKERRQ(ierr);
  if (nlen != 10) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Number %s is not ten digits",number);
  ierr = PetscStrlen(message,&mlen);CHKERRQ(ierr);
  if (mlen > 100) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Message  %s is too long",message);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (!rank) {
    int       sock;
    char      buff[1000],*body;
    PetscInt  i;
    SSL_CTX   *ctx;
    SSL       *ssl;
    PetscBool set;

    ierr = PetscOptionsGetString(NULL,NULL,"-tellmycell_user",Username,sizeof(Username),&set);CHKERRQ(ierr);
    if (!set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"You must pass in a tellmycell user name with -tellmycell_user <Username>");
    ierr = PetscOptionsGetString(NULL,NULL,"-tellmycell_password",Password,sizeof(Password),&set);CHKERRQ(ierr);
    if (!set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"You must pass in a tellmycell password with -tellmycell_password <Password>");
    ierr = PetscMalloc1(mlen+nlen+100,&body);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"User=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,Username);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&Password=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,Password);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&PhoneNumbers[]=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,number);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&");CHKERRQ(ierr);
    ierr = PetscStrcat(body,"Message=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,message);CHKERRQ(ierr);
    ierr = PetscStrlen(body,&blen);CHKERRQ(ierr);
    for (i=0; i<(int)blen; i++) {
      if (body[i] == ' ') body[i] = '+';
    }
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("app.tellmycell.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("POST","app.tellmycell.com/sending/messages?format=json",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr = PetscFree(body);CHKERRQ(ierr);
    if (flg) {
      char *found;
      ierr = PetscStrstr(buff,"\"success\":tr",&found);CHKERRQ(ierr);
      *flg = found ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  PetscFunctionReturn(0);
}
