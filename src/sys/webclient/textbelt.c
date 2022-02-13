
#include <petscwebclient.h>

/*@C
     PetscTextBelt - Sends an SMS to an American/Canadian phone number

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
.  number - the 10 digit telephone number
-  message - the message

   Output Parameter:
.   flg - PETSC_TRUE if the text was sent

   Options Database:
.   -textbelt <phonenumber[,message]> - sends a message to this number when the program ends

   Level: intermediate

   Notes:
    TextBelt is run for testing purposes only, please do not use this feature often

   As of November 2016 this service does not seem to be actually transmitting the SMS, which is unfortunate since it is such a great service. Consider
   registering and using PetscTellMyCell() instead. Or email us with other alternatives we might add or make a pull request.

   Developer Notes:
    I do not know how to make the buff[] long enough to receive the "success" string but short enough that the code does not hang
       waiting for part of the message to arrive that does not exist, hence the success flg may be improperly set to false even
       though the message was delivered.

.seealso: PetscTellMyCell(), PetscOpenSocket(), PetscHTTPRequest()
@*/
PetscErrorCode PetscTextBelt(MPI_Comm comm,const char number[],const char message[],PetscBool *flg)
{
  PetscErrorCode ierr;
  size_t         nlen,mlen,blen;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = PetscStrlen(number,&nlen);CHKERRQ(ierr);
  PetscCheckFalse(nlen != 10,comm,PETSC_ERR_ARG_WRONG,"Number %s is not ten digits",number);
  ierr = PetscStrlen(message,&mlen);CHKERRQ(ierr);
  PetscCheckFalse(mlen > 100,comm,PETSC_ERR_ARG_WRONG,"Message  %s is too long",message);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    int       sock;
    char      buff[474],*body;
    PetscInt  i;

    ierr = PetscMalloc1(mlen+nlen+100,&body);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"number=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,number);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&");CHKERRQ(ierr);
    ierr = PetscStrcat(body,"message=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,message);CHKERRQ(ierr);
    ierr = PetscStrlen(body,&blen);CHKERRQ(ierr);
    for (i=0; i<(int)blen; i++) {
      if (body[i] == ' ') body[i] = '+';
    }
    ierr = PetscOpenSocket("textbelt.com",80,&sock);CHKERRQ(ierr);
    ierr = PetscHTTPRequest("POST","textbelt.com/text",NULL,"application/x-www-form-urlencoded",body,sock,buff,sizeof(buff));CHKERRQ(ierr);
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
