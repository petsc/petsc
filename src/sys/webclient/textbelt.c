
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
  size_t         nlen,mlen,blen;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(number,&nlen));
  PetscCheck(nlen == 10,comm,PETSC_ERR_ARG_WRONG,"Number %s is not ten digits",number);
  PetscCall(PetscStrlen(message,&mlen));
  PetscCheck(mlen <= 100,comm,PETSC_ERR_ARG_WRONG,"Message  %s is too long",message);
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    int       sock;
    char      buff[474],*body;
    PetscInt  i;

    PetscCall(PetscMalloc1(mlen+nlen+100,&body));
    PetscCall(PetscStrcpy(body,"number="));
    PetscCall(PetscStrcat(body,number));
    PetscCall(PetscStrcat(body,"&"));
    PetscCall(PetscStrcat(body,"message="));
    PetscCall(PetscStrcat(body,message));
    PetscCall(PetscStrlen(body,&blen));
    for (i=0; i<(int)blen; i++) {
      if (body[i] == ' ') body[i] = '+';
    }
    PetscCall(PetscOpenSocket("textbelt.com",80,&sock));
    PetscCall(PetscHTTPRequest("POST","textbelt.com/text",NULL,"application/x-www-form-urlencoded",body,sock,buff,sizeof(buff)));
    close(sock);
    PetscCall(PetscFree(body));
    if (flg) {
      char *found;
      PetscCall(PetscStrstr(buff,"\"success\":tr",&found));
      *flg = found ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  PetscFunctionReturn(0);
}
