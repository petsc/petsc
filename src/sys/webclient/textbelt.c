
#include <petscwebclient.h>

#undef __FUNCT__
#define __FUNCT__ "PetscTextBelt"
/*@C
     PetscTextBelt - Sends an SMS to an American phone number

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
.  number - the 10 digit telephone number
-  message - the message

   Output Parameter:
.   flg - PETSC_TRUE if the text was sent

   Level: intermediate

   Notes: TextBelt is run for testing purposes only, please do not use this feature often

@*/
PetscErrorCode PetscTextBelt(MPI_Comm comm,const char number[],const char message[],PetscBool *flg)
{
  PetscErrorCode ierr;
  size_t         nlen,mlen,blen;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = PetscStrlen(number,&nlen);CHKERRQ(ierr);
  if (nlen != 10) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Number %s is not ten digits",number);
  ierr = PetscStrlen(message,&mlen);CHKERRQ(ierr);
  if (mlen > 100) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Message  %s is too long",message);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    int       sock;
    char      buff[186],*body;
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
