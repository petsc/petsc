function close(sreader)
%
%   O = close(sreader) - closes the socket connection created with sopen(socketnumber)
%
%   See $PETSC_DIR/share/petsc/matlab/@PetscOpenSocket/PetscOpenSocket.m
%
sclose(sreader.fd);
sreader.fd = 0;

