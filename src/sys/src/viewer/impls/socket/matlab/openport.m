function port = openport(number)
% $Id: closeport.c,v 1.14 1999/10/24 14:01:02 bsmith Exp $
%
%  port = openport(number)
%  Opens a port to receive matrices from Petsc.
% see closeport and receive
disp('You must build the openport mex file by doing: cd $PETSC_DIR; make BOPT=g matlabcodes')