function A = receive(port)
% $Id: closeport.c,v 1.14 1999/10/24 14:01:02 bsmith Exp $
%
%   A = receive(port)
%   Receives a matrix from a port opened with openport()
%see openport and closeport
disp('You must build the receive mex file by doing: cd $PETSC_DIR; make BOPT=g matlabcodes')