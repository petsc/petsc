function A = receive(port)
% $Id: receive.m,v 1.5 1999/11/23 18:08:05 bsmith Exp bsmith $
%
%   A = receive(port)
%   Receives a matrix from a port opened with openport()
%see openport and closeport
disp('You must build the receive mex file by doing: cd ${PETSC_DIR}; make BOPT=g matlabcodes')