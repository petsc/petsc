function [varargout] = PetscBinaryRead(inarg,comp,cnt)
%
%   [varargout] = PetscBinaryRead(inarg[,comp[,cnt]])
%
%  Reads in PETSc binary file matrices or vectors
%  emits as Matlab sparse matrice or vectors.
%
%  [,comp[,cnt]] indicates the comp and cnt are optional arguments
%  There are no [] in the arguments
%
%  Examples: A = PetscBinaryRead('myfile'); read from file 
%            b = PetscBinaryRead(1024);   read from socket
%            c = PetscBinaryRead();       read from default socket PETSc uses
%
%  Argument may be file name (string), socket number (integer)
%  or any Matlab class that provides the read() and close() methods
%  [We provide PetscOpenFile() and PetscOpenSocket() for binary files and sockets]
%
%  comp = 'complex' means the input file is complex
%  comp = 'cell' means return a Matlab cell array 
%         if cnt is given then cnt PETSc objects are read otherwise 
%         all objects are read in
%
%  Examples:  A = PetscBinaryRead('myfile','cell');  read all objects in file
%             A = PetscBinaryRead(1024,'cell',2);  read two objects from socket 
%   
if nargin < 2
  comp = 0;
end

if nargin == 0
  fd = PetscOpenSocket();
else if ischar(inarg) 
  fd = PetscOpenFile(inarg);
else if isnumeric(inarg)
  fd = PetscOpenSocket(inarg);
else % assume it is a PetscOpenFile or PetscOpenSocket object and handles read()
  fd = inarg;
end
end
end

if strcmp(comp,'cell')
  if nargin == 3
    narg = cnt;
  else
    narg   = 1000;  
  end
  result = cell(1);
else
  narg = nargout;
end

for l=1:narg
  header = double(read(fd,1,'int32'));
  if isempty(header)
    if strcmp(comp,'cell')
      varargout(1) = {result};
      return 
    else 
      disp('File/Socket does not have that many items')
    end
    return
  end
  if header == 1211216 % Petsc Mat Object 
    header = double(read(fd,3,'int32'));
    m      = header(1);
    n      = header(2);
    nz     = header(3);
    if (nz == -1)
      s   = read(fd,m*n,'double');
      A   = reshape(s,n,m)';
    else
      nnz = double(read(fd,m,'int32'));  %nonzeros per row
      sum_nz = sum(nnz);
      if(sum_nz ~=nz)
        str = sprintf('No-Nonzeros sum-rowlengths do not match %d %d',nz,sum_nz);
        error(str);
      end
      j   = double(read(fd,nz,'int32')) + 1;
      if strcmp(comp,'complex')
        s   = read(fd,2*nz,'double');
      else 
        s   = read(fd,nz,'double');
      end
      i   = ones(nz,1);
      cnt = 1;
      for k=1:m
        next = cnt+nnz(k)-1;
        i(cnt:next,1) = (double(k))*ones(nnz(k),1);
        cnt = next+1;
      end
      if strcmp(comp,'complex')
        A = sparse(i,j,complex(s(1:2:2*nz),s(2:2:2*nz)),m,n,nz);
      else
        A = sparse(i,j,s,m,n,nz);
      end
    end
    if strcmp(comp,'cell')
        result{l} = A;
    else 
      varargout(l) = {A};
    end
  
  elseif  header == 1211214 % Petsc Vec Object
    m = double(read(fd,1,'int32'));
    if strcmp(comp,'complex')
      v = read(fd,2*m,'double');
      v = complex(v(1:2:2*m),v(2:2:2*m));
    else
      v = read(fd,m,'double');
    end
    if strcmp(comp,'cell')
      result{l} = v;
    else 
      varargout(l) = {v};
    end

  elseif header == 1211219 % Petsc Bag Object
    b = PetscBagRead(fd);
    if strcmp(comp,'cell')
      result{l} = b;
    else 
      varargout(l) = {b};
    end

  else 
    disp(['Found unrecogonized header ' int2str(header) ' in file. If your file contains complex numbers'])
    disp(' then call PetscBinaryRead() with "complex" as the second argument')
    return
  end

end

% close the reader if we opened it

if nargin > 0
  if ischar(inarg) | isinteger(inarg) close(fd); end;
end
