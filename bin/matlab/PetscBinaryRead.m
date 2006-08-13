function [varargout] = PetscBinaryRead(inarg,comp,cnt)
%
%   [varargout] = PetscBinaryRead(inarg[,comp[,cnt]])
%
%  Reads in PETSc binary file matrices or vectors
%  emits as Matlab sparse matrice or vectors.
%
%  Argument may be file name (string), socket number (integer)
%  or any Matlab class that provides the read() and close() methods
%  [We provide freader() and sreader() for binary files and sockets]
%
%  comp = 'complex' means the input file is complex
%  comp = 'cell' means return a Matlab cell array 
%         if cnt is given then cnt PETSc objects are read otherwise 
%         all objects are read in
%   
if nargin == 1
  comp = 0;
end

if ischar(inarg) 
  fd = freader(inarg);
else if isnumeric(inarg)
       fd = sreader(inarg)
     else
       fd = inarg
     end
% otherwise assume it is a freader or sreader and handles read()
end

if comp == 'cell'
  if nargin == 3
    narg = cnt
  else
    narg   = 1000;  
  end
  result = cell(1);
else
  narg = nargout;
end

for l=1:narg
  header = read(fd,1,'int32');
  if isempty(header)
    if comp == 'cell'
      varargout(1) = {result};
      return 
    else 
      disp('File does not have that many items')
    end
    return
  end
  if header == 1211216 % Petsc Mat Object 
    header = read(fd,3,'int32');
    m      = header(1);
    n      = header(2);
    nz     = header(3);
    nnz = read(fd,m,'int32');  %nonzeros per row
    sum_nz = sum(nnz);
    if(sum_nz ~=nz)
      str = sprintf('No-Nonzeros sum-rowlengths do not match %d %d',nz,sum_nz);
      error(str);
    end
    j   = read(fd,nz,'int32') + 1;
    if comp == 'complex'
      s   = read(fd,2*nz,'double');
    else 
      s   = read(fd,nz,'double');
    end
    i   = ones(nz,1);
    cnt = 1;
    for k=1:m
      next = cnt+nnz(k)-1;
      i(cnt:next,1) = k*ones(nnz(k),1);
      cnt = next+1;
    end
    if comp == 'complex'
      A = sparse(i,j,complex(s(1:2:2*nz),s(2:2:2*nz)),m,n,nz);
    else
      A = sparse(i,j,s,m,n,nz);
    end
    if comp == 'cell'
      result{l} = A;
    else 
      varargout(l) = {A};
    end
  
  elseif  header == 1211214 % Petsc Vec Object
    m = read(fd,1,'int32');
    if comp == 'complex'
      v = read(fd,2*m,'double');
      v = complex(v(1:2:2*m),v(2:2:2*m));
    else
      v = read(fd,m,'double');
    end
    if comp == 'cell'
      result{l} = v;
    else 
      varargout(l) = {v};
    end

  elseif header == 1211219 % Petsc Bag Object
    b = PetscBagRead(fd);
    if comp == 'cell'
      result{l} = b;
    else 
      varargout(l) = {b};
    end

  else 
    disp('Found unrecongnized header in file. If your file contains complex numbers')
    disp(' then call PetscBinaryRead() with "complex" as the second argument')
    return
  end

end

% close the reader if we opened it
if ischar(inarg) or isinteger(inarg) close(fd); end;
