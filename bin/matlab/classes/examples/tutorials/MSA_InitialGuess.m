function[x] = MSA_InitialGuess(user,x)

%%% Get the n-dimensional array, actually its the same petsvec object with
%%% VecFromDM flag set to 1 if not set initially.
x1 = user.dm.VecGetArray(x); 
%% The initial guess is the average of the boundary conditions
mx = user.mx; my = user.my;
for(j = 1:my)
    for(i = 1:mx)
        a1 = ((j)*user.bottom(i+1)+(my-j+2)*user.top(i+1))/(my+2);
        a2 = ((i)*user.left(j+1)+(mx-i+2)*user.right(j+1))/(mx+2);
        x1(i,j) = (a1 + a2)/2.0;
    end
end
