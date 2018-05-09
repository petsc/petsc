function [m,n] = dfomn(nprob)

% Produces a fixed value of m and n for a given problem indexed by nprob.
% nprob cannot exceed 22. 

switch nprob

    case 1
        m = 10;
        n = 10;
    case 2
        m = 10;
        n = 10;
    case 3
        m = 10;
        n = 10;
    case 4
        m = 2;
        n = 2;
    case 5
        m = 3;
        n = 3;
    case 6
        m = 4;
        n = 4;
    case 7
        m = 2;
        n = 2;
    case 8
        m = 15;
        n = 3;
    case 9
        m = 11;
        n = 4;
    case 10
        m = 16;
        n = 3;
    case 11
        m = 31;
        n = 6;
    case 12
        m = 10;
        n = 3;
    case 13
        m = 10;
        n = 2;
    case 14
        m = 20;
        n = 4;
    case 15
        m = 2;
        n = 2;
    case 16
        m = 2;
        n = 2;
    case 17
        m = 33;
        n = 5;
    case 18
        m = 65;
        n = 11;
    case 19
        m = 12;
        n = 10;
    case 20
        m = 2;
        n = 2;
    case 21
        m = 2;
        n = 2;
    case 22
        m = 8;
        n = 8;
end