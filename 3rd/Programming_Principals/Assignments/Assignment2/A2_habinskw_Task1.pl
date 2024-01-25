% September 25, 2023
% Wyatt Habinski
% habinskw@mcmaster.ca
% 400338858

% TASK 1 -------------------------------------------------------------------------------------------------------------------------------------

% TASK 1.1
% A predicate elem/2 such that it returns true if the first argument given is present in the list that is the second argument given.
% Returns false if no such element is found.

% elem(X, Xs) :- false.
elem(X, [X|_]). % Base case -> element is the head of the list
elem(X, [_|Xs]) :- elem(X, Xs). % Recursive case -> if element is not the head, then recurse with the tail

% Test cases for elem
:- elem(a,[a,b,c,d,e]). % True
:- elem(c,[a,b,c,d,e]). % True
:- elem(e,[a,b,c,d,e]). % True
:- \+ elem(f,[a,b,c,d,e]). % False
:- \+ elem(two,[]). % False

% TASK 1.2
% A predicate pick/3 such that it returns true if the first argument is present in the list that is the second argument.
% Also, the list that is the third argument must equal the first list with one instance of the first argument element missing.
% Returns false if either of those conditions are not met.

% pick(X, Xs, Ys) :- false.
pick(X, [X|Xs], Xs). % Base case -> element is head of the first list, and the second list is the remaining tail
pick(X, [Y|Xs], [Y|Ys]) :- pick(X, Xs, Ys). % Recursive case -> if the element is not in the current head of the first list, traverse the remaining tail recursively

% Test cases for pick
:- pick(1, [1,2,3,4,5], [2,3,4,5]). % True
:- pick(1,[5,4,3,2,1],[5,4,3,2]). % True
:- pick(1,[1],[]). % True 
:- \+ pick(1,[2,3,4,5],[2,3,4,5]). % False
:- \+ pick(3,[1,2,3,4,5],[1,2,3,4,5]). % False

% Task 1.3
% A predicate permute/2 such that it returns true if the list that is the first argument, is a permutation of the list that is the second element.
% Returns false otherwise.
% Helper predicate select/3 -> a built in prolog predicate that selects an element X, from the list Ys, and unifies the remaining list without X.

% permute(Xs, Ys) :- false.
permute([], []). % Base case -> if the two lists are empty, the lists are the same sisze and are a permutation.
permute([X|Xs], Ys) :- select(X, Ys, NewYs), permute(Xs, NewYs). % Recursive case -> If the lists are not empty, we select and remove the head of the first list, remove it from the second while keeping its order, and recurse on the 2 new lists.

% Test cases for permute
:- permute([1,2,3],[3,2,1]). % True
:- permute([1,2,3],[1,2,3]). % True
:- permute([1,2,3], [2,3,1]). % True
:- \+ permute([1,2,3],[4,5,6]). % False
:- \+ permute([1,2,3,4],[1,2,3]). % False

% Task 1.4
% A predicate sorted/1 such that it returns true if the list that is the argument is sorted.
% Returns false if not sorted.

% sorted(Xs) :- false.
sorted([]). % Case to check if the given input was an empty list
sorted([_]). % Base case -> If the cardinality of the list is 1, then the list is sorted
sorted([X,Y|Rest]) :- X =< Y, sorted([Y|Rest]). % Recursive step -> If the list has multiple elements, check if the first element is <= to the second, if ture then recurse with the tail of the list.

% Test cases for sorted
:- sorted([1,2,3,4,5]). % True
:- sorted([-5,5,23,24,100]). % True
:- sorted([]). % True
:- \+ sorted([5,4,3,2,1]). % False
:- \+ sorted([5,1,2,3,4]). % False
:- \+ sorted([1,2,9,4,5]). % False

% Task 1.5
% A predicate naive_sort/2 such that it returns true if the list that is the second argument is a sorted version of the list that is the first argument
% Returns false otherwise
% Helper predicates permute/2 and sorted/1 defined above.

% naive_sort(Xs,Ys).
naive_sort(Xs, Ys) :- permute(Xs, Ys), sorted(Ys). % Case to check that the 2 lists are permutations of eachother, and if the second list is sorted.

% Test cases for naive_sort
:- naive_sort([3,1,2],[1,2,3]). % True
:- naive_sort([2],[2]). % True
:- \+ naive_sort([4,1,3,2],[4,1,3,2]). % False
:- \+ naive_sort([1,2],[1,2,3]). % False
:- \+ naive_sort([4,2,1],[1,2]). % False

