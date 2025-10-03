% September 25, 2023
% Wyatt Habinski
% habinskw@mcmaster.ca
% 400338858

:- use_module(library(clpb)).
:- use_module(library(clpfd)).

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

% --------------------------------------------------------------------------------------------------------------------------------------------

% TASK 2 -------------------------------------------------------------------------------------------------------------------------------------
% BONUS INCLUDED

% In the folowwing predicates, we define 0 as a liar / goblin, and 1 as a truth teller / gnome

% Clue 
clue(Alice, Bob) :- 
    sat(Alice =:= ~Alice * ~Bob).

% Solution
:- clue(0,1). % Alice is a goblin, and Bob is a gnome.

% Riddle 1
riddle1(Alice, Bob) :- 
    sat(Alice =:= ~Bob), % Alice says bob is a goblin
    sat(Bob =:= Alice * Bob). % Bob says alice and bob are gnomes

% Solution
:- riddle1(1,0). % Alice is a gnome, and Bob is a goblin

% Riddle 2
riddle2(Alice, Bob, Carol, Dave) :-
    sat(Alice =:= Dave), % Alice says Dave is a gnome
    sat(Bob =:= ~Carol * Alice), % Bob says Carol is a goblin, and Alice is a gnome
    sat(Carol =:= ~Carol + Carol), % Carol says they are either a goblin or a gnome
    sat(Dave =:= 
        (Alice * ~Alice * ~(~Bob * ~Carol)) + % Alice is a gnome and Alice is a goblin, but Bob and Carol are goblins is false
        (Alice * Alice * (~Bob * ~Carol)) + % Alice is a gnome and Bob and Carol are goblins, but Alice is a goblin is false
        (~Alice * ~Alice * (~Bob * ~Carol)) % Alice is a goblin and Bob and Carol are goblins, but Alice is a gnome is false
    ). % Dave says exactly 2 out of the 3 are true


% Solution 
:- riddle2(0,0,1,0).

% Riddle 3
is_creature(X) :- var(X).

is_statement(gnome(X)) :- is_creature(X). 

is_statement(goblin(X)) :- is_creature(X). 

is_statement(and(X,Y)) :-
    is_statement(X),
    is_statement(Y).

is_statement(or(X,Y)) :- 
    is_statement(X),
    is_statement(Y).

eval_statement(gnome(X), sat(X)). 
eval_statement(goblin(X), sat(~X)). 

eval_statement(and(X, Y), Result) :- 
    eval_statement(X, sat(Xresult)), 
    eval_statement(Y, sat(Yresult)), 
    Result = sat(Xresult * Yresult).

eval_statement(or(X, Y), Result) :- 
    eval_statement(X, sat(Xresult)), 
    eval_statement(Y, sat(Yresult)), 
    Result = sat(Xresult + Yresult).

% BONUS ------------------------
eval_statement(xor(X,Y), Result) :-
    eval_statement(X, sat(Xresult)), 
    eval_statement(Y, sat(Yresult)), 
    Result = sat(Xresult # Yresult).

eval_statement(implies(X,Y), Result) :-
    eval_statement(X, sat(Xresult)), 
    eval_statement(Y, sat(Yresult)), 
    Result = sat(~Xresult + Yresult).

eval_statement(not(X), Result) :-
    eval_statement(X, sat(Xresult)), 
    Result = sat(~Xresult).
% ------------------------------

goblins_or_gnomes([],[]).
goblins_or_gnomes([_|Gs],[]) :-
    goblins_or_gnomes(Gs,[]).
goblins_or_gnomes([G|Gs], [R|Rs]) :-
    eval_statement(R, sat(Result)),
    sat(G =:= Result),
    goblins_or_gnomes(Gs, Rs).

% Tests
% Clue
:- goblins_or_gnomes([A,B], [and(goblin(A), goblin(B))]).
% Riddle1
:- goblins_or_gnomes([A,B], [goblin(B), and(gnome(A), gnome(B))]).
% Riddle2
:- goblins_or_gnomes([A,B,C,D], [gnome(D), and(goblin(C), gnome(A)), or(gnome(C),goblin(C)), or(and(gnome(A), and(goblin(B),goblin(C))), and(goblin(A), and(goblin(B),goblin(C))))]).
% Others
:- goblins_or_gnomes([A,B],[goblin(B), and(gnome(A),gnome(B))]).
:- goblins_or_gnomes([A,B],[or(and(gnome(A),gnome(B)), and(goblin(A),goblin(B))), gnome(A)]).

% Bonus Tests
:- goblins_or_gnomes([A,B,C],[implies(gnome(A),gnome(B)), implies(goblin(C), gnome(B)), implies(gnome(A), goblin(B))]). % Implies
:- goblins_or_gnomes([A,B,C],[and(gnome(A),or(goblin(B),gnome(C))), xor(goblin(C), goblin(A)), goblin(A)]). % Xor
:- goblins_or_gnomes([A,B], [goblin(B), and(not(goblin(A)), not(goblin(B)))]). % Not

% --------------------------------------------------------------------------------------------------------------------------------------------

% TASK 3 -------------------------------------------------------------------------------------------------------------------------------------

boolean(true).
boolean(false).

is_expr(int(V)) :- V in inf..sup.
is_expr(bool(B)) :- boolean(B).
is_expr(add(X, Y)) :- is_expr(X), is_expr(Y).
is_expr(mul(X, Y)) :- is_expr(X), is_expr(Y).
is_expr(neg(X)) :- is_expr(X).
is_expr(and(X,Y)) :- is_expr(X), is_expr(Y).
is_expr(xor(X,Y)) :- is_expr(X), is_expr(Y).
is_expr(if(B,X,Y)) :- is_expr(B), is_expr(X), is_expr(Y).

is_val(V) :- boolean(V); V in inf..sup.

% Evaluator
% eval_expr(E,V) :- false

% Define tthe eval for int and boolean to maintain constraints
eval_expr(int(V), V) :- V in inf..sup. % Integer evaluator
eval_expr(bool(B), B):- boolean(B). % Boolean evaluator

eval_expr(add(X,Y), Sum) :- % Addition evaluator
    eval_expr(X, XVal), % Eval X
    eval_expr(Y, YVal), % Eval Y
    Sum #= XVal + YVal. % Add the values of eval X, and aval Y
eval_expr(mul(X,Y), Product) :- % Multiplication evaluator
    eval_expr(X, XVal), % Eval X
    eval_expr(Y, YVal), % Eval Y
    Product #= XVal * YVal. % Multiply the values of eval X, and eval Y

eval_expr(neg(X), NegX) :- % Integer negation evaluator
    eval_expr(X, XVal), % Eval X
    NegX #= -(XVal). % Negate the value of eval X

eval_expr(and(X, Y), true) :- % Logical AND evaluator 
    (eval_expr(X, true), eval_expr(Y, true)). % True if both eval X and eval Y are true
eval_expr(and(X, Y), false) :- 
    (eval_expr(X, false), eval_expr(Y, false)); % False if both eval X and eval Y are false
    (eval_expr(X, false), eval_expr(Y, true)); % False if eval X is false and eval Y true
    (eval_expr(X, true), eval_expr(Y, false)). % False if eval X is true and eval Y false

eval_expr(xor(X, Y), true) :- % Logical XOR evaluator
    (eval_expr(X, false), eval_expr(Y, true)); % True if eval X is false and eval Y is true
    (eval_expr(X, true), eval_expr(Y, false)). % True if eval X is true and eval Y is false
eval_expr(xor(X, Y), false) :- 
    (eval_expr(X, false), eval_expr(Y, false)); % False if both eval X and eval Y is false
    (eval_expr(X, true), eval_expr(Y, true)). % False if both eval X and eval Y is true

eval_expr(if(B, X, _), Result) :- % Logical if then else condition evaluator 
    eval_expr(B, true), % Eval B 
    eval_expr(X, Result). % If eval B is true, then eval X
eval_expr(if(B, _, Y), Result) :- 
    eval_expr(B, false), % Eval B
    eval_expr(Y, Result). % If eval B is false then eval Y

eval_expr(is_val(V), V) :- is_val(V). % Val evaluator

% Tests for Task 3

% Tests for student number 400338858
:- eval_expr(add(mul(int(757),int(382071)),int(111111111)),400338858). % (757 * 382 071) + 111 111 111
:- eval_expr(add(mul(int(123456789),int(5)),neg(int(216945087))),400338858). % (123456789 * 5) + neg(216945087)
:- eval_expr(add(mul(add(int(394),int(98765432)),int(5)),int(-93490272)),400338858). % ((394 + 98 765 432) * 5) + -93 490 272

% Int test
:- eval_expr(int(100),100).
:- \+ eval_expr(int(true),100). % Make sure type checking is working

% Bool tests
:- eval_expr(bool(true),true).
:- \+ eval_expr(bool(bool(false)),bool(true)). % Make sure type checking is working

% AND tests
:- eval_expr(and(bool(true),bool(false)),false).
:- eval_expr(and(bool(true),bool(true)), true).
:- \+ eval_expr(and(bool(false),bool(false)),true).
:- \+ eval_expr(and(bool(false),bool(true)),true).

:- eval_expr(and(and(bool(true),bool(true)),bool(false)),false).

% XOR tests
:- eval_expr(xor(bool(false),bool(true)),true).
:- eval_expr(xor(bool(true),bool(true)),false).
:- \+ eval_expr(xor(bool(false),bool(false)),true).
:- eval_expr(xor(bool(true),bool(false)),true).

:- \+ eval_expr(xor(xor(bool(true),bool(true)),bool(true)),false).

% IF tests
:- eval_expr(if(bool(true),int(10),neg(int(10))),10).
:- eval_expr(if(bool(false),int(10),neg(int(10))),-10).
:- eval_expr(if((and(bool(true),bool(false))),(mul(int(20),int(5))),(neg(int(100)))),-100).
:- eval_expr(if((xor(bool(true),bool(false))),(mul(int(20),int(5))),(neg(int(100)))),100).

% VAL tests
:- eval_expr(is_val(100),100). % Test integer
:- eval_expr(is_val(true),true). % Test boolean