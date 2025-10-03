% September 25, 2023
% Wyatt Habinski
% habinskw@mcmaster.ca
% 400338858

:- use_module(library(clpb)).

% TASK 2 -------------------------------------------------------------------------------------------------------------------------------------

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