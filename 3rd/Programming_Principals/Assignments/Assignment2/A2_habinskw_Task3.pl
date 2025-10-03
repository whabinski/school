% September 25, 2023
% Wyatt Habinski
% habinskw@mcmaster.ca
% 400338858

:- use_module(library(clpfd)).

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