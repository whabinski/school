:- module(typing, [expr/1, value/1, type/1, typed/2, sstep/2, mstep/2, tsstep/3, typederiv/3]).
%expr
expr(true_). 
expr(false_). 
expr(var(X)) :- atom(X).
expr(lambda(X,E)) :- expr(X), expr(E). 
expr(app(E1,E2)) :- expr(E1), expr(E2).
expr(pair(A,B)) :- expr(A), expr(B).
expr(fst(E)) :- expr(E).
expr(snd(E)) :- expr(E).
expr(and(P,Q)) :- expr(P), expr(Q).
expr(if_then_else(B,T,E)) :- expr(B), expr(T), expr(E).
expr(let(S, E1, E2)) :- expr(S), expr(E1), expr(E2).

%value
% terminal states that cant be further reduced
value(true_).
value(false_).
value(var(_)).
value(pair(A,B)) :- value(A), value(B). % if both elements in a pair are values, the pair itself is considered a value
value(lambda(_,_)). % lambda is simply a function. when nothing is being applied to it, its in a terminal state

%type
type(bool).
type(var).
type(func(T1, T2)) :- type(T1), type(T2). % a function with parameter t1, and body t2

%typed
typed(true_, bool). % true_ is of type bool
typed(false_, bool). % false_ is of type bool
typed(var(_), var). % var(_) is of type var 
typed(lambda(X, E), func(T1, T2)) :- typed(var(X), T1), typed(E,T2). % lambda(X,E) is of type func(X',E') if X is of type X' and E is of type E'
typed(app(E1, E2), T) :- typed(E1, func(T1, T)), typed(E2, T1). % app(E1, E2) is of type T if the return of func E1 is T, and the parameters of E1 is the same type as input E2
typed(pair(A, B), pair(T1, T2)) :- typed(A, T1), typed(B, T2). % pair(A,B) is of type pair(T1, T2) if A is of type T1 and B is of type T2
typed(fst(pair(E1,_)), T) :- typed(E1, T). % fst(app(E1,E2)) is of type T if E1 is of type T
typed(snd(pair(_,E2)), T) :- typed(E2, T). % snd(app(E1,E2)) is of type T if E2 is of type T
typed(if_then_else(X,Y,Z), T) :- typed(X,bool), typed(Y, T), typed(Z, T). % ite(X,Y,Z) is of type T if Y and Z are of type T
typed(and(P, Q), bool) :- typed(P, bool), typed(Q, bool). % and(P,Q) is of type bool if both P and Q are of type bool
typed(let(X, E1, E2), T) :- typed(X, ST), typed(E1, ST), typed(E2, T). %let(X,E1,E2) is of type T if E2 is of type T. Also X and E1 have to be of the same type


%sstep
% Rules are stated at bottom of file
sstep(app(E1, E2), app(ER, E2)) :- sstep(E1, ER). % Rule 1 
sstep(app(V, E), app(V, ER)) :- value(V), sstep(E, ER). % Rule 2
sstep(pair(A,B), pair(AR, B)) :- sstep(A, AR). % Rule 4
sstep(pair(V,B), pair(V,BR)) :- value(V), sstep(B, BR). % Rule 5
sstep(fst(pair(A, B)), A) :- value(A), value(B). % Rule 6
sstep(snd(pair(A, B)), B):- value(A), value(B). % Rule 7
sstep(and(P, Q), and(R, Q)) :- sstep(P, R). % Rule 8
sstep(and(true_, Q), Q). % Rule 9
sstep(and(false_, _), false_). % Rule 10
sstep(if_then_else(C, T, E), if_then_else(R, T, E)) :- sstep(C, R). % Rule 11
sstep(if_then_else(true_, T, _), T). % Rule 12
sstep(if_then_else(false_, _, E), E). % Rule 13
% need to implement rule 3, 14, 15 which correspond to lambda and 2 let reductions

%mstep
mstep(X, X) :- value(X).
mstep(X, Y) :- sstep(X, Z), mstep(Z, Y).

%tstep
tsstep(app(E1, E2), app(ER, E2), app_E1R(T)) :- tsstep(E1, ER, T).
tsstep(app(E1, E2), app(E1, ER), app_E2R(T)) :- tsstep(E2, ER, T).
tsstep(pair(A,B), pair(AR,B), pair_AR(T)) :- tsstep(A, AR, T).
tsstep(pair(A,B), pair(A,BR), pair_BR(T)) :- tsstep(B, BR, T).
tsstep(fst(pair(A, _)), A, fst_). 
tsstep(snd(pair(_, B)), B, snd_).
tsstep(and(true_, Q), Q, and_first_true). 
tsstep(and(false_, _), false_, and_first_false). 
tsstep(and(P, Q), and(R, Q), and_(T)) :- tsstep(P, R, T).
tsstep(if_then_else(true_, X, _), X, e_IfTrue).
tsstep(if_then_else(false_, _, Y), Y, e_IfFalse).
tsstep(if_then_else(Z, X, Y), if_then_else(W, X, Y), e_If(T)) :- tsstep(Z,W,T).
% need to implement let and lambda 

%typederiv
typederiv(true_, bool, t_True).
typederiv(false_, bool, t_False).
typederiv(var(_), var, t_Var).
typederiv(pair(E1, E2), pair(T1, T2), t_Pair(ED1, ED2)) :- typederiv(E1, T1, ED1), typederiv(E2, T2, ED2).
typederiv(fst(E), T, t_Fst(ED)) :- typederiv(E, pair(T, _), ED).
typederiv(snd(E), T, t_Snd(ED)) :- typederiv(E, pair(_, T), ED).
typederiv(and(P, Q), bool, t_And(PD, QD)) :- typederiv(P, bool, PD), typederiv(Q, bool, QD).
typederiv(if_then_else(X,Y,Z), T, t_If(XD, YD, ZD)) :- typederiv(X,bool,XD), typederiv(Y, T, YD), typederiv(Z, T, ZD).
% need to implement let, lambda, and app 




/*
Rule 1: app reduction
    -   app(e1, e2) can reduced to app(e1', e2) if e1 can reduce to e1'

Rule 2: app reduction
    -   app(e1, e2) can reduce to app(e1, e2') if e1 is in normal form, and e2 can reduce to e2'

Rule 3: lambda reduction
    - v2 applied to \x.e1 can reduce to e1[v2/x] which intuitively means v2 is substituted for x in e1.
        - can be reduced to that only when v2 is in normal form

Rule 4: pair reduction
    - pair(e1,e2) can reduce to pair(e1',e2) if e1 cn reduce to e1'

Rule 5: pair reduction
    - pair(e1,e2) can reduce to pair(e1,e2') if e1 is in normal form and e2 reduces to e2'

Rule 6: fst reduction
    - fst(pair(e1,e2)) reduces to e1 if both e1 and e2 are in normal form

Rule 7: snd reduction
    - snd((pair(e1,e2)) reduces to e2 if both e1 and e2 are in normal form

Rule 8: and reduction
    - and(e1,e2) reduces to and(e1',e2) if e1 can be reduced to e1'

Rule 9: and reduction
    - and(e1,e2) reduces to e2 if e1 is true

Rule 10: and reduction
    - and(e1,e2) reduces to false if e1 is false

Rule 11: if reduction
    - if(e1, e2, e3) reduces to if(e1', e2, e3) if e1 can reduce to e1'

Rule 12: if reduction
    - if(e1, e2, e3) reduces to e2 if e1 is true

Rule 13: if reduction
    - if(e1, e2, e3) reduces to e3 is e1 is false

Rule 14: let reduction
    - let(x,e1,e2) reduces to let(x,e1',e2) if e1 can reduce to e1'

Rule 15: let reduction
    - let(x, e1, e2) reduces to e2[e1/x] which intuitively means e1 is subsituted for x in e2
        - can be reduced only if e1 is in normal form
*/