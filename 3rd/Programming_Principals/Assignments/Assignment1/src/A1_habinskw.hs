-- Assignment 1
-- Wyatt Habinski
-- whabinski@mcmaster.ca
-- 400338858
-- September 15 2023

module A1_habinskw where

data Expr =
    EInt Integer
    | ECh Char
    | EBool Bool
    | EString String

    | EAdd Expr Expr -- addition of integers
    | EMul Expr Expr -- multiplication of integers
    | ENeg Expr -- negation of integers
    | ECat Expr Expr -- concatenation of strings
    | ECons Char Expr -- adding a character at the start of a string
    | EAnd Expr Expr -- AND of tho booleans
    | EXor Expr Expr -- XOR of two booleans
    | EIf Expr Expr Expr -- if then else
    | EShowInt Expr -- render an integer as a string
    deriving (Show)

data Val =
    VInt Integer
    | VBool Bool
    | VString String
    | VError -- something went wrong
    deriving (Show, Eq)

-- Function that evaluates a given Expr and returns a Val
evalExpr :: Expr -> Val
-- Evaluate an integer expression
evalExpr (EInt i) = VInt i
-- Evaluate a character expression as a string
evalExpr (ECh c) = VString [c]
-- Evaluate a boolean expression
evalExpr (EBool b) = VBool b
-- Evaluate a string expression
evalExpr (EString s) = VString s

-- Evaluate the negation of an integer expression
-- Returns VError if input is not valid
evalExpr (ENeg x) = 
    case evalExpr x of
        (VInt x)    -> VInt (-x)
        _           -> VError
-- Evaluate an integer expression and convert it to a string representation
-- Returns VError if input is not valid
evalExpr (EShowInt x) = 
    case evalExpr x of
        (VInt x)    -> VString (show x)
        _           -> VError

-- Evaluate the addition of two integer expressions
-- Returns VError if input is not valid
evalExpr (EAdd x y) =
    case (evalExpr x, evalExpr y) of
        (VInt x, VInt y)    -> VInt (x + y)
        _                   -> VError
-- Evaluate the multiplication of two integer expressions
-- Returns VError if input is not valid
evalExpr (EMul x y) = 
    case (evalExpr x, evalExpr y) of
        (VInt x, VInt y)    -> VInt (x * y)
        _                   -> VError
-- Evaluate the concatenation of two string expressions
-- Returns VError if input is not valid
evalExpr (ECat x y) = 
    case (evalExpr x, evalExpr y) of
        (VString x, VString y)  -> VString (x ++ y)
        _                       -> VError
-- Evaluate adding a character at the start of a string expression
-- Returns VError if input is not valid
evalExpr (ECons c s) =
    case evalExpr s of
        VString s   -> VString (c : s)
        _           -> VError

-- Evaluate the logical AND of two boolean expressions
-- Returns VError if input is not valid
evalExpr (EAnd a b) =
    case (evalExpr a, evalExpr b) of
        (VBool a, VBool b)  -> VBool(a && b)
        _                   -> VError
-- Evaluate the logical XOR of two boolean expressions
-- Returns VError if input is not valid
evalExpr (EXor a b) =
    case (evalExpr a, evalExpr b) of
        (VBool a, VBool b)  -> VBool(a /= b)
        _                   -> VError

-- Evaluate a conditional expression (if-then-else)
-- Returns VError if input is not valid
evalExpr (EIf c t f) = 
    case evalExpr c of
        VBool True  -> evalExpr t
        VBool False -> evalExpr f
        _           -> VError
