{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-|
Module      : 1JC3-Assign4.Assign_4.hs
Copyright   :  (c) Curtis D'Alves 2021
License     :  GPL (see the LICENSE file)
Maintainer  :  Wyatt Habinski
Date        :  December 7 2021
Stability   :  experimental
Portability :  portable

Description:
  Assignment 4 - McMaster CS 1JC3 2021
-}
module Assign_4 where

import Test.QuickCheck

-----------------------------------------------------------------------------------------------------------
-- INSTRUCTIONS              README!!!
-----------------------------------------------------------------------------------------------------------
-- 1) DO NOT DELETE/ALTER ANY CODE ABOVE THESE INSTRUCTIONS
-- 2) DO NOT REMOVE / ALTER TYPE DECLERATIONS (I.E THE LINE WITH THE :: ABOUT THE FUNCTION DECLERATION)
--    IF YOU ARE UNABLE TO COMPLETE A FUNCTION, LEAVE IT'S ORIGINAL IMPLEMENTATION (I.E. THROW AN ERROR)
-- 3) MAKE SURE THE PROJECT COMPILES (I.E. RUN STACK BUILD AND MAKE SURE THERE ARE NO ERRORS) BEFORE
--    SUBMITTING, FAILURE TO DO SO WILL RESULT IN A MARK OF 0
-- 4) REPLACE macid = "TODO" WITH YOUR ACTUAL MACID (EX. IF YOUR MACID IS jim THEN macid = "jim")
-----------------------------------------------------------------------------------------------------------

-- Name: TODO add name
-- Date: TODO add date
macid :: String
macid = "habinskw"

{- --------------------------------------------------------------------
 - Datatype: MathExpr
 - --------------------------------------------------------------------
 - Description: An Abstract Syntax Tree (AST) for encoding mathematical
 -              expressions
 - Example: The expression
 -                (abs (2*X + 1)) ^ 3
 -          can be encoded as
 -                (Func1 (Power 3) (Func1 Abs (Func2 Add (Func2 Mult (Coef 2) X) (Coef 1))))
 - --------------------------------------------------------------------
 -}
data MathExpr a =
    X
  | Coef a
  | Func1 UnaryOp (MathExpr a)
  | Func2 BinOp (MathExpr a) (MathExpr a)
  deriving (Eq,Show,Read)

data BinOp = Add | Mult
  deriving (Show,Eq,Read)

data UnaryOp = Cos | Sin | Abs | Power Int
  deriving (Show,Eq,Read)

{- -----------------------------------------------------------------
 - eval
 - -----------------------------------------------------------------
 - Description:
 -    
 -    the function of eval is to evalute a given math expression of type MathExpr at a given value of type float
 
 -    eval takes a MathExpr as the first input as well as a float as the second input
 -    the output of the function is a float of the answer 

 -    eval uses recurssion and pattern matching to the data mathExpr to iterate through the expression and evaluate at the given float
 -}
eval :: (Floating a, Eq a) => MathExpr a -> a -> a
eval X v = v
eval (Coef e) v = e
eval (Func1 ex e) v = 
    case ex of
      Cos       -> cos(eval e v)
      Sin       -> sin(eval e v)
      Abs       -> abs(eval e v)
      Power p   -> (eval e v)^^p
eval (Func2 ex e v) n = 
    case ex of
      Add       -> (eval e n) + (eval v n)
      Mult      -> (eval e n) * (eval v n)


{- -----------------------------------------------------------------
 - instance Num a => Num (MathExpr a)
 - -----------------------------------------------------------------
 - Description:
 -    Purpose is to define the following methods so that haskell's standard operations can be defined in terms of data MathExpr
 -    to be used 
 -}
instance Num a => Num (MathExpr a) where
  x + y         = Func2 Add x y
  x * y         = Func2 Mult x y
  negate x      = -X
  abs x         = Func1 Abs x
  fromInteger i = Coef (fromIntegral i) 
  signum _      = error "signum is left un-implemented"

{- -----------------------------------------------------------------
 - instance Fractional a => Fractional (MathExpr a)
 - -----------------------------------------------------------------
 - Description:
 -    Purpose is to define Haskell's implemented "recip" and "fromRational" methods to work in terms of a MathExpr
 -}
instance Fractional a => Fractional (MathExpr a) where
  recip e        = Func1 (Power (-1)) e
  fromRational e = Coef (fromRational e)

{- -----------------------------------------------------------------
 - instance Floating a => Floating (MathExpr a)
 - -----------------------------------------------------------------
 - Description:
 -    Purpose is to define Haskell's pi, sin, and cos functions to work in terms of MathExpr
 -}
instance Floating a => Floating (MathExpr a) where
  pi      = Coef pi
  sin     = Func1 Sin
  cos     = Func1 Cos
  log     = error "log is left un-implemented"
  asin _  = error "asin is left un-implemented"
  acos _  = error "acos is left un-implemented"
  atan _  = error "atan is left un-implemented"
  sinh _  = error "sinh is left un-implemented"
  cosh _  = error "cosh is left un-implemented"
  tanh _  = error "tanh is left un-implemented"
  asinh _ = error "asinh is left un-implemented"
  acosh _ = error "acosh is left un-implemented"
  atanh _ = error "atanh is left un-implemented"
  exp _   = error "exp is left un-implemented"
  sqrt _  = error "sqrt is left un-implemented"

{- -----------------------------------------------------------------
 - diff
 - -----------------------------------------------------------------
 - Description:
 -    
 -    Purpose is to find the derivative of a math expression 
 
 -    diff takes a MathExpr as an input and returns a modified version of it as another MathExpr as output. 

 -    diff recursively itterates through the MathExpr using pattern matching and applies the laws of derivatives 
 -    to each part of the expression
 
 -}
diff :: (Floating a, Eq a) => MathExpr a -> MathExpr a
diff X = 1
diff (Coef e) = 0
diff (Func1 ex e) = 
    case ex of
      Cos -> (sin(e) * (diff e))
      Sin -> cos(e) * (diff e)
      Abs -> (e/(abs(e))) * (diff e)
      Power n -> fromIntegral n*(e^^(n-1)) * (diff e)
diff (Func2 ex e v) = 
    case ex of
      Add -> (diff e) + (diff v)
      Mult -> ((diff e)*v) + ((diff v)*e)


        
{- -----------------------------------------------------------------
 - pretty
 - -----------------------------------------------------------------
 - Description:
 -    
 -    Purpose is to convert a type MathExpr to a readable string

 -    Takes a MathExpr as input and returns a String as output

 -    pretty recursively iterates through all parts of the MathExpr by pattern matching and returns the appropriate string for each part

 -}
pretty :: (Show a) => MathExpr a -> String
pretty X = "X"
pretty (Coef c) = show c
pretty (Func1 ex u0) = 
    case ex of
      Power n -> "(" ++ pretty u0 ++ " ^^ " ++ "(" ++ show n ++ ")" ++ ")"
      Cos     -> "cos" ++ "(" ++ pretty u0 ++ ")"
      Sin     -> "sin" ++ "(" ++ pretty u0 ++ ")"
      Abs     -> "abs" ++ "(" ++ pretty u0 ++ ")"
pretty (Func2 ex u0 u1) = 
    case ex of
      Add   -> "(" ++ pretty u0 ++ " + " ++ pretty u1 ++ ")"
      Mult  -> "(" ++ pretty u0 ++ " * " ++ pretty u1 ++ ")"

{- -----------------------------------------------------------------
 - Test Cases
 - -----------------------------------------------------------------
 -}
infix 4 =~
(=~) :: (Floating a,Ord a) => a -> a -> Bool
x =~ y = abs (x - y) <= 1e-4

{- EXAMPLE
 - Function: eval
 - Property: eval (Func2 Add (Coef x) X) y is correct for all x,y
 - Actual Test Result: Pass
 -}
evalProp0 :: (Float,Float) -> Bool
evalProp0 (x,y) = (x + y) =~ eval (Func2 Add (Coef x) X) y

runEvalProp0 :: IO ()
runEvalProp0 = quickCheck  evalProp0

{-
--------------------------------------
Function: eval
Test Case Number: 1
Input: (X*2) 7
Expected Output: 14.0
Actual Output: 14.0

Function: eval
Test Case Number: 2
Input: (abs (X + (7*X))) 8
Expected Output: 64.0
Actual Output: 64.0

Function: eval
Test Case Number: 3
Input: (cos(3+(2*(X+3)))) 2
Expected Output: 0.9074...
Actual Output: 0.9074...
--------------------------------------
Function: diff
Test Case Number: 1
Input: (1 + 3 + 4 + X + 7)
Expected Output: Func2 Add (Func2 Add (Func2 Add (Func2 Add (Coef 0.0) (Coef 0.0)) (Coef 0.0)) (Coef 1.0)) (Coef 0.0)
Actual Output: Func2 Add (Func2 Add (Func2 Add (Func2 Add (Coef 0.0) (Coef 0.0)) (Coef 0.0)) (Coef 1.0)) (Coef 0.0)

Function: diff
Test Case Number: 2
Input: (X^^2)
Expected Output: Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)
Actual Output: Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)

Function: diff
Test Case Number: 3
Input: diff (X^^4 * 2)
Expected Output: Func2 Add (Func2 Mult (Func2 Add (Func2 Mult (Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)) (Func2 Mult X X)) (Func2 Mult (Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)) (Func2 Mult X X))) (Coef 2.0)) (Func2 Mult (Coef 0.0) (Func2 Mult (Func2 Mult X X) (Func2 Mult X X)))
Actual Output: Func2 Add (Func2 Mult (Func2 Add (Func2 Mult (Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)) (Func2 Mult X X)) (Func2 Mult (Func2 Add (Func2 Mult (Coef 1.0) X) (Func2 Mult (Coef 1.0) X)) (Func2 Mult X X))) (Coef 2.0)) (Func2 Mult (Coef 0.0) (Func2 Mult (Func2 Mult X X) (Func2 Mult X X)))
--------------------------------------
Function: pretty
Test Case Number: 1
Input: (X*2)
Expected Output: "(X * 2)"
Actual Output: "(X * 2)"

Function: pretty
Test Case Number: 2
Input: Func2 Mult X (Coef 2)
Expected Output: "(X * 2)"
Actual Output: "(X * 2)"

Function: pretty
Test Case Number: 3
Input: Func2 Add (Func2 Mult (Func2 Mult (Func2 Mult X X) (Func2 Mult X X)) (Func2 Mult (Func2 Mult X X) X)) (Func2 Mult (Coef 3.0) (Func1 Sin X))
Expected Output: "((((X * X) * (X * X)) * ((X * X) * X)) + (3.0 * sin(X)))"
Actual Output: "((((X * X) * (X * X)) * ((X * X) * X)) + (3.0 * sin(X)))"
--------------------------------------
-}
