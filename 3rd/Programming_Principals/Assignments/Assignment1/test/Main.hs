-- | Tests for Assignment 1.
module Main where

import A1_habinskw
import Test.HUnit
import System.Exit

a1Tests :: Test 
a1Tests = TestList
  [ 
  -- Building macid via ECons and ECat
   TestCase (assertEqual "macid by concatenating abin and skw, then prepending an h " (A1_habinskw.evalExpr (ECons 'h' (ECat (EString "abin") (EString "skw")))) (VString "habinskw"))
  ,TestCase (assertEqual "macid by prepending h to ab, then prepending i to nskw, and finally concatenating the both together" (A1_habinskw.evalExpr (ECat (ECons 'h' (EString "ab")) (ECons 'i' (EString "nskw")))) (VString "habinskw"))
  ,TestCase (assertEqual "macid by concatenating k and w, followed by prepending s, n, i, b, a, and finally h" (A1_habinskw.evalExpr (ECons 'h' (ECons 'a' (ECons 'b' (ECons 'i' (ECons 'n' (ECons 's' (ECat (EString "k") (EString "w"))))))))) (VString "habinskw"))
  
  -- Building student # represented as a string via 3 computational steps
  , TestCase (assertEqual "student# in string representation by multiplication of 12345 and 6789, then addition of 416189795 and finally adding neg 99661142" (A1_habinskw.evalExpr (EShowInt (EAdd (ENeg (EInt 99661142)) (EAdd (EInt 416189795) (EMul (EInt 12345)(EInt 6789)))))) (VString "400338858"))
  , TestCase (assertEqual "student# in string representation by multiplication of 123456789 and 4, followed by the addition of 1, and finally adding the integer negation of 93488299" (A1_habinskw.evalExpr (EShowInt (EAdd (ENeg (EInt 93488299)) (EAdd (EInt 1) (EMul (EInt 123456789)(EInt 4)))))) (VString "400338858"))
  , TestCase (assertEqual "student# in string representation by adding 10 and 400338858, followed by the addition of (2 multiplied by the integer negation of 5)" (A1_habinskw.evalExpr (EShowInt (EAdd (EAdd (EInt 10) (EInt 400338858)) (EMul (EInt 2) (ENeg (EInt 5)))))) (VString "400338858"))
  
  -- Test EInt
  , TestCase (assertEqual "positive integer" (A1_habinskw.evalExpr (EInt 1)) (VInt 1))
  , TestCase (assertEqual "testing zero" (A1_habinskw.evalExpr (EInt 0)) (VInt 0))
  , TestCase (assertEqual "negative integer" (A1_habinskw.evalExpr (EInt (-1))) (VInt (-1)))

  -- Test EString
  , TestCase (assertEqual "test sentance" (A1_habinskw.evalExpr (EString "hello world")) (VString "hello world"))
  , TestCase (assertEqual "test single character" (A1_habinskw.evalExpr (EString "c")) (VString "c"))
  , TestCase (assertEqual "test blank string" (A1_habinskw.evalExpr (EString "")) (VString ""))

    -- Test ECH
    , TestCase (assertEqual "test character" (A1_habinskw.evalExpr (ECh 'c')) (VString "c"))
    , TestCase (assertEqual "test space char" (A1_habinskw.evalExpr (ECh ' ')) (VString " "))

    -- Test EBool
    , TestCase (assertEqual "test true bool" (A1_habinskw.evalExpr (EBool True)) (VBool True))
    , TestCase (assertEqual "test false bool" (A1_habinskw.evalExpr (EBool False)) (VBool False))

    -- Test EAdd
    , TestCase (assertEqual "Adding two even numbers" (A1_habinskw.evalExpr (EAdd (EInt 1) (EInt 2))) (VInt 3))
    , TestCase (assertEqual "Adding an even and an odd" (A1_habinskw.evalExpr (EAdd (EInt 1) (EInt (-2)))) (VInt (-1)))
    , TestCase (assertEqual "Nested addition" (A1_habinskw.evalExpr (EAdd (EInt 1) (EAdd (EInt 2) (EInt 3)))) (VInt 6))
    , TestCase (assertEqual "Adding to a non integer" (A1_habinskw.evalExpr (EAdd (EInt 1) (EString "not an int"))) VError)
    
    -- Test EMul
    , TestCase (assertEqual "Multiplying two even numbers" (A1_habinskw.evalExpr (EMul (EInt 1) (EInt 2))) (VInt 2))
    , TestCase (assertEqual "Multiplying an even and an odd" (A1_habinskw.evalExpr (EMul (EInt 1) (EInt (-2)))) (VInt (-2)))
    , TestCase (assertEqual "Nested Multiplication" (A1_habinskw.evalExpr (EMul (EInt 1) (EMul (EInt 2) (EInt 3)))) (VInt 6))
    , TestCase (assertEqual "Multiplying to a non integer" (A1_habinskw.evalExpr (EMul (EInt 1) (EString "not an int"))) VError)

    -- Test ENeg
    , TestCase (assertEqual "negating an integer" (A1_habinskw.evalExpr (ENeg (EInt 3))) (VInt (-3)))
    , TestCase (assertEqual "negating a negative integer" (A1_habinskw.evalExpr (ENeg (EInt (-3)))) (VInt 3))
    , TestCase (assertEqual "nesting negation of an integer" (A1_habinskw.evalExpr (ENeg (ENeg (EInt 3)))) (VInt 3))
    , TestCase (assertEqual "negating a non integer" (A1_habinskw.evalExpr (ENeg (EBool False))) VError)

    -- Test ECat
    , TestCase (assertEqual "concatenation of two strings" (A1_habinskw.evalExpr (ECat (EString "hello ") (EString "world"))) (VString "hello world"))
    , TestCase (assertEqual "concatenation of char and string" (A1_habinskw.evalExpr (ECat (ECh 'h') (EString "askell"))) (VString "haskell"))
    , TestCase (assertEqual "concatenation of string and blank string" (A1_habinskw.evalExpr (ECat (EString "this is the end" ) (EString ""))) (VString "this is the end"))
    , TestCase (assertEqual "nested concatenation" (A1_habinskw.evalExpr (ECat (EString "hello ") (ECat (EString "world ") (EString "again")))) (VString "hello world again"))
    , TestCase (assertEqual "concatenation of string and non (string or char)" (A1_habinskw.evalExpr (ECat (EString "error") (EInt 2))) VError)
  
    -- Test Cons
    , TestCase (assertEqual "char cons string" (A1_habinskw.evalExpr (ECons 'h' (EString "ello world"))) (VString "hello world"))
    , TestCase (assertEqual "char cons char" (A1_habinskw.evalExpr (ECons 'm' (ECh 'i'))) (VString "mi"))
    , TestCase (assertEqual "char cons blank string" (A1_habinskw.evalExpr (ECons 'a' (EString ""))) (VString "a"))
    , TestCase (assertEqual "nested cons" (A1_habinskw.evalExpr (ECons 'a' (ECons 'b' (ECh 'c')))) (VString "abc"))
    , TestCase (assertEqual "char cons non (string or char)" (A1_habinskw.evalExpr (ECons 'e' (EInt 2))) VError)

    -- Test EAnd
    , TestCase (assertEqual "true and true" (A1_habinskw.evalExpr (EAnd (EBool True) (EBool True))) (VBool True))
    , TestCase (assertEqual "true and false" (A1_habinskw.evalExpr (EAnd (EBool True) (EBool False))) (VBool False))
    , TestCase (assertEqual "nested evaluations" (A1_habinskw.evalExpr (EAnd (EBool True) (EAnd (EBool True) (EBool False)))) (VBool False))
    , TestCase (assertEqual "false and false" (A1_habinskw.evalExpr (EAnd (EBool False) (EBool False))) (VBool False))
    , TestCase (assertEqual "true and non bool" (A1_habinskw.evalExpr (EAnd (EBool True) (EString "error"))) VError)

    -- Test EXOR
    , TestCase (assertEqual "true and true" (A1_habinskw.evalExpr (EXor (EBool True) (EBool True))) (VBool False))
    , TestCase (assertEqual "true and false" (A1_habinskw.evalExpr (EXor (EBool True) (EBool False))) (VBool True))
    , TestCase (assertEqual "nested evaluations" (A1_habinskw.evalExpr (EXor (EBool True) (EXor (EBool True) (EBool False)))) (VBool False))
    , TestCase (assertEqual "false and false" (A1_habinskw.evalExpr (EXor (EBool False) (EBool False))) (VBool False))
    , TestCase (assertEqual "true and non bool" (A1_habinskw.evalExpr (EXor (EBool True) (EString "error"))) VError)

    -- Test EIf
    , TestCase (assertEqual "condition is true" (A1_habinskw.evalExpr (EIf (EBool True) (EInt 1) (EInt 0))) (VInt 1))
    , TestCase (assertEqual "condition is false" (A1_habinskw.evalExpr (EIf (EBool False) (EInt 1) (EInt 0))) (VInt 0))
    , TestCase (assertEqual "condition is nested" (A1_habinskw.evalExpr (EIf (EIf (EBool True) (EBool False) (EBool True)) (EInt 1) (EInt 0))) (VInt 0))
    , TestCase (assertEqual "condition is non boolean" (A1_habinskw.evalExpr (EIf (EString "error") (EInt 1) (EInt 0))) VError)

    -- Test EShowInt
    , TestCase (assertEqual "show an int" (A1_habinskw.evalExpr (EShowInt (EInt 1))) (VString "1"))
    , TestCase (assertEqual "show a negative int" (A1_habinskw.evalExpr (EShowInt (EInt (-1)))) (VString "-1"))
    , TestCase (assertEqual "show a computed int" (A1_habinskw.evalExpr (EShowInt (EAdd (EInt 10) (EInt 1)))) (VString "11"))
    , TestCase (assertEqual "show a non int" (A1_habinskw.evalExpr (EShowInt (EString "error"))) VError)
    
    -- Extreme Cases
    -- Add Mul Neg
    , TestCase (assertEqual "multiply by 3, the addition of 2 and int negation of 5" (A1_habinskw.evalExpr (EMul (EInt 3) (EAdd (EInt 2) (ENeg (EInt 5))))) (VInt (-9)))
    , TestCase (assertEqual "adding the integer negation of 9 and (100 multiplied by the integer negation of 13)" (A1_habinskw.evalExpr (EAdd (ENeg (EInt 9)) (EMul (EInt 100) (ENeg (EInt 13))))) (VInt (-1309)))

    -- Cons Cat
    , TestCase (assertEqual "cons an H to the concatination of ello  and world" (A1_habinskw.evalExpr (ECons 'H' (ECat (EString "ello ") (EString "world")))) (VString "Hello world"))
    , TestCase (assertEqual "concatenation of Hello  and (orld again cons w)" (A1_habinskw.evalExpr (ECat (EString "Hello ") (ECons 'w' (EString "orld again")))) (VString "Hello world again"))
    
    -- And Xor If
    , TestCase (assertEqual "If (True Xor False) then (True And True) else (False And False)" (A1_habinskw.evalExpr (EIf (EXor (EBool True) (EBool False)) (EAnd (EBool True) (EBool True)) (EAnd (EBool False) (EBool False)))) (VBool True))
    , TestCase (assertEqual "If (If (False Xor False) then (True And True) else (False And True)) then (EInt 3) else (EAdd 5 10)" (A1_habinskw.evalExpr (EIf (EIf (EXor (EBool False) (EBool False)) (EAnd (EBool True) (EBool True)) (EAnd (EBool False) (EBool False))) (EInt 3) (EAdd (EInt 5) (EInt 10)))) (VInt 15))
    ]

-- Run the test suite
main :: IO ()
main = do
    counts <- runTestTT a1Tests
    if errors counts + failures counts == 0 then
      exitSuccess
    else
      exitFailure