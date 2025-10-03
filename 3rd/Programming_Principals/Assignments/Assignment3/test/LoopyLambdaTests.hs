-- |
module LoopyLambdaTests (lambdaTests) where

import A3.LoopyLambda

import Test.HUnit

-- | Helper for testing α-equivalence of expressions.
assertAlphaEqual :: String -> Maybe Expr -> Maybe Expr -> Assertion
assertAlphaEqual msg (Just e1) (Just e2) = assertBool msg (alphaEq e1 e2)
assertAlphaEqual _ Nothing Nothing = pure ()
assertAlphaEqual msg _ _ = assertFailure msg

-- 25 Tests
lambdaTests :: Test
lambdaTests = TestList
  [   
    -- Rule 1
    -- App (Var "x") (PlusOne Zero) To test when e1 cannot be reduced
    TestCase $ assertAlphaEqual "When neither e1 nor e2 can be reduced" (A3.LoopyLambda.stepLoop (App (Var "x") (PlusOne Zero))) (Nothing)
    -- App (App (Lam "x" (Var "x")) (Zero)) (Zero) - To test when e1 can be reduced and e2 cannot
    ,TestCase $ assertAlphaEqual "When e1 can be reduced and e2 cannot" (A3.LoopyLambda.stepLoop (App (App (Lam "x" (Var "x")) (Zero)) (Zero))) (Just (App Zero Zero))
    -- App Zero (App (Lam "x" (Var "x")) (Zero) - To test when e1 cannot be reduced and e2 can
    ,TestCase $ assertAlphaEqual "When e1 cannot be reduced and e2 can" (A3.LoopyLambda.stepLoop (App (App (Lam "x" (Var "x")) (Zero)) (Zero))) (Just (App Zero Zero))
    -- App (App (Lam "x" (Var "x")) (Zero) (Loop Zero (Var "x") (Var "y")) - To test when both e1 and e2 are reducable
    ,TestCase $ assertAlphaEqual "When e1 cannot be reduced and e2 can" (A3.LoopyLambda.stepLoop (App (App (Lam "x" (Var "x")) (Zero)) (Loop Zero (Var "x") (Var "y")))) (Just (App Zero (Loop Zero (Var "x") (Var "y"))))
    -- App (App (Loop Zero (Var "a") Zero) (Var "b")) (App (Loop Zero (Var "c") Zero) (Var "d")) - Test what is to be reduced first in a nested App when both e1s can be reduced
    ,TestCase $ assertAlphaEqual "Nested Apps such that all e1s can be reduced" (A3.LoopyLambda.stepLoop (App (App (Loop Zero (Var "a") Zero) (Var "b")) (App (Loop Zero (Var "c") Zero) (Var "d")))) (Just (App (App (Var "a") (Var "b")) (App (Loop Zero (Var "c") Zero) (Var "d"))))

    -- Rule 2 
    -- App /x.x Zero - To test if lambda susbstitution works with a basic case with a bound variable
    ,TestCase $ assertAlphaEqual "Lambda Substitution to a bound variable" (A3.LoopyLambda.stepLoop (App (Lam "x" (Var "x")) (Zero))) (Just Zero)
    -- App /z.x Zero - To test if lambda susbstitution works with a basic case with a free variable
    ,TestCase $ assertAlphaEqual "Lambda Substitution to a free variable" (A3.LoopyLambda.stepLoop (App (Lam "z" (Var "x")) (Zero))) (Just (Var "x"))
    -- App /x.xy Zero - To test lambda substitution to both a free and bound variable 
    ,TestCase $ assertAlphaEqual "Lambda Substitution to a free variable both free and bound variables" (A3.LoopyLambda.stepLoop (App (Lam "x" (App (Var "x") (Var "y"))) (Zero))) (Just (App Zero (Var "y")))
    -- App /x.xyx - To test lambda substitution to a bound variable that has multiple instances in the body
    ,TestCase $ assertAlphaEqual "Lambda Substitution to a multiple occurances of a bound variable" (A3.LoopyLambda.stepLoop (App (Lam "x" (App (App (Var "x") (Var "y")) (Var "x"))) (Zero))) (Just (App (App Zero (Var "y")) Zero))
    -- App /xy.xy = /x/y.xy - To test lambda substituion to curried lambda function
    ,TestCase $ assertAlphaEqual "Lambda Substitution to curried labda functions" (A3.LoopyLambda.stepLoop (App (Lam "x" (Lam "y" (App (Var "x") (Var "y")))) Zero)) (Just (Lam "y" (App Zero (Var "y"))))

    -- Rule 3
    -- PlusOne Zero - Nothing to reduce
    ,TestCase $ assertAlphaEqual "Nothing to Reduce" (A3.LoopyLambda.stepLoop (PlusOne Zero)) (Nothing)
    -- PlusOne (Loop Zero (Var "x") Zero) - Reduce once
    ,TestCase $ assertAlphaEqual "1 + e can be reduced to 1 + e'" (A3.LoopyLambda.stepLoop (PlusOne (Loop Zero (Var "x") Zero))) (Just (PlusOne (Var "x")))
    -- PlusOne PlusOne PlusOne PlusOne PlusOne PlusOne Zero - Lots of PlusOnes
    ,TestCase $ assertAlphaEqual "Lots of PlusOnes that aren't reducable" (A3.LoopyLambda.stepLoop (PlusOne (PlusOne (PlusOne (PlusOne (PlusOne (PlusOne Zero))))))) (Nothing)

    -- Rule 4
    -- Loop Zero Zero Zero - e1 cant be reduced
    ,TestCase $ assertAlphaEqual "When e cannot be reduced" (A3.LoopyLambda.stepLoop (Loop (Var "x") Zero Zero)) (Nothing)
    -- Loop (App (Lam "x" (Var "x")) (Zero)) Zero Zero - When e1 can be reduced
    ,TestCase $ assertAlphaEqual "When e can be reduced" (A3.LoopyLambda.stepLoop (Loop (App (Lam "x" (Var "x")) (Zero)) Zero Zero)) (Just (Loop Zero Zero Zero))
    -- Loop (Loop Zero (PlusOne Zero) (Loop (Var "x") Zero Zero)) Zero Zero - Test nested loops
    ,TestCase $ assertAlphaEqual "Nested Loops" (A3.LoopyLambda.stepLoop (Loop (Loop Zero (PlusOne Zero) (Loop (Var "x") Zero Zero)) Zero Zero)) (Just (Loop (PlusOne Zero) Zero Zero))

    -- Rule 5
    -- Loop Zero (Var "x") (Var "y") -- When e1 is Zero and e2 is not reducable
    ,TestCase $ assertAlphaEqual "When e1 is Zero and e2 is not reducable" (A3.LoopyLambda.stepLoop (Loop Zero (Var "x") (Var "y") )) (Just (Var "x"))
    -- Loop Zero (Loop Zero (Var "x") (Var "y")) (Var "x") -- When e1 is Zero and e2 is reducable (no differnce)
    ,TestCase $ assertAlphaEqual "When e1 is Zero and e2 is reducable" (A3.LoopyLambda.stepLoop (Loop Zero (Loop Zero (Var "x") (Var "y")) (Var "x"))) (Just (Loop Zero (Var "x") (Var "y") ))
    
    -- Rule 6
    -- Loop (PlusOne Zero) (Var "x") (Var "y") - When e1 is of the form 'PlusOne e' and e3 is not reducable
    ,TestCase $ assertAlphaEqual "When e1 is of the form PlusOne e and e3 is not reducable" (A3.LoopyLambda.stepLoop (Loop (PlusOne Zero) (Var "x") (Var "y"))) (Just (App (Var "y") (Loop Zero (Var "x") (Var "y"))))
    -- Loop (PlusOne Zero) (Var "x") (Loop Zero (Var "x") (Var "y")) -- When e1 is of the form 'PlusOne e' and e3 is reducable
    ,TestCase $ assertAlphaEqual "When e1 is of the form PlusOne e and e3 is reducable" (A3.LoopyLambda.stepLoop (Loop (PlusOne Zero) (Var "x") (Loop Zero (Var "x") (Var "y")))) (Just (App (Loop Zero (Var "x") (Var "y")) (Loop Zero (Var "x") (Loop Zero (Var "x") (Var "y")))))

    -- To test if a Var can be reduced
    ,TestCase $ assertAlphaEqual "When there is nothing to reduce for a Var" (A3.LoopyLambda.stepLoop (Var "x")) (Nothing)
    -- To test if a Zero can be reduced
    ,TestCase $ assertAlphaEqual "When there is nothing to reduce for a Zero" (A3.LoopyLambda.stepLoop Zero) (Nothing)

    -- Student Number (Last 2 didgits) : 58
    -- Loop Zero (PlusOne 57) (Var "x") -- Using a helper function 'numberToExpr' such that it computes the number 57 as 57 successors to Zero
    -- Using rule 5
    ,TestCase $ assertAlphaEqual "Reduced to PlusOne 57" (A3.LoopyLambda.stepLoop (Loop Zero A3.LoopyLambda.studentNumber (Var "x"))) (Just A3.LoopyLambda.studentNumber)
    -- /x.(PlusOne x) 57
    -- Using rule 2
    ,TestCase $ assertAlphaEqual "Lambda substitution to reduce to 58" (A3.LoopyLambda.stepLoop (App (Lam "x" (PlusOne (Var "x"))) (A3.LoopyLambda.numberToExpr 57 Zero))) (Just A3.LoopyLambda.studentNumber)
    -- (PlusOne (Loop Zero (PlusOne (PlusOne (PlusOne (PlusOne 53)))) Zero))
    -- Using rule 3
    ,TestCase $ assertAlphaEqual "PlusOne e reduction" (A3.LoopyLambda.stepLoop (PlusOne (Loop Zero (PlusOne (PlusOne (PlusOne (PlusOne (A3.LoopyLambda.numberToExpr 53 Zero))))) Zero))) (Just A3.LoopyLambda.studentNumber)
    ]