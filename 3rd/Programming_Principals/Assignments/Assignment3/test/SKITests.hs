-- | Tests for question 1
module SKITests (skiTests) where

import A3.SKI
import Test.HUnit

-- 15 Test Cases
skiTests :: Test
skiTests = TestList [

     TestCase $ assertEqual "ski SKI" (A3.SKI.ski (App (App S K) I)) (Nothing) -- S with 2 arguments
    ,TestCase $ assertEqual "ski SK" (A3.SKI.ski (App S K)) (Nothing) -- S with 1 argument
    ,TestCase $ assertEqual "ski S" (A3.SKI.ski S) (Nothing) -- S with 0 arguments

    ,TestCase $ assertEqual "ski KI" (A3.SKI.ski (App K I)) (Nothing) -- K with 1 argument
    ,TestCase $ assertEqual "ski K" (A3.SKI.ski K) (Nothing) -- K with 0 arguments

    ,TestCase $ assertEqual "ski I" (A3.SKI.ski I) (Nothing) -- I with 0 arguments

    ,TestCase $ assertEqual "ski SIII" (A3.SKI.ski (App (App (App S I) I) I)) (Just (App (App I I) (App I I))) -- Test when S has 3 arguments
    ,TestCase $ assertEqual "ski SI(II)" (A3.SKI.ski (App S (App I (App I I)))) (Just (App S (App I I))) -- Test when S cant be reduced, if its second argument can be
    ,TestCase $ assertEqual "ski S(II)I" (A3.SKI.ski (App S (App (App I I) I))) (Just (App S (App I I))) -- Test when S cant be reduced, if its first argument can be

    ,TestCase $ assertEqual "ski KSI" (A3.SKI.ski (App (App K S) I)) (Just S) -- Test when K has 2 arguments
    ,TestCase $ assertEqual "ski KSI" (A3.SKI.ski (App (App K S) I)) (Just S) -- Test when K cant be reduced, if its argument can

    ,TestCase $ assertEqual "ski IS" (A3.SKI.ski (App I S)) (Just S) -- Test when I has an argument

    ,TestCase $ assertEqual "ski IS" (A3.SKI.ski (App (App (App S K) I) (App (App K I) S))) (Just (App (App K (App (App K I) S)) (App I (App (App K I) S)))) -- Test
    ,TestCase $ assertEqual "ski IS" (A3.SKI.ski (App (App K S) (App I (App (App (App S K) S) I)))) (Just S)-- Test
    ,TestCase $ assertEqual "ski IS" (A3.SKI.ski (App (App (App S K) I) K)) (Just (App (App K K) (App I K)))-- Test
    ]
