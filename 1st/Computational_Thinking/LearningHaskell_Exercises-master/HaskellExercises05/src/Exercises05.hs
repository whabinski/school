{-|
Module      : HaskellExercises05.Exercises05
Copyright   :  (c) Curtis D'Alves 2020
License     :  GPL (see the LICENSE file)
Maintainer  :  none
Stability   :  experimental
Portability :  portable

Description:
  Haskell exercise template Set 05 - McMaster CS 1JC3 2021
-}
module Exercises05 where

import Prelude hiding (take,drop,replicate,(!!),elem,and,or)

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
macid = "habinskw"

-- Exercise A
-----------------------------------------------------------------------------------------------------------
-- Implement the function split that takes a list and splits it in half and returns a tuple of the
-- two halves, WITHOUT USING TAKE / DROP
-- NOTE when the list is uneven, the second list is one element larger than the first
-- NOTE^2 when using take / drop, although convenient, you introduce redundant computation. A more
--        efficient implementation of this function can be done calling an auxilary function with
--        different parameters that recurses through the list directly
-----------------------------------------------------------------------------------------------------------
split :: [a] -> ([a],[a])
split xs = 
  let
    half = length xs `div` 2
    split' xs ys 0 = ((reverse xs),ys)
    split' xs (y:ys) n = split' (y:xs) ys (n-1)
    split' xs [] n = (xs,[])
   in split' [] xs half


-- Exercise B
-----------------------------------------------------------------------------------------------------------
-- Implement the function merge that takes two lists that (assuming both lists are already sorted) merges
-- them together in to a sorted list
-----------------------------------------------------------------------------------------------------------
merge :: (Ord a) => [a] -> [a] -> [a]
merge [] ys = ys
merge xs [] = xs
merge (x:xs) (y:ys)
  | x <= y = x : merge xs (y:ys)
  | otherwise = y : merge (x:xs) ys

-- Exercise C
-----------------------------------------------------------------------------------------------------------
-- Implement the function mergeSort that sorts a list by recusively splitting a list, and merging
-- the sorted lists back together
-- NOTE singleton and empty lists are already sorted
-----------------------------------------------------------------------------------------------------------
mergeSort :: (Ord a) => [a] -> [a]
mergeSort [] = []
mergeSort [x] = [x]
mergeSort xs = merge (mergeSort (fst(split xs))) (mergeSort (snd(split xs)))

-- Exercise D
-----------------------------------------------------------------------------------------------------------
-- Implement the function sortProp that tests if a list is sorted or not
-- NOTE you can use this with QuickCheck to test your mergSort function by calling
--      quickCheck (sortProp . mergeSort)
-----------------------------------------------------------------------------------------------------------
sortProp :: (Ord a) => [a] -> Bool
sortProp [] = True
sortProp [x] = True
sortProp (x:x1:xs)
  | x <= x1 = sortProp (x1:xs)
  | otherwise = False

-- Exercise E
-----------------------------------------------------------------------------------------------------------
-- Implement the Prelude function replicate that takes an Int n and a element and returns a list that
-- replicates that element n times
-----------------------------------------------------------------------------------------------------------
replicate :: Int -> a -> [a]

replicate 0 x = []
replicate n x
  | n < 0 = []
  | otherwise = replicate (n-1) x ++ [x]


-- Exercise F
-----------------------------------------------------------------------------------------------------------
-- Implement the Prelude function !! that selects the nth element of a list using recursion
-- NOTE throw an error when indexing out of bounds
-----------------------------------------------------------------------------------------------------------
(!!) :: [a] -> Int -> a
(!!) [] 0 = error "Out of bounds"
(!!) (x:xs) 0 = x
(!!) (x:xs) n = (!!) xs (n-1)

-- Exercise G
-----------------------------------------------------------------------------------------------------------
-- Implement the Prelude function elem that takes a value and a list and returns True if the value
-- is an element of the list
-----------------------------------------------------------------------------------------------------------
elem :: (Eq a) => a -> [a] -> Bool
elem e [] = False
elem e (x:xs)
  | e == x = True
  | otherwise = elem e xs
