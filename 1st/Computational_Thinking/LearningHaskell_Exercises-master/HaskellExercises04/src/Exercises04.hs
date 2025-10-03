{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-overlapping-patterns #-}
{-|
Module      : HaskellExercises04.Exercises04
Copyright   :  (c) Curtis D'Alves 2020
License     :  GPL (see the LICENSE file)
Maintainer  :  Wyatt Hainski
Date        :  October 18
Stability   :  experimental
Portability :  portable

Description:
  Haskell exercise template Set 04 - McMaster CS 1JC3 2021
-}
module Exercises04 where

import Prelude hiding (zip,take,drop)

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
-- Implement the Prelude function zip that takes two lists and combines them into a list of pairs
-- E.x.
--  zip [1,2,3] ['a','b'] == [(1,'a'),(2,'b')]
-- NOTE zip short circuits on the shortest list
-----------------------------------------------------------------------------------------------------------
zip :: [a] -> [b] -> [(a,b)]
zip [] _ = []
zip _ [] = []
zip [x] [y] = [(x,y)]
zip (x:xs) (y:ys) = (x,y) : zip xs ys

-- Exercise B
-----------------------------------------------------------------------------------------------------------
-- Implement the function mapWithIndex, that works just like map, however the function it "maps" takes
-- a tuple containing the index of each element of the list
-- E.x.
--  mapWithIndex (\(idx,x) -> idx + x) [0,0,0] == [0,1,2]
-- NOTE zip [0..] xs   creates a list of (index,element) tuples
-----------------------------------------------------------------------------------------------------------

mapWithIndex :: ((Int,a) -> b) -> [a] -> [b]
mapWithIndex f xs = map f xs'
  where xs' = zip[0..] xs

-- Exercise C
-----------------------------------------------------------------------------------------------------------
-- Implement the function mySum that works just like the Prelude function sum that adds up all the elements
-- in a list, but works on the following custom List data type instead
-----------------------------------------------------------------------------------------------------------
data List a = Cons a (List a)
            | Nil
  deriving (Show,Eq)

mySum :: (Num a) => List a -> a
mySum Nil = 0
mySum (Cons x xs) = x + mySum xs

-- Exercise D
-----------------------------------------------------------------------------------------------------------
-- Implement the operator +++ that works just like the Prelude operator ++ that puts two list together
-- (i.e. concatenates them), but works on the above custom List data type instead
-----------------------------------------------------------------------------------------------------------
(+++) :: List a -> List a -> List a
(+++) Nil ys = ys
(+++) (Cons x xs) ys = Cons x ((+++) xs ys)

-- Exercise E
-----------------------------------------------------------------------------------------------------------
-- Implement the function myReverse that works just like the Prelude function reverse that reverses the
-- order of all elements in a list, but works on the above custom List data type instead
-----------------------------------------------------------------------------------------------------------
myReverse :: List a -> List a
myReverse Nil = Nil
myReverse (Cons x xs) = (+++) (myReverse xs) (Cons x Nil)

-- Exercise F
-----------------------------------------------------------------------------------------------------------
-- Implement the function treeSum that works just like the Prelude function sum that adds up all the
-- elements in a list, but works on the following custom Tree data type instead
-----------------------------------------------------------------------------------------------------------
data Tree a = Node a (Tree a) (Tree a)
            | Empty
  deriving (Show,Eq)

treeSum :: Num a => Tree a -> a
treeSum Empty = 0
treeSum (Node a l r) = a + treeSum l + treeSum r


-- Exercise G
-----------------------------------------------------------------------------------------------------------
-- Implement the function treeHeight that returns the largest height of a Tree
-- E.x.
--              a
--            /  \
--           b    c
--               / \
--              d   e
-- has a height of 3 (elements d and e are both at "height" 3 in the tree)
-- NOTE the Empty Tree is of height 0
-----------------------------------------------------------------------------------------------------------
treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node a Empty Empty) = 1
treeHeight (Node a l r) = if treeHeight l + 1 > treeHeight r + 1 then treeHeight l + 1 else treeHeight r + 1
treeHeight (Node a l _) = treeHeight l + 1
treeHeight (Node a _ r) = treeHeight r + 1

-- Exercise H
-----------------------------------------------------------------------------------------------------------
-- Implement the Prelude functions take and drop that take / drop the first n elements of a list
-----------------------------------------------------------------------------------------------------------
take :: Int -> [a] -> [a]
take 0 _ = []
take _ [] = []
take n (x:xs) = x : take (n-1) xs

drop :: Int -> [a] -> [a]
drop 0 xs = xs
drop _ [] = []
drop n (x:xs) = drop (n-1) xs

-- Extra Challenge
-----------------------------------------------------------------------------------------------------------
-- Try and create a QuickCheck property that tests take and drop in combination
-- See the function takeDropProp in app/Main.hs for a solution
-----------------------------------------------------------------------------------------------------------
