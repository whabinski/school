{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_GHC -Wno-overlapping-patterns #-}
{-|
Module      : HaskellExercises03.Exercises03
Copyright   :  (c) Curtis D'Alves 2020
License     :  GPL (see the LICENSE file)
Maintainer  :  Wyatt Habinski
Stability   :  experimental
Portability :  portable
Date        :  October 4, 2021

Description:
  Haskell exercise template Set 03 - McMaster CS 1JC3 2021
-}
module Exercises03 where

-----------------------------------------------------------------------------------------------------------
-- INSTRUCTIONS              README!!!
-----------------------------------------------------------------------------------------------------------
-- 1) DO NOT DELETE/ALTER ANY CODE ABOVE THESE INSTRUCTIONS
-- 2) DO NOT REMOVE / ALTER TYPE DECLERATIONS (I.E. THE LINE WITH THE :: ABOUT THE FUNCTION DECLERATION)
--    OR DATA DECLERATIONS (I.E. data Tree a = ...)
--    IF YOU ARE UNABLE TO COMPLETE A FUNCTION, LEAVE IT'S ORIGINAL IMPLEMENTATION (I.E. THROW AN ERROR)
-- 3) MAKE SURE THE PROJECT COMPILES (I.E. RUN STACK BUILD AND MAKE SURE THERE ARE NO ERRORS) BEFORE
--    SUBMITTING, FAILURE TO DO SO WILL RESULT IN A MARK OF 0
-- 4) REPLACE macid = "TODO" WITH YOUR ACTUAL MACID (EX. IF YOUR MACID IS jim THEN macid = "jim")
-----------------------------------------------------------------------------------------------------------
macid = "habinskw"

-- Exercise A
-----------------------------------------------------------------------------------------------------------
-- Implement a function that computes the nth fibanacci number (see https://en.wikipedia.org/wiki/Fibonacci_number)
-- NOTE if the input is less than 0, simply return 0
-----------------------------------------------------------------------------------------------------------
fib :: (Integral a) => a -> a
fib n
  | n <= 0 = 0
  | n == 1 = 1
  | otherwise = fib (n-1) + fib(n-2)

-- Exercise B
-----------------------------------------------------------------------------------------------------------
-- Using the given Tree data type, encode the following tree in Haskell
--                d
--              /   \
--             b     f
--            / \   / \
--           a   c e   g
-----------------------------------------------------------------------------------------------------------
data Tree a = Node a (Tree a) (Tree a)
            | Leaf a
   deriving (Show,Eq)

exTree :: Tree Char
exTree = Node 'd' (Node 'b' (Leaf 'a') (Leaf 'c')) (Node 'f' (Leaf 'e') (Leaf 'g'))

-- Exercise C
-----------------------------------------------------------------------------------------------------------
-- Implement a function that multiplies two numbers encoded using the following Nat data type
-- NOTE start by creating an function add :: Nat -> Nat -> Nat
-----------------------------------------------------------------------------------------------------------

data Nat = Zero | Succ Nat
         deriving (Show,Eq)

add :: Nat -> Nat -> Nat
add Zero n = n
add (Succ m) n = Succ (add m n)


mult :: Nat -> Nat -> Nat
mult Zero _ = Zero
mult (Succ m) n = add n (mult n m)


-- Exercise D
-----------------------------------------------------------------------------------------------------------
-- Implement a function that determines whether or not a given tree is a search tree (is a "sorted" tree)
-- NOTE: this means for every Node v t0 t1, v is >= all values in tree t0 and v is < all values in tree t1
--       start by defining functions that test if a value is less/greater than all the values in a tree
----------------------------------------------------------------------------------

flattenTree :: Tree a -> [a]
flattenTree (Leaf n) = [n]
flattenTree (Node n t1 t2) = flattenTree t1 ++ [n] ++ flattenTree t2

isSortedList :: (Ord a) => [a] -> Bool
isSortedList [] = True
isSortedList [x] = True
isSortedList (x:y:xs) = (x <= y) && isSortedList (y:xs)

isSearchTree :: Ord a => Tree a -> Bool
isSearchTree tree = isSortedList $ flattenTree tree

-- Exercise E
-----------------------------------------------------------------------------------------------------------
-- Implement a function factors that takes an Int and returns a list off all its factors (i.e. all the
-- Int's bigger than 1 and less than that Int that are divisable without a remainder)
-- NOTE using list comprehension gives a pretty neat solution
-----------------------------------------------------------------------------------------------------------
factors :: Int -> [Int]
factors n = [x | x <- [1..(n-1)], n `mod` x == 0]

-- Exercise F
-----------------------------------------------------------------------------------------------------------
-- Implement a function pivot that takes a value and a list, then returns two lists in a tuple, with the
-- first list being all elements <= to the value, and the second list being all elements > the value
-- I.e.  pivot 3 [5,6,4,2,1,3]
--         = ([2,1,3],[5,6,4])
-- NOTE using list comprehensions gives a pretty neat solution
-----------------------------------------------------------------------------------------------------------

pivot :: Ord a => a -> [a] -> ([a],[a])
pivot v xs = (x,y)
              where
                x = [a | a <- xs, v >= a]
                y = [b | b <- xs, v < b]
