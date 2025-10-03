{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-|
Module      : 1JC3-Assign1.Assign_1.hs
Copyright   :  (c) Curtis D'Alves 2021
License     :  GPL (see the LICENSE file)
Maintainer  :  Wyatt Habinski
Stability   :  experimental
Portability :  portable
Date        : September 28, 2021

Description:
  Assignment 1 - McMaster CS 1JC3 2021
-}
module Assign_1 where

import Prelude hiding (sin, cos,tan)

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
macid :: String
macid = "habinskw"



{- -----------------------------------------------------------------
 - factorial
 - -----------------------------------------------------------------
 - Description:
 -    Computes the factorial of any Integer n
 - -----------------------------------------------------------------
 - |   Input     |                                                 |
 - |      n      | Integer input                                   |
 - -----------------------------------------------------------------
 - |   Output    |                                                 |
 - |      n <= 1 | 1                                               |
 - |      n >  1 | n * (n-1) ...  * 1   while (n-k) > 0            |
 - -----------------------------------------------------------------
 -}
factorial :: Integer -> Integer
factorial n = if n > 0
              then n * factorial (n-1)
              else 1

{- ------------------------------------------------------------------
 - sinTaylor
 - ------------------------------------------------------------------
 - Description:
 -   Using the 4th Taylor approximation of Sin(x) equation (f^k(a) / k!) * (x-a)**k
 -   Adding 5 iterations we have k ranging from 0 - 4
 -   f^k (where 0 <= k <= 4) we have sina, cosa, -sina, -cosa, and sina
 -   After inputting x, a, sin_a, and cos_a we add all iterations to achieve the output
 -}
sinTaylor :: Double -> Double -> Double -> Double -> Double
sinTaylor a cos_a sin_a x = (sin_a / fromIntegral (factorial 0))*((x - a)**0)
                          + (cos_a / fromIntegral (factorial 1))*((x - a)**1)
                          + (-sin_a / fromIntegral (factorial 2))*((x - a)**2)
                          + (-cos_a / fromIntegral (factorial 3))*((x - a)**3)
                          + (sin_a / fromIntegral (factorial 4))*((x - a)**4)

{- -----------------------------------------------------------------
 - fmod
 - -----------------------------------------------------------------
 - Description:
 - inputs 2 doubles to calculate and returns a double
 - modulus calculations involve 1st num - (1st num / 2nd num) when nums are integers
 - in the case of doubles, you need to round down the (1st num / 2 num) and from that change it back to a double
 - 
 - let z equal the double of (1st num / 2nd num)
 - then subract (z * 2nd num) from 1st num to find the remainder
 -}
fmod :: Double -> Double -> Double
fmod x y =
  let
    -- z is the largest integer s.t. z*y <= x
    -- HINT use floating point division, then round down
    z = fromIntegral(floor (x / y))
  in x - z*y

{- ----------------------------------------------------------------------
 - sinApprox
 - ----------------------------------------------------------------------
 - Description:
 -   Checks input to see what domain it is apart of using guard statements
 -   Once appropriate domain is found, it calls on sinTaylor using new values for a, cos_a, sin_a, and x using data that was given via chart
 -   Output would be sinTaylor at the given input
 -}

sinApprox :: Double -> Double
sinApprox x
          | con1 = sinTaylor 0 1 0 y
          | con2 = sinTaylor (pi/2) 0 1 y
          | con3 = sinTaylor pi (-1) 0 y
          | con4 = sinTaylor (3*pi/2) 0 (-1) y
          | con5 = sinTaylor (2*pi) 1 0 y
          | otherwise = error "help"
          where 
                y    = fmod x (2*pi)
                con1 = (y >= 0)         &&   (y < (pi/4))
                con2 = (y >= (pi/4))    &&   (y < (3*pi/4))
                con3 = (y >= (3*pi/4))  &&   (y < (5*pi/4))
                con4 = (y >= (5*pi/4))  &&   (y < (7*pi/4))
                con5 = (y >= (7*pi/4))  &&   (y < (2*pi))

{- ---------------------------------------------------------------------
 - sinApprox
 - ---------------------------------------------------------------------
 - Description:
 -   CosApprox will output the negation of sinApprox at the given input minus pi/2
 -   We call on the function sinApprox at the value of x - pi/2 for the output
 -}
cosApprox :: Double -> Double
cosApprox x = -sinApprox (x - (pi/2))

{- ---------------------------------------------------------------------
 - tanApprox
 - ---------------------------------------------------------------------
 - Description:
 -   tanApprox is the function of sinApprox / cosApprox at the given value x 
 -   x is inputed and sinApprox and cosApprox are both called on for the output
 -}
tanApprox :: Double -> Double
tanApprox x = sinApprox x/ cosApprox x

