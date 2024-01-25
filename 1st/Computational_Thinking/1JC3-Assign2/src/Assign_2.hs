{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-|
Module      : 1JC3-Assign2.Assign_2.hs
Copyright   :  (c) Curtis D'Alves 2021
License     :  GPL (see the LICENSE file)
Maintainer  :  Wyatt Habinski
Stability   :  experimental
Portability :  portable

Description:
  Assignment 2 - McMaster CS 1JC3 2021
-}
module Assign_2 where

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

-- Name: Wyatt Habinski
-- Date: October 26 2021
macid :: String
macid = "habinskw"

type Vector3D = (Double,Double,Double)

{- -----------------------------------------------------------------
 - getX
 - -----------------------------------------------------------------
 - Description:
    
 - getX takes a vector value as input and outputs just the x value component 
 - of the vector as double

 -}
getX :: Vector3D -> Double
getX (x,y,z) = x

{- -----------------------------------------------------------------
 - getY
 - -----------------------------------------------------------------
 - Description:
 
 - getY takes a vector value as input and outputs just the y value component 
 - of the vector as double

 -}
getY :: Vector3D -> Double
getY (x,y,z) = y

{- -----------------------------------------------------------------
 - getZ
 - -----------------------------------------------------------------
 - Description:
 
 - getZ takes a vector value as input and outputs just the z value component 
 - of the vector as double

 -}
getZ :: Vector3D -> Double
getZ (x,y,z) = z

{- -----------------------------------------------------------------
 - scalarMult
 - -----------------------------------------------------------------
 - Description:
 
 - sclarMult takes a scalar constant as a double and a vector value as inputs
 - and outputs a new vector value where each x,y,z component is multiplied by 
 - the scalar

 -}
scalarMult :: Double -> Vector3D -> Vector3D
scalarMult s v = (newX,newY,newZ)
  where 
    newX = s*getX v
    newY = s*getY v
    newZ = s*getZ v


{- -----------------------------------------------------------------
 - add
 - -----------------------------------------------------------------
 - Description:
 
 - add takes two vector values as inputs and outputs the new vector value
 - which is the addition of both x values, both y values, and both z values 
 - to make the new vector

 -}
add :: Vector3D -> Vector3D -> Vector3D
add v0 v1 = (newX, newY, newZ)
  where 
    newX = getX v0 + getX v1
    newY = getY v0 + getY v1
    newZ = getZ v0 + getZ v1

{- -----------------------------------------------------------------
 - innerProduct
 - -----------------------------------------------------------------
 - Description:
 -   TODO add comments
 -}
innerProduct :: Vector3D -> Vector3D -> Double
innerProduct v0 v1 = prodX + prodY + prodZ
  where
    prodX = getX v0 * getX v1
    prodY = getY v0 * getY v1
    prodZ = getZ v0 * getZ v1

{- -----------------------------------------------------------------
 - distance
 - -----------------------------------------------------------------
 - Description:
 
 - distance takes 2 vector values as inputs and computes the distance between
 - as output. The value outputed will be a double

 - The distance computed is the square root of ((x1-x2)^2 + (y1 - y2)^2 + (z1- z2)^2)
 
 - I first compute the new individual components by taking the component from the 
 - first vector and subtract it from the coresponding component in the second and 
 - squaring the difference

 - once I have computed all 3 components, I square root the sum to get value that 
 - corresponds to the distance between the two vectors

 -}
distance :: Vector3D -> Vector3D -> Double
distance v1 v2 = sqrt(newX + newY + newZ)
  where
    newX = (getX v1 - getX v2)^2
    newY = (getY v1 - getY v2)^2
    newZ = (getZ v1 - getZ v2)^2

{- ------------------------------------------------------------------------
 - maxDistance
 - ------------------------------------------------------------------------
 - Description:
 
 - maxDistance takes a list of vectors as input and scans through a list of vectors and outputs
 - the vector with the largest distance from the origin (0,0,0).

 - the base cases of nothing in the list [] and 1 element in the list [v] are first defined
 - then I recursively iterate through rest of the loop determine what is the largest distance

 - I use list splicing to retrieve the first 2 elements in the list in order to compare them.
 - if the first element that is spliced is larger or equal to the second then recursively go through 
 - the list again using that first vector to compare the rest of the list. otherwise use the second
 - element to recursively go through the list

 - the output is a vector with the largest distance from the origin

 -}
maxDistance :: [Vector3D] -> Vector3D
maxDistance []  = (0,0,0)
maxDistance [v] = v
maxDistance (v:v2:vs)
  | distance v (0,0,0) >= distance v2 (0,0,0) = maxDistance (v:vs)
  | otherwise = maxDistance (v2:vs)
  



{-
------------------TESTING----------------------

FUNCTION: scalarMult
TEST CASE NUMBER: 1
INPUT: 2.0 (1,1,1)
EXPECTED OUTPUT: (2.0,2.0,2.0)
ACTUAL OUTPUT: (2.0,2.0,2.0)

FUNCTION: scalarMult
TEST CASE NUMBER: 2
INPUT: 5.0 (1,2,3)
EXPECTED OUTPUT: (5.0,10.0,15.0)
ACTUAL OUTPUT: (5.0,10.0,15.0)

FUNCTION: scalarMult
TEST CASE NUMBER: 3
INPUT: 7.5 (-3,5,-7)
EXPECTED OUTPUT: (-22.5,37.5,-52.5)
ACTUAL OUTPUT: (-22.5,37.5,-52.5)

------------------------------------------------

FUNCTION: add
TEST CASE NUMBER: 1
INPUT: (1,1,1) (2,2,2)
EXPECTED OUTPUT: (3,3,3)
ACTUAL OUTPUT: (3.0,3.0,3.0)

FUNCTION: add
TEST CASE NUMBER: 2
INPUT: (1,2,3) (4,5,6)
EXPECTED OUTPUT: (5,7,9)
ACTUAL OUTPUT: (5.0,7.0,9.0)

FUNCTION: add
TEST CASE NUMBER: 3
INPUT: (-5,32,9) (12,-24,14)
EXPECTED OUTPUT: (7,8,23)
ACTUAL OUTPUT: (7.0,8.0,23.0)

------------------------------------------------

FUNCTION: innerProduct
TEST CASE NUMBER: 1
INPUT: (1,1,1) (2,2,2)
EXPECTED OUTPUT: 6.0
ACTUAL OUTPUT: 6.0

FUNCTION: innerProduct
TEST CASE NUMBER: 2
INPUT: (1.2,3.4,4.5) (6,7,8)
EXPECTED OUTPUT: 67.0
ACTUAL OUTPUT: 

FUNCTION: innerProduct
TEST CASE NUMBER: 3
INPUT: (-3.1,8.0,-5.2) (4.1,-9.3,-2.0)
EXPECTED OUTPUT: - 76.71
ACTUAL OUTPUT: -76.71

------------------------------------------------

FUNCTION: distance
TEST CASE NUMBER: 1
INPUT: (1,2,3) (4,5,6)
EXPECTED OUTPUT: sqrt(27)
ACTUAL OUTPUT: sqrt(27) = 5.2

FUNCTION: distance
TEST CASE NUMBER: 2
INPUT: (-4,8,-5) (4,9,-3)
EXPECTED OUTPUT: sqrt(11)
ACTUAL OUTPUT: sqrt(69) = 8.3

FUNCTION: distance
TEST CASE NUMBER: 3
INPUT: (-7.2,-9.4,11) (1.2,-4.3,6.7) 
EXPECTED OUTPUT: sqrt(115.06)
ACTUAL OUTPUT: sqrt(115.06) = 10.7

------------------------------------------------

FUNCTION: maxDistance
TEST CASE NUMBER: 1
INPUT: [(1,2,3),(4,5,6),(7,8,9)]
EXPECTED OUTPUT: (7,8,9)
ACTUAL OUTPUT: (7.0,8.0,9.0)

FUNCTION: maxDistance
TEST CASE NUMBER: 2
INPUT: []
EXPECTED OUTPUT: (0.0,0.0,0.0) 
ACTUAL OUTPUT: (0.0,0.0,0.0)

FUNCTION: maxDistance
TEST CASE NUMBER: 3
INPUT: [(-100,-100,-100),(-100,100,100),(100,-100,100),(100,100,-100),(-100,-100,100),(-100,100,-100),(100,-100,-100),(100,100,100)]
EXPECTED OUTPUT: (-100.0,-100.0,-100.0)
ACTUAL OUTPUT: (-100.0,-100.0,-100.0)

------------------------------------------------

-}