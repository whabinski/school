module A3.SKI where

-- * Question 1: Revenge of the Goblins and Gnomes

data SKI
    = S
    | K
    | I
    | App SKI SKI
    deriving (Show, Eq)

ski :: SKI -> Maybe SKI
ski S = Nothing -- No reduction possible with 0 arguments to S
ski K = Nothing -- No reduction possible with 0 arguments to K
ski I = Nothing -- No reduction possible with 0 arguments to I

ski (App (App (App S x) y) z) = Just (App (App x z) (App y z)) -- Reduce S with 3 arguments (3rd rule)
ski (App (App K x) _) = Just x -- Reduce K with 2 arguments (4th rule)
ski (App I x) = Just x -- Reduce I with 1 argument (5th rule)

ski (App f g) = -- 1st rule
    case ski f of -- Check if left expression can be reduced
        Just resultf -> Just (App resultf g) -- If it can be reduced then return with reduced step
        Nothing -> -- 2nd rule
            case ski g of -- If first argument is fully reduced, check if second argument can be reduced 
                Just resultg -> Just (App f resultg) -- If it can be reduced then return with reduced step
                Nothing -> Nothing -- Otherwise no further reduction is possible