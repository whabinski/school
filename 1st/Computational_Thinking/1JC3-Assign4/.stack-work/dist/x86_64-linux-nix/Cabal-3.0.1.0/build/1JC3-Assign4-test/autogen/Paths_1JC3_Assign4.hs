{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
module Paths_1JC3_Assign4 (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/bin"
libdir     = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/lib/x86_64-linux-ghc-8.8.4/1JC3-Assign4-0.1.0.0-KnsTBAY1AzAJ1bJX7MPGKJ-1JC3-Assign4-test"
dynlibdir  = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/lib/x86_64-linux-ghc-8.8.4"
datadir    = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/share/x86_64-linux-ghc-8.8.4/1JC3-Assign4-0.1.0.0"
libexecdir = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/libexec/x86_64-linux-ghc-8.8.4/1JC3-Assign4-0.1.0.0"
sysconfdir = "/home/dalvescb/Repos/1JC3_Spring2021/assign-templates/1JC3-Assign4/.stack-work/install/x86_64-linux-nix/283be8d7d199ff9d42779a800a1e41954b5d9ca8c9de1438b5c119fc88f16edd/8.8.4/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "1JC3_Assign4_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "1JC3_Assign4_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "1JC3_Assign4_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "1JC3_Assign4_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "1JC3_Assign4_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "1JC3_Assign4_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
