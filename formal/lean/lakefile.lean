import Lake
open Lake DSL

package «ZipperGenFormal» where

@[default_target]
lean_lib «ZipperGenFormal» where
  srcDir := "."
  roots := #[`ZipperGenParallel, `ZipperGenSpawn]
  globs := #[
    .submodules `MSCAgents,
    .submodules `ZipperGenParallel,
    .submodules `ZipperGenSpawn
  ]
