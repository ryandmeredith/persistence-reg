{
  inputs = {
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      flake-parts,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = lib.systems.flakeExposed;
        perSystem =
          { pkgs, ... }:
          {
            devShells.default =
              let
                python = pkgs.python3;
                dependencies = with python.pkgs; [
                  ipython
                  python-lsp-server
                  keras
                  gudhi
                ];
                tools = with pkgs; [
                  python
                  ruff
                  ty
                ];
              in
              pkgs.mkShell { packages = tools ++ dependencies; };
          };
      }
    );
}
