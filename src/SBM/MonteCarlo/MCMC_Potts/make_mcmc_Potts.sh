moduleName="MonteCarlo_Potts"
ext="$(python3-config --extension-suffix)"

file="$moduleName$ext"
dir="build/$(ls build | grep lib)"

python3 src/SBM/MonteCarlo/MCMC_Potts/setup_mcmc_Potts.py build_ext --inplace -v
rm $file
ln -s $dir/$file $file