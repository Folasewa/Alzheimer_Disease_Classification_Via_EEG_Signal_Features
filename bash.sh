base_dir="data"
output_dirs=("filtered_subjects" "preprocessed_filtered" "epochs_overlap" "plots")
virtualenv venv
source venv/bin/activate

mkdir -p "$base_dir"
mkdir -p "model"

#creates data folders under "data"
for dir in "${output_dirs[@]}"; do
    mkdir -p "$base_dir/$dir"
done

brew install git-annex

cd data
pip install datalad-installer
datalad-installer git-annex -m datalad/packages
pip install datalad

datalad install https://github.com/OpenNeuroDatasets/ds004504.git

cd ds004504

datalad get .

echo "Folder structure setup complete!"
