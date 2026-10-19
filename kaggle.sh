mkdir -p kaggle/casmi26-inference-assets/project
#mkdir -p kaggle/casmi26-inference-assets/MIST

cd /home/colleen/PycharmProjects/Envada-CASMI-2026

cp -a scripts kaggle/casmi26-inference-assets/project/
cp -a src kaggle/casmi26-inference-assets/project/

mkdir -p kaggle/casmi26-inference-assets/project/data/processed
cp -a data/processed/formula_index \
      kaggle/casmi26-inference-assets/project/data/processed/

cp -a data/processed/structure_index \
      kaggle/casmi26-inference-assets/project/data/processed/

cp -a data/processed/retrieval_index \
      kaggle/casmi26-inference-assets/project/data/processed/

if [ ! -d data/processed/pubchem_formula_index ]; then
  echo "Missing data/processed/pubchem_formula_index"
  echo "Run:"
  echo "  python scripts/build_pubchem_formula_candidates.py --source formula-index"
  echo "  python scripts/build_pubchem_inference_asset.py"
  exit 1
fi

cp -a data/processed/pubchem_formula_index \
      kaggle/casmi26-inference-assets/project/data/processed/

mkdir -p kaggle/casmi26-inference-assets/project/data/external

cp -a data/external/coconut \
      kaggle/casmi26-inference-assets/project/data/external/

#cp -a /home/colleen/PycharmProjects/MIST/src kaggle/casmi26-inference-assets/MIST/

#mkdir -p kaggle/casmi26-inference-assets/MIST/pretrained_models

#cp /home/colleen/PycharmProjects/MIST/pretrained_models/mist_contrastive_canopus_pretrain.ckpt \
#   kaggle/casmi26-inference-assets/MIST/pretrained_models/

#mkdir -p kaggle/casmi26-inference-assets/wheelhouse

#cd kaggle/casmi26-inference-assets/wheelhouse

#python -m pip download \
#  --only-binary=:all: \
#  --platform manylinux_2_28_x86_64 \
#  --python-version 313 \
#  --implementation cp \
#  --abi cp313 \
#  --no-deps \
#  rdkit==2026.9.1

#cd ../../../

kaggle datasets version \
  -m "update" \
  -p kaggle/casmi26-inference-assets \
  -t \
  -r zip