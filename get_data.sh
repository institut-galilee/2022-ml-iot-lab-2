#!/bin/bash

data_dir=./data/

# This list corresponds to the files used during 2020 SHL challenge
# lst=(
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_torso.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_bag.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_hips.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-train_hand.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-validation.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-test.zip"
# )

# The following files, one for train and one for validation, will be used during M2 course
lst=(
    "https://cloud.lipn.univ-paris13.fr/index.php/s/YY3AJzoc8QmnDc9/download"
    "https://cloud.lipn.univ-paris13.fr/index.php/s/nNGKXZnoQdaGkt7/download"
)

echo "[SHL Challenge] Downloading the shldataset. Depending on your bandwidth, this may take a little while."

for l in "${lst[@]}"
do
    echo "[SHL Challenge] downloading from " $l " into " $data_dir " ..."
    wget --content-disposition $l -N -P $data_dir/  # -N overwrite only if there is a new version in the server
done

if [ $? == "0" ]
then
    echo "[SHL Challenge] OK"
else
    echo "[SHL Challenge] Fatal Error"
fi
