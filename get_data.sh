#!/bin/bash

data_dir=./data

# This list corresponds to the files used during 2020 SHL challenge
# lst=(
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_torso.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_bag.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2019/challenge-2019-train_hips.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-train_hand.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-validation.zip"
#     "http://www.shl-dataset.org/wp-content/uploads/SHLChallenge2020/challenge-2020-test.zip"
#     "https://cloud.lipn.univ-paris13.fr/index.php/s/YY3AJzoc8QmnDc9/download"
#     "https://cloud.lipn.univ-paris13.fr/index.php/s/nNGKXZnoQdaGkt7/download"
# )

# The following files, one for train and one for validation, will be used during M2 course
lst=(
    "17y9iMIdW7cuYukMhH7wepFFZ0hJqHWHQ"
)

echo "[SHL Challenge] Downloading the shldataset. Depending on your bandwidth, this may take a little while."

for l in "${lst[@]}"
do
    echo "[SHL Challenge] downloading " $l " into " $data_dir " ..."
    # wget --content-disposition $l -N -P $data_dir/  # -N overwrite only if there is a new version in the server
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id='$l -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$l -O $data_dir/sample.zip -N -P $data_dir && rm -rf /tmp/cookies.txt
done

if [ $? == "0" ]
then
    echo "[SHL Challenge] OK"
else
    echo "[SHL Challenge] Fatal Error"
fi
