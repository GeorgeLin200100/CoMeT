#! /bin/bash

input_files=(
    "gpt3"
    "falcon"
    "llama2_70B"
    "llama2_1B"
    "llama2_3B"
    "llama2_7B"
    "llama2_13B"
)

prefix="exp4_input"

output_prefix="exp4_output"

# # base on the input_file of 512BW, create the input_file of other bandwidths if not exist
# # modify the last column of the input_file to match the bandwidth
# for bandwidth in 128 256 512 1024 2048; do
#     for input_file in "${input_files[@]}"; do
#         input_file_512BW="${prefix}_${input_file}_512BW.csv"
#         if [ ! -f ${input_file_512BW} ]; then
#             echo "Error: input_file_512BW ${input_file_512BW} does not exist"
#             exit 1
#         fi
#         # modify the last column of the input_file to match the bandwidth and save as new file
#         new_input_file="${prefix}_${input_file}_${bandwidth}BW.csv"
#         cp ${input_file_512BW} ${new_input_file}
#         sed -i "s/512/${bandwidth}/g" ${new_input_file}
#     done
# done

# for input_file in "${input_files[@]}"; do
#     output_file="${output_prefix}_${input_file}"
#     for bandwidth in 128 256 512 1024 2048; do
#         python3 test_liu.py -d 0 \
#             --input_file ${prefix}_${input_file}_${bandwidth}BW.csv \
#             -o ${output_file}_${bandwidth}BW.csv \
#             -od ${output_file}_${bandwidth}BW \
#             -cf exp4_cfg.cfg -r 10 -nf True \
#             --distribute-across-group --distribution-noise 0
#     done
# done

for input_file in "${input_files[@]}"; do
    for bandwidth in 128 256 512 1024 2048; do
        python3 plot_max_temperature.py \
            ${output_prefix}_${input_file}_${bandwidth}BW \
            --output ${output_prefix}_${input_file}_${bandwidth}BW_max_temp.png \
            --no-peaks
    done
done