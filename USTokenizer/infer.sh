test_json='path/to/data/test/meta.json' #test file
gt_file='exp/inference_res/gt.scp' #ground truth
result_file='exp/inference_res/res.scp' #inference result file
export CUDA_VISIBLE_DEVICES=0
python3 inference.py --cfg-path configs/decode_config.yaml --test_json_path $test_json \
                               --result_file $result_file  --gt_file $gt_file
                            
