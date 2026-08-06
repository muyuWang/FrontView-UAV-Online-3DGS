

CONFIG_PATH="configs/horizongs/HorizonGS_city_street_orb.yaml"
EXP_NAME="test_horizongs_city_street_latest"

export CUDA_VISIBLE_DEVICES=1
# python slam_new.py \
#   --config "${CONFIG_PATH}" \
#   --exp_name "${EXP_NAME}"

# RUN_DIR=$(ls -td Logs_horizongs/HorizonGS-city-street/*_"${EXP_NAME}" | head -n 1)
run_dir='/home/wmy/workspace_vla/Online-3DGS-Monocular/Logs_horizongs/HorizonGS-city-street/2026-07-17-16-07-17_test_horizongs_city_street_latest'
python render.py \
  --run_dir ${run_dir} \
  --output_dir ${run_dir} \
  --fps 15 \
  --device cuda:0 \
  --skip_novel \
  # --render_begin 173 \
  # --render_end 373 \