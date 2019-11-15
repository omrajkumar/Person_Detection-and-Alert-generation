python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix training_dir/model.ckpt-11270 \
    --output_directory person_detection_inference_graph
