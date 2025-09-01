parallel -j4 --colsep ' '          'CUDA_VISIBLE_DEVICES=$(({%}-1)) python svd_recover_dir-qwen.py --k_top {1}'          :::: params-dir.txt
