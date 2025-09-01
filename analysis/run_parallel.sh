parallel -j4 --colsep ' '          'CUDA_VISIBLE_DEVICES=$(({%}-1)) python svd_recover.py --k_top {1} --k_tail {2}'          :::: params.txt
