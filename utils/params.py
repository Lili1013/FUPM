


data_params = {
    'douban_book_movie_music_book':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/book/book_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/book/book_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/book/book_item_doc_emb_sbert.npy',
        'train_path': '../../datasets/douban_book_movie_music/book/train.csv',
        'test_path': '../../datasets/douban_book_movie_music/book/test.csv',
        'test_neg_path':'../../datasets/douban_book_movie_music/book/test.txt',
        'overlap_user_num': 901,
        'potential_pos_path':'../../datasets/douban_book_movie_music/book/user_pos_items_doc_emb_sbert_0.5.pkl',
        'u_neg_path':'../../datasets/douban_book_movie_music/book/user_neg_items_doc_emb_sbert_0.6.pkl'
    },
    'douban_book_movie_music_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/movie/movie_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/movie/movie_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/movie/movie_item_doc_emb_sbert.npy',
        'overlap_user_num': 901,
        'train_path': '../../datasets/douban_book_movie_music/movie/train.csv',
        'test_path': '../../datasets/douban_book_movie_music/movie/test.csv',
        'test_neg_path':'../../datasets/douban_book_movie_music/movie/test.txt',
        'potential_pos_path':'../../datasets/douban_book_movie_music/movie/user_pos_items_doc_emb_sbert_0.5.pkl',
        'u_neg_path':'../../datasets/douban_book_movie_music/movie/user_neg_items_doc_emb_sbert_0.6.pkl'
    },
    'douban_book_movie_music_music':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/music/music_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/music/music_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie_music/music/music_item_doc_emb_sbert.npy',
        'overlap_user_num': 901,
        'train_path': '../../datasets/douban_book_movie_music/music/train.csv',
        'test_path': '../../datasets/douban_book_movie_music/music/test.csv',
        'test_neg_path':'../../datasets/douban_book_movie_music/music/test.txt',
        'potential_pos_path':'../../datasets/douban_book_movie_music/music/user_pos_items_doc_emb_sbert_0.5.pkl',
        'u_neg_path':'../../datasets/douban_book_movie_music/music/user_neg_items_doc_emb_sbert_0.6.pkl'
    },
    'douban_book_music_book':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/book/book_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/book/book_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/book/book_item_doc_emb_sbert.npy',
        'train_path': '../../datasets/douban_book_music/book/train.csv',
        'test_path': '../../datasets/douban_book_music/book/test.csv',
        'potential_pos_path':'../../datasets/douban_book_music/book/user_pos_items_doc_emb_sbert_0.5.pkl',
        'test_neg_path':'../../datasets/douban_book_music/book/test.txt',
        'overlap_user_num': 909,
        # 'u_neg_path':'../datasets/douban_book_music/book/user_neg_items_doc_emb_sbert_0.5.pkl',
        # 'sim_item_path': '../datasets/douban_book_music/book/sim_items_0.5.pkl'
    },
    'douban_book_music_music':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/music/music_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/music/music_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/music/music_item_doc_emb_sbert.npy',
        'overlap_user_num': 909,
        'train_path': '../../datasets/douban_book_music/music/train.csv',
        'test_path': '../../datasets/douban_book_music/music/test.csv',
        'potential_pos_path':'../../datasets/douban_book_music/music/user_pos_items_doc_emb_sbert_0.5.pkl',
        'test_neg_path':'../../datasets/douban_book_music/music/test.txt',
        # 'u_neg_path':'../datasets/douban_book_music/music/user_neg_items_doc_emb_sbert_0.5.pkl',
        # 'sim_item_path': '../datasets/douban_book_music/music/sim_items_0.5.pkl'
    },
    'douban_book_movie_book':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/book/book_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/book/book_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/book/book_item_doc_emb_sbert.npy',
        'train_path': '../datasets/douban_book_movie/book/train.csv',
        'test_path': '../datasets/douban_book_movie/book/test.csv',
        'potential_pos_path':'../datasets/douban_book_movie/book/user_pos_items_doc_emb_sbert_0.5.pkl',
        'test_neg_path':'../datasets/douban_book_movie/book/test.txt',
        'overlap_user_num': 1580,
        # 'u_neg_path':'../datasets/douban_book_music/book/user_neg_items_doc_emb_sbert_0.5.pkl'
    },
    'douban_book_movie_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/movie/movie_review_data.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/movie/movie_user_doc_emb_sbert.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_movie/movie/movie_item_doc_emb_sbert.npy',
        'overlap_user_num': 1580,
        'train_path': '../datasets/douban_book_movie/movie/train.csv',
        'test_path': '../datasets/douban_book_movie/movie/test.csv',
        'potential_pos_path':'../datasets/douban_book_movie/movie/user_pos_items_doc_emb_sbert_0.5.pkl',
        'test_neg_path':'../datasets/douban_book_movie/movie/test.txt',
        # 'u_neg_path':'../datasets/douban_book_music/music/user_neg_items_doc_emb_sbert_0.5.pkl'
    },
    'amazon_book_movie_music_book':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/book/book_inter.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/book/book_user_doc_emb_256.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/book/book_item_doc_emb_256.npy',
        'overlap_user_num': 9126,
        'train_path': '../../datasets/amazon_book_movie_music/book/train.csv',
        'test_path': '../../datasets/amazon_book_movie_music/book/test.csv',
        'potential_pos_path':'../../datasets/amazon_book_movie_music/book/user_pos_items_doc_emb_256_0.5.pkl',
        'test_neg_path':'../../datasets/amazon_book_movie_music/book/test.txt',
        # 'u_neg_path':'../datasets/amazon_book_movie_music/book/user_neg_items_doc_emb_sbert_0.5.pkl'
    },
    'amazon_book_movie_music_movie':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/movie/movie_inter.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/movie/movie_user_doc_emb_256.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/movie/movie_item_doc_emb_256.npy',
        'overlap_user_num': 9126,
        'train_path': '../../datasets/amazon_book_movie_music/movie/train.csv',
        'test_path': '../../datasets/amazon_book_movie_music/movie/test.csv',
        'potential_pos_path':'../../datasets/amazon_book_movie_music/movie/user_pos_items_doc_emb_256_0.5.pkl',
        'test_neg_path':'../../datasets/amazon_book_movie_music/movie/test.txt',
        # 'u_neg_path':'../datasets/amazon_book_movie_music/book/user_neg_items_doc_emb_sbert_0.5.pkl'
    },
    'amazon_book_movie_music_music':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/music/music_inter.csv',
        'user_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/music/music_user_doc_emb_256.npy',
        'item_rev_emb_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_book_movie_music/music/music_item_doc_emb_256.npy',
        'overlap_user_num': 9126,
        'train_path': '../../datasets/amazon_book_movie_music/music/train.csv',
        'test_path': '../../datasets/amazon_book_movie_music/music/test.csv',
        'potential_pos_path':'../../datasets/amazon_book_movie_music/music/user_pos_items_doc_emb_256_0.5.pkl',
        'test_neg_path':'../../datasets/amazon_book_movie_music/music/test.txt',
        # 'u_neg_path':'../datasets/amazon_book_movie_music/book/user_neg_items_doc_emb_sbert_0.5.pkl'
    },
    'amazon_phone_sport_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/phone/phone_inter.csv',
        # 'rating_path': '../datasets/phone_sport/phone/phone_inter_new.csv',
        'overlap_user_num': 4998,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/phone/phone_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/phone/phone_item_doc_emb_256.npy',
        # 'user_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_review_feat.npy',
        # 'item_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_item_review_feat.npy',
        'train_path': '../../datasets/amazon_phone_sport/phone/train.csv',
        'test_path': '../../datasets/amazon_phone_sport/phone/test.csv',
        'test_neg_path': '../../datasets/amazon_phone_sport/phone/test.txt',
        'potential_pos_path':'../../datasets/amazon_phone_sport/phone/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/phone/train_new.csv',
        # 'test_path': '../datasets/phone_sport/phone/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/phone/test_new_doc_emb_256_0.6.txt',
        'u_neg_path':'../../datasets/amazon_phone_sport/phone/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_phone_sport_sport':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/sport/sport_inter.csv',
        # 'rating_path': '../datasets/phone_sport/sport/sport_inter_new.csv',
        'overlap_user_num': 4998,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/sport/sport_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_phone_sport/sport/sport_item_doc_emb_256.npy',
        # 'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_review_feat.npy',
        # 'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_item_review_feat.npy',
        'train_path': '../../datasets/amazon_phone_sport/sport/train.csv',
        'test_path': '../../datasets/amazon_phone_sport/sport/test.csv',
        'test_neg_path': '../../datasets/amazon_phone_sport/sport/test.txt',
        'potential_pos_path':'../../datasets/amazon_phone_sport/sport/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/sport/train_new.csv',
        # 'test_path': '../datasets/phone_sport/sport/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/sport/test_new_doc_emb_256_0.6.txt',
        'u_neg_path':'../../datasets/amazon_phone_sport/sport/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_elec_cloth_phone_elec':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/elec/elec_inter.csv',
        # 'rating_path': '../datasets/phone_sport/phone/phone_inter_new.csv',
        'overlap_user_num': 4188,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/elec/elec_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/elec/elec_item_doc_emb_256.npy',
        # 'user_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_review_feat.npy',
        # 'item_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_item_review_feat.npy',
        'train_path': '../datasets/amazon_elec_cloth_phone/elec/train.csv',
        'test_path': '../datasets/amazon_elec_cloth_phone/elec/test.csv',
        'test_neg_path': '../datasets/amazon_elec_cloth_phone/elec/test.txt',
        'potential_pos_path':'../datasets/amazon_elec_cloth_phone/elec/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/phone/train_new.csv',
        # 'test_path': '../datasets/phone_sport/phone/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/phone/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/phone/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_elec_cloth_phone_cloth':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/cloth/cloth_inter.csv',
        # 'rating_path': '../datasets/phone_sport/sport/sport_inter_new.csv',
        'overlap_user_num': 4188,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/cloth/cloth_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/cloth/cloth_item_doc_emb_256.npy',
        # 'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_review_feat.npy',
        # 'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_item_review_feat.npy',
        'train_path': '../datasets/amazon_elec_cloth_phone/cloth/train.csv',
        'test_path': '../datasets/amazon_elec_cloth_phone/cloth/test.csv',
        'test_neg_path': '../datasets/amazon_elec_cloth_phone/cloth/test.txt',
        'potential_pos_path':'../datasets/amazon_elec_cloth_phone/cloth/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/sport/train_new.csv',
        # 'test_path': '../datasets/phone_sport/sport/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/sport/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/sport/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_elec_cloth_phone_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/phone/phone_inter.csv',
        # 'rating_path': '../datasets/phone_sport/sport/sport_inter_new.csv',
        'overlap_user_num': 4188,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/phone/phone_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_elec_cloth_phone/phone/phone_item_doc_emb_256.npy',
        # 'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_review_feat.npy',
        # 'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_item_review_feat.npy',
        'train_path': '../datasets/amazon_elec_cloth_phone/phone/train.csv',
        'test_path': '../datasets/amazon_elec_cloth_phone/phone/test.csv',
        'test_neg_path': '../datasets/amazon_elec_cloth_phone/phone/test.txt',
        'potential_pos_path':'../datasets/amazon_elec_cloth_phone/phone/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/sport/train_new.csv',
        # 'test_path': '../datasets/phone_sport/sport/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/sport/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/sport/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_sport_cloth_phone_sport':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/sport/sport_inter.csv',
        # 'rating_path': '../datasets/phone_sport/phone/phone_inter_new.csv',
        'overlap_user_num': 2229,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/sport/sport_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/sport/sport_item_doc_emb_256.npy',
        # 'user_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_review_feat.npy',
        # 'item_rev_emb_path':'/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_item_review_feat.npy',
        'train_path': '../datasets/amazon_sport_cloth_phone/sport/train.csv',
        'test_path': '../datasets/amazon_sport_cloth_phone/sport/test.csv',
        'test_neg_path': '../datasets/amazon_sport_cloth_phone/sport/test.txt',
        'potential_pos_path':'../datasets/amazon_sport_cloth_phone/sport/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/phone/train_new.csv',
        # 'test_path': '../datasets/phone_sport/phone/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/phone/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/phone/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_sport_cloth_phone_cloth':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/cloth/cloth_inter.csv',
        # 'rating_path': '../datasets/phone_sport/sport/sport_inter_new.csv',
        'overlap_user_num': 2229,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/cloth/cloth_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/cloth/cloth_item_doc_emb_256.npy',
        # 'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_review_feat.npy',
        # 'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_item_review_feat.npy',
        'train_path': '../datasets/amazon_sport_cloth_phone/cloth/train.csv',
        'test_path': '../datasets/amazon_sport_cloth_phone/cloth/test.csv',
        'test_neg_path': '../datasets/amazon_sport_cloth_phone/cloth/test.txt',
        'potential_pos_path':'../datasets/amazon_sport_cloth_phone/cloth/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/sport/train_new.csv',
        # 'test_path': '../datasets/phone_sport/sport/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/sport/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/sport/user_neg_items_doc_emb_256_0.6.pkl'
    },
    'amazon_sport_cloth_phone_phone':{
        'rating_path': '/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/phone/phone_inter.csv',
        # 'rating_path': '../datasets/phone_sport/sport/sport_inter_new.csv',
        'overlap_user_num': 2229,
        'user_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/phone/phone_user_doc_emb_256.npy',
        'item_rev_emb_path':'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/amazon_sport_cloth_phone/phone/phone_item_doc_emb_256.npy',
        # 'user_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_review_feat.npy',
        # 'item_rev_emb_path': '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_item_review_feat.npy',
        'train_path': '../datasets/amazon_sport_cloth_phone/phone/train.csv',
        'test_path': '../datasets/amazon_sport_cloth_phone/phone/test.csv',
        'test_neg_path': '../datasets/amazon_sport_cloth_phone/phone/test.txt',
        'potential_pos_path':'../datasets/amazon_sport_cloth_phone/phone/user_pos_items_doc_emb_256_0.5.pkl',
        # 'train_path': '../datasets/phone_sport/sport/train_new.csv',
        # 'test_path': '../datasets/phone_sport/sport/test_new.csv',
        # 'test_neg_path': '../datasets/amazon_phone_sport/sport/test_new_doc_emb_256_0.6.txt',
        # 'u_neg_path':'../datasets/amazon_phone_sport/sport/user_neg_items_doc_emb_256_0.6.pkl'
    },

}

model_params = {
    'douban_book_movie_music_book':{
        'user_num' : 901,
        'item_num' : 3222,
    },
    'douban_book_movie_music_movie':{
        'user_num' : 901,
        'item_num' : 14501,
    },
    'douban_book_movie_music_music':{
        'user_num' : 901,
        'item_num' : 2546,
    },
    'douban_book_music_book':{
        'user_num' : 909,
        'item_num' : 3222,
    },
    'douban_book_music_music':{
        'user_num' : 909,
        'item_num' : 2546,
    },
    'douban_book_movie_book':{
        'user_num' : 1580,
        'item_num' : 3222,
    },
    'douban_book_movie_movie':{
        'user_num' : 1580,
        'item_num' : 8356,
    },
    'amazon_phone_sport_sport':{
        'user_num':4998,
        'item_num':22101
    },
    'amazon_phone_sport_phone':{
        'user_num':4998,
        'item_num':14618
    },
    'amazon_book_movie_music_book': {
        'user_num': 9126,
        'item_num': 163172
    },
    'amazon_book_movie_music_movie': {
        'user_num': 9126,
        'item_num': 44390
    },
    'amazon_book_movie_music_music': {
        'user_num': 9126,
        'item_num': 57448
    },
    'amazon_elec_cloth_phone_elec': {
        'user_num': 4188,
        'item_num': 34130
    },
    'amazon_elec_cloth_phone_cloth': {
        'user_num': 4188,
        'item_num': 24832
    },
    'amazon_elec_cloth_phone_phone': {
        'user_num': 4188,
        'item_num': 14553
    },
    'amazon_sport_cloth_phone_sport': {
        'user_num': 2229,
        'item_num': 15321
    },
    'amazon_sport_cloth_phone_cloth': {
        'user_num': 2229,
        'item_num': 16664
    },
    'amazon_sport_cloth_phone_phone': {
        'user_num': 2229,
        'item_num': 9789
    },


}
#
#
# data_params = {
#     'douban_book':{
#         'rating_path': 'datasets/debug_datasets/douban_book/book_review_data_new.csv',
#         'user_rev_emb_path': '../datasets/debug_datasets/douban_book/doc_embs/book_user_doc_emb_32.npy',
#         'item_rev_emb_path': '../datasets/debug_datasets/douban_book/doc_embs/book_item_doc_emb_32.npy',
#         'train_path': 'datasets/debug_datasets/douban_book/train.csv',
#         'test_path': 'datasets/debug_datasets/douban_book/test.csv',
#         'test_neg_path': 'datasets/debug_datasets/douban_book/test.txt',
#         'overlap_user_num': 6
#     },
#     'douban_movie':{
#         'rating_path': 'datasets/debug_datasets/douban_movie/movie_review_data_new.csv',
#         'user_rev_emb_path': 'datasets/debug_datasets/douban_movie/doc_embs/movie_user_doc_emb_32.npy',
#         'item_rev_emb_path': 'datasets/debug_datasets/douban_movie/doc_embs/movie_item_doc_emb_32.npy',
#         'train_path': 'datasets/debug_datasets/douban_movie/train.csv',
#         'test_path': 'datasets/debug_datasets/douban_movie/test.csv',
#         'test_neg_path': 'datasets/debug_datasets/douban_movie/test.txt',
#         'overlap_user_num': 6
#     },
#     'douban_music':{
#         'rating_path': 'datasets/debug_datasets/douban_music/music_review_data_new.csv',
#         'user_rev_emb_path': 'datasets/debug_datasets/douban_music/doc_embs/music_user_doc_emb_32.npy',
#         'item_rev_emb_path': 'datasets/debug_datasets/douban_music/doc_embs/music_item_doc_emb_32.npy',
#         'train_path': 'datasets/debug_datasets/douban_music/train.csv',
#         'test_path': 'datasets/debug_datasets/douban_music/test.csv',
#         'test_neg_path': 'datasets/debug_datasets/douban_music/test.txt',
#         'overlap_user_num': 6
#     }
# }
# model_params = {
#     'douban_book':{
#         'user_num' : 78,
#         'item_num' : 8776,
#     },
#     'douban_movie':{
#         'user_num' : 13,
#         'item_num' : 6054,
#     },
#     'douban_music':{
#         'user_num' : 95,
#         'item_num' : 8664,
#     },
# }